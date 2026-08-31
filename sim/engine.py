"""Discrete-event simulation of a continuous-batching LLM serving engine,
plus agent-side client callbacks that chain requests per topology.

Event kinds:
  'prefill'         -> request moved from queue into prefill→decode stage
  'decode'          -> one decode step for a request
  'agent_callback'  -> agent-side logic fires (submits next request)
"""

import heapq
from collections import deque
from dataclasses import dataclass, field

from kvcache import KVCacheManager


# ----------------------------- prefix cache ------------------------------- #

class PrefixCache:
    """Radix-style longest prefix matcher (hash-key approximation)."""

    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.table = {}
        self.order = deque()
        self.hits = 0
        self.lookups = 0
        self.tokens_saved = 0

    def insert(self, seg_ids, seg_tokens):
        cum = 0
        prefix = []
        for sid, tok in zip(seg_ids, seg_tokens):
            prefix.append(sid)
            cum += tok
            key = tuple(prefix)
            if key not in self.table:
                self.table[key] = cum
                self.order.append(key)
        while len(self.table) > self.capacity and self.order:
            old = self.order.popleft()
            self.table.pop(old, None)

    def longest_match(self, seg_ids):
        self.lookups += 1
        cum_best = 0
        prefix = []
        for sid in seg_ids:
            prefix.append(sid)
            key = tuple(prefix)
            if key in self.table:
                cum_best = self.table[key]
            else:
                break
        if cum_best:
            self.hits += 1
            self.tokens_saved += cum_best
        return cum_best


# ------------------------------- request ---------------------------------- #

@dataclass
class Request:
    req_id: int
    agent_id: int
    topology: str
    seg_ids: list
    seg_tokens: list
    prompt_tokens: int
    output_tokens: int
    on_complete: object = None
    meta: dict = field(default_factory=dict)

    arrival: float = 0.0
    prefill_done: float = 0.0
    decode_done: float = 0.0
    cached_tokens: int = 0

    def latency(self):
        return self.decode_done - self.arrival

    def ttft(self):
        return self.prefill_done - self.arrival


# -------------------------------- engine ---------------------------------- #

class Engine:
    def __init__(self, prefill_tok_s, decode_step_s, max_batch, cache=None,
                 weights_bytes=None, kv_bytes_per_token=None,
                 bandwidth_bytes_per_s=None):
        self.prefill_tok_s = prefill_tok_s
        self.decode_step_s = decode_step_s   # floor: weight-read time per step
        self.max_batch = max_batch
        self.cache = cache
        # memory-bandwidth contention model (optional)
        self.weights_bytes = weights_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.bandwidth = bandwidth_bytes_per_s

        self.queue = deque()
        self.running = []
        self.time = 0.0
        # events: (time, priority, kind, payload); priority 0=prefill/decode, 2=callback
        self.events = []
        self.completed = []
        self.trace = []
        self.total_prefill_tokens = 0
        self.prefill_busy_until = 0.0
        self._req_counter = 0
        self._seq = 0  # global tiebreaker for events

    def next_req_id(self):
        self._req_counter += 1
        return self._req_counter

    def _push(self, time, priority, kind, payload):
        # events: (time, priority, seq, kind, payload) -- seq breaks ties
        self._seq += 1
        heapq.heappush(self.events, (time, priority, self._seq, kind, payload))

    # ---------------- submission / scheduling ---------------- #

    def submit(self, req: Request, now: float):
        req.arrival = now
        self.queue.append(req)
        self._maybe_start_prefill(now)

    def _maybe_start_prefill(self, now):
        if not self.queue:
            return
        if len(self.running) >= self.max_batch:
            return
        if now < self.prefill_busy_until:
            return
        self._start_prefill(now)

    def _start_prefill(self, now):
        req = self.queue[0]  # peek; only pop if admitted
        compute_tokens = req.prompt_tokens
        if self.cache is not None:
            cached = self.cache.match_prefix(req.seg_ids, req.seg_tokens)
            req.cached_tokens = cached
            compute_tokens = max(0, req.prompt_tokens - cached)
            # admission control: full context footprint must fit in KV pool
            held = self.cache.try_allocate(req.seg_ids, req.seg_tokens, cached)
            if held is None:
                # pool full of live sequences + retained prefixes; block until
                # a completion frees blocks (event-driven retry, no polling).
                self.blocked_admissions = getattr(self, "blocked_admissions", 0) + 1
                return
            req.meta["held_blocks"] = held
        self.queue.popleft()
        started_from = max(now, self.prefill_busy_until)
        duration = compute_tokens / self.prefill_tok_s
        done_at = started_from + duration
        req.prefill_done = done_at
        req.meta["prefill_compute_tokens"] = compute_tokens
        self.prefill_busy_until = done_at
        self._push(done_at, 0, "prefill", req)
        self.total_prefill_tokens += compute_tokens

    # ---------------- event handlers ---------------- #

    def _handle_prefill(self, req, now):
        # admission (KV) was already granted at prefill start; decode batch
        # capacity was also checked. If max_batch filled since (rare race),
        # requeue and retry on next completion.
        if len(self.running) >= self.max_batch:
            self.queue.appendleft(req)
            return
        self.running.append(req)
        req.meta["decode_remaining"] = req.output_tokens
        # ensure a global decode ticker exists (use current step time)
        if not any(k == "decode_tick" for _, _, _, k, _ in self.events):
            self._push(now + self._current_step_time(), 0, "decode_tick", None)

    def _current_step_time(self):
        """Decode step duration under memory-bandwidth contention.

        Floor is `decode_step_s` (weight-read). If a bandwidth model is set,
        KV reads across the running batch can push the step longer.
        """
        if self.bandwidth is None:
            return self.decode_step_s
        kv_bytes = sum(r.prompt_tokens + (r.output_tokens - r.meta["decode_remaining"])
                       for r in self.running) * self.kv_bytes_per_token
        kv_time = kv_bytes / self.bandwidth
        return max(self.decode_step_s, kv_time)

    def _handle_decode_tick(self, now):
        """One batch-wide decode step: every running request produces 1 token."""
        step = self._current_step_time()
        finished = []
        for r in list(self.running):
            r.meta["decode_remaining"] -= 1
            if r.meta["decode_remaining"] <= 0:
                finished.append(r)
        for r in finished:
            r.decode_done = now
            self.running.remove(r)
            self.completed.append(r)
            if self.cache is not None and r.meta.get("held_blocks"):
                self.cache.release(r.meta["held_blocks"])
            if r.on_complete:
                r.on_complete(self, r, now)
        if finished:
            self._maybe_start_prefill(now)
        if self.running:
            self._push(now + step, 0, "decode_tick", None)

    def _handle_retry_start(self, now):
        self._maybe_start_prefill(now)

    # ---------------- main loop ---------------- #

    def step(self) -> bool:
        if not self.events:
            return False
        t, _, _, kind, payload = heapq.heappop(self.events)
        self.time = max(self.time, t)
        if kind == "prefill":
            self._handle_prefill(payload, t)
        elif kind == "decode_tick":
            self._handle_decode_tick(t)
        elif kind == "retry_start":
            self._handle_retry_start(t)
        elif kind == "agent_callback":
            payload(t)  # agent logic; payload is a callable(now)
        return True

    def snapshot(self):
        running_summary = {}
        for r in self.running:
            running_summary[r.topology] = running_summary.get(r.topology, 0) + 1
        snap = {
            "time": self.time,
            "queued": len(self.queue),
            "running": len(self.running),
            "completed": len(self.completed),
            "running_by_topology": dict(running_summary),
        }
        if self.cache is not None:
            snap["cache_blocks_used"] = self.cache.cached_block_count()
            snap["cache_evictions"] = self.cache.evictions
        return snap

    def run(self, record_trace: bool = True):
        while self.events:
            self.step()
            if record_trace:
                self.trace.append(self.snapshot())
        return self.completed
