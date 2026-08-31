"""Run the experiments: react vs orchestrator, cache on/off.

Fixed batch: all tasks submit at t=0; measure makespan + per-request stats.

Hardware defaults ~ 27B dense-equivalent on 1x H200 (tunable below).
"""

import json
import math
import random
import argparse
import statistics

from engine import Engine
from kvcache import KVCacheManager
from workloads import TaskAgent

# ---- 8xB200 + Kimi K2 (FP8) calibrated knobs ---- #
# Decode: SemiAnalysis B200 ~4,000 tok/s/GPU at conc 64 => ~32k tok/s/node.
#   With max_batch=128 decode steps, per-step time ~ max_batch/32000 s.
# Prefill: chunked prefill on 8xB200 for 32B-active MoE ~ 60k tok/s aggregate.
PREFILL_TOK_S = 60000.0    # aggregate prefill tokens/s (node)
DECODE_STEP_S = 0.004      # s per decode step; at batch 128 => 32k tok/s
MAX_BATCH = 128

# ---- KV cache budget: 8x192GB - 1000GB FP8 weights - 60GB overhead ---- #
KV_BUDGET_GB = 476.0
KV_BYTES_PER_TOKEN = 70272          # MLA: 61 * (512+64) * 2 bytes
KV_CAPACITY_TOKENS = int(KV_BUDGET_GB * 1e9 / KV_BYTES_PER_TOKEN)
BLOCK_SIZE = 16
NUM_BLOCKS = KV_CAPACITY_TOKENS // BLOCK_SIZE

# ---- memory-bandwidth contention (decode) ---- #
AGG_BANDWIDTH = 8 * 8e12            # 8x B200 @ 8 TB/s
WEIGHTS_BYTES = 1000e9              # FP8 1T params

# ---- agent-side knobs ---- #
TOOL_DELAY_MEAN = 0.6
ORCH_CAP = 8
SCHEMA_TOKENS_PER_GROUP = 64


def make_tool_delay(rng):
    """Exponential-ish tool delays; 'local' tools quicker, net APIs slower."""
    def fn(group):
        base = TOOL_DELAY_MEAN
        if group in ("local", "terminal", "memory"):
            base = TOOL_DELAY_MEAN * 0.3
        return rng.expovariate(1 / max(base, 0.01))
    return fn


def run_config(tasks, topology, cache_on, rng, record_trace=False,
               bw_contention=True):
    cache = KVCacheManager(NUM_BLOCKS, BLOCK_SIZE) if cache_on else None
    kw = dict(weights_bytes=WEIGHTS_BYTES,
              kv_bytes_per_token=KV_BYTES_PER_TOKEN,
              bandwidth_bytes_per_s=AGG_BANDWIDTH) if bw_contention else {}
    engine = Engine(PREFILL_TOK_S, DECODE_STEP_S, MAX_BATCH, cache=cache, **kw)
    agents = [
        TaskAgent(engine, t, topology, make_tool_delay(rng), ORCH_CAP,
                  SCHEMA_TOKENS_PER_GROUP)
        for t in tasks
    ]
    for a in agents:
        a.start(now=0.0)
    completed = engine.run(record_trace=record_trace)
    return engine, completed


def summarize(engine, completed):
    latencies = [r.latency() for r in completed]
    ttfts = [r.ttft() for r in completed]
    prompts = [r.prompt_tokens for r in completed]
    cached = [r.cached_tokens for r in completed]
    outputs = [r.meta.get("output_tokens0", 0) for r in completed]

    total_prompt_tokens = sum(r.prompt_tokens for r in completed)
    total_cached = sum(r.cached_tokens for r in completed)
    makespan = max(r.decode_done for r in completed) if completed else 0

    def pct(xs, p):
        xs = sorted(xs)
        k = max(0, math.ceil(p * len(xs)) - 1)
        return xs[k] if xs else 0

    out = {
        "n_requests": len(completed),
        "makespan_s": round(makespan, 3),
        "throughput_req_s": round(len(completed) / makespan, 3) if makespan else 0,
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "ttft_mean_s": round(statistics.mean(ttfts), 3),
        "latency_p50_s": round(pct(latencies, 0.5), 3),
        "latency_p90_s": round(pct(latencies, 0.9), 3),
        "ttft_p50_s": round(pct(ttfts, 0.5), 3),
        "ttft_p90_s": round(pct(ttfts, 0.9), 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached,
        "cache_hit_rate": round(total_cached / total_prompt_tokens, 4) if total_prompt_tokens else 0,
        "total_prefill_tokens": int(engine.total_prefill_tokens),
    }
    if engine.cache is not None:
        out["cache_block_hits"] = engine.cache.hits
        out["cache_evictions"] = engine.cache.evictions
        out["cache_blocks_total"] = NUM_BLOCKS
        out["cache_blocks_peak_used"] = engine.cache.peak_used
        out["blocked_admissions"] = getattr(engine, "blocked_admissions", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/token_stats_kimi.json")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--trace-out", default="results/trace_{name}.json")
    args = ap.parse_args()

    tasks = json.load(open(args.stats))
    # Use a uniform workload subset so topologies see identical tasks
    rng = random.Random(42)

    results = {}
    for topology in ["react", "orchestrator"]:
        for cache_on in [False, True]:
            name = f"{topology}_{'cache' if cache_on else 'nocache'}"
            rng = random.Random(42)  # same tool-delay stream per config
            engine, completed = run_config(tasks, topology, cache_on, rng,
                                           record_trace=True)
            s = summarize(engine, completed)
            results[name] = s
            print(f"\n{name}:")
            for k, v in s.items():
                print(f"  {k:24s}: {v}")
            # Save per-request table (compact) + trace snapshots
            with open(args.trace_out.format(name=name), "w") as f:
                json.dump({
                    "summary": s,
                    "requests": [
                        {
                            "arrival": round(r.arrival, 4),
                            "prefill_done": round(r.prefill_done, 4),
                            "decode_done": round(r.decode_done, 4),
                            "prompt_tokens": r.prompt_tokens,
                            "cached_tokens": r.cached_tokens,
                            "task": r.meta.get("task"),
                            "group": r.meta.get("group"),
                            "step": r.meta.get("step"),
                        }
                        for r in completed
                    ],
                    "trace": engine.trace[:: max(1, len(engine.trace)//4000)],
                }, f)
            print(f"  saved trace -> {args.trace_out.format(name=name)}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved summary ->", args.out)


if __name__ == "__main__":
    main()
