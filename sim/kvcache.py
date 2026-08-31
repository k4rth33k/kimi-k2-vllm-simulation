"""vLLM-faithful KV cache block manager with automatic prefix caching (APC).

Model of vLLM v1's KVCacheManager + BlockPool:
  - GPU KV memory is a fixed pool of `num_blocks` physical blocks, each holding
    `block_size` tokens (vLLM default 16).
  - A prompt is chunked into block-sized pieces; each block is identified by a
    hash chained over its parent (content-addressable, prefix-bound).
  - Cached blocks with refcount 0 live in a free queue (LRU). Allocation pops
    from the free-queue head; if that block is cached, it is evicted (hash
    removed) before reuse.
  - Blocks referenced by a running request have refcount > 0 and cannot be
    evicted. When a request finishes, its blocks' refcounts drop; blocks whose
    refcount hits 0 re-enter the free queue (tail, reverse order) but STAY
    CACHED until evicted -- that is what enables reuse by later requests.

This is a token-accounting simulation: we don't store KV tensors, only block
occupancy, so we can answer "how many prompt tokens were cache hits".
"""

from collections import OrderedDict


class Block:
    __slots__ = ("block_id", "hash", "refcount", "n_tokens")

    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = None        # content hash (chained); None => not cached
        self.refcount = 0
        self.n_tokens = 0       # tokens stored (== block_size for full blocks)


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.cached = {}                     # hash -> Block
        # free queue as OrderedDict[block_id -> Block], head = LRU.
        # Contains blocks with refcount == 0 (both never-used and cached).
        self.free = OrderedDict((b.block_id, b) for b in self.blocks)
        self.hits = 0               # blocks reused
        self.tokens_saved = 0
        self.evictions = 0
        self.peak_used = 0          # peak physically-occupied blocks

    def occupied(self):
        """Blocks physically holding KV = not in the free pool."""
        return len(self.blocks) - len(self.free)

    # ------------------------- hashing ------------------------- #

    def _hash(self, parent_hash, seg_id, block_index):
        return hash((parent_hash, seg_id, block_index))

    # ------------------------- lookup ------------------------- #

    def match_prefix(self, seg_ids, seg_tokens):
        """Return number of prompt tokens covered by cached full blocks."""
        parent = None
        cached_tokens = 0
        for block_index, (seg_id, n_tok) in enumerate(
                self._blocks_of(seg_ids, seg_tokens)):
            h = self._hash(parent, seg_id, block_index)
            blk = self.cached.get(h)
            if blk is None:
                break  # prefix chain broken; deeper blocks can't match
            cached_tokens += blk.n_tokens
            self.hits += 1
            self.tokens_saved += blk.n_tokens
            parent = h
        return cached_tokens

    def _blocks_of(self, seg_ids, seg_tokens):
        """Yield (seg_id, block_size) for each FULL block within a segment.

        Blocks never straddle segment boundaries; a trailing partial block is
        not cacheable (mirrors vLLM, which caches only full blocks).
        """
        for sid, n in zip(seg_ids, seg_tokens):
            for _ in range(n // self.block_size):
                yield (sid, self.block_size)

    # ----------------------- allocation ----------------------- #

    def free_count(self):
        return len(self.free)

    def blocks_needed(self, seg_tokens, cached_tokens):
        """Blocks this request must hold to decode: full prompt + one output
        block of headroom (decode grows into it). Cached prefix blocks are
        shared/refcounted, so they don't need fresh allocation."""
        prompt_blocks = sum(n // self.block_size for n in seg_tokens)
        cached_blocks = cached_tokens // self.block_size
        return max(0, prompt_blocks - cached_blocks) + 1  # +1 decode headroom

    def try_allocate(self, seg_ids, seg_tokens, cached_tokens):
        """Admission control: allocate only if the request's full context
        footprint fits. Returns held Blocks, or None if it doesn't fit."""
        need = self.blocks_needed(seg_tokens, cached_tokens)
        if need > self.free_count():
            return None
        held = []
        parent = None
        for block_index, (seg_id, n_tok) in enumerate(
                self._blocks_of(seg_ids, seg_tokens)):
            h = self._hash(parent, seg_id, block_index)
            existing = self.cached.get(h)
            if existing is not None:
                existing.refcount += 1
                self.free.pop(existing.block_id, None)
                held.append(existing)
                parent = h
                continue
            blk = self._alloc_block()
            if blk is None:
                # shouldn't happen given the check, but be safe
                break
            blk.hash = h
            blk.n_tokens = n_tok
            blk.refcount = 1
            self.cached[h] = blk
            held.append(blk)
            parent = h
        # one extra block for decode growth
        extra = self._alloc_block()
        if extra is not None:
            extra.refcount = 1
            held.append(extra)
        self.peak_used = max(self.peak_used, self.occupied())
        return held

    def _alloc_block(self):
        while self.free:
            _, blk = self.free.popitem(last=False)   # LRU head
            if blk.hash is not None:
                # evict cached block before reuse; memory stays occupied
                self.cached.pop(blk.hash, None)
                blk.hash = None
                self.evictions += 1
            blk.n_tokens = 0
            return blk
        return None

    def release(self, held_blocks):
        """Request finished: drop refs; refcount-0 blocks return to free queue
        (staying cached). Reverse order so deepest blocks evict first."""
        for blk in reversed(held_blocks):
            blk.refcount -= 1
            if blk.refcount == 0:
                self.free[blk.block_id] = blk

    # ------------------------- stats ------------------------- #

    def cached_block_count(self):
        return len(self.cached)
