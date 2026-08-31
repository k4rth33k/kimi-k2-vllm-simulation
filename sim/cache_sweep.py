"""Sweep KV cache budget to show hit-rate vs capacity (the eviction cliff)."""

import json
import random

from engine import Engine
from kvcache import KVCacheManager
from workloads import TaskAgent
import run_experiments as R


def run(tasks, topology, cache_gb, rng):
    num_blocks = int(cache_gb * 1e9 / R.KV_BYTES_PER_TOKEN) // R.BLOCK_SIZE
    cache = KVCacheManager(num_blocks, R.BLOCK_SIZE)
    engine = Engine(R.PREFILL_TOK_S, R.DECODE_STEP_S, R.MAX_BATCH, cache=cache)

    def tool_delay(group):
        base = R.TOOL_DELAY_MEAN
        if group in ("local", "terminal", "memory"):
            base = R.TOOL_DELAY_MEAN * 0.3
        return rng.expovariate(1 / max(base, 0.01))

    agents = [TaskAgent(engine, t, topology, tool_delay, R.ORCH_CAP,
                        R.SCHEMA_TOKENS_PER_GROUP) for t in tasks]
    for a in agents:
        a.start(now=0.0)
    engine.run(record_trace=False)
    s = R.summarize(engine, engine.completed)
    return {
        "cache_gb": cache_gb,
        "hit_rate": s["cache_hit_rate"],
        "makespan_s": s["makespan_s"],
        "evictions": s["cache_evictions"],
        "blocks": num_blocks,
    }


def main():
    tasks = json.load(open("results/token_stats_kimi.json"))[:80]
    budgets = [10, 25, 50, 100, 200, 476]
    out = {}
    for topo in ["react", "orchestrator"]:
        rows = []
        for gb in budgets:
            r = run(tasks, topo, gb, random.Random(42))
            rows.append(r)
            print(f"{topo} cache={gb}GB: hit={r['hit_rate']:.3f} "
                  f"makespan={r['makespan_s']:.0f}s evict={r['evictions']}")
        out[topo] = rows
    json.dump(out, open("results/cache_sweep.json", "w"), indent=2)
    print("saved cache_sweep.json")


if __name__ == "__main__":
    main()
