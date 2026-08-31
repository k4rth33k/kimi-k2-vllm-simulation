"""Sensitivity sweeps over key knobs (task subset for speed)."""

import json
import random

from engine import Engine, PrefixCache
from workloads import TaskAgent
import run_experiments as R


def run(tasks, topology, cache_on, rng, overrides):
    prefill = overrides.get("prefill", R.PREFILL_TOK_S)
    decode = overrides.get("decode", R.DECODE_STEP_S)
    max_batch = overrides.get("max_batch", R.MAX_BATCH)
    tool_mean = overrides.get("tool_mean", R.TOOL_DELAY_MEAN)
    orch_cap = overrides.get("orch_cap", R.ORCH_CAP)

    cache = PrefixCache() if cache_on else None
    engine = Engine(prefill, decode, max_batch, cache=cache)

    def tool_delay(group):
        base = tool_mean
        if group in ("local", "terminal", "memory"):
            base = tool_mean * 0.3
        return rng.expovariate(1 / max(base, 0.01))

    agents = [TaskAgent(engine, t, topology, tool_delay, orch_cap,
                        R.SCHEMA_TOKENS_PER_GROUP) for t in tasks]
    for a in agents:
        a.start(now=0.0)
    engine.run(record_trace=False)
    return R.summarize(engine, engine.completed)


def main():
    tasks = json.load(open("results/token_stats_kimi.json"))[:60]  # subset
    sweeps = [
        ("tool_mean", [0.2, 0.6, 1.5]),
        ("max_batch", [64, 128, 256]),
        ("prefill", [10000, 20000, 40000]),
        ("orch_cap", [2, 8, 16]),
        ("decode", [0.005, 0.010, 0.020]),
    ]
    out = {}
    for knob, values in sweeps:
        for v in values:
            for topo in ["react", "orchestrator"]:
                for cache_on in [False, True]:
                    name = f"{topo}_{'cache' if cache_on else 'nocache'}"
                    rng = random.Random(42)
                    over = {knob: v}
                    try:
                        cfg = run(tasks, topo, cache_on, rng, over)
                        out.setdefault(knob, {}).setdefault(str(v), {})[name] = {
                            "makespan_s": cfg["makespan_s"],
                            "latency_mean_s": cfg["latency_mean_s"],
                        }
                        print(f"{knob}={v} {name}: makespan={cfg['makespan_s']}s")
                    except Exception as e:
                        print(f"{knob}={v} {name}: ERROR {e}")
    json.dump(out, open("results/sweeps.json", "w"), indent=2)
    print("saved sweeps")


if __name__ == "__main__":
    main()
