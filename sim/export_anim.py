"""Export compact animation data for the blog from simulation traces.

Output per config: public/blog-data/<name>.json
  meta : summary stats (makespan, hit rate, evictions, ...)
  req  : per-request rows [arrival, prefill_start, prefill_done, decode_done,
         prompt_tokens, cached_tokens, tool_group_id]
  snaps: engine snapshots [t, queued, running, completed, cache_used, evictions]
  groups: tool group id -> name

Times in milliseconds (ints) to keep the files small.
"""

import json
import os

PREFILL_TOK_S = 60000.0
OUT_DIR = "results/blog-data"

CONFIGS = ["react_nocache", "react_cache", "orchestrator_nocache", "orchestrator_cache"]


def ms(x):
    return int(round(x * 1000))


def export(name):
    tr = json.load(open(f"results/trace_{name}.json"))
    reqs = tr["requests"]

    groups = sorted({r["group"] or "_none" for r in reqs})
    gid = {g: i for i, g in enumerate(groups)}

    rows = []
    for r in reqs:
        compute = r["prompt_tokens"] - r["cached_tokens"]
        prefill_start = r["prefill_done"] - compute / PREFILL_TOK_S
        rows.append([
            ms(r["arrival"]),
            ms(prefill_start),
            ms(r["prefill_done"]),
            ms(r["decode_done"]),
            r["prompt_tokens"],
            r["cached_tokens"],
            gid[r["group"] or "_none"],
        ])
    rows.sort(key=lambda x: x[0])

    snaps = []
    for s in tr["trace"]:
        snaps.append([
            ms(s["time"]),
            s["queued"],
            s["running"],
            s["completed"],
            s.get("cache_blocks_used", 0),
            s.get("cache_evictions", 0),
        ])

    out = {
        "meta": tr["summary"],
        "groups": groups,
        "req": rows,
        "snaps": snaps,
    }
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{name}: {len(rows)} requests, {len(snaps)} snaps, "
          f"{os.path.getsize(path)/1e6:.2f} MB")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for c in CONFIGS:
        export(c)
