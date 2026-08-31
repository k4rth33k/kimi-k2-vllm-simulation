# Agent topology vs serving — a vLLM-style simulation with Kimi K2 on 8×B200

This repo contains the code used to create the blog post
[Agent architectures & model serving](https://www.kartheeksurampudi.com/blog/agent-architectures-model-serving).

Discrete-event simulation of two agent topologies (**ReAct** vs **Orchestrator
+ parallel subagents**) replaying real LLM request sequences extracted from the
[Toolathlon-Trajectories](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories)
dataset (`kimi-k2-0905` runs), served by a vLLM-faithful engine
(continuous batching, chunked prefill, paged prefix KV cache with
hash-chained blocks, LRU eviction) calibrated for **Kimi K2 FP8 on 8×B200**.

Full findings and numbers are in [REPORT.md](REPORT.md).

## Repository layout

```
analysis/
  analyze_trajectories.py   # extract per-request token sequences from the dataset
sim/
  workloads.py              # ReAct / orchestrator agent request generators
  kvcache.py                # vLLM-faithful paged prefix cache (APC)
  engine.py                 # discrete-event continuous-batching engine
  run_experiments.py        # main experiment: {react,orchestrator} × {cache,nocache}
  cache_sweep.py            # hit-rate vs cache budget (the "eviction cliff")
  sweep.py                  # sensitivity sweeps over engine/agent knobs
  export_anim.py            # compact per-request animation data for the blog
REPORT.md                   # full writeup of results
```

## 1. Setup

Requires Python 3.10+. The simulator (`sim/`) is stdlib-only; the trajectory
analysis needs `tiktoken`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken
```

## 2. Download the dataset

The simulation is driven by the Toolathlon trajectories (51 JSONL files,
~2 GB total). Only the three `kimi-k2-0905` files (~90 MB) are needed to
reproduce the reported numbers.

Using `hf` (Hugging Face CLI, `pip install -U "huggingface_hub[cli]"`):

```bash
hf download hkust-nlp/Toolathlon-Trajectories \
  --repo-type dataset \
  --include "kimi-k2-0905*" \
  --local-dir toolathon_trajectories
```

or with the legacy CLI: `huggingface-cli download ...` with the same args.

To grab the full dataset instead, drop the `--include` filter. The folder must
end up at `toolathon_trajectories/` in the repo root (the analysis globs
`toolathon_trajectories/*.jsonl`). The dataset is CC-BY-4.0, from
[HKUST-NLP](https://github.com/hkust-nlp/Toolathlon).

## 3. Run the pipeline

All commands run from the repo root.

```bash
# 1. Tokenize trajectories into per-request sequences (kimi-k2-0905 runs)
python analysis/analyze_trajectories.py \
  --runs kimi-k2-0905 --out results/token_stats_kimi.json

# 2. Headline experiment: ReAct vs orchestrator, cache on/off (295 tasks)
python sim/run_experiments.py
#   -> results/summary.json, results/trace_{config}.json

# 3. Cache-budget sweep (eviction cliff, 80-task subset)
python sim/cache_sweep.py
#   -> results/cache_sweep.json

# 4. (optional) Sensitivity sweeps over engine knobs
python sim/sweep.py
#   -> results/sweeps.json

# 5. (optional) Compact animation data for the blog
python sim/export_anim.py
#   -> results/blog-data/{react,orchestrator}_{nocache,cache}.json
```

Headline result (makespan for 295 tasks submitted at t=0, shared 476 GB KV pool):

| Configuration          | Makespan | Speedup | Cache hit |
|------------------------|----------|---------|-----------|
| ReAct, no cache        | 6,693s   | 1.0×    | —         |
| ReAct, cache           | 669s     | 10.0×   | 91.1%     |
| Orchestrator, no cache | 2,998s   | 2.2×    | —         |
| Orchestrator, cache    | 455s     | 14.7×   | 87.2%     |

## Notes

- The dataset folder (`toolathon_trajectories/`) and virtualenv are git-ignored;
  only the simulation code and the report are tracked.
- Hardware model, engine parameters, and caveats are documented at the top of
  `sim/run_experiments.py` and in `REPORT.md`.
