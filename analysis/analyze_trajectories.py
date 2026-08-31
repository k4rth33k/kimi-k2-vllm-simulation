"""Extract per-request token sequences from toolathon trajectories.

For each trajectory we reconstruct the sequence of LLM requests the ReAct
agent made. For each assistant turn we record:
  - output_tokens: tokens in the assistant message (content + tool call args)
  - tool_result_tokens: tokens of subsequent tool messages (until next assistant/user)
  - group: tool-server prefix invoked by this turn (or None for plain response)

The system prompt and the initial user/task message form the fixed prefix.

Usage (from the repo root):
    python analysis/analyze_trajectories.py --runs kimi-k2-0905 --out results/token_stats_kimi.json
"""

import json
import glob
import argparse
from collections import defaultdict

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def tok_len(text: str) -> int:
    if not text:
        return 0
    return len(ENC.encode(text, disallowed_special=()))


def msg_tokens(msg: dict) -> int:
    n = tok_len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        n += tok_len(fn.get("name", ""))
        n += tok_len(fn.get("arguments", ""))
    return n


def tool_group(name: str) -> str:
    if "-" in name:
        return name.split("-")[0]
    if "_" in name:
        return name.split("_")[0]
    return name


def reconstruct_requests(rec: dict) -> dict | None:
    msgs = rec.get("messages")
    if msgs is None:
        return None
    if isinstance(msgs, str):
        msgs = json.loads(msgs)

    cfg = json.loads(rec["config"]) if isinstance(rec.get("config"), str) else rec.get("config", {})
    sys_prompts = cfg.get("system_prompts", {})
    system_text = sys_prompts.get("agent", "")
    system_tokens = tok_len(system_text)

    requests = []
    tool_groups_used = set()
    task_tokens = 0            # tokens of the first user message (task description)
    prefix_assistant_msgs = 0  # sanity counter
    pending_tool_tokens = 0    # tool tokens accumulated since last assistant turn

    for msg in msgs:
        role = msg.get("role")
        if role == "user":
            t = msg_tokens(msg)
            if not requests and task_tokens == 0:
                task_tokens = t
            else:
                pending_tool_tokens += t
        elif role == "tool":
            pending_tool_tokens += msg_tokens(msg)
        elif role == "assistant":
            out_toks = msg_tokens(msg)
            groups = []
            for tc in msg.get("tool_calls") or []:
                g = tool_group(tc["function"]["name"])
                groups.append(g)
                tool_groups_used.add(g)
            requests.append({
                "output_tokens": out_toks,
                "tool_result_tokens": pending_tool_tokens,
                "group": groups[0] if groups else None,
            })
            pending_tool_tokens = 0
            prefix_assistant_msgs += 1

    stats = json.loads(rec["key_stats"]) if isinstance(rec.get("key_stats"), str) else rec.get("key_stats", {})

    return {
        "task_name": rec.get("task_name"),
        "model_run": rec.get("modelname_run"),
        "system_tokens": system_tokens,
        "task_tokens": task_tokens,
        "tool_groups_used": sorted(tool_groups_used),
        "n_requests": len(requests),
        "requests": requests,
        "reported_input_tokens": stats.get("input_tokens"),
        "reported_output_tokens": stats.get("output_tokens"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="kimi-k2-0905",
                    help="model run prefix to keep (comma separated), or 'all'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wanted = None if args.runs == "all" else set(args.runs.split(","))

    out = []
    files = sorted(glob.glob("toolathon_trajectories/*.jsonl"))
    for f in files:
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                run = rec.get("modelname_run", "")
                model = "_".join(run.split("_")[:-1])
                if wanted and model not in wanted:
                    continue
                r = reconstruct_requests(rec)
                if r and r["n_requests"] > 0:
                    out.append(r)

    with open(args.out, "w") as fh:
        json.dump(out, fh)

    n = len(out)
    avg_req = sum(r["n_requests"] for r in out) / max(n, 1)
    print(f"Extracted {n} trajectories, avg {avg_req:.1f} requests/task")


if __name__ == "__main__":
    main()
