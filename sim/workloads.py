"""Agent definitions that drive the engine for a single task under a given
topology."""

from engine import Request, Engine


class TaskAgent:
    """Chains LLM requests for ONE trajectory under react or orchestrator."""

    def __init__(self, engine: Engine, task: dict, topology: str,
                 tool_delay_fn, orch_cap: int = 4,
                 schema_tokens_per_group: int = 64):
        self.engine = engine
        self.task = task
        self.topology = topology
        self.tool_delay_fn = tool_delay_fn
        self.orch_cap = orch_cap
        self.schema_tokens = schema_tokens_per_group
        self.task_id = task["task_name"]

        if topology == "react":
            self.kind = "react"
            self.requests = task["requests"]
            self.segs = [("sys", task["system_tokens"]),
                         (f"task:{self.task_id}", task["task_tokens"])]
            self.step_idx = 0
        else:
            self.kind = "orchestrator"
            self.by_group, self.group_order = self._group_requests(task["requests"])
            self.step_pos = {g: 0 for g in self.group_order}
            self.group_segs = {g: [("sys", task["system_tokens"]),
                                    (f"task:{self.task_id}", task["task_tokens"]),
                                    (f"schema:{g}", self.schema_tokens)]
                                for g in self.group_order}
            self.active = 0
            self.next_group_idx = 0
            # Orchestrator pays one up-front planning call to decompose the task
            # before launching subagents.
            self.plan_pending = True
            self.planner_output_tokens = min(200, max(40, len(self.group_order) * 30))

    # --------------------------- grouping (orch) --------------------------- #

    @staticmethod
    def _group_requests(requests):
        group_order = []
        by_group = {}
        for r in requests:
            g = r["group"] or "_final"
            if g not in by_group:
                by_group[g] = []
                group_order.append(g)
            by_group[g].append(r)
        return by_group, group_order

    # ------------------------------ entry ------------------------------ #

    def start(self, now=0.0):
        if self.kind == "react":
            self._next_react(now)
        else:
            if self.plan_pending:
                self._plan_request(now)
            else:
                self._orch_launch(now)

    # --------------------------- Planner call --------------------------- #

    def _plan_request(self, now):
        req = Request(
            req_id=self.engine.next_req_id(),
            agent_id=id(self),
            topology="orchestrator",
            seg_ids=["sys", f"task:{self.task_id}", "plan"],
            seg_tokens=[self.task["system_tokens"], self.task["task_tokens"], 24],
            prompt_tokens=self.task["system_tokens"] + self.task["task_tokens"] + 24,
            output_tokens=self.planner_output_tokens,
            on_complete=lambda e, rq, n: self._plan_done(e, rq, n),
            meta={"task": self.task_id, "group": "_planner", "step": 0},
        )
        self.engine.submit(req, now)

    def _plan_done(self, engine, req, now):
        self.plan_pending = False
        engine._push(now + 0.05, 2, "agent_callback",
                     lambda t: self._orch_launch(t))

    # ------------------------------- ReAct ------------------------------- #

    def _next_react(self, now):
        if self.step_idx >= len(self.requests):
            return
        r = self.requests[self.step_idx]
        idx = self.step_idx
        req = Request(
            req_id=self.engine.next_req_id(),
            agent_id=id(self),
            topology="react",
            seg_ids=[s[0] for s in self.segs],
            seg_tokens=[s[1] for s in self.segs],
            prompt_tokens=sum(s[1] for s in self.segs),
            output_tokens=max(1, r["output_tokens"]),
            on_complete=self._react_done,
            meta={"task": self.task_id, "step": idx, "group": r["group"]},
        )
        self.segs.append((f"{self.task_id}:asst:{idx}", r["output_tokens"]))
        self.segs.append((f"{self.task_id}:tool:{idx}", r["tool_result_tokens"]))
        self.step_idx += 1
        self.engine.submit(req, now)

    def _react_done(self, engine, req, now):
        delay = self.tool_delay_fn(req.meta.get("group"))
        engine._push(now + delay, 2, "agent_callback",
                     lambda t: self._next_react(t))

    # ----------------------------- Orchestrator ----------------------------- #

    def _orch_launch(self, now):
        while self.active < self.orch_cap and self.next_group_idx < len(self.group_order):
            g = self.group_order[self.next_group_idx]
            self.active += 1
            self.next_group_idx += 1
            self._next_in_group(g, now)

    def _next_in_group(self, g, now):
        pos = self.step_pos[g]
        if pos >= len(self.by_group[g]):
            self.active -= 1
            self._orch_launch(now)
            return
        r = self.by_group[g][pos]
        segs = self.group_segs[g]
        idx = pos
        req = Request(
            req_id=self.engine.next_req_id(),
            agent_id=id(self),
            topology="orchestrator",
            seg_ids=[s[0] for s in segs],
            seg_tokens=[s[1] for s in segs],
            prompt_tokens=sum(s[1] for s in segs),
            output_tokens=max(1, r["output_tokens"]),
            on_complete=lambda e, rq, n: self._orch_done(e, rq, n, g),
            meta={"task": self.task_id, "group": g, "step": idx},
        )
        segs.append((f"{self.task_id}:{g}:asst:{idx}", r["output_tokens"]))
        segs.append((f"{self.task_id}:{g}:tool:{idx}", r["tool_result_tokens"]))
        self.step_pos[g] += 1
        self.engine.submit(req, now)

    def _orch_done(self, engine, req, now, group):
        delay = self.tool_delay_fn(group)
        engine._push(now + delay, 2, "agent_callback",
                     lambda t: self._next_in_group(group, t))
