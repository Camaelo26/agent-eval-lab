"""FastAPI surface over the harness.

The API exists so evals can be triggered from CI, a dashboard or a chat command
instead of only a terminal. It stores runs in memory, which is fine for a single
worker and honestly wrong for anything else - see the note in `RUN_STORE`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from agents import AGENTS, get_agent
from evalkit import gate, judge as judge_mod
from evalkit.dataset import load_jsonl
from evalkit.runner import run_eval

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
TRAPS = ROOT / "data" / "judge_trap_set.jsonl"

app = FastAPI(
    title="agent-eval-lab",
    version="0.1.0",
    description="Evaluation, judge auditing and regression gating for LLM agents.",
)

# In-process store. A second uvicorn worker would not see these runs, so this is
# a single-worker deployment only. Swapping in Redis or Postgres is the obvious
# next step and is deliberately not done here: the interesting part of this
# project is the evaluation logic, not the persistence layer.
RUN_STORE: dict[str, dict] = {}


class RunRequest(BaseModel):
    agent: str = Field(default="v2", description="Which agent build to evaluate")
    judge: Literal["auto", "rubric", "anthropic"] = "rubric"
    compare_to_baseline: bool = True
    include_cases: bool = False


class GateView(BaseModel):
    passed: bool
    violations: list[str]
    checked_against_baseline: bool


class RunSummary(BaseModel):
    run_id: str
    suite: str
    judge: str
    n_cases: int
    pass_rate: float
    mean_judge_score: float
    mean_groundedness: float
    mean_tool_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    by_tag: dict[str, float]
    failures: list[str]
    judge_audit: dict | None
    gate: GateView


def _load_traps() -> list[dict]:
    rows = []
    for line in TRAPS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "agents": sorted(AGENTS), "cases": len(load_jsonl(GOLDEN))}


@app.get("/agents")
def list_agents() -> dict:
    return {"agents": sorted(AGENTS)}


@app.post("/evals/run", response_model=RunSummary)
def run(request: RunRequest) -> RunSummary:
    try:
        agent = get_agent(request.agent)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        judge = judge_mod.get_judge(request.judge)
    except RuntimeError as exc:  # anthropic judge without a key
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cases = load_jsonl(GOLDEN)
    audit = judge_mod.audit_judge(judge, _load_traps())
    report = run_eval(
        agent, cases, judge, suite=f"support-agent:{request.agent}", judge_audit=audit
    )

    baseline = (
        gate.load_baseline(ROOT / "baselines" / "support-agent.json")
        if request.compare_to_baseline
        else None
    )
    result = gate.check(report, gate.Thresholds(), baseline)

    run_id = uuid.uuid4().hex[:12]
    payload = report.to_dict()
    if not request.include_cases:
        payload.pop("results", None)
    payload["run_id"] = run_id
    payload["gate"] = {
        "passed": result.passed,
        "violations": result.violations,
        "checked_against_baseline": result.checked_against_baseline,
    }
    RUN_STORE[run_id] = payload

    return RunSummary(
        run_id=run_id,
        suite=report.suite,
        judge=report.judge,
        n_cases=report.n_cases,
        pass_rate=report.pass_rate,
        mean_judge_score=report.mean_judge_score,
        mean_groundedness=report.mean_groundedness,
        mean_tool_score=report.mean_tool_score,
        p50_latency_ms=report.p50_latency_ms,
        p95_latency_ms=report.p95_latency_ms,
        total_cost_usd=report.total_cost_usd,
        by_tag=report.by_tag,
        failures=report.failures,
        judge_audit=report.judge_audit,
        gate=GateView(
            passed=result.passed,
            violations=result.violations,
            checked_against_baseline=result.checked_against_baseline,
        ),
    )


@app.get("/evals/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id not in RUN_STORE:
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    return RUN_STORE[run_id]


@app.get("/evals")
def list_runs() -> dict:
    return {
        "runs": [
            {"run_id": rid, "suite": r["suite"], "pass_rate": r["pass_rate"]}
            for rid, r in RUN_STORE.items()
        ]
    }


@app.get("/judge/audit")
def judge_audit(name: Literal["auto", "rubric", "anthropic"] = "rubric") -> dict:
    judge = judge_mod.get_judge(name)
    audit = judge_mod.audit_judge(judge, _load_traps())
    return {
        "judge": judge.name,
        "total": audit.total,
        "accuracy": round(audit.accuracy, 4),
        "false_pass": audit.false_pass,
        "false_fail": audit.false_fail,
        "false_pass_rate": round(audit.false_pass_rate, 4),
        "trustworthy": audit.trustworthy(),
    }


@app.get("/metrics", response_class=Response)
def prometheus_metrics() -> Response:
    """Prometheus exposition of the most recent run.

    Eval scores belong on the same dashboard as latency and error rate. If they
    live in a notebook nobody opens, quality regressions are found by users.
    """
    if not RUN_STORE:
        return Response("# no runs yet\n", media_type="text/plain; version=0.0.4")
    latest = list(RUN_STORE.values())[-1]
    lines = [
        "# HELP agent_eval_pass_rate Share of golden cases the judge passed.",
        "# TYPE agent_eval_pass_rate gauge",
        f"agent_eval_pass_rate{{suite=\"{latest['suite']}\"}} {latest['pass_rate']}",
        "# HELP agent_eval_groundedness Mean share of answer tokens supported by context.",
        "# TYPE agent_eval_groundedness gauge",
        f"agent_eval_groundedness{{suite=\"{latest['suite']}\"}} {latest['mean_groundedness']}",
        "# HELP agent_eval_p95_latency_ms p95 agent latency over the suite.",
        "# TYPE agent_eval_p95_latency_ms gauge",
        f"agent_eval_p95_latency_ms{{suite=\"{latest['suite']}\"}} {latest['p95_latency_ms']}",
        "# HELP agent_eval_gate_passed 1 when the regression gate passed.",
        "# TYPE agent_eval_gate_passed gauge",
        f"agent_eval_gate_passed{{suite=\"{latest['suite']}\"}} "
        f"{1 if latest['gate']['passed'] else 0}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
