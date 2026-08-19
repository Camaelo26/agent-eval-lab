"""Run an agent over a golden dataset and produce a report."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import metrics
from .dataset import GoldenCase
from .judge import Judge, JudgeAudit


@dataclass
class AgentResponse:
    """What an agent under test must return."""

    answer: str
    tools_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# An agent is any callable taking (question, context) and returning AgentResponse.
Agent = Callable[[str, list[str]], AgentResponse]


@dataclass
class CaseResult:
    case_id: str
    tags: list[str]
    passed: bool
    judge_score: float
    judge_reason: str
    exact_match: float
    token_f1: float
    groundedness: float
    tool_score: float
    latency_ms: float
    cost_usd: float
    answer: str


@dataclass
class EvalReport:
    suite: str
    judge: str
    n_cases: int
    pass_rate: float
    mean_judge_score: float
    mean_token_f1: float
    mean_groundedness: float
    mean_tool_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    by_tag: dict[str, float]
    failures: list[str]
    judge_audit: dict | None = None
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["results"] = [asdict(r) for r in self.results]
        return payload


def run_eval(
    agent: Agent,
    cases: list[GoldenCase],
    judge: Judge,
    cost_model: metrics.CostModel | None = None,
    suite: str = "default",
    judge_audit: JudgeAudit | None = None,
) -> EvalReport:
    cost_model = cost_model or metrics.CostModel(input_per_mtok=3.0, output_per_mtok=15.0)
    results: list[CaseResult] = []

    for case in cases:
        start = time.perf_counter()
        response = agent(case.question, case.context)
        latency_ms = (time.perf_counter() - start) * 1000

        verdict = judge.score(case.question, response.answer, case.expected, case.context)
        results.append(
            CaseResult(
                case_id=case.id,
                tags=case.tags,
                passed=verdict.passed,
                judge_score=verdict.score,
                judge_reason=verdict.reason,
                exact_match=metrics.exact_match(response.answer, case.expected),
                token_f1=metrics.token_f1(response.answer, case.expected),
                groundedness=metrics.groundedness(response.answer, case.context),
                tool_score=metrics.tool_call_score(response.tools_used, case.expected_tools),
                latency_ms=round(latency_ms, 3),
                cost_usd=cost_model.cost(response.input_tokens, response.output_tokens),
                answer=response.answer,
            )
        )

    latencies = [r.latency_ms for r in results]
    return EvalReport(
        suite=suite,
        judge=judge.name,
        n_cases=len(results),
        pass_rate=_mean([1.0 if r.passed else 0.0 for r in results]),
        mean_judge_score=_mean([r.judge_score for r in results]),
        mean_token_f1=_mean([r.token_f1 for r in results]),
        mean_groundedness=_mean([r.groundedness for r in results]),
        mean_tool_score=_mean([r.tool_score for r in results]),
        p50_latency_ms=round(metrics.percentile(latencies, 50), 3),
        p95_latency_ms=round(metrics.percentile(latencies, 95), 3),
        total_cost_usd=round(sum(r.cost_usd for r in results), 6),
        by_tag=_pass_rate_by_tag(results),
        failures=[r.case_id for r in results if not r.passed],
        judge_audit=_audit_dict(judge_audit),
        results=results,
    )


def _pass_rate_by_tag(results: list[CaseResult]) -> dict[str, float]:
    """Aggregate pass rate per tag.

    An overall pass rate of 90% can still hide a category that is at 0%. Slicing
    by tag is what turns the report from a scoreboard into a bug report.
    """
    buckets: dict[str, list[float]] = {}
    for result in results:
        for tag in result.tags:
            buckets.setdefault(tag, []).append(1.0 if result.passed else 0.0)
    return {tag: round(_mean(scores), 4) for tag, scores in sorted(buckets.items())}


def _audit_dict(audit: JudgeAudit | None) -> dict | None:
    if audit is None:
        return None
    return {
        "total": audit.total,
        "correct": audit.correct,
        "false_pass": audit.false_pass,
        "false_fail": audit.false_fail,
        "accuracy": round(audit.accuracy, 4),
        "false_pass_rate": round(audit.false_pass_rate, 4),
        "trustworthy": audit.trustworthy(),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
