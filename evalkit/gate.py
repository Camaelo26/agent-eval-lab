"""Regression gate. Turns a report into a CI pass/fail decision.

Two rules, and the second one is the one that matters:

1. Absolute floors: pass rate, groundedness and p95 latency must clear a bar.
2. Relative regression: a metric may not drop more than `tolerance` below the
   stored baseline, even if it still clears the absolute floor.

Rule 2 exists because absolute floors alone let quality bleed out slowly. A
suite that drifts 87% -> 86% -> 85% never trips a 0.80 floor and ships a much
worse agent five releases later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .runner import EvalReport


@dataclass(frozen=True)
class Thresholds:
    min_pass_rate: float = 0.80
    min_groundedness: float = 0.50
    max_p95_latency_ms: float = 5000.0
    max_cost_usd: float = 1.00
    # A metric may not fall more than this below the baseline.
    regression_tolerance: float = 0.03
    require_trustworthy_judge: bool = True


@dataclass
class GateResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    checked_against_baseline: bool = False

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def check(
    report: EvalReport,
    thresholds: Thresholds | None = None,
    baseline: dict | None = None,
) -> GateResult:
    thresholds = thresholds or Thresholds()
    violations: list[str] = []

    if report.pass_rate < thresholds.min_pass_rate:
        violations.append(
            f"pass_rate {report.pass_rate:.3f} below floor {thresholds.min_pass_rate:.3f}"
        )
    if report.mean_groundedness < thresholds.min_groundedness:
        violations.append(
            f"groundedness {report.mean_groundedness:.3f} below floor "
            f"{thresholds.min_groundedness:.3f}"
        )
    if report.p95_latency_ms > thresholds.max_p95_latency_ms:
        violations.append(
            f"p95 latency {report.p95_latency_ms:.1f}ms over budget "
            f"{thresholds.max_p95_latency_ms:.1f}ms"
        )
    if report.total_cost_usd > thresholds.max_cost_usd:
        violations.append(
            f"suite cost ${report.total_cost_usd:.4f} over budget "
            f"${thresholds.max_cost_usd:.2f}"
        )

    audit = report.judge_audit
    if thresholds.require_trustworthy_judge and audit is not None and not audit["trustworthy"]:
        violations.append(
            f"judge failed its own audit (accuracy {audit['accuracy']:.2f}, "
            f"false_pass_rate {audit['false_pass_rate']:.2f}) - scores not trusted"
        )

    checked_baseline = False
    if baseline:
        checked_baseline = True
        for metric in ("pass_rate", "mean_judge_score", "mean_groundedness", "mean_tool_score"):
            previous = baseline.get(metric)
            if previous is None:
                continue
            current = getattr(report, metric)
            drop = previous - current
            if drop > thresholds.regression_tolerance:
                violations.append(
                    f"{metric} regressed {drop:.3f} (baseline {previous:.3f} -> "
                    f"{current:.3f}, tolerance {thresholds.regression_tolerance:.3f})"
                )

    return GateResult(
        passed=not violations,
        violations=violations,
        checked_against_baseline=checked_baseline,
    )


def load_baseline(path: str | Path) -> dict | None:
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def save_baseline(report: EvalReport, path: str | Path) -> None:
    """Store only the aggregate metrics, never the raw answers.

    Baselines get committed to the repo. Keeping model output out of them stops
    the diff from being unreadable and stops anyone from eyeballing the baseline
    instead of running the suite.
    """
    payload = {
        "suite": report.suite,
        "judge": report.judge,
        "n_cases": report.n_cases,
        "pass_rate": report.pass_rate,
        "mean_judge_score": report.mean_judge_score,
        "mean_groundedness": report.mean_groundedness,
        "mean_tool_score": report.mean_tool_score,
        "p95_latency_ms": report.p95_latency_ms,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
