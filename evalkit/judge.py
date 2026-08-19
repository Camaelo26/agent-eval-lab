"""LLM-as-judge, plus the part everybody skips: auditing the judge.

An LLM judge is a model scoring a model. Trusting its numbers without evidence
just moves the unverified step one level up. So this module ships two things:

1. A judge interface with a deterministic rubric judge as the default, and an
   Anthropic-backed judge when ANTHROPIC_API_KEY is set.
2. `audit_judge`, which runs the judge against a trap set of cases whose correct
   verdict is already known (some deliberately wrong answers, some subtly wrong,
   some right) and reports the judge's own accuracy, false-pass rate and
   false-fail rate.

A judge that has not passed its audit is not allowed to gate a release.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from . import metrics


@dataclass(frozen=True)
class Verdict:
    passed: bool
    score: float  # 0.0 - 1.0
    reason: str


class Judge(Protocol):
    name: str

    def score(self, question: str, answer: str, expected: str, context: list[str]) -> Verdict: ...


class RubricJudge:
    """Deterministic judge. No network, no API key, reproducible in CI.

    It combines the deterministic metrics into a single verdict using an
    explicit rubric. It is weaker than a real LLM judge on paraphrase, and it is
    the default anyway, because a weak-but-reproducible baseline is more useful
    in a regression gate than a strong-but-drifting one.
    """

    name = "rubric"

    def __init__(self, pass_threshold: float = 0.6) -> None:
        self.pass_threshold = pass_threshold

    def score(self, question: str, answer: str, expected: str, context: list[str]) -> Verdict:
        del question  # rubric judge scores the answer against the reference only
        if metrics.is_refusal(expected):
            # Correct behaviour is a refusal; grade on whether it refused.
            refused = metrics.is_refusal(answer)
            return Verdict(
                passed=refused,
                score=1.0 if refused else 0.0,
                reason="expected a refusal; agent "
                + ("refused" if refused else "answered anyway"),
            )

        overlap = metrics.token_f1(answer, expected)
        grounded = metrics.groundedness(answer, context) if context else overlap
        # Weighted toward reference overlap: groundedness alone is satisfied by
        # any answer that parrots the context without answering the question.
        score = 0.7 * overlap + 0.3 * grounded
        if metrics.is_refusal(answer):
            score = min(score, 0.2)  # refusing an answerable question is a failure
        return Verdict(
            passed=score >= self.pass_threshold,
            score=round(score, 4),
            reason=f"token_f1={overlap:.2f} groundedness={grounded:.2f}",
        )


class AnthropicJudge:
    """Real LLM judge. Used only when ANTHROPIC_API_KEY is present.

    Temperature is pinned to 0 and the rubric is passed as a system prompt so
    two runs on the same input stay as close as the API allows. It still drifts
    across model versions, which is exactly why `audit_judge` exists.
    """

    name = "anthropic"

    SYSTEM = (
        "You grade a support agent's answer against a reference answer.\n"
        "Return ONLY compact JSON: {\"score\": <0.0-1.0>, \"reason\": \"<12 words max>\"}.\n"
        "Score 1.0 when the answer conveys the same facts as the reference, even if "
        "worded differently. Score 0.0 when it contradicts the reference or invents "
        "facts absent from the context. Paraphrase is not an error. Missing a "
        "required fact is."
    )

    def __init__(self, model: str = "claude-sonnet-5", pass_threshold: float = 0.6) -> None:
        self.model = model
        self.pass_threshold = pass_threshold

    def score(self, question: str, answer: str, expected: str, context: list[str]) -> Verdict:
        import httpx  # imported lazily so the package works with no network deps

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("AnthropicJudge requires ANTHROPIC_API_KEY")

        user = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{chr(10).join(context) if context else '(none)'}\n\n"
            f"REFERENCE ANSWER:\n{expected}\n\n"
            f"AGENT ANSWER:\n{answer}"
        )
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 200,
                "temperature": 0,
                "system": self.SYSTEM,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        parsed = _parse_json_object(text)
        score = float(parsed.get("score", 0.0))
        return Verdict(
            passed=score >= self.pass_threshold,
            score=score,
            reason=str(parsed.get("reason", ""))[:120],
        )


def get_judge(name: str = "auto") -> Judge:
    """Pick a judge. 'auto' upgrades to the LLM judge only if a key is present."""
    if name == "rubric":
        return RubricJudge()
    if name == "anthropic":
        return AnthropicJudge()
    if name == "auto":
        return AnthropicJudge() if os.environ.get("ANTHROPIC_API_KEY") else RubricJudge()
    raise ValueError(f"unknown judge: {name!r}")


@dataclass(frozen=True)
class JudgeAudit:
    """How much the judge itself can be trusted."""

    total: int
    correct: int
    false_pass: int  # judge passed an answer the trap set marks as bad
    false_fail: int  # judge failed an answer the trap set marks as good

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def false_pass_rate(self) -> float:
        return self.false_pass / self.total if self.total else 0.0

    def trustworthy(self, min_accuracy: float = 0.8, max_false_pass: float = 0.1) -> bool:
        return self.accuracy >= min_accuracy and self.false_pass_rate <= max_false_pass


def audit_judge(judge: Judge, trap_cases: list[dict]) -> JudgeAudit:
    """Score the judge against cases with a known correct verdict.

    Each trap case carries `should_pass`. False passes are weighted heavier in
    practice than false fails: a judge that waves bad answers through makes the
    whole harness worse than having no harness, because it manufactures
    confidence.
    """
    correct = false_pass = false_fail = 0
    for case in trap_cases:
        verdict = judge.score(
            case["question"], case["answer"], case["expected"], case.get("context", [])
        )
        if verdict.passed == case["should_pass"]:
            correct += 1
        elif verdict.passed:
            false_pass += 1
        else:
            false_fail += 1
    return JudgeAudit(
        total=len(trap_cases), correct=correct, false_pass=false_pass, false_fail=false_fail
    )


def _parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge returned no JSON object: {text[:120]!r}")
    return json.loads(text[start : end + 1])
