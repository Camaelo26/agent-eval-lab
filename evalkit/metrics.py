"""Deterministic metrics.

Everything here is pure and testable. No model calls, no network, no clock.
The judge (which IS non-deterministic) lives in judge.py and is kept separate on
purpose so a CI failure can always be traced to one side or the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no evidence. Kept small and explicit rather than pulling a
# stopword list from a package, so the behaviour is auditable.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that
    the to was were what when where which who will with you your do does did not
    can could should would this these those there their they i we he she""".split()
)

_REFUSAL_MARKERS = (
    "not in the",
    "not covered",
    "no information",
    "cannot answer",
    "can't answer",
    "don't have",
    "do not have",
    "not documented",
    "unable to answer",
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def exact_match(prediction: str, expected: str) -> float:
    """1.0 only on a normalized string match. Brittle by design."""
    return 1.0 if _normalize(prediction) == _normalize(expected) else 0.0


def token_f1(prediction: str, expected: str) -> float:
    """Bag-of-tokens F1. Partial credit where exact match gives none."""
    pred = content_tokens(prediction)
    gold = content_tokens(expected)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    overlap = _multiset_overlap(pred, gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def groundedness(prediction: str, context: list[str]) -> float:
    """Share of the answer's content tokens that appear in the retrieved context.

    This is a hallucination proxy, not a hallucination detector. It catches the
    common failure (the model inventing a number or a name that is nowhere in the
    docs) and misses the subtle one (the model recombining real tokens into a
    false claim). The limitation is the point of the metric, not a bug in it.
    """
    pred = content_tokens(prediction)
    if not pred:
        return 0.0
    supported = set()
    for chunk in context:
        supported.update(content_tokens(chunk))
    if not supported:
        return 0.0
    hits = sum(1 for token in pred if token in supported)
    return hits / len(pred)


def is_refusal(prediction: str) -> bool:
    lowered = prediction.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def tool_call_score(actual: list[str], expected: list[str]) -> float:
    """Order-insensitive F1 over tool names.

    Order-insensitive because most agent frameworks parallelise tool calls, so
    penalising order would flag correct runs. Extra tools still cost precision,
    which is what we actually care about: an agent that calls every tool it owns
    is not solving the task, it is brute-forcing it.
    """
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    overlap = len(set(actual) & set(expected))
    if overlap == 0:
        return 0.0
    precision = overlap / len(set(actual))
    recall = overlap / len(set(expected))
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class CostModel:
    """Per-million-token pricing, so the harness reports dollars not tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. p95 latency, not mean latency.

    Mean latency hides the tail, and the tail is what users actually experience
    and what pages the on-call engineer.
    """
    if not values:
        return 0.0
    if not 0 < pct <= 100:
        raise ValueError("pct must be in (0, 100]")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), _ceil(pct / 100 * len(ordered))))
    return ordered[rank - 1]


def _ceil(value: float) -> int:
    as_int = int(value)
    return as_int if as_int == value else as_int + 1


def _normalize(text: str) -> str:
    return " ".join(tokenize(text))


def _multiset_overlap(left: list[str], right: list[str]) -> int:
    remaining = list(right)
    count = 0
    for token in left:
        if token in remaining:
            remaining.remove(token)
            count += 1
    return count
