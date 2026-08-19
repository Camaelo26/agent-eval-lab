"""Golden dataset loading and validation.

A golden case is the contract between "what the agent did" and "what it should
have done". If the dataset is sloppy, every number downstream is decoration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REQUIRED_FIELDS = ("id", "question", "context", "expected")


@dataclass(frozen=True)
class GoldenCase:
    """One labelled test case."""

    id: str
    question: str
    context: list[str]
    expected: str
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # True when the correct behaviour is to refuse / say "not in the docs".
    unanswerable: bool = False

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "GoldenCase":
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"case {raw.get('id', '<no id>')} missing fields: {missing}")
        if not isinstance(raw["context"], list):
            raise ValueError(f"case {raw['id']}: context must be a list of strings")
        return GoldenCase(
            id=str(raw["id"]),
            question=raw["question"],
            context=[str(c) for c in raw["context"]],
            expected=raw["expected"],
            expected_tools=[str(t) for t in raw.get("expected_tools", [])],
            tags=[str(t) for t in raw.get("tags", [])],
            unanswerable=bool(raw.get("unanswerable", False)),
        )


def load_jsonl(path: str | Path) -> list[GoldenCase]:
    """Read a .jsonl dataset. Blank lines and # comments are skipped."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for line_no, line in enumerate(_lines(Path(path)), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        case = GoldenCase.from_dict(raw)
        if case.id in seen:
            raise ValueError(f"{path}:{line_no} duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contained no cases")
    return cases


def _lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def filter_by_tag(cases: list[GoldenCase], tag: str) -> list[GoldenCase]:
    return [c for c in cases if tag in c.tags]
