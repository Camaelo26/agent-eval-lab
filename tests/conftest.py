import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def golden_path(root: Path) -> Path:
    return root / "data" / "golden.jsonl"


@pytest.fixture(scope="session")
def trap_cases(root: Path) -> list[dict]:
    rows = []
    for line in (root / "data" / "judge_trap_set.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows
