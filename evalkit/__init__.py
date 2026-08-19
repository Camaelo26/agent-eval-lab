"""evalkit - a small, dependency-free evaluation harness for LLM agents.

The public surface is deliberately narrow:

    from evalkit import dataset, judge, metrics, runner, gate

Nothing in `evalkit` imports FastAPI. The HTTP layer in `app/` depends on the
harness, never the other way around, so the whole suite runs in CI with nothing
installed but pytest.
"""

from . import dataset, gate, judge, metrics, runner

__all__ = ["dataset", "gate", "judge", "metrics", "runner"]
__version__ = "0.1.0"
