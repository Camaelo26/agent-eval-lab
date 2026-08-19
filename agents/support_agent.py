"""The agent under test.

Two versions ship on purpose:

* `agent_v2` is the current agent. It refuses when retrieval is weak and it
  selects tools from the question.
* `agent_v1` is the previous agent, kept as a known-bad build. It answers even
  when it has no supporting context and it calls every tool it owns.

Keeping a known-bad build in the repo is what makes the harness falsifiable. A
test suite that only ever sees a good agent has never proved it can fail.

Neither version calls a model. The harness is the deliverable here, not the
agent, and a deterministic agent means every number in the test suite is
reproducible on any machine with no API key.
"""

from __future__ import annotations

from evalkit import metrics
from evalkit.runner import AgentResponse

# Weak-retrieval cutoff. Below this the agent says it does not know instead of
# answering from the closest-but-irrelevant chunk.
REFUSAL_THRESHOLD = 0.12

REFUSAL_TEXT = "That is not covered in the documentation I have access to."

TOOL_TRIGGERS: dict[str, tuple[str, ...]] = {
    "lookup_order": ("refund", "order", "shipment", "delivery", "return"),
    "billing_api": ("invoice", "charge", "billing", "receipt", "payment", "card"),
    "auth_service": ("password", "login", "sign", "locked", "2fa", "authenticate"),
}


def _select_tools(question: str) -> list[str]:
    tokens = set(metrics.tokenize(question))
    tools = ["search_docs"]
    for tool, triggers in TOOL_TRIGGERS.items():
        if tokens & set(triggers):
            tools.append(tool)
    return tools


def _best_chunk(question: str, context: list[str]) -> tuple[str, float]:
    """Score each chunk by content-token overlap with the question."""
    query = set(metrics.content_tokens(question))
    if not query or not context:
        return "", 0.0
    best_text, best_score = "", 0.0
    for chunk in context:
        chunk_tokens = set(metrics.content_tokens(chunk))
        if not chunk_tokens:
            continue
        # Coverage of the question, damped by chunk length so a long chunk does
        # not win just by containing more words.
        score = len(query & chunk_tokens) / len(query)
        score *= 1 / (1 + 0.02 * max(0, len(chunk_tokens) - len(query)))
        if score > best_score:
            best_text, best_score = chunk, score
    return best_text, best_score


def agent_v2(question: str, context: list[str]) -> AgentResponse:
    """Current agent: retrieves, refuses on weak evidence, picks tools."""
    chunk, score = _best_chunk(question, context)
    if score < REFUSAL_THRESHOLD:
        answer = REFUSAL_TEXT
    else:
        answer = chunk
    return AgentResponse(
        answer=answer,
        tools_used=_select_tools(question),
        input_tokens=_estimate_tokens(question) + sum(_estimate_tokens(c) for c in context),
        output_tokens=_estimate_tokens(answer),
    )


def agent_v1(question: str, context: list[str]) -> AgentResponse:
    """Previous agent, kept as a regression fixture.

    Two defects, both realistic: it never refuses, and it fans out to every tool
    on every request. The gate should catch both.
    """
    chunk, _ = _best_chunk(question, context)
    answer = chunk or "Please contact support."
    return AgentResponse(
        answer=answer,
        tools_used=["search_docs", "lookup_order", "billing_api", "auth_service"],
        input_tokens=_estimate_tokens(question) + sum(_estimate_tokens(c) for c in context),
        output_tokens=_estimate_tokens(answer),
    )


AGENTS = {"v1": agent_v1, "v2": agent_v2}


def get_agent(name: str):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; available: {sorted(AGENTS)}")
    return AGENTS[name]


def _estimate_tokens(text: str) -> int:
    """~4 characters per token. Good enough for a cost budget, not for billing."""
    return max(1, len(text) // 4)
