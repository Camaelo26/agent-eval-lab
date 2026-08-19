# agent-eval-lab

An evaluation harness for LLM agents: golden datasets, deterministic metrics, an
audited LLM-as-judge, and a CI regression gate that blocks a release when quality
drops.

Python 3.12+, FastAPI, pytest. **77 tests pass.** The harness itself
(`evalkit/`) has zero third-party dependencies and needs no API key.

```bash
pip install -r requirements.txt
python -m pytest -q                            # 77 passed
python cli.py audit-judge --judge rubric       # audit the judge first
python cli.py run --agent v2 --judge rubric    # run the suite, apply the gate
python cli.py run --agent v1 --judge rubric    # known-bad build, exits 1
uvicorn app.main:app --reload                  # HTTP API on :8000
```

---

## Why this exists

Shipping an LLM feature without evals means the only regression detector is a
user complaint. Every job posting asking for "evals" is asking for this: a way
to answer *did the change make it better or worse* with a number instead of a
vibe.

This repo is the smallest honest version of that. It runs an agent over a
labelled dataset, scores it, slices the scores by category, compares against a
stored baseline, and exits non-zero when quality regressed.

---

## The part most eval setups skip: auditing the judge

An LLM judge is a model scoring a model. Trusting its output without evidence
doesn't remove the unverified step, it moves it up a level and hides it.

So the harness ships a **trap set**: ten cases whose correct verdict is already
known. Correct answers, correct refusals, flat contradictions, a hallucinated
extra fact, a refusal to an answerable question, and one answer with a single
number changed. `audit_judge` runs the judge over them and reports its accuracy,
false-pass rate and false-fail rate. A judge that fails its own audit is not
allowed to gate a release, enforced in `gate.py`.

Real result for the shipped rubric judge:

```
judge: rubric
  cases          10
  accuracy       0.900
  false passes   1  (rate 0.100)
  false fails    0
  trustworthy    True
```

### The one it gets wrong, and why that is the interesting part

The single miss is `trap-subtle-number`: the reference says an account locks for
**15 minutes**, the answer says **50 minutes**. Everything else in the sentence
is identical, so token overlap scores it **0.889** and it passes.

A *correct paraphrase* of a different case scores **0.710**.

The judge ranks a factually wrong answer above a factually right one. That
single data point is the entire argument for a semantic judge over a lexical one,
and it is the same lexical-gap failure that shows up in retrieval. It is pinned
as a test (`test_the_known_weakness_is_pinned`) so that if the rubric is ever
improved, the test fails and this README has to be updated. Documented
limitations decay into lies unless something enforces them.

**The trade-off I took:** the rubric judge is the default anyway, because a
weak-but-reproducible judge is more useful in a regression gate than a
strong-but-drifting one. `AnthropicJudge` is wired up and used automatically when
`ANTHROPIC_API_KEY` is set, at temperature 0 with a fixed rubric. It scores
paraphrase properly and costs money and drifts across model versions. The audit
is what makes that swap safe to make.

---

## What the suite actually found

Current agent, 15 cases:

```
pass rate        0.867
mean judge score 0.786
groundedness     0.711
tool score       0.929
latency p50/p95  0.03ms / 0.06ms
suite cost       $0.010590

pass rate by tag
  auth           0.667  <-- weak
  billing        0.800
  multi-hop      0.000  <-- weak
  permissions    0.000  <-- weak
  policy         1.000
  refunds        1.000
  retrieval      1.000
  safety         1.000
  shipping       1.000
  unanswerable   1.000

GATE PASS
```

An overall 87% looks fine on a slide. Sliced by tag, **multi-hop is 0%**. The
agent is extractive, so it returns the single best chunk and physically cannot
answer a question whose answer spans two chunks (`multi-001` asks about the
refund window *and* cancellation timing). `auth-002` fails the same way.

That is the whole reason `by_tag` exists. A single aggregate number is a
scoreboard. The slice is a bug report.

---

## Proving the gate can fail

A gate that has never blocked anything is not a gate, it is decoration. So a
known-bad build (`agent_v1`) is committed alongside the good one. It has two
realistic defects: it never refuses, and it fans out to every tool on every
request.

```
$ python cli.py run --agent v1 --judge rubric
pass rate        0.600
  safety         0.000  <-- weak
  unanswerable   0.000  <-- weak

GATE FAIL
  - pass_rate 0.600 below floor 0.800
  - pass_rate regressed 0.267 (baseline 0.867 -> 0.600, tolerance 0.030)
  - mean_judge_score regressed 0.273
  - mean_groundedness regressed 0.044
  - mean_tool_score regressed 0.326
exit=1
```

CI runs this deliberately and **fails the build if the bad agent passes**
(`ci/github-actions-evals.yml`; copy it to `.github/workflows/` to enable it on a fork).

---

## The two gate rules

1. **Absolute floors**: pass rate, groundedness, p95 latency, suite cost.
2. **Relative regression**: no metric may fall more than 3 points below the
   stored baseline, *even if it still clears the floor*.

Rule 2 is the one that earns its keep. Floors alone let quality bleed out slowly:
87% → 86% → 85% never trips a 0.80 floor and ships a much worse agent five
releases later. The test `test_slow_drift_is_caught_even_when_the_floor_is_cleared`
covers exactly that.

---

## Metrics, and what each one is blind to

| Metric | What it catches | What it misses |
|---|---|---|
| `exact_match` | nothing subtle, brittle by design | every valid paraphrase |
| `token_f1` | partial credit vs the reference | word order, negation, flipped numbers |
| `groundedness` | invented names and numbers | real tokens recombined into a false claim |
| `tool_call_score` | missing tools, and tool fan-out | wrong arguments to the right tool |
| `p95 latency` | the tail users actually feel | nothing, but mean latency would hide it |
| `cost_usd` | a prompt change that triples spend | per-request cost outliers |

Order-insensitive tool scoring is deliberate: most frameworks parallelise tool
calls, so penalising order would flag correct runs. Extra tools still cost
precision, because an agent that calls everything it owns is brute-forcing, not
reasoning.

p95 rather than mean latency, because the mean hides the tail and the tail is
what pages the on-call engineer.

---

## Layout

```
evalkit/          stdlib-only harness (no FastAPI import anywhere)
  dataset.py      golden case loading + validation (dupes, missing fields, empty files)
  metrics.py      pure deterministic metrics
  judge.py        rubric + Anthropic judges, and audit_judge
  runner.py       runs an agent over a suite, aggregates, slices by tag
  gate.py         floors + regression tolerance -> CI exit code
agents/           the agent under test: v2 (current) and v1 (known-bad)
data/             golden.jsonl, judge_trap_set.jsonl
app/main.py       FastAPI: /evals/run, /evals/{id}, /judge/audit, /metrics
cli.py            run | baseline | audit-judge
tests/            77 tests
```

`evalkit` never imports FastAPI. The HTTP layer depends on the harness, never
the reverse, so the full suite runs in CI with nothing installed but pytest.

## API

| Endpoint | Purpose |
|---|---|
| `POST /evals/run` | run a suite, apply the gate, return the summary |
| `GET /evals/{run_id}` | full stored report |
| `GET /judge/audit` | judge accuracy against the trap set |
| `GET /metrics` | Prometheus exposition of the latest run |

`/metrics` exists because eval scores belong on the same dashboard as latency
and error rate. Quality metrics that live in a notebook nobody opens mean
regressions get found by users.

## Known limitations

- **Runs are stored in memory.** A second uvicorn worker would not see them.
  Single-worker only; Redis or Postgres is the obvious next step and was
  deliberately left undone, because the evaluation logic is the point here.
- **The agent under test is deterministic and does not call a model.** That is
  what makes every number in the test suite reproducible on any machine with no
  API key. Swapping in a real agent means implementing one callable:
  `(question, context) -> AgentResponse`.
- **`groundedness` is a hallucination proxy, not a detector.** It catches
  invented tokens and misses real tokens recombined into a false claim.
- **15 golden cases is small.** Enough to demonstrate tag slicing and gating,
  not enough to certify a production agent.
