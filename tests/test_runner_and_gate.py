import pytest

from agents import agent_v1, agent_v2
from evalkit import gate
from evalkit.dataset import load_jsonl
from evalkit.judge import RubricJudge, audit_judge
from evalkit.runner import AgentResponse, run_eval


@pytest.fixture(scope="module")
def cases(request):
    root = request.path.parent.parent
    return load_jsonl(root / "data" / "golden.jsonl")


@pytest.fixture(scope="module")
def report_v2(cases):
    return run_eval(agent_v2, cases, RubricJudge(), suite="v2")


@pytest.fixture(scope="module")
def report_v1(cases):
    return run_eval(agent_v1, cases, RubricJudge(), suite="v1")


class TestRunner:
    def test_every_case_produces_a_result(self, report_v2, cases):
        assert report_v2.n_cases == len(cases)
        assert len(report_v2.results) == len(cases)

    def test_aggregates_are_in_range(self, report_v2):
        for value in (
            report_v2.pass_rate,
            report_v2.mean_judge_score,
            report_v2.mean_groundedness,
            report_v2.mean_tool_score,
        ):
            assert 0.0 <= value <= 1.0

    def test_latency_percentiles_are_ordered(self, report_v2):
        assert report_v2.p50_latency_ms <= report_v2.p95_latency_ms

    def test_pass_rate_is_sliced_by_tag(self, report_v2):
        assert report_v2.by_tag
        assert "billing" in report_v2.by_tag

    def test_report_serialises_to_plain_json_types(self, report_v2):
        payload = report_v2.to_dict()
        assert isinstance(payload["results"], list)
        assert isinstance(payload["results"][0], dict)

    def test_current_agent_beats_the_known_bad_build(self, report_v1, report_v2):
        # The whole harness is pointless if it cannot rank a good build above a
        # bad one. This is the single most important assertion in the repo.
        assert report_v2.pass_rate > report_v1.pass_rate
        assert report_v2.mean_tool_score > report_v1.mean_tool_score

    def test_the_known_bad_build_fails_the_unanswerable_cases(self, report_v1):
        unanswerable = [r for r in report_v1.results if "unanswerable" in r.tags]
        assert unanswerable
        assert all(not r.passed for r in unanswerable), (
            "agent_v1 never refuses, so every unanswerable case must fail"
        )

    def test_the_current_build_refuses_unanswerable_questions(self, report_v2):
        unanswerable = [r for r in report_v2.results if "unanswerable" in r.tags]
        assert unanswerable
        assert all(r.passed for r in unanswerable)

    def test_cost_is_reported_in_dollars(self, report_v2):
        assert report_v2.total_cost_usd > 0


class TestGate:
    def test_a_healthy_report_passes_with_no_baseline(self, report_v2):
        result = gate.check(report_v2, gate.Thresholds())
        assert result.passed, result.violations

    def test_the_known_bad_build_is_blocked(self, report_v1):
        result = gate.check(report_v1, gate.Thresholds())
        assert not result.passed
        assert result.exit_code == 1

    def test_slow_drift_is_caught_even_when_the_floor_is_cleared(self, report_v2):
        # pass_rate stays above the 0.80 floor but drops 5 points from baseline.
        baseline = {"pass_rate": report_v2.pass_rate + 0.05}
        result = gate.check(report_v2, gate.Thresholds(min_pass_rate=0.5), baseline)
        assert not result.passed
        assert any("regressed" in v for v in result.violations)

    def test_a_drop_inside_tolerance_is_allowed(self, report_v2):
        baseline = {"pass_rate": report_v2.pass_rate + 0.01}
        result = gate.check(report_v2, gate.Thresholds(regression_tolerance=0.03), baseline)
        assert result.passed, result.violations

    def test_an_untrustworthy_judge_blocks_the_release(self, cases):
        class AlwaysPass:
            name = "always-pass"

            def score(self, question, answer, expected, context):
                from evalkit.judge import Verdict

                return Verdict(passed=True, score=1.0, reason="")

        traps = [
            {"question": "q", "answer": "wrong", "expected": "right", "should_pass": False}
        ] * 5
        audit = audit_judge(AlwaysPass(), traps)
        report = run_eval(agent_v2, cases, AlwaysPass(), suite="v2", judge_audit=audit)
        result = gate.check(report, gate.Thresholds())
        assert not result.passed
        assert any("judge failed its own audit" in v for v in result.violations)

    def test_latency_budget_is_enforced(self, report_v2):
        result = gate.check(report_v2, gate.Thresholds(max_p95_latency_ms=0.0))
        assert not result.passed
        assert any("latency" in v for v in result.violations)

    def test_cost_budget_is_enforced(self, report_v2):
        result = gate.check(report_v2, gate.Thresholds(max_cost_usd=0.0))
        assert not result.passed
        assert any("cost" in v for v in result.violations)

    def test_baseline_round_trips_without_raw_answers(self, report_v2, tmp_path):
        path = tmp_path / "baseline.json"
        gate.save_baseline(report_v2, path)
        loaded = gate.load_baseline(path)
        assert loaded["pass_rate"] == report_v2.pass_rate
        assert "results" not in loaded

    def test_missing_baseline_loads_as_none(self, tmp_path):
        assert gate.load_baseline(tmp_path / "nope.json") is None


class TestAgentContract:
    def test_agents_return_the_expected_shape(self):
        response = agent_v2("How do I reset my password?", ["Choose Forgot Password."])
        assert isinstance(response, AgentResponse)
        assert isinstance(response.tools_used, list)
        assert response.input_tokens > 0

    def test_tool_selection_reacts_to_the_question(self):
        auth = agent_v2("I forgot my password", ["Choose Forgot Password."])
        shipping = agent_v2("Where is my order", ["Delivery takes five days."])
        assert "auth_service" in auth.tools_used
        assert "lookup_order" in shipping.tools_used

    def test_weak_retrieval_triggers_a_refusal(self):
        response = agent_v2("What is the capital of Mongolia?", ["Delivery takes five days."])
        assert "not covered" in response.answer.lower()
