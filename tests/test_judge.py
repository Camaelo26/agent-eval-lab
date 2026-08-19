import pytest

from evalkit.judge import RubricJudge, audit_judge, get_judge


@pytest.fixture
def rubric():
    return RubricJudge()


class TestRubricJudge:
    def test_identical_answer_passes(self, rubric):
        verdict = rubric.score(
            "How long is the refund window?",
            "Annual plans are refundable within 30 days.",
            "Annual plans are refundable within 30 days.",
            ["Annual plans are refundable within 30 days."],
        )
        assert verdict.passed
        assert verdict.score > 0.9

    def test_unrelated_answer_fails(self, rubric):
        verdict = rubric.score(
            "How long is the refund window?",
            "Delivery takes three to five business days.",
            "Annual plans are refundable within 30 days.",
            ["Annual plans are refundable within 30 days."],
        )
        assert not verdict.passed

    def test_refusing_an_answerable_question_fails(self, rubric):
        verdict = rubric.score(
            "How long does delivery take?",
            "I have no information about that.",
            "Three to five business days.",
            ["Standard delivery arrives in three to five business days."],
        )
        assert not verdict.passed

    def test_correct_refusal_on_an_unanswerable_question_passes(self, rubric):
        verdict = rubric.score(
            "Which competitor is cheapest?",
            "That is not covered in the documentation I have access to.",
            "That is not covered in the documentation I have access to.",
            ["Standard delivery arrives in three to five business days."],
        )
        assert verdict.passed

    def test_answering_an_unanswerable_question_fails(self, rubric):
        verdict = rubric.score(
            "Which competitor is cheapest?",
            "Their enterprise tier is 19 dollars per seat.",
            "That is not covered in the documentation I have access to.",
            ["Standard delivery arrives in three to five business days."],
        )
        assert not verdict.passed

    def test_parroting_the_context_without_answering_does_not_pass(self, rubric):
        # Groundedness alone would score this 1.0. The rubric weights reference
        # overlap higher precisely so that this case still fails.
        verdict = rubric.score(
            "How do I download a past invoice?",
            "Support hours are Monday to Friday, 8am to 6pm Central Time.",
            "Go to Settings, Billing, Invoice History and download the PDF.",
            ["Support hours are Monday to Friday, 8am to 6pm Central Time."],
        )
        assert not verdict.passed

    def test_verdict_is_deterministic(self, rubric):
        args = (
            "How long is the refund window?",
            "Refundable within 30 days of the charge.",
            "Annual plans are refundable within 30 days of the charge date.",
            ["Annual plans are refundable within 30 days of the charge date."],
        )
        first = rubric.score(*args)
        second = rubric.score(*args)
        assert first == second


class TestGetJudge:
    def test_rubric_is_selectable_by_name(self):
        assert get_judge("rubric").name == "rubric"

    def test_auto_falls_back_to_rubric_without_an_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert get_judge("auto").name == "rubric"

    def test_auto_upgrades_when_a_key_is_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert get_judge("auto").name == "anthropic"

    def test_unknown_judge_is_rejected(self):
        with pytest.raises(ValueError):
            get_judge("vibes")


class TestJudgeAudit:
    def test_the_shipped_rubric_judge_passes_its_own_audit(self, trap_cases):
        # If this ever fails, the harness has stopped being trustworthy and the
        # gate is supposed to block releases until the judge is fixed.
        audit = audit_judge(RubricJudge(), trap_cases)
        assert audit.total == len(trap_cases)
        assert audit.trustworthy(), (
            f"accuracy={audit.accuracy:.2f} false_pass_rate={audit.false_pass_rate:.2f}"
        )

    def test_the_known_weakness_is_pinned(self):
        # Documented failure, not an accident. An overlap-based judge cannot see
        # a single flipped number: "50 minutes" vs "15 minutes" shares every
        # other token, so it scores 0.89 while a CORRECT paraphrase scores 0.71.
        # A judge that ranks a wrong fact above a right one is exactly the reason
        # a semantic judge is worth its cost. Pinned as a test so that if the
        # rubric is ever improved, this test fails and the README gets updated.
        rubric = RubricJudge()
        wrong_number = rubric.score(
            "How long does an account stay locked?",
            "The account locks for 50 minutes after five failed sign-in attempts.",
            "The account locks for 15 minutes after five failed sign-in attempts.",
            ["An account locks for 15 minutes after five consecutive failed attempts."],
        )
        correct_paraphrase = rubric.score(
            "How do I reset my password?",
            "Hit Forgot Password at sign-in, type your work email, and use the link "
            "within 60 minutes before it expires.",
            "Use Forgot Password on the sign-in screen and enter your work email. "
            "The link expires in 60 minutes.",
            ["Choose Forgot Password on the sign-in screen. The link is valid 60 minutes."],
        )
        assert wrong_number.passed is True, "known weakness: flipped numbers slip through"
        assert wrong_number.score > correct_paraphrase.score

    def test_a_judge_that_passes_everything_is_caught(self, trap_cases):
        class AlwaysPass:
            name = "always-pass"

            def score(self, question, answer, expected, context):
                from evalkit.judge import Verdict

                return Verdict(passed=True, score=1.0, reason="")

        audit = audit_judge(AlwaysPass(), trap_cases)
        assert not audit.trustworthy()
        assert audit.false_pass > 0

    def test_a_judge_that_fails_everything_is_caught(self, trap_cases):
        class AlwaysFail:
            name = "always-fail"

            def score(self, question, answer, expected, context):
                from evalkit.judge import Verdict

                return Verdict(passed=False, score=0.0, reason="")

        audit = audit_judge(AlwaysFail(), trap_cases)
        assert not audit.trustworthy()
        assert audit.false_fail > 0
