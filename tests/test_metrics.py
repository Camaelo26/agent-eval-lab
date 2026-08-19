from evalkit import metrics


class TestExactMatch:
    def test_ignores_case_and_punctuation(self):
        assert metrics.exact_match("Thirty days.", "thirty days") == 1.0

    def test_different_text_scores_zero(self):
        assert metrics.exact_match("thirty days", "sixty days") == 0.0


class TestTokenF1:
    def test_identical_text_scores_one(self):
        assert metrics.token_f1("refund within 30 days", "refund within 30 days") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert metrics.token_f1("delivery times", "password reset") == 0.0

    def test_partial_overlap_gives_partial_credit(self):
        score = metrics.token_f1("refunds take 30 days", "refunds take 30 business days")
        assert 0.5 < score < 1.0

    def test_stopwords_do_not_inflate_the_score(self):
        # Two answers that share only filler words must not look similar.
        assert metrics.token_f1("it is in the of a", "the of a it is in") == 1.0
        assert metrics.token_f1("the account is locked", "the parcel is shipped") == 0.0

    def test_empty_prediction_scores_zero(self):
        assert metrics.token_f1("", "some expected answer") == 0.0


class TestGroundedness:
    def test_answer_copied_from_context_is_fully_grounded(self):
        context = ["Annual plans are refundable within 30 days of the charge date."]
        assert metrics.groundedness("refundable within 30 days", context) == 1.0

    def test_invented_content_lowers_the_score(self):
        context = ["Annual plans are refundable within 30 days."]
        grounded = metrics.groundedness("refundable within 30 days by wire transfer only", context)
        assert grounded < 1.0

    def test_fully_invented_answer_scores_zero(self):
        assert metrics.groundedness("quantum tunnelling", ["delivery takes five days"]) == 0.0

    def test_no_context_scores_zero(self):
        assert metrics.groundedness("anything", []) == 0.0


class TestRefusalDetection:
    def test_detects_common_refusal_phrasings(self):
        assert metrics.is_refusal("That is not covered in the documentation.")
        assert metrics.is_refusal("I have no information about that.")

    def test_normal_answer_is_not_a_refusal(self):
        assert not metrics.is_refusal("Delivery takes three to five business days.")


class TestToolCallScore:
    def test_exact_set_match_scores_one(self):
        assert metrics.tool_call_score(["a", "b"], ["b", "a"]) == 1.0

    def test_extra_tools_cost_precision(self):
        # Calling every tool you own is not the same as solving the task.
        score = metrics.tool_call_score(["a", "b", "c", "d"], ["a", "b"])
        assert 0.0 < score < 1.0

    def test_missing_tools_cost_recall(self):
        assert metrics.tool_call_score(["a"], ["a", "b"]) < 1.0

    def test_no_tools_expected_or_used_scores_one(self):
        assert metrics.tool_call_score([], []) == 1.0

    def test_calling_tools_when_none_expected_scores_zero(self):
        assert metrics.tool_call_score(["a"], []) == 0.0


class TestCostModel:
    def test_cost_is_priced_per_million_tokens(self):
        model = metrics.CostModel(input_per_mtok=3.0, output_per_mtok=15.0)
        assert model.cost(1_000_000, 0) == 3.0
        assert model.cost(0, 1_000_000) == 15.0
        assert model.cost(500_000, 100_000) == 1.5 + 1.5


class TestPercentile:
    def test_p50_of_a_known_series(self):
        assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p95_tracks_the_tail_not_the_mean(self):
        values = [1.0] * 99 + [500.0]
        assert metrics.percentile(values, 95) == 1.0
        assert metrics.percentile(values, 100) == 500.0

    def test_empty_series_is_zero(self):
        assert metrics.percentile([], 95) == 0.0
