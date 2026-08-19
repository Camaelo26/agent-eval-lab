import pytest

from evalkit.dataset import GoldenCase, filter_by_tag, load_jsonl


class TestLoadJsonl:
    def test_loads_the_shipped_golden_set(self, golden_path):
        cases = load_jsonl(golden_path)
        assert len(cases) >= 10
        assert all(isinstance(c, GoldenCase) for c in cases)

    def test_every_case_has_context_and_an_expected_answer(self, golden_path):
        for case in load_jsonl(golden_path):
            assert case.context, f"{case.id} has no context"
            assert case.expected.strip(), f"{case.id} has no expected answer"

    def test_ids_are_unique(self, golden_path):
        ids = [c.id for c in load_jsonl(golden_path)]
        assert len(ids) == len(set(ids))

    def test_the_suite_actually_covers_unanswerable_questions(self, golden_path):
        # A suite of only answerable questions cannot detect over-answering,
        # which is the failure mode that puts wrong facts in front of users.
        cases = load_jsonl(golden_path)
        assert any(c.unanswerable for c in cases)

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "small.jsonl"
        path.write_text(
            "# a comment\n\n"
            '{"id": "a", "question": "q", "context": ["c"], "expected": "e"}\n',
            encoding="utf-8",
        )
        assert len(load_jsonl(path)) == 1

    def test_missing_required_field_is_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "question": "q"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing fields"):
            load_jsonl(path)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "dupe.jsonl"
        row = '{"id": "a", "question": "q", "context": ["c"], "expected": "e"}\n'
        path.write_text(row + row, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate case id"):
            load_jsonl(path)

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("# only a comment\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no cases"):
            load_jsonl(path)


class TestFilterByTag:
    def test_filters_to_the_requested_tag(self, golden_path):
        cases = load_jsonl(golden_path)
        billing = filter_by_tag(cases, "billing")
        assert billing
        assert all("billing" in c.tags for c in billing)
