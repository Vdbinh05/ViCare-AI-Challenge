import json
from types import SimpleNamespace

import pytest

from benchmarks.check_demo import CASES, check_case
from benchmarks import evaluate_generation as benchmark
from src.config import Settings
from src.generation import Answer
from src.safety import DISCLAIMER


def test_fixed_denominator_and_private_text_on_mixed_results(tmp_path, monkeypatch):
    """Even an application exception, transport error and invalid judge format keep their row."""
    questions = [{"id": f"d{i:03d}", "source_id": f"validation-{i}", "gold_doc_ids": [f"doc-{i}"],
                  "question": f"private-question-{i}"} for i in range(10)]
    validation = [{"id": i, "answer": "private-reference"} for i in range(10)]
    monkeypatch.setattr(benchmark, "load_vihealthqa", lambda: {"validation": validation})
    monkeypatch.setattr(benchmark, "prepare_dataset", lambda ds: ([], [], questions))
    monkeypatch.setattr(benchmark, "dev_identity", lambda *args: {
        "corpus_sha256": "test-signature", "ordered_questions_sha256": "test-questions", "chunks": 10})
    model = SimpleNamespace(device="cpu")
    monkeypatch.setattr(benchmark, "RecordedSearch", lambda *_: SimpleNamespace(
        last=None, retriever=SimpleNamespace(embedder=SimpleNamespace(model=model), hybrid_backend="locked")))
    asked = []

    def ask(q, method):
        asked.append((q, method))
        if q.endswith("-2"):
            raise RuntimeError("private-exception")
        if q.endswith("-1"):
            return Answer("unavailable", "private-error", method, reason_code="GENERATION_FAILURE")
        return Answer("refused", "private-refusal", method)

    monkeypatch.setattr(benchmark, "Assistant", lambda *args, **kwargs: SimpleNamespace(
        ask=ask, generate=SimpleNamespace(prepare=lambda: None), answerability_check=lambda *_: ""))
    installed = SimpleNamespace(model=Settings().llm_model, digest="test-digest")
    monkeypatch.setattr(benchmark, "Client", lambda **kwargs: SimpleNamespace(
        list=lambda: SimpleNamespace(models=[installed]), _client=SimpleNamespace(close=lambda: None)))

    def judge(_, model, question, reference, prediction):
        assert reference == "private-reference"
        if question.endswith("-3"):
            return None, "private-invalid"
        if question.endswith("-4"):
            raise OSError("private-transport")
        return 1, "1"

    monkeypatch.setattr(benchmark, "judge", judge)
    output = tmp_path / "evaluation.json"
    result = benchmark.run(output, settings=Settings())
    assert asked == [(row["question"], "Hybrid") for row in questions]
    assert (result["n_questions"], result["raw_judge_ones"], result["judge_hits"]) == (10, 7, 6)
    assert result["invalid_judge_format"] == result["judge_request_errors"] == 1
    assert result["judge_skipped_application_exception"] == 1
    assert result["technical_failures"] == 2
    assert result["refusals"] == 8
    assert result["answered"] == result["judge_positive_answers"] == 0
    assert result["answer_rate"] == result["supported_answer_yield"] == 0
    assert result["judge_positive_refusals"] == 6
    assert result["judge_by_status"]["unavailable"]["ones"] == 1
    assert json.loads(output.read_text(encoding="utf8"))["rows"] == result["rows"]
    for path in (output, output.with_suffix(".csv")):
        assert "private-" not in path.read_text(encoding="utf8")


def test_answer_yield_keeps_full_denominator_and_separates_safe_actions_from_answers():
    rows = [{"status": status, "judge_score": score} for status, score in [
        ("answered", 1), ("answered", 0), ("answered", None),
        ("refused", 1), ("refused", 1), ("refused", 0), ("refused", None),
        ("emergency", 1), ("unavailable", 1), ("invalid", 1),
    ]]
    result = benchmark.response_metrics(rows)
    assert result["answered"] == 3 and result["answer_rate"] == .3
    assert result["judge_positive_answers"] == 1 and result["supported_answer_yield"] == .1
    assert result["judge_positive_refusals"] == 2 and result["judge_positive_emergencies"] == 1
    assert result["judge_by_status"]["answered"] == {"total": 3, "ones": 1, "zeros": 1, "unscored": 1}
    assert sum(counts["total"] for counts in result["judge_by_status"].values()) == 10


@pytest.mark.parametrize("quote,passed", [
    ("Nhịp tim ảnh hưởng đến kết quả đo lưu huyết não.", True),
    ("Tài liệu mô tả phép đo lưu huyết não.\nNhịp tim ảnh hưởng đến kết quả đo lưu huyết não.", True),
    ("Tài liệu mô tả phép đo lưu huyết não.\n"
     "Nhịp tim ảnh hưởng đến kết quả đo lưu huyết não.\n"
     "Huyết áp là một chỉ số sức khỏe.", True),
    ("Kết quả đo lưu huyết não bị ảnh hưởng bởi huyết áp.", True),
    ("Huyết áp bị ảnh hưởng bởi kết quả đo lưu huyết não.", False),
    ("Kết quả đo lưu huyết não được ghi lại.\n"
     "Giấc ngủ bị ảnh hưởng bởi tiếng ồn.\n"
     "Huyết áp là một chỉ số sức khỏe.", False),
    ("Kết quả đo lưu huyết não được ghi lại."
     "Giấc ngủ bị ảnh hưởng bởi tiếng ồn."
     "Huyết áp là một chỉ số sức khỏe.", False),
])
def test_demo_checks_measurement_relation_separately_from_publication(quote, passed):
    source = SimpleNamespace(doc_id="alternative", chunk_id="alternative#2", url="", text=quote)
    reply = Answer("answered", f"Tài liệu [1] ghi “{quote}”\n\n{DISCLAIMER}", "Hybrid",
                   hits=[SimpleNamespace(chunk=source)], reason_code="SUPPORTED", citation_ids=["1"])
    row = check_case(SimpleNamespace(ask=lambda *_: reply), CASES[0])
    assert row["passed"] is passed
    assert row["answer"] is None
    assert row["sources"][0]["chunk_id"] == "alternative#2"
