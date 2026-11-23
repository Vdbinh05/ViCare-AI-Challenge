from types import SimpleNamespace

import pytest

from src import evaluation
from src.config import Settings
from src.dataset import Chunk, dev_identity, prepare_dataset, question_sha256
from src.evaluation import METHODS, WARMUP, coverage, recall_at_k, run_paired, summarize
from src.retrieval import Hit


def test_recall_at_k_matches_notebook_binary_any_gold_per_question():
    gold = ["d1", "d2"]
    retrieved = ["d0", "d2", "d3"]
    assert recall_at_k(gold, retrieved, 1) == 0
    assert recall_at_k(gold, retrieved, 2) == 1
    assert recall_at_k(["d4"], retrieved, 10) == 0
    assert recall_at_k(gold, [], 10) == 0


def test_coverage_denominator_includes_uncovered_questions():
    questions = [{"gold_doc_ids": ["d1", "d2"]}, {"gold_doc_ids": ["d9"]}, {"gold_doc_ids": ["d3"]}]
    assert coverage(questions, {"d1", "d3"}) == pytest.approx(2 / 3)
    assert coverage(questions, set()) == 0


def test_dev_hash_is_sensitive_to_order_and_original_question_whitespace():
    questions = [{"question": "Câu 1"}, {"question": "Câu 2"}]
    assert question_sha256(questions) != question_sha256(questions[::-1])
    assert question_sha256(questions) != question_sha256([{"question": " Câu 1"}, questions[1]])


def test_dev_identity_stops_comparison_with_other_data():
    ds = {
        "train": [{"id": 1, "question": "Câu hỏi", "answer": "Văn bản tham khảo", "link": None}],
        "validation": [{"id": 2, "question": "ViHealthQA?", "answer": "Văn bản tham khảo", "link": None}],
        "test": [],
    }
    with pytest.raises(ValueError, match="Dừng so sánh"):
        dev_identity(ds, *prepare_dataset(ds))


def test_paired_benchmark_warms_up_then_measures_each_method_in_the_same_question_order():
    calls = []

    def search(question, method, k):
        calls.append((question, method, k))
        doc_id = "d1" if method == "Dense" else "d2"
        chunk = Chunk(f"{doc_id}#0", doc_id, "", "Văn bản tổng hợp.")
        return SimpleNamespace(hits=[Hit(chunk, 0.99)])

    dev = [{"id": "qa1", "question": "Câu 1", "source_id": "s1", "gold_doc_ids": ["d1"]},
           {"id": "qa2", "question": "Câu 2", "source_id": "s2", "gold_doc_ids": ["d2"]}]
    rows = run_paired(dev, SimpleNamespace(search=search))
    assert calls == [*((question, method, 10) for question in WARMUP for method in METHODS),
                     ("Câu 1", "Dense", 10), ("Câu 1", "Hybrid", 10),
                     ("Câu 2", "Hybrid", 10), ("Câu 2", "Dense", 10)]
    assert [(row["id"], row["method"], row["hit@1"]) for row in rows] == [
        ("qa1", "Dense", 1), ("qa1", "Hybrid", 0), ("qa2", "Hybrid", 1), ("qa2", "Dense", 0),
    ]
    assert all(row["latency_ms"] >= 0 for row in rows)
    metrics = summarize(rows, dev, {"d1", "d2"})
    for method in METHODS:
        assert metrics[method]["recall@1"] == 0.5
        assert metrics[method]["coverage"] == 1
        assert metrics[method]["latency_ms"]["p95"] >= metrics[method]["latency_ms"]["median"]


def test_evaluation_rejects_a_changed_persisted_chunk_with_unchanged_id(monkeypatch, tmp_path):
    original = Chunk("a#0", "a", "https://example.org", "Nội dung gốc")
    altered = Chunk("a#0", "a", "https://example.org", "Nội dung đã thay đổi")
    monkeypatch.setattr(evaluation, "load_vihealthqa", lambda: {})
    monkeypatch.setattr(evaluation, "prepare_dataset", lambda _: ([], [original], []))
    monkeypatch.setattr(evaluation, "dev_identity", lambda *args: {"corpus_sha256": "qa-only"})
    monkeypatch.setattr(evaluation, "open_retriever", lambda *args, **kwargs: SimpleNamespace(chunks=[altered]))
    with pytest.raises(ValueError, match="không khớp kho dev"):
        evaluation.measure(Settings(), tmp_path)
    assert not list(tmp_path.iterdir())
