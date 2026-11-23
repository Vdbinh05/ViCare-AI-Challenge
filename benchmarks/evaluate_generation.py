"""Measure the actual application on dev[:10]; publish metrics and IDs only."""
import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from ollama import Client

from src.config import DATASET_REVISION, DEFAULT_LLM_MODEL, EMBEDDING_REVISION, PROJECT_ROOT, Settings
from src.dataset import dev_identity, load_vihealthqa, prepare_dataset
from src.generation import Answer, Assistant, validate_answerability_result
from src.retrieval import dense_candidate_count, open_retriever


JUDGE = """Bạn là giám khảo đối chiếu câu trả lời với đáp án tham khảo.
CÂU HỎI: {q}
ĐÁP ÁN THAM KHẢO: {gold}
CÂU TRẢ LỜI CẦN CHẤM: {pred}
Chấm 1 nếu câu trả lời thể hiện đúng ý chính theo đáp án tham khảo,
ngược lại chấm 0. Nếu là từ chối khi đáp án có thể trả lời, chấm 0.
Chỉ trả lời một ký tự 0 hoặc 1, không giải thích."""
N_JUDGE = 10
NOTEBOOK_REFUSAL = "Tôi không tìm thấy thông tin này trong kho tài liệu."
DEFAULT_OUTPUT = PROJECT_ROOT / ".private" / "reruns" / "generation.json"
JUDGE_OPTIONS = {"temperature": 0, "num_predict": 8}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


class RecordedSearch:
    def __init__(self, settings, signature, hybrid_backend):
        options = {"hybrid_backend": "locked"} if hybrid_backend == "locked" else {}
        self.retriever = open_retriever(settings, expected_signature=signature, **options)
        self.last = None

    def search(self, question, method, k):
        self.last = self.retriever.search(question, method, k)
        return self.last


def judge(client, model, question, reference, prediction):
    prompt = JUDGE.format(q=question, gold=str(reference or "")[:1500], pred=prediction)
    response = client.chat(model=model, options=JUDGE_OPTIONS,
                           messages=[{"role": "user", "content": prompt}], stream=False)
    raw = (response.message.content or "").strip()
    return (int(raw) if re.fullmatch(r"[01]", raw) else None), raw


def latency_summary(rows, field):
    values = [row[field] for row in rows]
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": statistics.quantiles(values, n=100, method="inclusive")[94]}


def response_metrics(rows):
    """Separate answering coverage from the legacy judge, which can reward refusals."""
    by_status = {}
    for row in rows:
        counts = by_status.setdefault(row["status"], {"total": 0, "ones": 0, "zeros": 0, "unscored": 0})
        counts["total"] += 1
        score = row["judge_score"]
        counts["ones" if score == 1 else "zeros" if score == 0 else "unscored"] += 1
    answered = by_status.get("answered", {}).get("total", 0)
    positive_answers = by_status.get("answered", {}).get("ones", 0)
    return {
        "answered": answered,
        "answer_rate": answered / len(rows) if rows else None,
        "judge_positive_answers": positive_answers,
        "supported_answer_yield": positive_answers / len(rows) if rows else None,
        "supported_answer_yield_definition": (
            "status == answered and raw Judge == 1, divided by all questions. "
            "A descriptive model-judged answer yield, not verified clinical accuracy. "
            "Refusals, emergencies and failures receive no yield credit; "
            "the correctness of safe refusals is not assessed by this metric."
        ),
        "judge_positive_refusals": by_status.get("refused", {}).get("ones", 0),
        "judge_positive_emergencies": by_status.get("emergency", {}).get("ones", 0),
        "judge_by_status": by_status,
    }


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list)
                             else value for key, value in row.items()})
    temporary.replace(path)


def run(output=DEFAULT_OUTPUT, method="Hybrid", run_label="rerun", save_review_inputs=False,
        hybrid_backend="selected", settings=None):
    settings = settings or Settings.from_env()
    output = Path(output)
    if output.suffix.lower() != ".json" or not re.fullmatch(r"[A-Za-z0-9_-]+", run_label):
        raise ValueError("Cần đường dẫn .json và tên lượt chạy chỉ gồm chữ, số, '_' hoặc '-'.")
    if settings.llm_model != DEFAULT_LLM_MODEL:
        raise ValueError(f"Protocol yêu cầu {DEFAULT_LLM_MODEL}.")
    ds = load_vihealthqa()
    corpus, chunks, dev = prepare_dataset(ds)
    identity = dev_identity(ds, corpus, chunks, dev)
    questions = dev[:N_JUDGE]
    wanted = {entry["source_id"] for entry in questions}
    references = {f"validation-{row['id']}": row["answer"] for row in ds["validation"]
                  if f"validation-{row['id']}" in wanted}
    del corpus, chunks, ds, dev
    private_path = PROJECT_ROOT / ".private" / "generation" / run_label / "dev10.json"
    if output.resolve() == private_path.resolve():
        raise ValueError("Tệp chỉ số phải khác tệp đối chiếu riêng.")
    if len(questions) != N_JUDGE:
        raise ValueError("Sai số lượng câu hỏi dev[:10].")
    search = RecordedSearch(settings, identity["corpus_sha256"], hybrid_backend)
    assistant = Assistant(settings, retriever_factory=lambda: search)
    # Observe the production check verbatim, and decode its returned value after ask().
    check_answerability = assistant.answerability_check
    evidence_raw = None

    def record_check(question, hits):
        nonlocal evidence_raw
        evidence_raw = check_answerability(question, hits)
        return evidence_raw

    assistant.answerability_check = record_check
    assistant.generate.prepare()
    client = Client(host=settings.ollama_host, trust_env=False, follow_redirects=False,
                    timeout=httpx.Timeout(settings.timeout, connect=min(3, settings.timeout)))
    # Include the model digest so future users can distinguish tags that changed weights.
    models = client.list().models
    digest = next(item.digest for item in models if item.model == settings.llm_model)
    records, public_rows = [], []
    notebook_literal_refusals = 0
    try:
        for entry in questions:
            key, question = entry["id"], entry["question"]
            search.last = None
            evidence_raw = None
            started = time.perf_counter()
            # The only input passed to the assistant is the question and fixed method.
            application_error = None
            try:
                response = assistant.ask(question, method)
            except Exception as exc:
                application_error = type(exc).__name__
                response = Answer("unavailable", "", method, reason_code="UNCAUGHT_APPLICATION_ERROR")
            generation_ms = round(1000 * (time.perf_counter() - started), 1)
            reference = references[entry["source_id"]]
            started = time.perf_counter()
            error, raw, score = None, None, None
            if application_error is None:
                try:
                    score, raw = judge(client, settings.llm_model, question, reference, response.answer)
                except Exception as exc:
                    # Only an exception class is published.
                    error = type(exc).__name__
            judge_ms = round(1000 * (time.perf_counter() - started), 1)
            hits = search.last.hits if search.last is not None else []
            docs = [h.chunk.doc_id for h in hits]
            evidence = validate_answerability_result(evidence_raw, hits) if evidence_raw is not None else None
            notebook_literal_refusals += int(NOTEBOOK_REFUSAL in response.answer)
            public = {
                "id": key, "source_id": entry["source_id"], "gold_doc_ids": entry["gold_doc_ids"],
                "retrieved_doc_ids": docs, "gold_found": bool(set(entry["gold_doc_ids"]) & set(docs)),
                "retrieved_chunk_ids": [h.chunk.chunk_id for h in hits],
                "evidence_verdict": evidence.verdict if evidence else None,
                "evidence_reason": evidence.reason if evidence else None,
                "evidence_doc_ids": [docs[int(i) - 1] for i in evidence.supporting_ids] if evidence else [],
                "visible_doc_ids": [h.chunk.doc_id for h in response.hits],
                "cited_doc_ids": [response.hits[int(i) - 1].chunk.doc_id for i in response.citation_ids],
                "cited_chunk_ids": [response.hits[int(i) - 1].chunk.chunk_id for i in response.citation_ids],
                "top_dense_score": search.last.top_dense_score if search.last is not None else None,
                "status": response.status, "reason_code": response.reason_code,
                "judge_score": score, "judge_error_type": error, "application_error_type": application_error,
                "generation_ms": generation_ms,
                "credited_score": int(score == 1 and response.status not in {"unavailable", "invalid"}),
                "judge_ms": judge_ms,
            }
            public_rows.append(public)
            if save_review_inputs:
                records.append({**public, "question": question, "gold": reference, "prediction": response.answer,
                                "judge_raw": raw, "citation_ids": response.citation_ids,
                                "hits": [{"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id,
                                          "text": h.chunk.text, "score": h.score} for h in hits]})
                write_json(private_path, records)
            print(key, response.status, response.reason_code, "judge", score, flush=True)
    finally:
        client._client.close()  # Ollama 0.5.4 has no public close/context manager.
    scores = [r["judge_score"] for r in public_rows]
    refusals = sum(r["status"] == "refused" for r in public_rows)
    codes = Counter(r["reason_code"] for r in public_rows)
    credited = sum(r["credited_score"] for r in public_rows)
    judge_errors = sum(r["judge_error_type"] is not None for r in public_rows)
    skipped = sum(r["application_error_type"] is not None for r in public_rows)
    code_paths = sorted((PROJECT_ROOT / "src").glob("*.py")) + [Path(__file__)]
    result = {
        "finished_utc": datetime.now(timezone.utc).isoformat(), "run_label": run_label,
        "platform": platform.platform(), "python": platform.python_version(),
        "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix():
                          hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                          for path in code_paths},
        "dataset_revision": DATASET_REVISION, "corpus_sha256": identity["corpus_sha256"],
        "dev100_ordered_questions_sha256": identity["ordered_questions_sha256"],
        "selection": "dev[:10]", "method": method, "candidate_pool": settings.candidate_pool,
        "hybrid_configuration": hybrid_backend if method == "Hybrid" else None,
        "runtime_backend": search.retriever.hybrid_backend if method == "Hybrid" else "dense_hnsw",
        "indexed_chunks": identity["chunks"],
        "dense_lookup_depth_chunks": (dense_candidate_count(settings.top_k) if method == "Dense"
                                      else max(settings.hybrid_dense_depth, settings.candidate_pool)),
        "display_budget_documents": settings.top_k,
        "embedding_revision": EMBEDDING_REVISION, "embedding_device": str(search.retriever.embedder.model.device),
        "dense_weight": settings.dense_weight, "sparse_weight": settings.sparse_weight,
        "hybrid_dense_depth": settings.hybrid_dense_depth, "rrf_k": settings.rrf_k,
        "top_k": settings.top_k, "dense_threshold": settings.threshold,
        "generation_model": settings.llm_model, "judge_model": settings.llm_model, "model_digest": digest,
        "judge_prompt_sha256": hashlib.sha256(JUDGE.encode()).hexdigest(),
        "judge_options": JUDGE_OPTIONS, "judge_gold_max_chars": 1500,
        "generation_temperature": settings.temperature, "generation_max_output_tokens": settings.max_output_tokens,
        "timeout_seconds": settings.timeout,
        "n_questions": N_JUDGE, "judge_hits": credited, "judge_rate": credited / N_JUDGE,
        "score_definition": "All 10 questions. Invalid judge, judge error or application failure scores zero.",
        **response_metrics(public_rows),
        "raw_judge_ones": scores.count(1), "raw_judge_zeros": scores.count(0),
        "invalid_judgments_including_errors": scores.count(None) - skipped,
        "invalid_judge_format": scores.count(None) - skipped - judge_errors,
        "judge_request_errors": judge_errors, "judge_skipped_application_exception": skipped,
        "refusals": refusals, "refusal_rate": refusals / N_JUDGE,
        "refusal_definition": "status == refused; excludes technical failures and emergency redirects",
        "notebook_literal_refusals": notebook_literal_refusals,
        "notebook_literal_refusal_rate": notebook_literal_refusals / N_JUDGE,
        "technical_failures": sum(r["status"] == "unavailable" for r in public_rows),
        "status_counts": Counter(r["status"] for r in public_rows), "reason_counts": codes,
        "latency_ms": {"ask": latency_summary(public_rows, "generation_ms"),
                       "judge": latency_summary(public_rows, "judge_ms")},
        "rows": public_rows,
    }
    write_json(output, result)
    write_csv(output.with_suffix(".csv"), public_rows)
    print(f"Legacy Judge {credited}/{N_JUDGE}, answered {result['answered']}/{N_JUDGE}, "
          f"Judge-positive answers {result['judge_positive_answers']}/{N_JUDGE}, "
          f"Judge-positive refusals {result['judge_positive_refusals']}/{refusals}, "
          f"invalid {scores.count(None)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Public metric JSON and sibling CSV")
    parser.add_argument("--method", choices=("Hybrid", "Dense"), default="Hybrid")
    parser.add_argument("--hybrid-backend", choices=("selected", "locked"), default="selected")
    parser.add_argument("--run-label", default="rerun")
    parser.add_argument("--save-review-inputs", action="store_true",
                        help="Save restricted benchmark texts under .private/generation/<run-label>/")
    run(**vars(parser.parse_args()))
