"""Run bootcamp section 8 on locked dev[:10] with the local application and judge."""
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from ollama import Client

from src.config import DATASET_REVISION, PROJECT_ROOT, Settings
from src.dataset import dev_identity, load_vihealthqa, prepare_dataset
from src.generation import Assistant
from src.retrieval import open_retriever


JUDGE = """Bạn là giám khảo đối chiếu câu trả lời với đáp án tham khảo.
CÂU HỎI: {q}
ĐÁP ÁN THAM KHẢO: {gold}
CÂU TRẢ LỜI CẦN CHẤM: {pred}
Chấm 1 nếu câu trả lời thể hiện đúng ý chính theo đáp án tham khảo,
ngược lại chấm 0. Nếu là từ chối khi đáp án có thể trả lời, chấm 0.
Chỉ trả lời một ký tự 0 hoặc 1, không giải thích."""
N_JUDGE = 10
NOTEBOOK_REFUSAL = "Tôi không tìm thấy thông tin này trong kho tài liệu."


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


class RecordedSearch:
    def __init__(self, settings):
        self.retriever = open_retriever(settings)
        self.last = None

    def search(self, question, method, k):
        self.last = self.retriever.search(question, method, k)
        return self.last


def judge(client, model, question, reference, prediction):
    prompt = JUDGE.format(q=question, gold=str(reference or "")[:1500], pred=prediction)
    response = client.chat(model=model, options={"temperature": 0, "num_predict": 8},
                           messages=[{"role": "user", "content": prompt}], stream=False)
    raw = (response.message.content or "").strip()
    return (int(raw) if re.fullmatch(r"[01]", raw) else None), raw


def run():
    settings = Settings.from_env()
    ds = load_vihealthqa()
    corpus, chunks, dev = prepare_dataset(ds)
    identity = dev_identity(ds, corpus, chunks, dev)
    questions = dev[:N_JUDGE]
    wanted = {entry["source_id"] for entry in questions}
    references = {f"validation-{row['id']}": row["answer"] for row in ds["validation"]
                  if f"validation-{row['id']}" in wanted}
    del corpus, chunks, ds, dev
    private_path = PROJECT_ROOT / ".private" / "generation" / "dev10.json"
    output = PROJECT_ROOT / "results" / "submission" / "generation.json"
    if len(questions) != N_JUDGE:
        raise ValueError("Sai số lượng câu hỏi dev[:10].")
    search = RecordedSearch(settings)
    assistant = Assistant(settings, retriever_factory=lambda: search)
    assistant.generate.prepare()
    client = Client(host=settings.ollama_host, trust_env=False, follow_redirects=False,
                    timeout=httpx.Timeout(settings.timeout, connect=min(3, settings.timeout)))
    # Include the model digest so future users can distinguish tags that changed weights.
    models = client.list().models
    digest = next(item.digest for item in models if item.model == settings.llm_model)
    records, public_rows = [], []
    try:
        for entry in questions:
            key, question = entry["id"], entry["question"]
            search.last = None
            started = time.perf_counter()
            # The only input passed to the assistant is the question and fixed method.
            response = assistant.ask(question, "Hybrid")
            generation_ms = round(1000 * (time.perf_counter() - started), 1)
            reference = references[entry["source_id"]]
            started = time.perf_counter()
            error = None
            try:
                score, raw = judge(client, settings.llm_model, question, reference, response.answer)
            except Exception as exc:
                # No server bodies, queries or responses in console or shared artifacts.
                error, raw, score = type(exc).__name__, None, None
            judge_ms = round(1000 * (time.perf_counter() - started), 1)
            hits = search.last.hits if search.last is not None else []
            docs = [h.chunk.doc_id for h in hits]
            public = {
                "id": key, "source_id": entry["source_id"], "gold_doc_ids": entry["gold_doc_ids"],
                "retrieved_doc_ids": docs, "gold_found": bool(set(entry["gold_doc_ids"]) & set(docs)),
                "visible_doc_ids": [h.chunk.doc_id for h in response.hits],
                "cited_doc_ids": [response.hits[int(i) - 1].chunk.doc_id for i in response.citation_ids],
                "top_dense_score": search.last.top_dense_score if search.last is not None else None,
                "status": response.status, "reason_code": response.reason_code,
                "judge_score": score, "judge_error_type": error, "generation_ms": generation_ms,
                "judge_ms": judge_ms,
            }
            public_rows.append(public)
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
    notebook_literal_refusals = sum(NOTEBOOK_REFUSAL in r["prediction"] for r in records)
    codes = Counter(r["reason_code"] for r in public_rows)
    result = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": DATASET_REVISION, "corpus_sha256": identity["corpus_sha256"],
        "dev100_ordered_questions_sha256": identity["ordered_questions_sha256"],
        "selection": "dev[:10]", "method": "Hybrid", "candidate_pool": settings.candidate_pool,
        "hybrid_dense_depth": settings.hybrid_dense_depth, "rrf_k": settings.rrf_k,
        "top_k": settings.top_k, "dense_threshold": settings.threshold,
        "generation_model": settings.llm_model, "judge_model": settings.llm_model, "model_digest": digest,
        "judge_prompt_sha256": hashlib.sha256(JUDGE.encode()).hexdigest(),
        "judge_options": {"temperature": 0, "num_predict": 8}, "judge_gold_max_chars": 1500,
        "n_questions": N_JUDGE, "judge_hits": scores.count(1), "judge_rate": scores.count(1) / N_JUDGE,
        "invalid_judgments_including_errors": scores.count(None),
        "judge_request_errors": sum(r["judge_error_type"] is not None for r in public_rows),
        "refusals": refusals, "refusal_rate": refusals / N_JUDGE,
        "refusal_definition": "status == refused; excludes technical failures and emergency redirects",
        "notebook_literal_refusals": notebook_literal_refusals,
        "notebook_literal_refusal_rate": notebook_literal_refusals / N_JUDGE,
        "technical_failures": sum(r["status"] == "unavailable" for r in public_rows),
        "reason_counts": codes, "rows": public_rows,
    }
    write_json(output, result)
    print(f"Judge {scores.count(1)}/{N_JUDGE}, refusals {refusals}/{N_JUDGE}, invalid {scores.count(None)}")


if __name__ == "__main__":
    run()
