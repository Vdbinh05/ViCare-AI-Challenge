"""Paired Dense and Hybrid retrieval benchmark on the verified original dev set."""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .config import CHUNK_SIZE, EMBEDDING_MODEL, EMBEDDING_REVISION, OVERLAP, Settings
from .dataset import dev_identity, load_vihealthqa, prepare_dataset
from .retrieval import dense_candidate_count, open_retriever

K_VALUES = (1, 3, 5, 10)
METHODS = ("Dense", "Hybrid")
WARMUP = (
    "Tiêm phòng viêm gan B",
    "Tập thể dục ảnh hưởng thế nào đến giấc ngủ?",
    "Tại sao cần khám răng định kỳ?",
)
LIMITATION = (
    "The evaluation corpus contains reference answers from the dataset splits. "
    "Recall@K measures retrieval of known reference documents. It does not measure "
    "real-world medical correctness and is not an independent generalization test."
)
REPRODUCE = ["python -m src.retrieval build", "python -m src.evaluation"]


def recall_at_k(gold_ids, retrieved_ids, k):
    if type(k) is not int or k < 1:
        raise ValueError("K phải là số nguyên dương.")
    return int(bool(set(gold_ids).intersection(retrieved_ids[:k])))


def coverage(questions, indexed_doc_ids):
    if not questions:
        raise ValueError("Cần câu hỏi để đo độ phủ.")
    indexed = set(indexed_doc_ids)
    return statistics.mean(bool(indexed.intersection(q["gold_doc_ids"])) for q in questions)


def summarize(rows, questions, indexed_doc_ids):
    cover = coverage(questions, indexed_doc_ids)
    results = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        latency = [row["latency_ms"] for row in selected]
        p95 = statistics.quantiles(latency, n=100, method="inclusive")[94] if len(latency) > 1 else latency[0]
        results[method] = {
            **{f"recall@{k}": statistics.mean(row[f"hit@{k}"] for row in selected) for k in K_VALUES},
            "coverage": cover,
            "latency_ms": {"mean": statistics.mean(latency), "median": statistics.median(latency), "p95": p95},
        }
    return results


def run_paired(questions, retriever):
    for question in WARMUP:
        for method in METHODS:
            retriever.search(question, method, max(K_VALUES))
    rows = []
    for index, q in enumerate(questions):
        for method in METHODS[::(-1 if index % 2 else 1)]:
            started = perf_counter()
            hits = retriever.search(q["question"], method, max(K_VALUES)).hits
            elapsed = (perf_counter() - started) * 1000
            gold = q["gold_doc_ids"]
            found = [hit.chunk.doc_id for hit in hits]
            rows.append({
                "id": q["id"], "source_id": q["source_id"], "method": method,
                "gold_doc_ids": gold, "retrieved_doc_ids": found,
                "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in hits],
                **{f"hit@{k}": recall_at_k(gold, found, k) for k in K_VALUES},
                "latency_ms": elapsed,
            })
    return rows


def paired_changes(rows):
    hits = {method: {row["id"]: row for row in rows if row["method"] == method} for method in METHODS}
    dense, hybrid = hits["Dense"], hits["Hybrid"]
    if dense.keys() != hybrid.keys():
        raise ValueError("Hai phương pháp phải có cùng bộ câu hỏi.")
    changes = {}
    for k in K_VALUES:
        key = f"hit@{k}"
        changes[key] = {
            "wins": [qid for qid in dense if hybrid[qid][key] > dense[qid][key]],
            "losses": [qid for qid in dense if hybrid[qid][key] < dense[qid][key]],
        }
    return changes


def table(report):
    dense, hybrid = (report["metrics"][method] for method in METHODS)
    dense_pool = report["settings"]["dense"]["candidate_pool_at_eval_k"]
    hybrid_pool = report["settings"]["hybrid"]["candidate_pool_per_branch"]
    hybrid_depth = report["settings"]["hybrid"].get("dense_query_depth", hybrid_pool)
    hybrid_backend = report["settings"]["hybrid"].get("backend", "locked")
    dense_stage = "cosine chính xác" if hybrid_backend == "exact" else "HNSW"
    lines = [
        "# So sánh truy xuất Dense và Hybrid", "", "Trạng thái **COMPLETED**.", "",
        "| Chỉ số | Dense | Hybrid | Hybrid − Dense |",
        "|---|---:|---:|---:|",
    ]
    for metric in [*(f"recall@{k}" for k in K_VALUES), "coverage"]:
        difference = (hybrid[metric] - dense[metric]) * 100
        label = {"coverage": "Coverage"}.get(metric, metric.capitalize())
        lines.append(f"| {label} | {dense[metric]:.1%} | {hybrid[metric]:.1%} | {difference:+.1f} đ.% |")
    for statistic, label in (("mean", "Trung bình"), ("median", "Trung vị"), ("p95", "p95")):
        a, b = (result["latency_ms"][statistic] for result in (dense, hybrid))
        lines.append(f"| {label} (ms) | {a:.2f} | {b:.2f} | {b - a:+.2f} ms |")
    lines += [
        "", f"Đo trên {report['n_questions']} câu dev theo thứ tự đã xác minh, không gọi LLM.",
        f"Dense lấy {dense_pool} đoạn khi đo top 10. Hybrid lấy {hybrid_depth} đoạn từ {dense_stage}, "
        f"giữ {hybrid_pool} đoạn theo cosine và lấy tối đa {hybrid_pool} tài liệu từ BM25. "
        "RRF hợp nhất theo mã tài liệu.",
        "Cả hai phương pháp khởi động bằng ba câu ngoài dev. Thứ tự câu hỏi giống nhau "
        "và phương pháp chạy trước được luân phiên theo câu.",
        "", "Coverage là tỷ lệ câu có ít nhất một tài liệu chuẩn trong chỉ mục dùng chung.",
        "", f"> {LIMITATION}", "",
    ]
    return "\n".join(lines)


def observed_examples(rows):
    dense = {row["id"]: row for row in rows if row["method"] == "Dense"}
    hybrid = [row for row in rows if row["method"] == "Hybrid"]
    lines = [
        "# Trường hợp quan sát được", "",
        "Mỗi dòng là một câu dev thật, định danh theo CSV. Thành công nghĩa là tìm thấy "
        "ít nhất một tài liệu chuẩn trong 10 tài liệu đầu. Dùng mã bộ dữ liệu để đối chiếu "
        "mà không sao chép nội dung câu hỏi hoặc tài liệu.", "",
    ]
    for success, heading in ((1, "Hybrid tìm thấy tài liệu chuẩn"), (0, "Hybrid bỏ sót tài liệu chuẩn")):
        matches = sorted((row for row in hybrid if row["hit@10"] == success),
                         key=lambda row: dense[row["id"]]["hit@10"] == success)
        lines += [f"## {heading}", ""]
        for row in matches[:3]:
            base = dense[row["id"]]
            gold = ", ".join(row["gold_doc_ids"])
            lines.append(
                f"- `{row['id']}` (`{row['source_id']}`), tài liệu chuẩn `{gold}`. "
                f"Dense hit@10 = {base['hit@10']}, Hybrid hit@10 = {row['hit@10']}. "
                f"Top 3 Dense `{', '.join(base['retrieved_doc_ids'][:3])}`. "
                f"Top 3 Hybrid `{', '.join(row['retrieved_doc_ids'][:3])}`."
            )
        if not matches:
            lines.append("Không quan sát được trường hợp thuộc nhóm này trong lần chạy.")
        lines.append("")
    return "\n".join(lines)


def write_chart(path, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for offset, method, color in ((-0.19, "Dense", "#2563a0"), (0.19, "Hybrid", "#008675")):
        values = [metrics[method][f"recall@{k}"] for k in K_VALUES]
        bars = ax.bar([i + offset for i in range(len(K_VALUES))], values, width=.34, color=color, label=method)
        ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=4)
    ax.set_xticks(range(len(K_VALUES)), [f"@{k}" for k in K_VALUES])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Tỷ lệ tìm thấy tài liệu chuẩn")
    ax.set_title("Recall@K trên 100 câu dev ViHealthQA")
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_results(output, report, rows):
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        (stage / "comparison.md").write_text(table(report), encoding="utf-8")
        (stage / "examples.md").write_text(observed_examples(rows), encoding="utf-8")
        (stage / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        with (stage / "per_question.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "|".join(value) if isinstance(value, list) else value
                                 for key, value in row.items()})
        write_chart(stage / "recall.png", report["metrics"])
        for name in ("comparison.md", "examples.md", "per_question.csv", "recall.png", "evaluation.json"):
            (stage / name).replace(output / name)


def measure(settings, output):
    ds = load_vihealthqa()
    corpus, chunks, dev = prepare_dataset(ds)
    identity = dev_identity(ds, corpus, chunks, dev)
    retriever = open_retriever(settings, expected_signature=identity["corpus_sha256"])
    if {c.chunk_id: c for c in retriever.chunks} != {c.chunk_id: c for c in chunks}:
        raise ValueError("Các đoạn thực tế trong ChromaDB không khớp kho dev đã xác minh.")
    rows = run_paired(dev, retriever)
    import torch

    device = str(retriever.embedder.model.device)
    report = {
        "status": "COMPLETED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction": REPRODUCE,
        "n_questions": len(dev), "identity": identity,
        "settings": {
            "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION, "normalized": True,
            "dense": {
                "backend": "ChromaDB HNSW",
                "distance": "squared L2",
                "candidate_pool_at_eval_k": dense_candidate_count(max(K_VALUES)),
                "query_cleaning": "conservative greeting stripping (>=4 words preserved)",
            },
            "hybrid": {
                "backend": retriever.hybrid_backend,
                "version": f"Hybrid RRF, {retriever.hybrid_backend}, retain {settings.candidate_pool}",
                "candidate_pool_per_branch": settings.candidate_pool,
                "dense_query_depth": max(settings.candidate_pool, settings.hybrid_dense_depth),
                "rrf_k": settings.rrf_k,
                "dense_weight": settings.dense_weight,
                "sparse_weight": settings.sparse_weight,
                "bm25_granularity": "document-level",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
                "query_cleaning": "conservative greeting stripping (>=4 words preserved)",
                "tokenization": "strictly accented Vietnamese stopwords, accented unigrams/bigrams, adaptive diacritic-free fold",
                "sparse_backend": "rank-bm25 BM25Okapi (k1=1.2, b=0.75)",
            },
            "ks": K_VALUES, "development_seed": 2026, "warmups_per_method": len(WARMUP),
            "latency": "perf_counter, per-query retrieval only, alternating first method per pair",
            "generation": False, "similarity_refusal": False,
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "embedding_device": device,
                     "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                     "torch_threads": torch.get_num_threads()},
        "versions": {"python": platform.python_version(), **{pkg: version(pkg) for pkg in (
            "torch", "transformers", "sentence-transformers", "chromadb", "rank-bm25", "datasets",
            "numpy", "pyarrow",
        )}},
        "metrics": summarize(rows, dev, retriever.indexed_doc_ids),
        "paired_hybrid_vs_dense": paired_changes(rows),
        "limitation": LIMITATION,
    }
    write_results(output, report, rows)
    return report


def main():
    parser = argparse.ArgumentParser(description="Đo Dense và Hybrid trên dev gốc.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--index-dir", type=Path, default=Path(os.getenv("VICARE_INDEX_DIR", "chroma_db")))
    args = parser.parse_args()
    try:
        report = measure(Settings(index_dir=args.index_dir), args.output)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"NOT YET RUN — {reason}", file=sys.stderr)
        output = args.output / "evaluation.json"
        existing = {}
        if output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if existing.get("status") != "COMPLETED":
            args.output.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "status": "NOT YET RUN", "reason": reason, "reproduction": REPRODUCE,
                "limitation": LIMITATION,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    print(table(report))
    print(f"Kết quả đã lưu tại {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
