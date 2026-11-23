"""Two fixed query variants on the already-used NoDiac50 selection examples."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from benchmarks.retrieval_study import load_sets, manifest_hash, utc_now
from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION, PROJECT_ROOT, Settings
from src.retrieval import open_retriever


VARIANTS = ("original", "curly_quotes")
CUTOFFS = (1, 3, 5, 10)
SOURCE_FILES = (
    "src/config.py", "src/dataset.py", "src/embeddings.py", "src/exact_search.py",
    "src/retrieval.py", "src/safety.py", "benchmarks/retrieval_study.py",
    "benchmarks/evaluate_unicode_robustness.py",
)


def summarize(rows):
    if len(rows) != 50 or len({row["id"] for row in rows}) != 50:
        raise ValueError("Every variant must retain all 50 distinct question IDs")
    counts = {f"@{k}": sum(
        row["error_type"] is None
        and bool(set(row["gold_doc_ids"]) & set(row["retrieved_doc_ids"][:k]))
        for row in rows) for k in CUTOFFS}
    return {"n": 50, "errors": sum(row["error_type"] is not None for row in rows),
            "hit_counts": counts, "recall": {key: value / 50 for key, value in counts.items()}}


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output path to preserve previous measurements")
    settings = Settings.from_env()
    budgets = (settings.candidate_pool, settings.hybrid_dense_depth, settings.rrf_k,
               settings.dense_weight, settings.sparse_weight)
    if budgets != (80, 200, 40, .5, .5):
        raise ValueError("Use the locked candidate budgets and RRF settings")
    sets, expected_chunks, identity = load_sets("diagnose")
    questions = sets["nodiac50"]
    if len(questions) != 50 or len({q["id"] for q in questions}) != 50:
        raise ValueError("Locked NoDiac50 must contain exactly 50 distinct IDs")
    retriever = open_retriever(settings, expected_signature=identity["corpus_sha256"],
                               hybrid_backend="exact")
    if {c.chunk_id: c for c in expected_chunks} != {c.chunk_id: c for c in retriever.chunks}:
        raise ValueError("Stored passage mapping differs from the canonical corpus")

    rows = []
    for question in questions:
        inputs = (question["question"], "“" + question["question"] + "”")
        for variant, text in zip(VARIANTS, inputs):
            row = {"id": question["id"], "source_id": question["source_id"], "variant": variant,
                   "gold_doc_ids": list(question["gold_doc_ids"]),
                   "retrieved_doc_ids": [], "error_type": None}
            try:
                result = retriever.search(text, "Hybrid", 10)
                row["retrieved_doc_ids"] = [hit.chunk.doc_id for hit in result.hits]
            except Exception as exc:
                # Preserve the denominator without exporting query text from exception messages.
                row["error_type"] = type(exc).__name__
            rows.append(row)

    sources = {name: hashlib.sha256((PROJECT_ROOT / name).read_text("utf-8").encode()).hexdigest()
               for name in SOURCE_FILES}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "created_utc": utc_now(), "split": "nodiac50", "n_questions": 50, "variants": list(VARIANTS),
        "interpretation": "Robustness on previously used tuning examples; not a holdout or an independent gain.",
        "protocol": "All 50 locked NoDiac50 IDs; unchanged normalized question and the same text wrapped in U+201C/U+201D; production Hybrid search at k=10; errors are misses.",
        "metadata": {
            "source_commit": commit, "source_sha256": sources, "hash_encoding": "UTF-8, LF line endings",
            "sets_sha256": manifest_hash(), "dataset_revision": identity["dataset_revision"],
            "corpus_sha256": identity["corpus_sha256"],
            "e5": EMBEDDING_MODEL, "e5_revision": EMBEDDING_REVISION,
            "e5_device": str(retriever.embedder.model.device),
            "backend": "exact", "method": "Hybrid", "k": 10,
            "candidate_pool": settings.candidate_pool, "hybrid_dense_depth": settings.hybrid_dense_depth,
            "rrf_k": settings.rrf_k, "dense_weight": settings.dense_weight,
            "sparse_weight": settings.sparse_weight,
            "documents": len(retriever.indexed_doc_ids), "chunks": len(retriever.chunks),
            "platform": platform.platform(), "python": platform.python_version(),
            "packages": {name: version(name) for name in (
                "numpy", "torch", "chromadb", "sentence-transformers", "datasets", "rank-bm25")},
        },
        "metrics": {variant: summarize([row for row in rows if row["variant"] == variant])
                    for variant in VARIANTS},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["metrics"], ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    return int(any(metric["errors"] for metric in report["metrics"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
