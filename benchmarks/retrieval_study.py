"""Paired retrieval evaluation. Query/answer/passage text is never written to the reports."""
import argparse
from collections import Counter, defaultdict
from copy import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import DEFAULT_HYBRID_BACKEND, EMBEDDING_MODEL, EMBEDDING_REVISION, PROJECT_ROOT, Settings
from src.dataset import dev_identity, load_vihealthqa, prepare_dataset
from src.evaluation import WARMUP
from src.exact_search import ExactCosine
from src.retrieval import open_retriever, strip_query_greetings, unique_hits
from src.safety import normalize_vietnamese

PUBLISHED = PROJECT_ROOT / "results" / "colab" / "retrieval"
CUTOFFS = (1, 3, 5, 10, 20, 50)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text("utf-8"))


def manifest_hash():
    # Universal newlines produce the same digest on CRLF checkouts and Colab's LF checkout.
    return hashlib.sha256((PUBLISHED / "sets.json").read_text("utf-8").encode("utf-8")).hexdigest()


def rss_peak_mb():
    import sys

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("peak_working_set", ctypes.c_size_t), ("other", ctypes.c_size_t * 7)]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        size = counters.peak_working_set
    else:
        import resource
        size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return round(size / 2**20, 1)


def latency(values):
    return {name: round(float(value), 3) for name, value in (
        ("mean", statistics.mean(values)), ("median", statistics.median(values)),
        ("p95", np.percentile(values, 95)),
    )}


def describe(rows):
    ranks = ("dense_rank", "bm25_rank", "rrf_rank", "rank")
    return {
        "n": len(rows), "errors": sum(r.get("error") is not None for r in rows),
        "coverage": sum(r["covered"] for r in rows) / len(rows),
        "candidate_union_coverage": sum(r["union_hit"] for r in rows) / len(rows),
        "candidate50_document_coverage": sum(r["candidate50_hit"] for r in rows) / len(rows),
        "recall": {f"@{k}": sum(0 < (r["rank"] or 0) <= k for r in rows) / len(rows)
                   for k in (1, 3, 5, 10)},
        "hit_counts": {rank: {f"@{k}": sum(0 < (r[rank] or 0) <= k for r in rows) for k in CUTOFFS}
                       for rank in ranks},
        "error_groups_at_5": dict(Counter(
            "gold_at_5" if 0 < (r["rank"] or 0) <= 5 else
            "outside_union" if not r["union_hit"] else
            "removed_before_50" if not r["candidate50_hit"] else "inside_50_outside_5"
            for r in rows)),
        "mean_dense_duplicate_chunks": statistics.mean(len(r["dense_discarded_chunk_ids"]) for r in rows),
        "mean_candidate50_documents": statistics.mean(r["candidate50_documents"] for r in rows),
        "mean_candidate50_chunks": statistics.mean(r["candidate50_chunks"] for r in rows),
        "latency_ms": latency([r["latency_ms"] for r in rows]),
        "peak_process_rss_mib": max(r["peak_rss_mib"] for r in rows),
    }


def save(output, split, rows, meta, aliases=None):
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    expected = meta["expected_count"]
    for name, group in groups.items():
        if len(group) != expected or len({r["id"] for r in group}) != expected:
            raise RuntimeError(f"Incomplete {name}: {len(group)} / {expected}")
    metrics = {name: describe(group) for name, group in groups.items()}
    for alias, target in (aliases or {}).items():
        metrics[alias] = metrics[target]
    with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(output / f"{split}.json", {
        "set": split, "created_utc": utc_now(), "metrics": metrics,
        "aliases_same_invocation": aliases or {}, "metadata": meta,
    })


def load_sets(phase):
    ds = load_vihealthqa()
    corpus, chunks, dev = prepare_dataset(ds)
    identity = dev_identity(ds, corpus, chunks, dev)
    if phase == "final":
        return {"dev100": dev}, chunks, identity
    locked = load_json(PUBLISHED / "sets.json")
    if (locked["dataset_revision"] != identity["dataset_revision"]
            or locked["corpus_sha256"] != identity["corpus_sha256"]):
        raise ValueError("Locked split/corpus mismatch")
    key = "confirmation100" if phase == "confirm" else "train200"
    source_split = "test" if phase == "confirm" else "train"
    raw = {f"{source_split}-{r['id']}": r["question"] for r in ds[source_split]}
    queries = []
    for row in locked[key]:
        text = raw[row["source_id"]]
        digest = hashlib.sha256(normalize_vietnamese(text).encode()).hexdigest()
        if digest != row["question_sha256"]:
            raise ValueError("Locked question identity changed")
        queries.append({**row, "question": text})
    sets = {key: queries}
    if key == "train200":
        sets["nodiac50"] = [{**q, "question": normalize_vietnamese(q["question"])} for q in queries[:50]]
    return sets, chunks, identity


def token_report(embedder, texts, prefix):
    model = embedder.model
    limit = int(model.max_seq_length)
    token_lengths = []
    for start in range(0, len(texts), 100):
        encoded = model.tokenizer([prefix + t for t in texts[start:start + 100]], add_special_tokens=True,
                                  truncation=False, return_length=True, verbose=False)
        token_lengths.extend(encoded["length"])
    clipped = sum(n > limit for n in token_lengths)
    return {"n": len(texts), "effective_limit": limit, "includes_prefix_and_special_tokens": True,
            "min": min(token_lengths), "p50": float(np.percentile(token_lengths, 50)),
            "p95": float(np.percentile(token_lengths, 95)), "max": max(token_lengths),
            "clipped": clipped, "fraction_clipped": clipped / len(texts)}


def gold_rank(hits, gold):
    return next((i for i, h in enumerate(hits, 1) if h.chunk.doc_id in gold), None)


def row_for(q, name, result, trace, elapsed, rt, parents):
    gold = set(q["gold_doc_ids"])
    dense_all, bm25, fused = (trace[k] for k in ("dense", "sparse", "fused"))
    dense = unique_hits(dense_all)
    dense_kept = {h.chunk.chunk_id for h in dense}
    rank_before = gold_rank(fused, gold)
    candidate = fused[:50]
    docs = [h.chunk.doc_id for h in candidate]
    retrieved = result.hits
    data = {
        "id": q["id"], "source_id": q["source_id"], "gold_doc_ids": sorted(gold), "method": name,
        "covered": bool(gold & rt.indexed_doc_ids),
        "dense_rank": gold_rank(dense, gold), "bm25_rank": gold_rank(bm25, gold),
        "rrf_rank": rank_before if name != "dense" else None,
        "rank": gold_rank(trace.get("ranked", fused), gold),
        "union_hit": rank_before is not None, "candidate50_hit": 0 < (rank_before or 0) <= 50,
        "dense_chunks": len(dense_all), "dense_documents": len(dense), "bm25_documents": len(bm25),
        "union_documents": len(fused), "candidate50_documents": len(candidate), "candidate50_doc_ids": docs,
        "candidate50_chunks": sum(len(parents[d]) for d in docs),
        "dense_discarded_chunk_ids": [h.chunk.chunk_id for h in dense_all if h.chunk.chunk_id not in dense_kept],
        "retrieved_doc_ids": [h.chunk.doc_id for h in retrieved],
        "retrieved_chunk_ids": [h.chunk.chunk_id for h in retrieved],
        "top_dense_score": result.top_dense_score, "latency_ms": round(elapsed, 3),
        "peak_rss_mib": rss_peak_mb(), "error": None,
    }
    return data


def setup(phase):
    start = perf_counter()
    sets, expected_chunks, identity = load_sets(phase)
    backend = DEFAULT_HYBRID_BACKEND if phase in {"final", "confirm"} else "locked"
    rt = open_retriever(Settings.from_env(), expected_signature=identity["corpus_sha256"], hybrid_backend=backend)
    if {c.chunk_id: c for c in expected_chunks} != {c.chunk_id: c for c in rt.chunks}:
        raise ValueError("Stored passage/parent mapping differs from canonical corpus")
    parents = defaultdict(list)
    for c in rt.chunks:
        parents[c.doc_id].append(c)
    meta = {
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {p: version(p) for p in ("chromadb", "sentence-transformers", "torch", "numpy")},
        "e5": EMBEDDING_MODEL, "e5_revision": EMBEDDING_REVISION, "e5_device": str(rt.embedder.model.device),
        "torch_threads": __import__("torch").get_num_threads(), "openblas_threads_env": os.getenv("OPENBLAS_NUM_THREADS"),
        "dataset_revision": identity["dataset_revision"], "corpus_sha256": identity["corpus_sha256"],
        "documents": len(parents), "chunks": len(rt.chunks), "mapping_errors": 0,
        "index_load_and_validation_seconds": round(perf_counter()-start, 3),
        "candidate_units": "Dense standalone HNSW100 chunks; Hybrid HNSW200 -> sorted 80 chunks, BM25 80 documents, RRF 40 equal weights. Candidate50 = first 50 distinct documents from this ranking and every indexed passage of those documents.",
    }
    # Protect the public study from accidental local configuration overrides.
    if (rt.settings.candidate_pool, rt.settings.hybrid_dense_depth, rt.settings.rrf_k,
            rt.settings.dense_weight, rt.settings.sparse_weight) != (80, 200, 40, .5, .5):
        raise ValueError("Study requires the locked retrieval budgets and RRF settings.")
    return sets, rt, parents, meta


def diagnose(output):
    sets, rt, parents, meta = setup("diagnose")
    start = perf_counter()
    exact = ExactCosine(rt.collection, rt.chunks)
    meta.update(exact_load_seconds=round(perf_counter()-start, 3),
                exact_matrix_mib=round(exact.matrix.nbytes / 2**20, 3),
                stored_vector_max_norm_error=exact.max_norm_error)
    meta["passage_tokens"] = token_report(rt.embedder, [c.text for c in rt.chunks], "passage: ")
    for text in WARMUP:
        v = rt.embedder.query(text)
        for fn in (rt.dense_search, exact.search):
            rt.search_encoded(text, v, "Hybrid", 10, dense_search=fn)
    for split, questions in sets.items():
        rows = []
        for i, q in enumerate(questions):
            start = perf_counter()
            text = strip_query_greetings(q["question"].strip())
            vector = rt.embedder.query(text)
            enc_ms = (perf_counter() - start) * 1000
            for name, fn in (("locked", rt.dense_search), ("exact", exact.search))[::(-1 if i % 2 else 1)]:
                trace = {}
                start = perf_counter()
                result = rt.search_encoded(text, vector, "Hybrid", 10, trace=trace, dense_search=fn)
                elapsed = enc_ms + (perf_counter() - start) * 1000
                rows.append(row_for(q, name, result, trace, elapsed, rt, parents))
        save(output, split, rows, {**meta, "expected_count": len(questions),
             "query_tokens": token_report(rt.embedder, [strip_query_greetings(q["question"]) for q in questions],
                                          "query: ")})
        print(f"{split} paired diagnosis complete", flush=True)

def rerank(output):
    import torch
    from benchmarks.passage_reranker import PassageReranker, RERANKER, RERANKER_REVISION

    stage1 = load_json(output / "train200.json")["metrics"]["locked"]["error_groups_at_5"]
    if stage1.get("inside_50_outside_5", 0) <= stage1.get("removed_before_50", 0):
        raise ValueError("No pre-registered basis for the reranker")
    torch.set_num_threads(int(os.getenv("VICARE_TORCH_THREADS", "4")))
    started = perf_counter()
    sets, chunks, identity = load_sets("rerank")
    model = PassageReranker(chunks)
    metadata = {
        "dataset_revision": identity["dataset_revision"], "corpus_sha256": identity["corpus_sha256"],
        "reranker": RERANKER, "revision": RERANKER_REVISION, "device": "cpu", "dtype": "float32",
        "max_pair_tokens": 384, "batch_size": 8, "activation": "raw logits",
        "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
        "setup_seconds": round(perf_counter() - started, 3),
        "latency_protocol": "paired locked full retrieval latency plus wall time of all CE passage batches, excluding evaluator and disk IO; final deployment is timed separately.",
    }
    deadline = perf_counter() + 40 * 60
    for split, questions in sets.items():
        baseline = {r["id"]: r for line in (output / f"{split}.jsonl").read_text("utf-8").splitlines()
                    if (r := json.loads(line))["method"] == "locked"}
        rows = []
        # A partial run is ineligible; retain completed IDs if any exception occurs.
        partial = output / f"{split}_rerank_partial.jsonl"
        try:
            with partial.open("w", encoding="utf-8") as log:
                for q in questions:
                    start = perf_counter()
                    if start > deadline:
                        raise TimeoutError("Pre-registered CE time budget exhausted")
                    base = baseline[q["id"]]
                    ranking = model.rank(strip_query_greetings(q["question"]), base["candidate50_doc_ids"])
                    ce_ms = (perf_counter() - start) * 1000
                    rank = gold_rank(ranking, set(q["gold_doc_ids"])) or base["rrf_rank"]
                    row = {**base, "method": "rerank", "rank": rank,
                           "retrieved_doc_ids": [h.chunk.doc_id for h in ranking[:10]],
                           "retrieved_chunk_ids": [h.chunk.chunk_id for h in ranking[:10]],
                           "latency_ms": round(base["latency_ms"] + ce_ms, 3),
                           "rerank_only_ms": round(ce_ms, 3), "peak_rss_mib": rss_peak_mb()}
                    rows.append(row)
                    log.write(json.dumps(row, separators=(",", ":")) + "\n")
                    log.flush()
                    if len(rows) % 25 == 0:
                        print(f"{split}: scored {len(rows)} / {len(questions)}", flush=True)
        except Exception as exc:
            write_json(output / "rerank_failure.json", {"time_utc": utc_now(), "split": split,
                       "completed": len(rows), "expected": len(questions), "error_type": type(exc).__name__})
            raise
        save(output, split + "_rerank", rows, {**metadata, "expected_count": len(questions)})
        partial.unlink()


def select(output):
    train = load_json(output / "train200.json")["metrics"]
    nodiac = load_json(output / "nodiac50.json")["metrics"]
    for split, store in (("train200", train), ("nodiac50", nodiac)):
        file = output / f"{split}_rerank.json"
        if file.exists():
            store.update(load_json(file)["metrics"])
    baseline = train["locked"]["hit_counts"]["rank"]
    eligible = []
    reasons = {}
    for name in ("exact", "rerank"):
        if name not in train or name not in nodiac:
            reasons[name] = ["incomplete or untested"]
            continue
        m, n = train[name], nodiac[name]
        counts = m["hit_counts"]["rank"]
        failures = []
        if counts["@5"] < baseline["@5"] + 2:
            failures.append("train200 gain < 2 hits at 5")
        if any(counts[f"@{k}"] < baseline[f"@{k}"] - 2 for k in (1, 3, 10)):
            failures.append("train200 loss > 2 hits at another monitored cutoff")
        if n["hit_counts"]["rank"]["@5"] < nodiac["locked"]["hit_counts"]["rank"]["@5"] - 1:
            failures.append("nodiac50 loss > 1 hit at 5")
        if m["errors"] or n["errors"] or max(m["latency_ms"]["p95"], n["latency_ms"]["p95"]) >= 5000:
            failures.append("errors or p95 >= 5 s")
        reasons[name] = failures
        if not failures:
            eligible.append((-counts["@5"], -counts["@1"], m["latency_ms"]["p95"], name))
    choice = min(eligible)[3] if eligible else "locked"
    result = {"selected": choice, "locked_at_utc": utc_now(), "ineligible_reasons": reasons,
              "selection_sets": ["train200", "nodiac50"], "current_study_evaluated_before_lock": ["train200", "nodiac50"],
              "historically_examined": ["train200", "nodiac50", "dev100", "previous_confirm100"],
              "sets_sha256": manifest_hash(), "hash_encoding": "UTF-8, LF line endings"}
    # A freeze is a one-way decision. Use a separate output directory for a new study.
    with (output / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


def final(phase, output):
    decision = load_json(PUBLISHED / "selection.json")
    chosen = decision["selected"]
    if decision["sets_sha256"] != manifest_hash():
        raise ValueError("The locked question-set manifest has changed")
    if chosen != DEFAULT_HYBRID_BACKEND:
        raise ValueError("The deployed Hybrid backend differs from the pre-evaluation freeze")
    sets, rt, parents, meta = setup(phase)
    locked = copy(rt)
    locked.hybrid_backend = "locked"
    methods = [("locked", "Hybrid", locked)]
    aliases = {"selected": "locked"} if chosen == "locked" else {}
    if chosen != "locked":
        methods.append(("selected", "Hybrid", rt))
    if phase == "final":
        v2 = copy(locked)
        v2.settings = replace(locked.settings, hybrid_dense_depth=80)
        methods.extend([("dense", "Dense", rt), ("hybrid_v2", "Hybrid", v2)])
    for question in WARMUP:
        for _, method, backend in methods:
            backend.search(question, method, 10)
    for split, questions in sets.items():
        rows = []
        for i, q in enumerate(questions):
            rotation = i % len(methods)
            for name, method, backend in methods[rotation:] + methods[:rotation]:
                trace = {}
                start = perf_counter()
                result = backend.search(q["question"], method, 10, trace=trace)
                elapsed = (perf_counter() - start) * 1000
                rows.append(row_for(q, name, result, trace, elapsed, rt, parents))
        save(output, split, rows, {**meta, "expected_count": len(questions), "selection": chosen,
             "order": "round-robin rotation of methods for each question",
             "budgets_by_method": {"dense": "HNSW100 chunks",
                     "hybrid_v2": "HNSW80 -> 80 chunks, BM25 80 documents",
                     "locked": "HNSW200 -> 80 chunks, BM25 80 documents",
                     "selected": f"{chosen}; depth 200 -> 80 chunks, BM25 80 documents"},
             "warmup": len(WARMUP), "query_tokens": token_report(rt.embedder, [
                 strip_query_greetings(q["question"]) for q in questions], "query: ")},
             aliases=aliases)
        print(f"{split} complete; {len(questions)} questions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["diagnose", "rerank", "select", "final", "confirm"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"final", "confirm"}:
        final(args.phase, args.output)
    else:
        {"diagnose": diagnose, "rerank": rerank, "select": select}[args.phase](args.output)


if __name__ == "__main__":
    main()
