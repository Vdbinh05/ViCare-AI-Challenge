"""ViHealthQA preprocessing ported from original notebook cells 11–15."""
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    CHUNK_SIZE,
    DATASET,
    DATASET_REVISION,
    DEFAULT_HF_CACHE,
    EMBEDDING_MODEL,
    OVERLAP,
)

PII = re.compile(r"(?<!\w)(?:\+?84|0)\d{9,10}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
NOTEBOOK_CORPUS_PREFIX = "35902972d43e0841"
EXPECTED_DEV_SHA256 = "5a0f84a1a3f4834af5eb21c2eee5ddb17d977f7fec321fba65e1c5606e3ab8fa"
EXPECTED_SPLITS = [("train", 7009), ("validation", 993), ("test", 2013)]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    text: str


def _resolve_datasets_cache_dir():
    ds_env = os.getenv("HF_DATASETS_CACHE")
    if ds_env:
        return Path(ds_env)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "datasets"
    project_ds = DEFAULT_HF_CACHE / "datasets"
    if project_ds.is_dir():
        return project_ds
    return None


def load_vihealthqa():
    if not os.getenv("HF_HOME") and DEFAULT_HF_CACHE.is_dir():
        os.environ["HF_HOME"] = str(DEFAULT_HF_CACHE)
    from datasets import DownloadConfig, load_dataset

    cache_dir = _resolve_datasets_cache_dir()
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    token = os.getenv("HF_TOKEN") or False

    # Local-first: avoid network/expired-token failures when the pinned snapshot is already cached.
    try:
        return load_dataset(
            DATASET,
            revision=DATASET_REVISION,
            cache_dir=cache_dir_str,
            download_config=DownloadConfig(
                cache_dir=cache_dir_str,
                local_files_only=True,
                token=False,
            ),
            token=False,
        )
    except (OSError, ValueError, ConnectionError):
        pass

    return load_dataset(
        DATASET,
        revision=DATASET_REVISION,
        cache_dir=cache_dir_str,
        token=token,
    )


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("Cần 0 ≤ phần chồng lấn < kích thước đoạn.")
    words = text.split()
    parts = []
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return parts


def prepare_dataset(ds):
    if not {"train", "validation"} <= set(ds):
        raise ValueError("Bộ dữ liệu cần có tập train và validation.")
    corpus, text_to_id, gold = [], {}, {}
    for split, rows in ds.items():
        for row in rows:
            if not {"id", "question", "answer", "link"} <= row.keys():
                raise ValueError(f"Dữ liệu thiếu cột bắt buộc trong tập {split}.")
            text = PII.sub("[đã ẩn]", str(row.get("answer") or "")).strip()
            if not text:
                continue
            key = f"{split}-{row['id']}"
            if text not in text_to_id:
                text_to_id[text] = key
                corpus.append({"doc_id": key, "url": str(row.get("link") or ""), "text": text})
            gold[key] = text_to_id[text]
    if not corpus:
        raise ValueError("Kho tài liệu trống.")
    validation = list(ds["validation"])
    selected = random.Random(2026).sample(validation, k=min(100, len(validation)))
    dev = []
    for i, row in enumerate(selected):
        key = f"validation-{row['id']}"
        if key in gold and str(row.get("question") or "").strip():
            dev.append({"id": f"d{i:03d}", "source_id": key, "question": row["question"],
                        "gold_doc_ids": [gold[key]]})
    chunks = [Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], doc["url"], part)
              for doc in corpus for i, part in enumerate(chunk_text(doc["text"]))]
    return corpus, chunks, dev


def chunk_dict(chunk):
    data = asdict(chunk)
    data["id"] = data.pop("chunk_id")
    return data


def corpus_sha256(chunks):
    hasher = hashlib.sha256(f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{OVERLAP}|l2|".encode())
    for chunk in chunks:
        hasher.update(json.dumps(chunk_dict(chunk), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def question_sha256(dev):
    ordered = json.dumps([q["question"] for q in dev], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def dev_identity(ds, corpus, chunks, dev):
    checksum = corpus_sha256(chunks)
    questions = question_sha256(dev)
    splits = [(name, len(ds[name])) for name in ds]
    if (splits != EXPECTED_SPLITS or len(corpus) != 9895 or len(chunks) != 11227
            or not checksum.startswith(NOTEBOOK_CORPUS_PREFIX)
            or len(dev) != 100 or questions != EXPECTED_DEV_SHA256):
        raise ValueError(
            "Không tái tạo được bộ dev gốc. "
            f"Split={splits}, tài liệu={len(corpus)}, đoạn={len(chunks)}, câu={len(dev)}, "
            f"corpus_sha256={checksum}, questions_sha256={questions}. Dừng so sánh."
        )
    return {
        "dataset": DATASET, "dataset_revision": DATASET_REVISION,
        "split_counts": dict(splits), "documents": len(corpus), "chunks": len(chunks),
        "corpus_sha256": checksum, "ordered_questions_sha256": questions,
        "questions": [{"id": q["id"], "source_id": q["source_id"], "gold_doc_ids": q["gold_doc_ids"]}
                      for q in dev],
    }
