"""Dense baseline, lexical retrieval and weighted reciprocal rank fusion."""
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import DEFAULT_HYBRID_BACKEND, EMBEDDING_MODEL, EMBEDDING_REVISION, INDEX_COMMAND, Settings
from .dataset import Chunk, corpus_sha256, load_vihealthqa, prepare_dataset
from .embeddings import Embedder
from .safety import normalize_vietnamese

MANIFEST = "vicare-index.json"
BM25_K1 = 1.2
BM25_B = 0.75

_VIETNAMESE_DIACRITIC_LETTERS = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđ"
)

# Strictly accented Vietnamese grammatical particles only (matched on NFC-accented words so
# medical compounds like "tử cung", "cổ tử cung", "đo huyết áp", "ợ chua", "sâu răng" are preserved).
ACCENTED_STOPWORDS = frozenset({
    "của", "và", "những", "các", "một", "này", "đó", "thì", "mà", "với",
    "trong", "khi", "như", "tại", "từ", "về", "đã", "đang", "sẽ", "rất",
    "để", "cho", "bị", "được", "là", "ạ", "nhé", "nha", "ơi", "vậy",
    "thế", "nên", "nếu", "vì", "nhưng", "tuy", "hoặc", "cũng", "chào",
})

_GREETING_PREFIX_RES = (
    re.compile(
        r"^\s*(?:dạ\s+)?(?:xin\s+)?(?:kính\s+)?(?:chào|thưa|gửi)\s+(?:các\s+)?(?:bác\s*sĩ|bs|chuyên\s*gia|chương\s*trình|nhà\s*thuốc)[\s,!.:;-]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:bác\s*sĩ|bs)\s+(?:cho\s+(?:em|tôi|mình|cháu)\s+hỏi|ơi|tư\s+vấn\s+(?:giúp\s+)?(?:em|tôi|mình|cháu)?)\s*[\s,!.:;-]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:xin\s+)?(?:cho\s+(?:em|tôi|mình|cháu)\s+hỏi|em\s+muốn\s+hỏi|tôi\s+muốn\s+hỏi)[\s,!.:;-]*",
        re.IGNORECASE,
    ),
)

_GREETING_SUFFIX_RES = (
    re.compile(
        r"[\s,!.:;-]*(?:em\s+|tôi\s+|mình\s+|cháu\s+|xin\s+)?(?:chân\s+thành\s+)?cảm\s+ơn\s*(?:bác\s*sĩ|bs|nhiều|ạ|rất\s+nhiều)*[\s!?.]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\s,!.:;-]*(?:mong|nhờ)\s+(?:bác\s*sĩ|bs)\s+(?:tư\s+vấn|giải\s+đáp|trả\s+lời)(?:\s+giúp(?:\s+em|\s+tôi|\s+mình)?)?[\s!?.]*$",
        re.IGNORECASE,
    ),
)


class MissingIndex(RuntimeError):
    def __init__(self):
        super().__init__(f"Chưa có chỉ mục ViCare đầy đủ. Chạy {INDEX_COMMAND} tại thư mục dự án.")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    top_dense_score: float | None
    method: str


def dense_candidate_count(k):
    # The UI supports 1–10 sources. Slice a single fixed ranking at every cutoff.
    return max(k * 10, 100)


def unique_hits(hits):
    seen = set()
    result = []
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(hit)
    return result


def strip_query_greetings(text):
    """Conservatively remove conversational greeting prefixes/suffixes if >=4 words remain."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    for _ in range(3):
        prev = cleaned
        for pattern in _GREETING_PREFIX_RES:
            cleaned = pattern.sub("", cleaned).strip()
        for pattern in _GREETING_SUFFIX_RES:
            cleaned = pattern.sub("", cleaned).strip()
        if cleaned == prev:
            break
    if len(cleaned.split()) >= 4:
        return cleaned
    return text.strip()


def has_vietnamese_diacritics(text):
    """Recognize marked Vietnamese letters, not Unicode punctuation or emoji."""
    return isinstance(text, str) and not _VIETNAMESE_DIACRITIC_LETTERS.isdisjoint(
        unicodedata.normalize("NFC", text).casefold()
    )


def _extract_words(text, *, remove_accented_stopwords=True):
    raw_words = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.casefold()))
    exact_out = []
    fold_out = []
    for word in raw_words:
        if remove_accented_stopwords and word in ACCENTED_STOPWORDS:
            continue
        folded = normalize_vietnamese(word)
        if not folded:
            continue
        exact_out.append(word)
        fold_out.append(folded)
    if not exact_out and raw_words:
        for word in raw_words:
            folded = normalize_vietnamese(word)
            if folded:
                exact_out.append(word)
                fold_out.append(folded)
    return exact_out, fold_out


def tokenize_accented(text):
    exact, _ = _extract_words(text, remove_accented_stopwords=True)
    return [f"e:{word}" for word in exact] + [f"eb:{a} {b}" for a, b in zip(exact, exact[1:])]


def tokenize_folded(text):
    _, folded = _extract_words(text, remove_accented_stopwords=True)
    return [f"f:{word}" for word in folded] + [f"fb:{a} {b}" for a, b in zip(folded, folded[1:])]


def _tokenize_document(text):
    exact, folded = _extract_words(text, remove_accented_stopwords=True)
    return (
        [f"e:{word}" for word in exact]
        + [f"eb:{a} {b}" for a, b in zip(exact, exact[1:])]
        + [f"f:{word}" for word in folded]
        + [f"fb:{a} {b}" for a, b in zip(folded, folded[1:])]
    )


def tokenize(text):
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.casefold()))
    folded = normalize_vietnamese(text).split()
    return ([f"exact:{word}" for word in words] + [f"fold:{word}" for word in folded]
            + [f"bigram:{left} {right}" for left, right in zip(folded, folded[1:])])


class SparseIndex:
    """Document-level BM25Okapi index over ChromaDB chunks with adaptive diacritic routing."""

    def __init__(self, chunks, *, k1=BM25_K1, b=BM25_B):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        grouped_texts = defaultdict(list)
        self.chunks_by_doc = defaultdict(list)
        first_chunk = {}
        for chunk in self.chunks:
            if chunk.doc_id not in first_chunk:
                first_chunk[chunk.doc_id] = chunk
            grouped_texts[chunk.doc_id].append(chunk.text)
            self.chunks_by_doc[chunk.doc_id].append(chunk)
        self.items = [first_chunk[doc_id] for doc_id in first_chunk]
        self.postings = defaultdict(list)
        token_lists = []
        for i, chunk in enumerate(self.items):
            doc_text = " ".join(grouped_texts[chunk.doc_id])
            tokens = _tokenize_document(doc_text)
            token_lists.append(tokens)
            for token in set(tokens):
                self.postings[token].append(i)
        self.bm25 = BM25Okapi(token_lists, k1=k1, b=b) if self.postings else None

    def _best_passage(self, doc_id, query_tokens):
        chunks = self.chunks_by_doc[doc_id]
        if len(chunks) == 1:
            return chunks[0]
        return min(chunks, key=lambda chunk: (
            -len(query_tokens.intersection(_tokenize_document(chunk.text))), chunk.chunk_id,
        ))

    def search(self, query, limit):
        cleaned = strip_query_greetings(query)
        raw_tokens = tokenize_accented(cleaned) if has_vietnamese_diacritics(cleaned) else tokenize_folded(cleaned)
        tokens = list(dict.fromkeys(raw_tokens))
        indices = sorted({i for token in tokens for i in self.postings.get(token, ())})
        if not indices or self.bm25 is None:
            return []
        scores = self.bm25.get_batch_scores(tokens, indices)
        ranked = [Hit(self.items[i], float(score)) for i, score in zip(indices, scores)]
        ranked = sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.doc_id, hit.chunk.chunk_id))[:limit]
        query_tokens = set(tokens)
        return [Hit(self._best_passage(hit.chunk.doc_id, query_tokens), hit.score) for hit in ranked]


def fuse(dense, sparse, *, k, rrf_k=60, dense_weight=0.5, sparse_weight=0.5):
    scores = defaultdict(float)
    representatives = {}
    for branch, weight in ((dense, dense_weight), (sparse, sparse_weight)):
        if weight == 0:
            continue
        for rank, hit in enumerate(unique_hits(branch), 1):
            doc_id = hit.chunk.doc_id
            scores[doc_id] += weight / (rrf_k + rank)
            # Dense runs first; a sparse document rank must not replace its matching passage.
            representatives.setdefault(doc_id, hit.chunk)
    ranked_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [Hit(representatives[doc_id], scores[doc_id]) for doc_id in ranked_ids[:k]]


def parse_chunk(chunk_id, text, metadata):
    if not isinstance(metadata, dict):
        return None
    doc_id, url = metadata.get("doc_id"), metadata.get("url")
    if not all(isinstance(value, str) and value.strip() for value in (chunk_id, doc_id, text)):
        return None
    return Chunk(chunk_id, doc_id, url if isinstance(url, str) else "", text)


def parse_dense(response):
    if not isinstance(response, dict):
        return []
    columns = [(response.get(name) or [[]])[0] or []
               for name in ("ids", "documents", "metadatas", "distances")]
    hits = []
    for chunk_id, text, metadata, distance in zip(*columns):
        chunk = parse_chunk(chunk_id, text, metadata)
        if not chunk or type(distance) not in (float, int) or not math.isfinite(distance):
            continue
        hits.append(Hit(chunk, 1.0 - float(distance) / 2.0))
    return hits


class Retriever:
    def __init__(self, collection, embedder, chunks, settings, *, hybrid_backend=DEFAULT_HYBRID_BACKEND):
        self.collection = collection
        self.embedder = embedder
        self.chunks = list(chunks)
        self.settings = settings
        self.indexed_doc_ids = {chunk.doc_id for chunk in self.chunks}
        self.sparse = SparseIndex(self.chunks)
        self.hybrid_backend = hybrid_backend
        if hybrid_backend == "exact":
            from .exact_search import ExactCosine
            self.exact = ExactCosine(collection, self.chunks)
        elif hybrid_backend != "locked":
            raise ValueError("Unknown Hybrid retrieval backend")

    def search(self, question, method, k, *, trace=None):
        if not isinstance(method, str) or method not in {"Dense", "Hybrid"} or type(k) is not int or k < 1:
            raise ValueError("Phương pháp hoặc số nguồn truy xuất không hợp lệ.")
        if not isinstance(question, str) or not question.strip() or not self.chunks:
            return SearchResult([], None, method)
        question = strip_query_greetings(question.strip())
        vector = self.embedder.query(question)
        return self.search_encoded(question, vector, method, k, trace=trace)

    def dense_search(self, vector, depth):
        return parse_dense(self.collection.query(
            query_embeddings=vector, n_results=min(len(self.chunks), depth),
            include=["documents", "metadatas", "distances"],
        ))

    def search_encoded(self, question, vector, method, k, *, trace=None, dense_search=None):
        """Rank a cleaned query. A paired evaluator can supply the same encoded vector to both searches."""
        pool = self.settings.candidate_pool
        depth = dense_candidate_count(k) if method == "Dense" else max(pool, self.settings.hybrid_dense_depth)
        if dense_search is None:
            dense_search = (self.exact.search if method == "Hybrid" and self.hybrid_backend == "exact"
                            else self.dense_search)
        dense = dense_search(vector, depth)
        top_score = max((hit.score for hit in dense), default=None)
        if method == "Dense":
            hits = unique_hits(dense)
            if trace is not None:
                trace.update(dense=dense, sparse=[], fused=hits)
            return SearchResult(hits[:k], top_score, method)
        dense = sorted(dense, key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:pool]
        sparse = self.sparse.search(question, pool)
        hits = fuse(dense, sparse, k=2 * pool, rrf_k=self.settings.rrf_k,
                    dense_weight=self.settings.dense_weight, sparse_weight=self.settings.sparse_weight)
        if trace is not None:
            trace.update(dense=dense, sparse=sparse, fused=hits)
        return SearchResult(hits[:k], top_score, method)


def write_manifest(path, manifest):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as temp:
        temp_path = Path(temp.name)
        json.dump(manifest, temp, indent=2)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def persistent_client(path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


def build_index(settings, chunks, embedder=None):
    if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Cần danh sách đoạn khác rỗng với ID duy nhất.")
    signature = corpus_sha256(chunks)
    name = f"vicare_{signature[:16]}"
    manifest = {"name": name, "corpus_sha256": signature, "count": len(chunks),
                "model": EMBEDDING_MODEL, "model_revision": EMBEDDING_REVISION, "space": "l2"}
    collection = persistent_client(settings.index_dir).get_or_create_collection(
        name=name, configuration={"hnsw": {"space": "l2"}}, embedding_function=None,
        metadata={"corpus_sha256": signature, "model_revision": EMBEDDING_REVISION},
    )
    if (collection.metadata or {}).get("model_revision") != EMBEDDING_REVISION:
        raise RuntimeError("Chỉ mục cũ không có phiên bản E5 tương ứng. Chọn VICARE_INDEX_DIR mới.")
    expected = {chunk.chunk_id for chunk in chunks}
    existing = set(collection.get(include=[])["ids"])
    if not existing <= expected:
        raise RuntimeError("Chỉ mục chứa ID ngoài kho dữ liệu. Chọn VICARE_INDEX_DIR mới.")
    pending = [chunk for chunk in chunks if chunk.chunk_id not in existing]
    if pending:
        embedder = embedder or Embedder()
    print(f"ChromaDB {name}: có {len(existing)}/{len(chunks)} đoạn", flush=True)
    for offset in range(0, len(pending), 256):
        batch = pending[offset:offset + 256]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embedder.passages([chunk.text for chunk in batch]),
            metadatas=[{"doc_id": chunk.doc_id, "url": chunk.url} for chunk in batch],
        )
        print(f"Đã lập chỉ mục {len(existing) + offset + len(batch)}/{len(chunks)} đoạn", flush=True)
    if set(collection.get(include=[])["ids"]) != expected:
        raise RuntimeError("ID trong chỉ mục không khớp kho dữ liệu.")
    write_manifest(settings.index_dir / MANIFEST, manifest)
    return manifest


def open_retriever(settings, *, expected_signature=None, embedder=None, hybrid_backend=DEFAULT_HYBRID_BACKEND):
    manifest_path = settings.index_dir / MANIFEST
    if not manifest_path.is_file():
        raise MissingIndex()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["model"] != EMBEDDING_MODEL or manifest["model_revision"] != EMBEDDING_REVISION
                or manifest["space"] != "l2" or type(manifest["count"]) is not int
                or manifest["count"] < 1
                or manifest["name"] != f"vicare_{manifest['corpus_sha256'][:16]}"):
            raise MissingIndex()
        if expected_signature is not None and manifest["corpus_sha256"] != expected_signature:
            raise MissingIndex()
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise MissingIndex() from exc

    from chromadb.errors import NotFoundError

    try:
        collection = persistent_client(settings.index_dir).get_collection(
            manifest["name"], embedding_function=None,
        )
    except NotFoundError as exc:
        raise MissingIndex() from exc
    metadata = collection.metadata or {}
    if (collection.count() != manifest["count"] or metadata.get("corpus_sha256") != manifest["corpus_sha256"]
            or metadata.get("model_revision") != EMBEDDING_REVISION):
        raise MissingIndex()
    chunks = []
    for offset in range(0, manifest["count"], 1000):
        batch = collection.get(limit=1000, offset=offset, include=["documents", "metadatas"])
        for fields in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            chunk = parse_chunk(*fields)
            if chunk is None:
                raise MissingIndex()
            chunks.append(chunk)
    if len({chunk.chunk_id for chunk in chunks}) != manifest["count"]:
        raise MissingIndex()
    return Retriever(collection, embedder or Embedder(), chunks, settings, hybrid_backend=hybrid_backend)


def main():
    parser = argparse.ArgumentParser(description="Lập chỉ mục ViCare.")
    parser.add_argument("action", choices=["build"])
    parser.parse_args()
    try:
        _, chunks, _ = prepare_dataset(load_vihealthqa())
        build_index(Settings.from_env(), chunks)
    except Exception as exc:
        print(f"Không thể lập chỉ mục ViCare. {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
