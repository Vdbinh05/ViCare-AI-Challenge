import os
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.dataset import Chunk, chunk_text, dev_identity, load_vihealthqa, prepare_dataset
from src.embeddings import Embedder
from src.generation import Assistant
from src.retrieval import (
    ACCENTED_STOPWORDS, BM25_B, BM25_K1, Hit, Retriever, SparseIndex, build_index, fuse,
    open_retriever, parse_dense, strip_query_greetings, tokenize, tokenize_accented,
    tokenize_folded, unique_hits,
)


def hit(doc_id, chunk_num=0, score=1.0, text="Đo huyết áp định kỳ"):
    return Hit(Chunk(f"{doc_id}#{chunk_num}", doc_id, "https://example.org/health", text), score)


def test_tokenization_normalizes_case_diacritics_and_punctuation():
    accented = set(tokenize("  HUYẾT ÁP, ĐƯỜNG! "))
    unaccented = set(tokenize("huyet ap duong"))
    assert accented & unaccented
    assert set(tokenize("Huyết áp?")) == set(tokenize("huyết áp"))
    assert not tokenize(" !?")


def test_bm25_exact_matches_and_never_invents_an_unmatched_document():
    chunks = [
        hit("train-1", text="Bệnh viêm gan B và HBsAg").chunk,
        hit("train-2", text="Huyết áp và vận động").chunk,
        hit("train-3", text="Tài liệu về dinh dưỡng").chunk,
        hit("train-4", text="Viêm gan và chức năng gan").chunk,
    ]
    index = SparseIndex(chunks)
    results = index.search("HBsAg", 4)
    assert results[0].chunk == chunks[0]
    assert index.search("mãkhôngcótrongkho", 4) == []
    assert index.search("viem gan", 4)[0].chunk.doc_id in {"train-1", "train-4"}
    assert index.search("viem gan", 4) == index.search("viem gan", 4)


def test_dense_output_and_sparse_index_use_the_same_chunk_and_metadata():
    chunk = hit("train-7", 2, text="Xét nghiệm HBsAg").chunk
    dense = parse_dense({
        "ids": [[chunk.chunk_id]], "documents": [[chunk.text]],
        "metadatas": [[{"doc_id": chunk.doc_id, "url": chunk.url}]],
        "distances": [[0.24]],
    })
    sparse = SparseIndex([chunk, hit("train-8", text="Tài liệu 1").chunk,
                          hit("train-9", text="Tài liệu 2").chunk]).search("HBsAg", 5)
    assert dense[0].chunk == sparse[0].chunk == chunk
    assert dense[0].score == pytest.approx(0.88)


def test_dense_parser_disregards_invalid_core_metadata_but_preserves_missing_urls():
    response = {
        "ids": [["missing#0", "valid#0", "bad-url#1", "bad-distance#0"]],
        "documents": [["Thiếu mã", "Văn bản hợp lệ", "URL bất thường", "Điểm lỗi"]],
        "metadatas": [[None, {"doc_id": "valid"}, {"doc_id": "bad-url", "url": 7},
                       {"doc_id": "bad-distance"}]],
        "distances": [[0.1, 0.2, 0.2, float("nan")]],
    }
    assert [item.chunk for item in parse_dense(response)] == [
        Chunk("valid#0", "valid", "", "Văn bản hợp lệ"),
        Chunk("bad-url#1", "bad-url", "", "URL bất thường"),
    ]


def test_retrieval_modes_keep_distinct_candidate_pools_and_cosine_score():
    chunks = [hit("a", text="Bệnh HBsAg").chunk, hit("b", text="Tài liệu sức khỏe").chunk,
              hit("c", text="Thông tin về ăn uống").chunk]
    chunks.extend(hit(f"qa-padding-{i}", text=f"qa_seed_{i}").chunk for i in range(197))
    candidate_counts = []
    embedded_questions = []

    def collection_query(**kwargs):
        candidate_counts.append(kwargs["n_results"])
        assert kwargs["query_embeddings"] == [[1.0, 0.0]]
        returned = chunks[:3]
        return {"ids": [[chunk.chunk_id for chunk in returned]],
                "documents": [[chunk.text for chunk in returned]],
                "metadatas": [[{"doc_id": chunk.doc_id, "url": chunk.url} for chunk in returned]],
                "distances": [[0.12, 0.16, 0.2]]}

    def embed(question):
        embedded_questions.append(question)
        return [[1.0, 0.0]]

    collection = SimpleNamespace(count=lambda: 200, query=collection_query)
    retriever = Retriever(collection, SimpleNamespace(query=embed), chunks, Settings())
    dense = retriever.search("HBsAg", "Dense", 10)
    hybrid = retriever.search("HBsAg", "Hybrid", 10)
    assert candidate_counts == [100, 200]
    assert embedded_questions == ["HBsAg", "HBsAg"]
    assert dense.top_dense_score == hybrid.top_dense_score == pytest.approx(0.94)
    assert dense.hits[0].score == pytest.approx(0.94)
    assert hybrid.hits[0].score == pytest.approx(1 / (Settings().rrf_k + 1))


def test_hybrid_truncates_sorted_chunks_before_document_fusion():
    chunks = [hit("other").chunk, hit("best", 1).chunk, hit("best", 0).chunk]

    def query(**options):
        assert options["n_results"] == 3
        return {"ids": [[c.chunk_id for c in chunks]], "documents": [[c.text for c in chunks]],
                "metadatas": [[{"doc_id": c.doc_id} for c in chunks]], "distances": [[0.3, 0.1, 0.1]]}

    settings = Settings(candidate_pool=2, top_k=2, hybrid_dense_depth=3)
    retriever = Retriever(SimpleNamespace(query=query), SimpleNamespace(query=lambda q: [[1.0]]), chunks, settings)
    result = retriever.search("unmatchedsynthetictoken", "Hybrid", 2)
    assert [h.chunk.chunk_id for h in result.hits] == ["best#0"]


def test_unique_hits_keeps_first_chunk_and_input_rank():
    first = hit("b", 1, 0.92)
    other = hit("a", 0, 0.91)
    assert list(unique_hits([first, hit("b", 2, 0.90), other])) == [first, other]


def test_rrf_uses_unique_document_ranks_and_handles_missing_branches():
    dense = [hit("a", 0), hit("a", 1), hit("b")]
    sparse = [hit("b"), hit("c")]
    result = fuse(dense, sparse, k=5, rrf_k=60, dense_weight=0.5, sparse_weight=0.5)
    assert [item.chunk.doc_id for item in result] == ["b", "a", "c"]
    assert result[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert result[1].score == pytest.approx(0.5 / 61)
    assert result[2].score == pytest.approx(0.5 / 62)


def test_rrf_tie_break_repeatability_and_depth():
    dense, sparse = [hit("b")], [hit("a")]
    rankings = [fuse(dense, sparse, k=2) for _ in range(5)]
    assert all(ranking == rankings[0] for ranking in rankings)
    assert rankings[0][0].score == rankings[0][1].score
    assert len(fuse(dense, sparse, k=1)) == 1
    assert fuse([], [], k=5) == []


def test_dataset_keeps_first_source_and_gold_for_duplicate_cleaned_documents():
    ds = OrderedDict([
        ("train", [{"id": 17, "question": "Câu hỏi", "answer": "  Hỏi qa@example.org  ",
                    "link": "https://example.org/first"}]),
        ("validation", [{"id": 21, "question": "  Câu hỏi thứ hai  ",
                         "answer": "Hỏi qa2@example.org", "link": "https://example.org/second"},
                        {"id": 22, "question": "Bỏ qua", "answer": None, "link": None}]),
        ("test", [{"id": 11, "question": "Kiểm tra", "answer": "Một nội dung mới", "link": None}]),
    ])
    corpus, chunks, dev = prepare_dataset(ds)
    assert len(corpus) == 2
    assert chunks[0] == Chunk("train-17#0", "train-17", "https://example.org/first", "Hỏi [đã ẩn]")
    assert chunks[1] == Chunk("test-11#0", "test-11", "", "Một nội dung mới")
    assert len(dev) == 1
    assert dev[0]["question"] == "  Câu hỏi thứ hai  "
    assert dev[0]["gold_doc_ids"] == ["train-17"]


def test_chunking_stops_at_final_word_and_validates_overlap():
    assert chunk_text("a\tb c\nd e f g", 4, 1) == ["a b c d", "d e f g"]
    assert chunk_text(" \n ", 4, 1) == []
    assert chunk_text("a b c d", 4, 1) == ["a b c d"]
    with pytest.raises(ValueError):
        chunk_text("a b", 4, 4)


def test_integration_runtime_imports_for_dataset_and_e5():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    assert callable(load_dataset)
    assert callable(SentenceTransformer)


def test_integration_real_chroma_and_real_bm25_with_fixed_test_vectors(tmp_path):
    chunks = [hit("qa-a", text="HBsAg và viêm gan B").chunk,
              hit("qa-b", text="Đo huyết áp").chunk, hit("qa-c", text="Tài liệu dinh dưỡng").chunk]
    settings = Settings(index_dir=tmp_path / "chroma")
    embedder = SimpleNamespace(passages=lambda texts: [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                               query=lambda question: [[1.0, 0.0]])
    manifest = build_index(settings, chunks, embedder)
    assert build_index(settings, chunks, embedder) == manifest
    retriever = open_retriever(settings, expected_signature=manifest["corpus_sha256"], embedder=embedder)
    dense = retriever.search("HBsAg", "Dense", 1)
    hybrid = retriever.search("HBsAg", "Hybrid", 2)
    assert {chunk.chunk_id for chunk in retriever.chunks} == {chunk.chunk_id for chunk in chunks}
    assert dense.top_dense_score == pytest.approx(1.0)
    assert dense.hits[0].chunk == hybrid.hits[0].chunk == chunks[0]


@pytest.mark.skipif(os.getenv("VICARE_QA_REAL_INDEX") != "1", reason="Cần ViHealthQA, E5 và chỉ mục đầy đủ")
def test_integration_real_dataset_real_e5_and_full_chroma_index():
    ds = load_vihealthqa()
    corpus, chunks, dev = prepare_dataset(ds)
    identity = dev_identity(ds, corpus, chunks, dev)
    assert identity["corpus_sha256"].startswith("35902972d43e0841")
    assert len(dev) == 100

    embedder = Embedder()
    retriever = open_retriever(Settings.from_env(), expected_signature=identity["corpus_sha256"], embedder=embedder)
    # This smoke case is separate from the protected evaluation questions.
    question = "Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?"
    dense_vector = embedder.query(question)[0]
    passage_vector = embedder.passages(["Thông tin tiêm chủng viêm gan B."])[0]
    assert len(dense_vector) == len(passage_vector) == 384
    assert sum(value ** 2 for value in dense_vector) == pytest.approx(1.0, abs=1e-5)
    assert sum(value ** 2 for value in passage_vector) == pytest.approx(1.0, abs=1e-5)
    indexed = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    assert indexed == {chunk.chunk_id: chunk for chunk in chunks}
    for method in ("Dense", "Hybrid"):
        result = retriever.search(question, method, 5)
        assert len(result.hits) == len({hit.chunk.doc_id for hit in result.hits}) == 5
        assert result.top_dense_score is not None
        assert all(indexed[hit.chunk.chunk_id] == hit.chunk for hit in result.hits)
    def generator_not_needed(*args, **kwargs):
        pytest.fail("Câu ngoài miền không được chuyển tới mô hình")

    assistant = Assistant(Settings.from_env(), lambda: retriever, generator_not_needed)
    out_of_domain = assistant.ask("Thủ đô của nước Úc là gì?", "Hybrid")
    assert out_of_domain.status == "refused"
    assert not out_of_domain.hits


def test_hybrid_v2_doc_level_bm25_deduplicates_multichunk_docs_and_breaks_ties_deterministically():
    c1_a = hit("doc-b", 0, text="Tổng quan bệnh lý tiêu hóa ban đầu").chunk
    c1_b = hit("doc-b", 1, text="Triệu chứng trào ngược dạ dày thực quản kéo dài").chunk
    c2 = hit("doc-a", 0, text="Chế độ dinh dưỡng và luyện tập cơ bản").chunk
    c3 = hit("doc-c", 0, text="Triệu chứng trào ngược dạ dày thực quản kéo dài").chunk

    index = SparseIndex([c1_a, c1_b, c2, c3])
    assert index.k1 == BM25_K1 == pytest.approx(1.2)
    assert index.b == BM25_B == pytest.approx(0.75)
    # Document-level grouping: 4 input chunks -> 3 unique documents
    assert len(index.items) == 3
    assert {item.doc_id for item in index.items} == {"doc-a", "doc-b", "doc-c"}

    # Query matches text located in chunk #1 of doc-b, returning doc-b's preserved first-chunk metadata once
    results = index.search("trào ngược dạ dày", 5)
    doc_ids = [item.chunk.doc_id for item in results]
    assert len(doc_ids) == len(set(doc_ids))
    assert "doc-b" in doc_ids
    doc_b_hit = next(item for item in results if item.chunk.doc_id == "doc-b")
    assert doc_b_hit.chunk == c1_a

    # Deterministic tie-breaking across identical documents
    tie_a = hit("tie-a", 0, text="Xét nghiệm đường huyết lúc đói").chunk
    tie_b = hit("tie-b", 0, text="Xét nghiệm đường huyết lúc đói").chunk
    tie_idx = SparseIndex([tie_b, tie_a])
    first_run = tie_idx.search("đường huyết", 2)
    assert [item.chunk.doc_id for item in first_run] == ["tie-a", "tie-b"]
    assert tie_idx.search("đường huyết", 2) == first_run


def test_hybrid_v2_accented_stopwords_preserve_vietnamese_medical_compounds():
    # Pure accented grammatical particles are stripped
    toks_filler = set(tokenize_accented("Các triệu chứng của bệnh là gì và rất phổ biến"))
    assert "e:các" not in toks_filler
    assert "e:của" not in toks_filler
    assert "e:và" not in toks_filler
    assert "e:triệu" in toks_filler and "e:chứng" in toks_filler

    # Medical compounds whose diacritic-folded forms collide with stopwords MUST be preserved
    for phrase, required_tokens in (
        ("viêm cổ tử cung", ("e:cổ", "e:tử", "e:cung", "eb:tử cung")),
        ("cơ thể mệt mỏi", ("e:cơ", "e:thể", "eb:cơ thể")),
        ("đo huyết áp định kỳ", ("e:đo", "e:huyết", "e:áp", "eb:huyết áp")),
        ("sâu răng ở trẻ em", ("e:sâu", "e:răng", "eb:sâu răng")),
        ("ợ chua sau ăn", ("e:ợ", "e:chua", "eb:ợ chua")),
    ):
        toks = set(tokenize_accented(phrase))
        for expected in required_tokens:
            assert expected in toks, f"Missing {expected!r} from {phrase!r}: {toks}"
    assert "tử" not in ACCENTED_STOPWORDS
    assert "đo" not in ACCENTED_STOPWORDS
    assert "sâu" not in ACCENTED_STOPWORDS


def test_hybrid_v2_adaptive_diacritic_routing_handles_accented_and_diacritic_free_queries():
    chunks = [
        hit("d-hbv", text="Bệnh viêm gan B và xét nghiệm HBsAg định kỳ").chunk,
        hit("d-bp", text="Yếu tố nguy cơ gây tăng huyết áp cao ở người lớn").chunk,
        hit("d-other", text="Chăm sóc da mặt hằng ngày đúng cách").chunk,
    ]
    index = SparseIndex(chunks)

    # Accented query -> matches cleanly via accented unigrams/bigrams
    assert index.search("viêm gan B", 3)[0].chunk.doc_id == "d-hbv"
    assert index.search("huyết áp cao", 3)[0].chunk.doc_id == "d-bp"

    # Diacritic-free queries -> automatically route to folded tokens and match accurately
    assert index.search("viem gan b", 3)[0].chunk.doc_id == "d-hbv"
    assert index.search("huyet ap cao", 3)[0].chunk.doc_id == "d-bp"
    assert "f:huyet" in tokenize_folded("huyet ap cao")
    assert "fb:huyet ap" in tokenize_folded("huyet ap cao")


def test_hybrid_v2_strip_query_greetings_is_conservative_and_preserves_short_queries():
    long_q = "Chào bác sĩ, bác sĩ cho em hỏi huyết áp cao có nguy hiểm không ạ? Xin cảm ơn bác sĩ!"
    cleaned = strip_query_greetings(long_q)
    assert cleaned == "huyết áp cao có nguy hiểm không ạ?"
    assert "chào" not in cleaned.casefold()
    assert "cảm ơn" not in cleaned.casefold()

    # Short query (<4 words remaining after stripping) MUST stay intact
    short_q = "Chào bác sĩ, huyết áp?"
    assert strip_query_greetings(short_q) == short_q

    # Pure medical query without greetings stays untouched and no words are added
    plain_q = "Những yếu tố nào liên quan đến tăng huyết áp?"
    assert strip_query_greetings(plain_q) == plain_q
