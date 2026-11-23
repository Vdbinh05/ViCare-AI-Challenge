"""Source-span regressions use synthetic text and scripted model responses only."""
import json

import pytest

from src.config import Settings
from src.dataset import Chunk
from src.generation import ModelUnavailable, OllamaGenerator
from src.retrieval import Hit


def scripted_generator(monkeypatch, selection, question, relation="SAME"):
    generator = OllamaGenerator(Settings())
    responses = iter([selection, {"question": question}, {"relation": relation}])
    calls = []

    def structured(system, data, schema, max_tokens):
        calls.append({"system": system, "data": data, "schema": schema})
        return next(responses)

    monkeypatch.setattr(generator, "_structured", structured)
    return generator, calls


def source(text, number=1):
    return Hit(Chunk(f"synthetic-{number}#0", f"synthetic-{number}", "", text), .99)


@pytest.mark.parametrize(("excerpt", "question"), [
    ("Quy trình chỉ áp dụng cho nhóm A.\n\nCác bước gồm X và Y.",
     "Các bước của quy trình dành cho nhóm A gồm những gì?"),
    ("Quy trình được thực hiện ở bên trái. Các bước gồm X và Y.",
     "Quy trình ở bên trái gồm những bước nào?"),
    ("Tài liệu nói về người mắc hội chứng thử nghiệm A. Các biểu hiện gồm X và Y.",
     "Cần ghi nhận gì ở người mắc hội chứng thử nghiệm A?"),
])
def test_context_and_answer_are_checked_and_quoted_as_the_same_verbatim_span(monkeypatch, excerpt, question):
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 2}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "ANSWER" and reply["citation_ids"] == ["1"]
    assert reply["answer"] == f"Tài liệu [1] ghi “{excerpt}”"
    assert calls[1]["data"] == "Đoạn tài liệu: " + excerpt
    assert calls[2]["data"]["source_span"] == excerpt
    assert [entry["id"] for entry in calls[0]["data"]["nguon"]] == [1, 2]
    assert all(entry["so_nguon"] == 1 for entry in calls[0]["data"]["nguon"])


def test_three_sentence_span_preserves_source_number_and_original_whitespace(monkeypatch):
    excerpt = "Phạm vi là nhóm A.  Bước đầu là X.\nBước cuối là Y."
    question = "Quy trình dành cho nhóm A gồm những gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 2, "end_id": 4}, question)
    reply = json.loads(generator(question, [source("Nguồn không liên quan."), source(excerpt, 2)]))
    assert reply["citation_ids"] == ["2"]
    assert reply["answer"] == f"Tài liệu [2] ghi “{excerpt}”"
    assert calls[2]["data"]["source_span"] == excerpt


def test_unselected_preceding_sentence_is_never_hidden_scope_evidence(monkeypatch):
    hidden = "Câu trước chứa QA_OUTSIDE_SPAN_92."
    chosen = "Câu được chọn ghi dữ liệu tổng hợp."
    question = "Câu được chọn ghi gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 2, "end_id": 2}, question)
    reply = json.loads(generator(question, [source(hidden + " " + chosen)]))
    assert calls[1]["data"] == "Đoạn tài liệu: " + chosen
    assert calls[2]["data"]["source_span"] == chosen
    assert "QA_OUTSIDE_SPAN_92" not in reply["answer"]


@pytest.mark.parametrize("selection", [
    {"start_id": True, "end_id": 1}, {"start_id": "1", "end_id": 1},
    {"start_id": 1}, {"start_id": 0, "end_id": 1}, {"start_id": 1, "end_id": 0},
    {"start_id": -1, "end_id": 1}, {"start_id": 2, "end_id": 1},
    {"start_id": 1, "end_id": 6}, {"start_id": 1, "end_id": 4},
    {"start_id": 4, "end_id": 5},  # Adjacent IDs from different sources.
])
def test_malformed_oversized_or_cross_source_span_fails_closed(monkeypatch, selection):
    generator, calls = scripted_generator(monkeypatch, selection, "Câu hỏi tổng hợp?")
    hits = [source("Câu một. Câu hai. Câu ba. Câu bốn."), source("Câu nguồn khác.", 2)]
    with pytest.raises(ModelUnavailable) as error:
        generator("Câu hỏi tổng hợp?", hits)
    assert error.value.code == "malformed_response"
    assert len(calls) == 1


def test_zero_span_is_an_explicit_evidence_refusal(monkeypatch):
    generator, calls = scripted_generator(monkeypatch, {"start_id": 0, "end_id": 0}, "Câu hỏi?")
    reply = json.loads(generator("Câu hỏi?", [source("Một câu nguồn.")]))
    assert reply["decision"] == "REFUSE" and reply["reason_code"] == "INSUFFICIENT_EVIDENCE"
    assert len(calls) == 1


@pytest.mark.parametrize(("excerpt", "question"), [
    ("Bước đầu là đọc hướng dẫn. Liều tham khảo là 20 mg.", "Quy trình gồm những gì?"),
    ("Bước đầu là đọc hướng dẫn. Liên hệ qa@example.org.", "Quy trình gồm những gì?"),
    ("Bước đầu là đọc hướng dẫn. Tôi đang dùng phương pháp này.", "Quy trình gồm những gì?"),
    ("Bước đầu là đọc hướng dẫn. Xem https://example.org.", "Quy trình gồm những gì?"),
    ("Bước đầu là đọc hướng dẫn. <b>Thông tin khác.</b>", "Quy trình gồm những gì?"),
    ("Bước đầu là đọc hướng dẫn. Thông tin khác [9].", "Quy trình gồm những gì?"),
    ("phần cuối của một câu. Các bước gồm X và Y.", "Quy trình gồm những gì?"),
    ("Dữ liệu chưa hoàn tất... Phần tiếp theo.", "Quy trình gồm những gì?"),
    ("Quy trình được thực hiện ở bên phải. Các bước gồm X và Y.",
     "Quy trình ở bên trái gồm những bước nào?"),
    ("Tài liệu mô tả nhật ký. Có thể ghi lại giấc ngủ trong nhật ký.",
     "Có nên ghi lại giấc ngủ trong nhật ký không?"),
    ("Tài liệu nói về người mắc hội chứng thử nghiệm B. Các biểu hiện gồm X và Y.",
     "Cần ghi nhận gì ở người mắc hội chứng thử nghiệm A?"),
])
def test_every_selected_sentence_remains_subject_to_safety_and_scope_guards(monkeypatch, excerpt, question):
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 2}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "REFUSE" and reply["citation_ids"] == []
    assert len(calls) == 1


def test_semantic_relation_rejection_still_blocks_a_valid_span(monkeypatch):
    excerpt = "Đây là một phép đo. Yếu tố X ảnh hưởng đến kết quả đo."
    question = "Yếu tố nào gây bệnh Y?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 2}, question, "DIFFERENT")
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "REFUSE" and reply["citation_ids"] == []
    assert len(calls) == 3


@pytest.mark.parametrize("excerpt", [
    "Mẹ bạn có các triệu chứng A và B.",
    "Con của chị đã gặp tình trạng A.",
    "Theo thông tin bạn cung cấp, mẹ bạn bị A.",
    "Tài liệu ghi: bố của bạn đang theo dõi tình trạng A.",
    "Bạn có các dấu hiệu A.",
    "Anh trai của bạn từng gặp tình trạng A.",
])
def test_another_person_case_assertion_cannot_be_reused_as_general_evidence(monkeypatch, excerpt):
    question = "Tài liệu mô tả điều gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 1}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "REFUSE" and reply["citation_ids"] == []
    assert len(calls) == 1


def test_family_narrative_in_later_sentence_is_still_blocked(monkeypatch):
    question = "Tài liệu mô tả điều gì?"
    excerpt = "Đây là đoạn ghi nhận. Con bạn có biểu hiện A."
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 2}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "REFUSE" and len(calls) == 1


@pytest.mark.parametrize("excerpt", [
    "Bố mẹ mang yếu tố A có thể truyền đặc điểm B.",
    "Trẻ có các biểu hiện A cần được đánh giá theo tài liệu.",
    "Con của người mang yếu tố A cần được đánh giá theo tài liệu.",
    "Nếu mẹ bạn có biểu hiện A, hãy liên hệ người hướng dẫn.",
    "Khi con bạn có dấu hiệu A, người hướng dẫn sẽ xem xét thông tin.",
    "Bạn nên trao đổi với người hướng dẫn.",
    "Mẹ bạn cần trao đổi với người hướng dẫn.",
    "Bạn có thể ghi lại thông tin.",
    "Vì vậy, người học nên ghi lại thông tin.",
])
def test_generic_population_conditional_information_and_normal_advice_remain_eligible(monkeypatch, excerpt):
    question = "Tài liệu mô tả điều gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 1}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "ANSWER" and reply["citation_ids"] == ["1"]
    assert len(calls) == 3


@pytest.mark.parametrize("excerpt", [
    "Khi các biểu hiện trên kéo dài, cần ghi nhận lại.",
    "Trong trường hợp này, các bước gồm X và Y.",
    "Các kết quả đó cần được đối chiếu.",
    "Bệnh này được mô tả trong tài liệu.",
    "Những dấu hiệu đã nêu cần được đối chiếu.",
])
def test_opening_backward_reference_requires_visible_antecedent(monkeypatch, excerpt):
    question = "Tài liệu mô tả điều gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 1}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "REFUSE" and len(calls) == 1


def test_backward_reference_can_be_quoted_with_its_visible_context(monkeypatch):
    question = "Tài liệu mô tả điều gì?"
    context = "Nhóm A có các biểu hiện X và Y."
    dependent = "Khi các biểu hiện trên kéo dài, cần ghi nhận lại."
    hit = source(context + " " + dependent)
    for selection, expected, call_count in [
        ({"start_id": 2, "end_id": 2}, "REFUSE", 1),
        ({"start_id": 1, "end_id": 2}, "ANSWER", 3),
    ]:
        generator, calls = scripted_generator(monkeypatch, selection, question)
        reply = json.loads(generator(question, [hit]))
        assert reply["decision"] == expected and len(calls) == call_count
        if expected == "ANSWER":
            assert hit.chunk.text in reply["answer"]


@pytest.mark.parametrize("excerpt", [
    "Các biểu hiện trên da được mô tả bằng ký hiệu A.",
    "Các chỉ số trên nhãn cần được đọc kỹ.",
    "Các biểu hiện trên lâm sàng được ghi nhận trong tài liệu.",
])
def test_spatial_tren_is_not_treated_as_a_backward_reference(monkeypatch, excerpt):
    question = "Tài liệu mô tả điều gì?"
    generator, calls = scripted_generator(monkeypatch, {"start_id": 1, "end_id": 1}, question)
    reply = json.loads(generator(question, [source(excerpt)]))
    assert reply["decision"] == "ANSWER" and len(calls) == 3
