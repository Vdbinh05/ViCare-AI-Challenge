"""Local HTTP fault fixtures. No model, shared daemon or index is required."""
import json
import os
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import render_sources
from src.config import PROJECT_ROOT, Settings
from src.dataset import Chunk
from src.generation import Assistant, ModelUnavailable
from src.retrieval import Hit, SearchResult


@pytest.fixture
def endpoint():
    model = Settings().llm_model
    state = SimpleNamespace(tags={"models": [{"name": model, "model": model}]},
                            posts=[], calls=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.calls.append((self.path, None))
            if self.path == "/api/tags":
                self.respond(state.tags)
            else:
                self.respond({"error": "unexpected route"}, 404)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, data))
            if self.path != "/api/chat" or not state.posts:
                self.respond({"error": "unexpected request"}, 404)
                return
            status, value = state.posts.pop(0)
            self.respond(value, status)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        state.host = f"http://127.0.0.1:{server.server_port}"
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join(timeout=2)


def envelope(content):
    return {"model": Settings().llm_model, "done": True,
            "message": {"role": "assistant", "content": content}}


@pytest.mark.parametrize(("tags", "error_code"), [
    ({"version": "other-app"}, "wrong_service"),
    ({"models": [{"name": "some-other:1b", "model": "some-other:1b"}]}, "missing_model"),
])
def test_readiness_identifies_foreign_listener_and_missing_model(endpoint, monkeypatch, tags, error_code):
    from src.ollama_runtime import ensure_ready

    start_process = Mock(side_effect=AssertionError("An existing listener must be left alone"))
    monkeypatch.setattr(subprocess, "Popen", start_process)
    endpoint.tags = tags
    with pytest.raises(ModelUnavailable) as error:
        ensure_ready(Settings(ollama_host=endpoint.host))
    assert error.value.code == error_code
    assert all(path == "/api/tags" for path, _ in endpoint.calls)
    start_process.assert_not_called()


def test_readiness_requires_real_nonempty_chat_after_model_is_listed(endpoint):
    from src.ollama_runtime import ensure_ready

    settings = Settings(ollama_host=endpoint.host, timeout=2)
    endpoint.posts = [(200, envelope("  ")), (200, envelope("OK"))]
    with pytest.raises(ModelUnavailable) as error:
        ensure_ready(settings)
    assert error.value.code == "malformed_response"
    ensure_ready(settings)
    chats = [body for path, body in endpoint.calls if path == "/api/chat"]
    assert len(chats) == 2
    assert all(body["model"] == settings.llm_model and body.get("stream") is False for body in chats)


def test_chat_failure_keeps_sources_and_same_assistant_recovers(endpoint, caplog):
    excerpt = "Văn bản dùng trong kiểm thử."
    hit = Hit(Chunk("synthetic#0", "synthetic", "", excerpt), 0.99)
    distractor = Hit(Chunk("other#0", "other", "", "Nội dung không được duyệt QA_OTHER_92."), .995)
    result = SearchResult([distractor, hit], .995, "Hybrid")
    assistant = Assistant(Settings(ollama_host=endpoint.host, timeout=3),
                          retriever_factory=lambda: SimpleNamespace(search=lambda *_: result))
    question = "Dữ liệu tổng hợp QA_PRIVATE_741 gồm những gì?"
    stage3 = {"verdict": "SUFFICIENT", "supporting_ids": ["2"], "evidence_quotes": [excerpt]}
    endpoint.posts = [
        (500, {"error": f"synthetic server echoed {question}"}),
        (200, envelope("")),
        (200, envelope("unparseable synthetic verdict")),
        (200, envelope(json.dumps(stage3))),
        (200, envelope(json.dumps({"id": 1}))),
        (200, envelope(json.dumps({"question": question}))),
        (200, envelope(json.dumps({"relation": "SAME"}))),
    ]
    for _ in range(3):
        reply = assistant.ask(question, "Hybrid")
        assert (reply.status, reply.reason_code) == ("unavailable", "GENERATION_FAILURE")
        assert reply.hits == result.hits and reply.citation_ids == []
        assert excerpt in render_sources(reply.hits, reply.method, reply.citation_ids)
        assert "QA_PRIVATE_741" not in reply.answer

    reply = assistant.ask(question, "Hybrid")
    assert (reply.status, reply.reason_code) == ("answered", "SUPPORTED")
    assert reply.citation_ids == ["2"] and reply.hits == result.hits
    selection = endpoint.calls[-3][1]["messages"][1]["content"]
    assert excerpt in selection and "QA_OTHER_92" not in selection
    assert not endpoint.posts
    assert "QA_PRIVATE_741" not in caplog.text


@pytest.mark.parametrize(("relation", "reason"), [
    ("DIFFERENT", "INSUFFICIENT_EVIDENCE"), (True, "GENERATION_FAILURE"),
])
def test_source_only_check_isolated_and_rejects_different_or_malformed_scope(endpoint, relation, reason):
    excerpt = "Câu nguồn thứ nhất."
    other = "Câu nguồn tổng hợp thứ hai."
    question = "Câu hỏi tổng hợp QA_PRIVATE_729?"
    hit = Hit(Chunk("qa#0", "qa", "", f"{excerpt} {other}"), .99)
    result = SearchResult([hit], .99, "Dense")
    assistant = Assistant(Settings(ollama_host=endpoint.host, timeout=3),
                          retriever_factory=lambda: SimpleNamespace(search=lambda *_: result))
    stage3 = {"verdict": "SUFFICIENT", "supporting_ids": ["1"], "evidence_quotes": [excerpt]}
    scripted = [stage3, {"id": 1}, {"question": "Câu nào được trích?"}, {"relation": relation}]
    endpoint.posts = [(200, envelope(json.dumps(item))) for item in scripted]
    reply = assistant.ask(question, "Dense")
    assert reply.reason_code == reason and reply.citation_ids == [] and reply.hits == [hit]
    chats = [data for path, data in endpoint.calls if path == "/api/chat"]
    assert len(chats) == 4
    isolated = json.loads(chats[2]["messages"][1]["content"])
    assert excerpt in isolated and other not in isolated and question not in isolated


@pytest.mark.parametrize(("question", "excerpt"), [
    ("Cần ghi nhận gì ở người mắc hội chứng thử nghiệm A?", "Ghi nhận các biểu hiện của hội chứng B."),
    ("Đoạn ghi nhận điều gì?", "phần cuối của câu đã bị tách khỏi đoạn đầu."),
])
def test_unusable_selected_sentence_refuses_before_followup_calls(endpoint, question, excerpt):
    hit = Hit(Chunk("qa#0", "qa", "", excerpt), .99)
    result = SearchResult([hit], .99, "Hybrid")
    assistant = Assistant(Settings(ollama_host=endpoint.host, timeout=3),
                          retriever_factory=lambda: SimpleNamespace(search=lambda *_: result))
    stage3 = {"verdict": "SUFFICIENT", "supporting_ids": ["1"], "evidence_quotes": [excerpt]}
    endpoint.posts = [(200, envelope(json.dumps(data))) for data in (stage3, {"id": 1})]
    reply = assistant.ask(question, "Hybrid")
    assert reply.reason_code == "INSUFFICIENT_EVIDENCE" and reply.citation_ids == []
    assert len(endpoint.calls) == 2 and not endpoint.posts


def test_retrieval_exception_echoing_input_keeps_query_out_of_terminal(caplog):
    question = "Dữ liệu tổng hợp QA_PRIVATE_319."

    def fail_search(text, method, k):
        raise RuntimeError(text)

    assistant = Assistant(Settings(), lambda: SimpleNamespace(search=fail_search))
    reply = assistant.ask(question, "Dense")
    assert reply.reason_code == "RETRIEVAL_FAILURE"
    assert "QA_PRIVATE_319" not in reply.answer
    assert "RuntimeError" in caplog.text
    assert "QA_PRIVATE_319" not in caplog.text


def test_foreign_cwd_with_shadow_index_uses_project_index(tmp_path, monkeypatch):
    monkeypatch.delenv("VICARE_INDEX_DIR", raising=False)
    (tmp_path / "chroma_db").mkdir()
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env().index_dir.resolve() == PROJECT_ROOT / "chroma_db"


@pytest.mark.parametrize(("question", "sentence", "compatible"), [
    ("Có nên dùng nhật ký để theo dõi giấc ngủ?", "Có thể ghi lại giấc ngủ trong nhật ký.", False),
    ("Có thể ghi giấc ngủ vào nhật ký không?", "Có thể ghi nhật ký sau khi thức dậy.", True),
    ("Có nên ghi nhật ký giấc ngủ?", "Tài liệu khuyến nghị ghi nhật ký giấc ngủ.", True),
    ("Vị trí ở mặt trong khác thế nào?", "Đoạn tài liệu mô tả mặt ngoài.", False),
    ("Dấu đánh ở bên trái nằm ở đâu?", "Dấu đánh nằm ở bên trái.", True),
])
def test_explicit_scope_keeps_recommendation_and_site(question, sentence, compatible):
    from src.generation import _scope_compatible

    assert _scope_compatible(question, sentence) is compatible


@pytest.mark.skipif(os.getenv("VICARE_QA_REAL_OLLAMA") != "1", reason="Cần model Ollama cục bộ")
@pytest.mark.parametrize(("question", "decision"), [
    ("Có thể ghi lại giấc ngủ trong nhật ký không?", "ANSWER"),
    ("Có nên ghi lại giấc ngủ trong nhật ký không?", "REFUSE"),
])
def test_local_model_distinguishes_possibility_and_recommendation(question, decision):
    from src.generation import OllamaGenerator, parse_and_validate_output

    excerpt = "Có thể ghi lại giấc ngủ trong nhật ký sau khi thức dậy."
    hit = Hit(Chunk("scope#0", "scope", "", excerpt), 0.99)
    reply = OllamaGenerator(Settings.from_env())(question, [hit])
    valid, parsed = parse_and_validate_output(reply, 1)
    assert valid and parsed["decision"] == decision
    if decision == "ANSWER":
        assert excerpt in parsed["answer"] and parsed["citation_ids"] == ["1"]


@pytest.fixture
def launch_ui(monkeypatch):
    import app

    demo = Mock()
    demo.launch.side_effect = lambda **kwargs: (
        demo, f"http://127.0.0.1:{kwargs['server_port']}/", None
    )
    monkeypatch.setattr(app, "Assistant", Mock())
    monkeypatch.setattr(app, "build_app", Mock(return_value=demo))
    return app, demo


def test_launch_default_skips_occupied_ports_and_prints_listening_url(launch_ui, monkeypatch, capsys):
    app, demo = launch_ui
    monkeypatch.setattr(app, "port_in_use", lambda port: port in {7860, 7861})
    _, url, _ = app.launch_app(prevent_thread_lock=True)
    assert url == "http://127.0.0.1:7862/"
    options = demo.launch.call_args.kwargs
    assert options["server_name"] == "127.0.0.1" and options["share"] is False
    output = capsys.readouterr().out
    assert url in output and f"PID={os.getpid()}" in output
    demo.launch.assert_called_once()


def test_explicit_port_conflict_detects_real_listener_without_fallback(launch_ui):
    app, demo = launch_ui
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(OSError, match=str(port)):
            app.launch_app(port=port, prevent_thread_lock=True)
    demo.launch.assert_not_called()
    app.Assistant.assert_not_called()


def test_unrelated_gradio_oserror_propagates_without_port_retry(launch_ui, monkeypatch):
    app, demo = launch_ui
    monkeypatch.setattr(app, "port_in_use", lambda port: False)
    demo.launch.side_effect = OSError("synthetic asset read failure")
    with pytest.raises(OSError, match="asset read failure"):
        app.launch_app(prevent_thread_lock=True)
    demo.launch.assert_called_once()


def test_race_after_initial_port_check_is_rechecked_and_reuses_prepared_demo(launch_ui, monkeypatch):
    app, demo = launch_ui
    probe = Mock(side_effect=[False, True, False])
    monkeypatch.setattr(app, "port_in_use", probe)
    demo.launch.side_effect = [OSError("Cannot find empty port in range: 7860-7860."),
                               (demo, "http://127.0.0.1:7861/", None)]
    assert app.launch_app(prevent_thread_lock=True)[1] == "http://127.0.0.1:7861/"
    assert [call.args[0] for call in probe.call_args_list] == [7860, 7860, 7861]
    app.Assistant.return_value.generate.prepare.assert_called_once()
    app.build_app.assert_called_once()
