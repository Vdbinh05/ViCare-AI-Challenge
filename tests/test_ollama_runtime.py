from types import SimpleNamespace

import httpx
import ollama
import pytest

from src.config import Settings
from src import ollama_runtime as runtime


def test_tags_unavailable_http_does_not_imply_foreign_service(monkeypatch, caplog):
    def unavailable(url, **options):
        assert options["timeout"] == 3.0
        return httpx.Response(503, text="SYNTHETIC_PRIVATE")

    monkeypatch.setattr(httpx, "get", unavailable)
    with pytest.raises(runtime.ModelUnavailable) as failure:
        runtime.ensure_ready(Settings(timeout=45))
    assert failure.value.code == "http_error"
    assert "503" in str(failure.value)
    assert "SYNTHETIC_PRIVATE" not in caplog.text


@pytest.mark.parametrize(("status", "body", "code"), [
    (404, "an unrelated resource vanished", "http_error"),
    (404, "404 page not found", "wrong_service"),
    (404, "model 'qwen3:4b-instruct' not found, try pulling it first", "missing_model"),
    (500, "model requires more system memory than is available", "model_load"),
])
def test_chat_classifies_server_errors_without_logging_body(monkeypatch, caplog, status, body, code):
    def fail(**kwargs):
        raise ollama.ResponseError(body + " SYNTHETIC_PRIVATE", status)

    monkeypatch.setattr(ollama, "Client", lambda **kwargs: SimpleNamespace(chat=fail))
    with pytest.raises(runtime.ModelUnavailable) as failure:
        runtime.chat(Settings(), [], 10)
    assert failure.value.code == code
    assert "SYNTHETIC_PRIVATE" not in str(failure.value) + caplog.text


def test_probe_rejects_completed_generation_for_different_model(monkeypatch):
    response = SimpleNamespace(model="other", done=True, message=SimpleNamespace(role="assistant", content="OK"))
    monkeypatch.setattr(ollama, "Client", lambda **kwargs: SimpleNamespace(chat=lambda **opts: response))
    with pytest.raises(runtime.ModelUnavailable) as failure:
        runtime.chat(Settings(), [], 10)
    assert failure.value.code == "malformed_response"


def test_concurrent_start_rechecks_listener_under_lock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event, local
    from unittest.mock import Mock

    callers = Barrier(2)
    active = Event()
    worker = local()
    process = Mock(pid=130)
    process.poll.return_value = None

    def installed(settings):
        if not getattr(worker, "first_check_done", False):
            worker.first_check_done = True
            callers.wait(timeout=3)
            return False
        return active.is_set()

    def start(settings):
        active.set()
        return process

    launch = Mock(side_effect=start)
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime, "_installed_or_absent", installed)
    monkeypatch.setattr(runtime, "_start_server", launch)
    monkeypatch.setattr(runtime, "chat", lambda *args: "OK")
    s = Settings(ollama_host="http://127.0.0.1:11451")
    with ThreadPoolExecutor(max_workers=2) as callers_pool:
        results = list(callers_pool.map(runtime.ensure_ready, (s, s)))
    launch.assert_called_once_with(s)
    assert all(s.llm_model in result for result in results)
