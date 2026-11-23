"""Local Ollama readiness and transport. User input and server bodies stay out of logs."""
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .config import PROJECT_ROOT
from .safety import strip_hidden_reasoning

logger = logging.getLogger(__name__)
_owned_servers = {}


class ModelUnavailable(RuntimeError):
    """Public service diagnostic. The cause may contain untrusted server text."""

    def __init__(self, message, code="http_error", *, status_code=None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def log_failure(stage, exc, started):
    """Keep stack locations and OS error codes without printing private payloads."""
    frames = traceback.extract_tb(exc.__traceback__)
    chain = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        label = type(exc).__name__
        details = [exc.code] if isinstance(exc, ModelUnavailable) else []
        number = getattr(exc, "status_code", None) or getattr(exc, "errno", None)
        if isinstance(number, int):
            details.append(str(number))
        chain.append(f"{label}({','.join(details)})" if details else label)
        exc = exc.__cause__ or exc.__context__
    logger.warning("%s failed elapsed=%.3fs errors=%s stack=%s", stage,
                   time.monotonic() - started, " <- ".join(chain),
                   " > ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in frames))


def _timeout(settings):
    return httpx.Timeout(settings.timeout, connect=min(3.0, settings.timeout))


def _transport_error(exc):
    if isinstance(exc, httpx.ConnectTimeout):
        return ModelUnavailable("Chưa kết nối được Ollama trong thời gian chờ kết nối.", "connect_timeout")
    if isinstance(exc, httpx.TimeoutException):
        return ModelUnavailable("Ollama phản hồi quá chậm. Có thể mô hình đang nạp hoặc xử lý yêu cầu.", "timeout")
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return ModelUnavailable("Chưa kết nối được Ollama tại địa chỉ đã cấu hình.", "connection")
    return ModelUnavailable("Ollama gặp lỗi giao thức khi trao đổi dữ liệu.", "wrong_service")


def check_installed(settings):
    url = settings.ollama_host.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, trust_env=False, follow_redirects=False, timeout=min(3.0, settings.timeout))
    except httpx.ReadTimeout as exc:
        raise ModelUnavailable("Ollama phản hồi quá chậm khi đọc danh sách mô hình.", "timeout") from exc
    except httpx.RequestError as exc:
        raise _transport_error(exc) from exc
    if 300 <= response.status_code < 400 or response.status_code in {404, 405}:
        raise ModelUnavailable(f"Endpoint không cung cấp danh sách mô hình Ollama (HTTP {response.status_code}).",
                               "wrong_service", status_code=response.status_code)
    if response.status_code != 200:
        raise ModelUnavailable(f"Yêu cầu danh sách mô hình /api/tags gặp lỗi HTTP {response.status_code}.",
                               "http_error", status_code=response.status_code)
    try:
        models = response.json()["models"]
        if not isinstance(models, list):
            raise TypeError()
        names = [model["name"] for model in models]
        if not all(isinstance(name, str) for name in names):
            raise TypeError()
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelUnavailable("Dịch vụ tại endpoint trả về danh sách mô hình sai định dạng Ollama.",
                               "wrong_service") from exc
    if settings.llm_model not in names:
        raise ModelUnavailable(f"Ollama chưa có mô hình {settings.llm_model} tại endpoint đã cấu hình.",
                               "missing_model")


def chat(settings, messages, max_tokens, *, response_format=None):
    from ollama import Client, ResponseError
    from pydantic import ValidationError

    started = time.monotonic()
    client = Client(host=settings.ollama_host, timeout=_timeout(settings),
                    trust_env=False, follow_redirects=False)
    try:
        try:
            response = client.chat(
                model=settings.llm_model, messages=messages, stream=False, format=response_format,
                options={"temperature": settings.temperature, "num_predict": max_tokens, "num_ctx": 4096},
            )
        except (ConnectionError, httpx.RequestError) as exc:
            raise _transport_error(exc) from exc
        except ResponseError as exc:
            body = str(exc.error).lower()
            if exc.status_code == 404 and any(
                    f"model {q}{settings.llm_model.lower()}{q} not found" in body for q in ("'", '"', "")):
                raise ModelUnavailable(f"Ollama chưa có mô hình {settings.llm_model}.", "missing_model") from exc
            if exc.status_code == 404 and "page not found" in body:
                raise ModelUnavailable("Endpoint gặp lỗi vì thiếu tuyến Ollama /api/chat (HTTP 404).",
                                       "wrong_service") from exc
            if exc.status_code >= 500 and any(term in body
                    for term in ("load model", "loading model", "model runner", "llama runner",
                                 "out of memory", "available memory", "model requires more")):
                raise ModelUnavailable(f"Ollama gặp lỗi nạp mô hình (HTTP {exc.status_code}).", "model_load") from exc
            raise ModelUnavailable(f"Ollama gặp lỗi tại /api/chat (HTTP {exc.status_code}).", "http_error") from exc
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise ModelUnavailable("Phản hồi /api/chat sai định dạng.", "malformed_response") from exc
        text = strip_hidden_reasoning(response.message.content or "")
        if response.model != settings.llm_model:
            raise ModelUnavailable("Phản hồi /api/chat sai tên mô hình đã cấu hình.", "malformed_response")
        if response.message.role != "assistant" or response.done is not True or not text:
            raise ModelUnavailable("Phản hồi /api/chat thiếu nội dung hoàn chỉnh.", "malformed_response")
        logger.info("POST /api/chat model=%s elapsed=%.3fs", settings.llm_model, time.monotonic() - started)
        return text
    except ModelUnavailable as exc:
        log_failure("POST /api/chat", exc, started)
        raise
    finally:
        # Ollama 0.5.x exposes no public close/context manager on Client.
        transport = getattr(client, "_client", None)
        if transport is not None:
            transport.close()


def _configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _start_server(settings):
    explicit_exe = os.getenv("VICARE_OLLAMA_EXE")
    existing_exe = Path(r"D:\ollama\ollama.exe")
    exe = (_configured_path(explicit_exe) if explicit_exe else
           existing_exe if existing_exe.is_file() else shutil.which("ollama"))
    if not exe or not Path(exe).is_file():
        raise ModelUnavailable("Không tìm thấy chương trình Ollama. Kiểm tra VICARE_OLLAMA_EXE.", "missing_executable")
    configured_store = os.getenv("VICARE_OLLAMA_MODELS") or os.getenv("OLLAMA_MODELS")
    stores = ([_configured_path(configured_store)] if configured_store else
              [Path(r"D:\ollama_models"), Path.home() / ".ollama" / "models"])
    store = next((p for p in stores if (p / "blobs").is_dir() and (p / "manifests").is_dir()), None)
    if store is None:
        raise ModelUnavailable("Không tìm thấy kho mô hình hiện có. Kiểm tra VICARE_OLLAMA_MODELS.", "missing_store")
    host = urlsplit(settings.ollama_host)
    bind_host = f"[{host.hostname}]" if host.hostname == "::1" else host.hostname
    env = {**os.environ, "OLLAMA_HOST": f"{bind_host}:{host.port or 80}", "OLLAMA_MODELS": str(store)}
    try:
        process = subprocess.Popen([str(exe), "serve"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise ModelUnavailable("Không thể khởi chạy chương trình Ollama đã cấu hình.", "spawn_failed") from exc
    _owned_servers[settings.ollama_host] = process
    logger.info("Started Ollama PID=%d endpoint=%s executable=%s model_store=%s",
                process.pid, settings.ollama_host, exe, store)
    return process


def _installed_or_absent(settings):
    try:
        check_installed(settings)
    except ModelUnavailable as exc:
        cause = exc
        while cause is not None:
            if isinstance(cause, ConnectionRefusedError):
                return False
            cause = cause.__cause__ or cause.__context__
        raise
    return True


def ensure_ready(settings):
    """At startup only: reuse or start local serve, verify tags, then verify nonstream chat."""
    from filelock import FileLock, Timeout  # Installed with the existing Hugging Face stack.

    started = time.monotonic()
    host = urlsplit(settings.ollama_host)
    service = "Dùng dịch vụ Ollama đang chạy"
    try:
        if not _installed_or_absent(settings):
            if host.hostname not in {"127.0.0.1", "localhost", "::1"} or host.scheme != "http":
                raise ModelUnavailable("Chưa kết nối được Ollama tại endpoint đã cấu hình.", "connection")
            lock = Path(tempfile.gettempdir()) / f"vicare-ollama-{host.port or 80}.lock"
            with FileLock(str(lock), timeout=20):
                if not _installed_or_absent(settings):
                    process = _owned_servers.get(settings.ollama_host)
                    if process is None or process.poll() is not None:
                        process = _start_server(settings)
                    deadline = time.monotonic() + 12
                    while not _installed_or_absent(settings):
                        if process.poll() is not None:
                            # A separately started daemon may have won the bind race.
                            if _installed_or_absent(settings):
                                break
                            raise ModelUnavailable(
                                f"Ollama serve đã thoát (mã {process.returncode}) trước khi API sẵn sàng.", "spawn_failed")
                        if time.monotonic() >= deadline:
                            raise ModelUnavailable("Hết thời gian chờ Ollama mở cổng dịch vụ.", "startup_timeout")
                        time.sleep(0.2)
                    if process.poll() is None:
                        service = f"Đã khởi chạy Ollama (PID {process.pid})"
        chat(settings, [{"role": "user", "content": "Reply with OK."}], 12)
        logger.info("%s endpoint=%s model=%s elapsed=%.3fs",
                    service, settings.ollama_host, settings.llm_model, time.monotonic() - started)
        return f"{service}. Mô hình {settings.llm_model} sẵn sàng."
    except Timeout as exc:
        raise ModelUnavailable("Hết thời gian chờ phiên ViCare khác khởi động Ollama.", "startup_timeout") from exc
    except ModelUnavailable as exc:
        log_failure("Ollama startup", exc, started)
        raise
