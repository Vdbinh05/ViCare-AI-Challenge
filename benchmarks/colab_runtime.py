"""Linux setup for the pinned submission notebook. Run from the source checkout."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen


UV_VERSION = "0.11.12"
UV_SHA = "9acdecddacba550ee616c02bb4616d894352022550c5977524556fd5077ce1d4"
PYTHON_VERSION = "3.11.14"
OLLAMA_VERSION = "0.40.17"
OLLAMA_SHA = "1df9d6228bac5a22e420b242708c0d10e5b886db97bab99d446a41760f4e79af"
MODEL = "qwen3:4b-instruct"
MODEL_SHA = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
HOST = "http://127.0.0.1:11434"
ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = build_opener(ProxyHandler({}))


def fetch(url, path, sha):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "20", "--max-time", "1800", "--retry", "3",
                        "--output", str(temporary), url], check=True)
        with temporary.open("rb") as data:
            digest = hashlib.sha256()
            for block in iter(lambda: data.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha:
            raise RuntimeError(f"Sai SHA-256 khi tải {path.name}.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install(work):
    directory = work / ("uv-" + UV_VERSION)
    uv = directory / "uv"
    if not uv.exists():
        directory.mkdir(exist_ok=True)
        archive = directory / "uv.tar.gz"
        fetch(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
              "uv-x86_64-unknown-linux-gnu.tar.gz", archive, UV_SHA)
        with tarfile.open(archive, "r:gz") as bundle:
            with bundle.extractfile("uv-x86_64-unknown-linux-gnu/uv") as binary:
                stage = directory / "uv.tmp"
                with stage.open("wb") as output:
                    shutil.copyfileobj(binary, output)
                stage.chmod(0o755)
                stage.replace(uv)
        archive.unlink()
    venv = work / "venv"
    python = venv / "bin" / "python"
    env = {**os.environ, "UV_CACHE_DIR": str(work / "uv-cache"),
           "UV_PYTHON_INSTALL_DIR": str(work / "python"), "UV_NO_PROGRESS": "1",
           "UV_PYTHON_BIN_DIR": str(work / "bin")}
    if not python.exists():
        subprocess.run([uv, "venv", "--managed-python", "--python", PYTHON_VERSION, venv],
                       env=env, check=True)
    actual = subprocess.check_output([python, "-c", "import platform; print(platform.python_version())"],
                                     text=True).strip()
    if actual != PYTHON_VERSION:
        raise RuntimeError(f"Python môi trường riêng là {actual}, cần {PYTHON_VERSION}.")
    subprocess.run([uv, "pip", "install", "--python", python, "--torch-backend", "cpu",
                    "-r", ROOT / "requirements.txt", "-r", ROOT / "benchmarks/colab-constraints.txt"],
                   env=env, check=True)
    subprocess.run([uv, "pip", "check", "--python", python], env=env, check=True)


def get_api(path):
    with LOOPBACK.open(HOST + path, timeout=4) as response:
        return json.load(response)


def ensure_linux_server(work):
    try:
        version = get_api("/api/version")["version"]
    except URLError as exc:
        if not isinstance(exc.reason, ConnectionRefusedError):
            raise
    else:
        if version != OLLAMA_VERSION:
            raise RuntimeError(f"Ollama đang chạy là {version}, cần {OLLAMA_VERSION}.")
        return "đã chạy"

    import zstandard

    directory = work / ("ollama-" + OLLAMA_VERSION)
    binary = directory / "bin" / "ollama"
    if not binary.is_file():
        archive = work / "ollama-linux-amd64.tar.zst"
        fetch(f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/{archive.name}",
              archive, OLLAMA_SHA)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            stage = Path(tmp) / "release"
            stage.mkdir()
            with archive.open("rb") as file, zstandard.ZstdDecompressor().stream_reader(file) as data:
                with tarfile.open(fileobj=data, mode="r|") as tar:
                    tar.extractall(stage, filter="data")
            if not (stage / "bin" / "ollama").is_file():
                raise RuntimeError("Gói Ollama thiếu bin/ollama.")
            stage.rename(directory)
        archive.unlink()
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434",
           "OLLAMA_MODELS": str(work / "models"), "OLLAMA_NO_CLOUD": "1"}
    with (work / "ollama.log").open("ab") as log:
        server = subprocess.Popen([binary, "serve"], env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=log, start_new_session=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            version = get_api("/api/version")["version"]
        except URLError as exc:
            if not isinstance(exc.reason, ConnectionRefusedError):
                raise
        else:
            if version != OLLAMA_VERSION:
                raise RuntimeError("Phiên bản Ollama lắng nghe khác bản đã ghim.")
            return "khởi động trong runtime"
        if server.poll() is not None:
            raise RuntimeError(f"Ollama thoát mã {server.returncode}. Xem {work / 'ollama.log'}.")
        time.sleep(0.5)
    raise TimeoutError("Quá 60 giây chờ API Ollama sẵn sàng.")


def model_digest():
    return next((item["digest"].removeprefix("sha256:") for item in get_api("/api/tags")["models"]
                 if item["name"] == MODEL), None)


def prepare_ollama(work):
    from filelock import FileLock
    from src.config import Settings
    from src.ollama_runtime import chat

    settings = Settings.from_env()
    if settings.ollama_host != HOST or settings.llm_model != MODEL:
        raise RuntimeError("Sai địa chỉ Ollama hoặc tên model của notebook.")
    with FileLock(str(work / "ollama-setup.lock"), timeout=2400):
        service = ensure_linux_server(work)
        if model_digest() is None:
            # Check the public tag before downloading large blobs; recheck the local manifest afterward.
            manifest = Request("https://registry.ollama.ai/v2/library/qwen3/manifests/4b-instruct",
                               headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
            with urlopen(manifest, timeout=30) as response:
                if hashlib.sha256(response.read()).hexdigest() != MODEL_SHA:
                    raise RuntimeError("Tag Qwen trên registry đã đổi digest.")
            request = Request(HOST + "/api/pull", data=json.dumps({"model": MODEL, "stream": True}).encode(),
                              headers={"Content-Type": "application/json"})
            deadline = time.monotonic() + 1800
            status = None
            with LOOPBACK.open(request, timeout=120) as response:
                for line in response:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Quá 30 phút tải model.")
                    progress = json.loads(line)
                    if "error" in progress:
                        raise RuntimeError("Ollama không tải được tag đã ghim.")
                    status = progress.get("status")
            if status != "success":
                raise RuntimeError("Ollama chưa báo tải model thành công.")
        if model_digest() != MODEL_SHA:
            raise RuntimeError("Digest Qwen trên máy khác bản đã đánh giá.")
        answer = chat(settings, [{"role": "user", "content": "Reply with OK."}], 12)
        loaded = next((m for m in get_api("/api/ps")["models"] if m.get("name") == MODEL), {})
        print(json.dumps({"service": service, "version": OLLAMA_VERSION, "model": MODEL,
                          "digest": MODEL_SHA, "smoke_answer": answer, "size_bytes": loaded.get("size"),
                          "size_vram_bytes": loaded.get("size_vram")},
                         ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "ollama"))
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        sys.exit("Notebook cần runtime Linux x86_64.")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.action == "install":
        install(work)
    else:
        prepare_ollama(work)


if __name__ == "__main__":
    main()
