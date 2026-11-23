"""Shared settings for the research demo and fixed comparison."""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DATASET = "tarudesu/ViHealthQA"
DATASET_REVISION = "d90a62deedfc822a15de3f35535bb41474e4d3f4"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_HYBRID_BACKEND = "exact"
CHUNK_SIZE = 200
OVERLAP = 40
INDEX_COMMAND = "python -m src.retrieval build"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_CACHE = PROJECT_ROOT / ".cache" / "hf"

# VICARE_LLM_MODEL overrides the locally evaluated default.
DEFAULT_LLM_MODEL = "qwen3:4b-instruct"

# Uncalibrated demonstration dense-similarity gate retained from the original notebook;
# NOT clinically validated.
DEFAULT_MIN_DENSE_SCORE = 0.86
DEFAULT_MAX_OUTPUT_TOKENS = 384
MAX_ALLOWED_OUTPUT_TOKENS = 2048


@dataclass(frozen=True)
class Settings:
    index_dir: Path = PROJECT_ROOT / "chroma_db"
    # Uncalibrated demonstration threshold retained from the original notebook (NOT clinically validated).
    threshold: float = DEFAULT_MIN_DENSE_SCORE
    candidate_pool: int = 80
    rrf_k: int = 40
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    top_k: int = 5
    max_question_chars: int = 1000
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LLM_MODEL
    timeout: float = 45.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    llm_model: str | None = field(default=None, repr=False, compare=False)
    hybrid_dense_depth: int = 200

    def __post_init__(self):
        if self.llm_model is not None:
            if not isinstance(self.llm_model, str):
                raise ValueError("Tên mô hình LLM phải là chuỗi ký tự.")
            if self.ollama_model != DEFAULT_LLM_MODEL and self.ollama_model != self.llm_model:
                raise ValueError("Không truyền đồng thời llm_model và ollama_model với giá trị khác nhau.")
            resolved_model = self.llm_model
        else:
            if not isinstance(self.ollama_model, str):
                raise ValueError("Tên mô hình LLM phải là chuỗi ký tự.")
            resolved_model = self.ollama_model

        resolved_model = resolved_model.strip()
        object.__setattr__(self, "ollama_model", resolved_model)
        object.__setattr__(self, "llm_model", resolved_model)

        integers = (
            self.candidate_pool,
            self.hybrid_dense_depth,
            self.rrf_k,
            self.top_k,
            self.max_question_chars,
            self.max_output_tokens,
        )
        weights = (self.dense_weight, self.sparse_weight)
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("Kích thước và số hạng cấu hình phải là số nguyên dương.")
        if self.max_output_tokens > MAX_ALLOWED_OUTPUT_TOKENS:
            raise ValueError(f"Số token đầu ra tối đa không được vượt quá {MAX_ALLOWED_OUTPUT_TOKENS}.")
        if self.top_k > self.candidate_pool or self.top_k > 10:
            raise ValueError("Số nguồn hiển thị phải từ 1 đến 10 và trong nhóm ứng viên.")
        if not -1 <= self.threshold <= 1:
            raise ValueError("Ngưỡng cosine phải nằm trong khoảng từ -1 đến 1.")
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
            raise ValueError("Trọng số RRF phải hữu hạn, không âm và có ít nhất một giá trị dương.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Thời gian chờ Ollama phải hữu hạn và lớn hơn 0.")
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0):
            raise ValueError("Nhiệt độ sinh văn bản phải nằm trong khoảng từ 0 đến 2.")
        try:
            host = urlsplit(self.ollama_host)
            port = host.port
        except ValueError as exc:
            raise ValueError("Địa chỉ Ollama không hợp lệ.") from exc
        if (host.scheme != "http" or host.hostname not in {"localhost", "127.0.0.1", "::1"}
                or host.username is not None or host.password is not None or host.path not in {"", "/"}
                or host.query or host.fragment or port == 0 or not resolved_model):
            raise ValueError("Dùng địa chỉ Ollama trên máy này và chỉ định tên mô hình.")

    @classmethod
    def from_env(cls):
        try:
            configured_model = (
                os.getenv("VICARE_LLM_MODEL")
                or os.getenv("VICARE_OLLAMA_MODEL")
                or DEFAULT_LLM_MODEL
            )
            index_dir = PROJECT_ROOT / os.getenv("VICARE_INDEX_DIR", "chroma_db")
            return cls(
                index_dir=index_dir,
                threshold=float(os.getenv("VICARE_MIN_DENSE_SCORE", str(DEFAULT_MIN_DENSE_SCORE))),
                candidate_pool=int(os.getenv("VICARE_CANDIDATE_POOL", "80")),
                hybrid_dense_depth=int(os.getenv("VICARE_HYBRID_DENSE_DEPTH", "200")),
                rrf_k=int(os.getenv("VICARE_RRF_K", "40")),
                dense_weight=float(os.getenv("VICARE_DENSE_WEIGHT", "0.5")),
                sparse_weight=float(os.getenv("VICARE_SPARSE_WEIGHT", "0.5")),
                top_k=int(os.getenv("VICARE_TOP_K", "5")),
                max_question_chars=int(os.getenv("VICARE_MAX_QUESTION_CHARS", "1000")),
                ollama_host=os.getenv("VICARE_OLLAMA_HOST", "http://127.0.0.1:11434"),
                ollama_model=configured_model,
                timeout=float(os.getenv("VICARE_OLLAMA_TIMEOUT", "45")),
                max_output_tokens=int(os.getenv("VICARE_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
            )
        except ValueError as exc:
            raise ValueError("Cấu hình ViCare không hợp lệ. Kiểm tra biến VICARE_* theo .env.example.") from exc
