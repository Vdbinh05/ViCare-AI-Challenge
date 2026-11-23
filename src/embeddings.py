"""SentenceTransformer adapter with local-first HF cache loading and E5 conventions."""
import os
from pathlib import Path

from .config import DEFAULT_HF_CACHE, EMBEDDING_MODEL, EMBEDDING_REVISION


def _candidate_hub_caches():
    candidates = []
    hub_env = os.getenv("HF_HUB_CACHE")
    if hub_env:
        candidates.append(Path(hub_env))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    project_hub = DEFAULT_HF_CACHE / "hub"
    if project_hub.is_dir():
        candidates.append(project_hub)

    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _has_local_snapshot(hub_cache, model_id, revision):
    snapshot_dir = (
        hub_cache
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    return snapshot_dir.is_dir() and (snapshot_dir / "config.json").is_file()


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer

        threads = int(os.getenv("VICARE_TORCH_THREADS", "8"))
        if threads < 1:
            raise ValueError("VICARE_TORCH_THREADS phải là số nguyên dương.")
        torch.set_num_threads(threads)

        model_kwargs = {"use_safetensors": True}
        candidates = _candidate_hub_caches()

        # 1. Local-first: load pinned snapshot directly from a candidate cache without network calls.
        for cache_folder in candidates:
            if _has_local_snapshot(cache_folder, EMBEDDING_MODEL, EMBEDDING_REVISION):
                self.model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    revision=EMBEDDING_REVISION,
                    cache_folder=str(cache_folder),
                    local_files_only=True,
                    token=False,
                    model_kwargs=model_kwargs,
                )
                return

        # 2. Try default Hugging Face cache offline before hitting the network.
        try:
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                revision=EMBEDDING_REVISION,
                local_files_only=True,
                token=False,
                model_kwargs=model_kwargs,
            )
            return
        except (OSError, ValueError):
            pass

        # 3. Network fallback for public model: avoid sending expired global OAuth tokens
        # unless HF_TOKEN is explicitly provided in the environment.
        preferred_cache = str(candidates[0]) if candidates else None
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            cache_folder=preferred_cache,
            token=os.getenv("HF_TOKEN") or False,
            model_kwargs=model_kwargs,
        )

    def passages(self, texts):
        if not texts:
            return []
        return self.model.encode(
            ["passage: " + text for text in texts], batch_size=64,
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

    def query(self, question):
        return self.model.encode(
            ["query: " + question], normalize_embeddings=True, show_progress_bar=False,
        ).tolist()
