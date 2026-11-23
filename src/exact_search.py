"""Exact cosine over normalized E5 vectors from the persistent Chroma collection."""
import numpy as np

from .retrieval import Hit


class ExactCosine:
    """Rank existing Chroma vectors by cosine, resolving ties by lexical chunk ID."""

    def __init__(self, collection, chunks):
        by_id = {c.chunk_id: c for c in chunks}
        ids = []
        parts = []
        for offset in range(0, len(by_id), 512):
            block = collection.get(limit=512, offset=offset, include=["embeddings"])
            values = np.asarray(block["embeddings"], dtype=np.float32)
            if values.ndim != 2 or len(values) != len(block["ids"]):
                raise ValueError("Unaligned Chroma IDs and vectors")
            ids.extend(block["ids"])
            parts.append(values)
        if len(ids) != len(by_id) or set(ids) != by_id.keys():
            raise ValueError("Vector identity mismatch")
        order = np.argsort(ids, kind="stable")
        self.chunks = [by_id[ids[i]] for i in order]
        self.matrix = np.concatenate(parts)[order]
        norms = np.linalg.norm(self.matrix, axis=1)
        self.max_norm_error = float(np.max(abs(norms - 1)))
        if self.max_norm_error > 1e-4 or not np.isfinite(norms).all():
            raise ValueError("Chroma did not contain normalized finite E5 vectors")
        self.matrix /= norms[:, None]

    def search(self, vector, depth):
        q = np.asarray(vector[0], dtype=np.float32)
        norm = np.linalg.norm(q)
        if q.shape != (self.matrix.shape[1],) or not np.isfinite(norm) or abs(norm - 1) > 1e-4:
            raise ValueError("Invalid query vector")
        scores = self.matrix @ (q / norm)
        order = np.argsort(-scores, kind="stable")[:depth]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]
