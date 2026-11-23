"""Two fixed alternatives to the locked document ranking used by the retrieval study."""
from collections import defaultdict

from src.retrieval import Hit

RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class PassageReranker:
    """Score every passage of the first 50 documents; preserve the winning passage as evidence."""
    def __init__(self, chunks):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.embeddings import _candidate_hub_caches

        self.torch = torch
        local = next((cache / f"models--{RERANKER.replace('/', '--')}" / "snapshots" / RERANKER_REVISION
                      for cache in _candidate_hub_caches()
                      if (cache / f"models--{RERANKER.replace('/', '--')}" / "snapshots" /
                          RERANKER_REVISION / "model.safetensors").is_file()), None)
        if local is None:
            # This optional experiment only runs with a complete local checkpoint.
            raise FileNotFoundError(f"Missing cached checkpoint {RERANKER}@{RERANKER_REVISION}")
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
        ).to("cpu").eval()
        self.by_doc = defaultdict(list)
        for chunk in chunks:
            self.by_doc[chunk.doc_id].append(chunk)

    def rank(self, question, candidate_ids):
        pairs = [chunk for doc in candidate_ids for chunk in self.by_doc[doc]]
        if len(set(candidate_ids)) != len(candidate_ids) or not all(self.by_doc[d] for d in candidate_ids):
            raise ValueError("Candidate mapping is incomplete")
        pairs.sort(key=lambda c: (len(c.text), c.chunk_id))
        best = {}
        with self.torch.inference_mode():
            for offset in range(0, len(pairs), 8):
                batch = pairs[offset:offset + 8]
                inputs = self.tokenizer([question] * len(batch), [c.text for c in batch],
                                        padding=True, truncation=True, max_length=384, return_tensors="pt")
                logits = self.model(**inputs).logits
                if logits.shape != (len(batch), 1) or not self.torch.isfinite(logits).all():
                    raise ValueError("Nonfinite or unexpected cross-encoder logits")
                for c, value in zip(batch, logits[:, 0].tolist()):
                    current = best.get(c.doc_id)
                    if current is None or (-value, c.chunk_id) < (-current.score, current.chunk.chunk_id):
                        best[c.doc_id] = Hit(c, value)
        ce_order = sorted(best, key=lambda doc: (-best[doc].score, doc))
        ce_ranks = {doc: rank for rank, doc in enumerate(ce_order, 1)}
        scores = {d: .3 / (20 + i) + .7 / (20 + ce_ranks[d]) for i, d in enumerate(candidate_ids, 1)}
        return [Hit(best[d].chunk, scores[d]) for d in sorted(scores, key=lambda d: (-scores[d], d))]
