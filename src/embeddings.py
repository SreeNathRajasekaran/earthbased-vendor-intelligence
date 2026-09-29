"""Text encoders: Sentence Transformers with a TF-IDF fallback, plus an on-disk embedding cache."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class BaseEncoder:
    name: str = "base"
    fallback_reason: str | None = None
    strong_threshold: float = 0.5
    weak_threshold: float = 0.3

    def fit(self, corpus: list[str]) -> "BaseEncoder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEncoder(BaseEncoder):
    strong_threshold, weak_threshold = 0.50, 0.30

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional heavy import
        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")


class TfidfEncoder(BaseEncoder):
    """Lexical fallback: cosine similarity over TF-IDF vectors (no network needed)."""
    name = "tfidf-fallback"
    strong_threshold, weak_threshold = 0.30, 0.12

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", max_features=6000)
        self.fitted = False

    def fit(self, corpus: list[str]) -> "TfidfEncoder":
        self.vectorizer.fit(list(corpus))
        self.fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfEncoder must be fitted before encoding.")
        return normalize(self.vectorizer.transform(list(texts))).toarray().astype("float32")


def get_encoder(backend: str | None = None, model_name: str | None = None) -> BaseEncoder:
    """Return a Sentence Transformers encoder, or the TF-IDF fallback if unavailable."""
    backend = (backend or os.getenv("EB_EMBEDDING_BACKEND", "auto")).lower()
    model_name = model_name or os.getenv("EB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    if backend in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 - any import/download failure triggers fallback
            logger.warning("Sentence Transformers unavailable (%s); using TF-IDF fallback.", exc)
            enc = TfidfEncoder()
            enc.fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
            return enc
    return TfidfEncoder()


def encode_with_cache(encoder: BaseEncoder, texts: list[str], cache_dir: Path, tag: str) -> np.ndarray:
    """Encode texts, caching neural embeddings on disk keyed by model + content hash."""
    if isinstance(encoder, TfidfEncoder):
        return encoder.encode(texts)
    digest = hashlib.sha256((encoder.name + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{tag}_{digest}.npy"
    if path.exists():
        return np.load(path)
    emb = encoder.encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
