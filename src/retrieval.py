"""Vector search over products and vendor catalogues (FAISS if installed, NumPy otherwise)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import BaseEncoder, encode_with_cache


class VectorIndex:
    """Inner-product index over L2-normalised vectors (inner product = cosine similarity)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.backend = "numpy"
        self._faiss = None
        try:
            import faiss  # type: ignore
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss, self.backend = index, "faiss"
        except Exception:  # noqa: BLE001
            pass

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        k = min(k, len(self.vectors))
        if self._faiss is not None:
            D, I = self._faiss.search(q, k)
            keep = I[0] >= 0
            return D[0][keep], I[0][keep]
        sims = self.vectors @ q[0]
        idx = np.argsort(-sims)[:k]
        return sims[idx], idx

    def similarities(self, query: np.ndarray) -> np.ndarray:
        return self.vectors @ query.reshape(-1).astype("float32")


@dataclass
class MarketplaceIndex:
    encoder: BaseEncoder
    products: pd.DataFrame
    vendor_ids: list[str]
    product_index: VectorIndex
    vendor_index: VectorIndex

    @classmethod
    def build(cls, products: pd.DataFrame, profile: pd.DataFrame, encoder: BaseEncoder, artifact_dir: Path):
        products = products.reset_index(drop=True)
        p_texts = (products["product_name"] + ". " + products["subcategory"] + ". " + products["category"] + ". "
                   + products["description"]).tolist()
        v_texts = profile["catalogue_description"].fillna("").tolist()
        encoder.fit(p_texts + v_texts)
        P = encode_with_cache(encoder, p_texts, artifact_dir, "products")
        V = encode_with_cache(encoder, v_texts, artifact_dir, "vendors")
        return cls(encoder, products, profile["vendor_id"].tolist(), VectorIndex(P), VectorIndex(V))

    @property
    def backend_label(self) -> str:
        return f"{self.encoder.name} + {self.product_index.backend}"

    def embed_query(self, text: str) -> np.ndarray:
        return self.encoder.encode([text])[0]

    def search_products(self, query: str, k: int = 50) -> pd.DataFrame:
        sims, idx = self.product_index.search(self.embed_query(query), k)
        res = self.products.iloc[idx].copy()
        res["similarity"] = sims.astype(float)
        return res.reset_index(drop=True)

    def vendor_similarity(self, query: str) -> pd.Series:
        return pd.Series(self.vendor_index.similarities(self.embed_query(query)), index=self.vendor_ids, dtype=float)

    def vendor_vector(self, vendor_id: str) -> np.ndarray:
        return self.vendor_index.vectors[self.vendor_ids.index(vendor_id)]

    def strength_label(self, similarity: float) -> str:
        if similarity >= self.encoder.strong_threshold:
            return "Strong textual match"
        if similarity >= self.encoder.weak_threshold:
            return "Moderate textual match"
        return "Weak match (verify)"
