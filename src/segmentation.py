"""K-Means vendor segmentation on latent factor scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from .config import FACTOR_KEYS, SEED

POSITIVE = {"operational_reliability": "Strong operational reliability",
            "commercial_potential": "High commercial traction",
            "catalogue_market_fit": "Strong catalogue-market fit",
            "customer_product_fit": "Strong customer fit"}
NEGATIVE = {"operational_reliability": "Operational-risk signals",
            "commercial_potential": "Limited commercial traction",
            "catalogue_market_fit": "Catalogue-demand misalignment",
            "customer_product_fit": "Weaker customer fit"}


@dataclass
class SegmentationResults:
    k_grid: pd.DataFrame
    chosen_k: int
    elbow_k: int
    selection_note: str
    labels: pd.Series
    centroids: pd.DataFrame
    descriptions: pd.DataFrame
    silhouette: float
    embedding_2d: pd.DataFrame


def _kneedle(k: np.ndarray, inertia: np.ndarray) -> int:
    x = (k - k.min()) / (k.max() - k.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    return int(k[np.argmax(1 - x - y)])


def describe_centroid(centroid: pd.Series, threshold: float = 0.35) -> str:
    """Neutral analytical description from a cluster's mean factor z-scores."""
    strong = centroid[centroid >= threshold].sort_values(ascending=False)
    weak = centroid[centroid <= -threshold].sort_values()
    parts = [POSITIVE[k] for k in strong.index[:2] if k in POSITIVE] + [NEGATIVE[k] for k in weak.index[:2] if k in NEGATIVE]
    return " + ".join(parts[:3]) if parts else "Near-average profile across all factors"


def run_segmentation(scores: pd.DataFrame, k_range: range = range(2, 9), seed: int = SEED,
                     tolerance: float = 0.02) -> SegmentationResults:
    feats = [k for k in FACTOR_KEYS if k in scores.columns]
    X = scores[feats].to_numpy()
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X, km.labels_))})
    grid = pd.DataFrame(rows)
    elbow_k = _kneedle(grid["k"].to_numpy(dtype=float), grid["inertia"].to_numpy())
    best = grid.loc[grid["silhouette"].idxmax()]
    chosen = int(best["k"])
    note = f"k={chosen} has the highest silhouette ({best['silhouette']:.3f}); the elbow of the inertia curve is at k={elbow_k}."
    if chosen == 2:
        near = grid[(grid["k"] >= 3) & (grid["silhouette"] >= best["silhouette"] - tolerance)]
        if not near.empty:
            chosen = int(near["k"].min())
            note = (f"k=2 has the highest silhouette ({best['silhouette']:.3f}), but k={chosen} is within {tolerance} "
                    f"and gives an operationally useful split, so k={chosen} is used. Elbow: k={elbow_k}.")
    km = KMeans(n_clusters=chosen, n_init=20, random_state=seed).fit(X)
    centroids = pd.DataFrame(km.cluster_centers_, columns=feats)
    anchor = "commercial_potential" if "commercial_potential" in feats else feats[0]
    order = centroids.sort_values(anchor, ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series([remap[l] for l in km.labels_], index=scores.index, name="cluster")
    centroids = scores[feats].groupby(labels).mean()
    sil_samples = silhouette_samples(X, labels.to_numpy())
    desc = pd.DataFrame({
        "cluster": centroids.index,
        "label": [f"Cluster {c}" for c in centroids.index],
        "description": [describe_centroid(centroids.loc[c]) for c in centroids.index],
        "size": labels.value_counts().reindex(centroids.index).to_numpy(),
        "mean_silhouette": [float(sil_samples[labels.to_numpy() == c].mean()) for c in centroids.index],
    })
    emb = PCA(n_components=2, random_state=seed).fit_transform(X)
    embedding = pd.DataFrame({"vendor_id": scores.index, "component_1": emb[:, 0], "component_2": emb[:, 1]})
    return SegmentationResults(k_grid=grid, chosen_k=chosen, elbow_k=elbow_k, selection_note=note, labels=labels,
                               centroids=centroids, descriptions=desc,
                               silhouette=float(silhouette_score(X, labels.to_numpy())), embedding_2d=embedding)
