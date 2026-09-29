"""End-to-end pipeline: data -> features -> latent factors -> segments -> vector index.

Usage:
    python -m src.pipeline --regenerate --evaluate
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ALL_SCORE_KEYS, ARTIFACT_DIR, DATA_DIR, FACTOR_LABELS, SEED
from .data_generator import generate_marketplace
from .embeddings import get_encoder
from .feature_engineering import build_vendor_metrics
from .latent_analysis import LatentResults, run_latent_analysis
from .preprocessing import clean_tables, load_ground_truth, load_tables, tables_exist, validate_tables
from .recommendation import category_coverage_table
from .retrieval import MarketplaceIndex
from .segmentation import SegmentationResults, run_segmentation

logger = logging.getLogger(__name__)


@dataclass
class MarketplaceSystem:
    tables: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    latent: LatentResults
    segmentation: SegmentationResults
    profile: pd.DataFrame
    index: MarketplaceIndex
    category_table: pd.DataFrame
    subcategory_table: pd.DataFrame
    ground_truth: pd.DataFrame | None
    data_dir: Path
    artifact_dir: Path


def build_vendor_profile(vendors, metrics, latent: LatentResults, seg: SegmentationResults) -> pd.DataFrame:
    base = ["vendor_id", "vendor_name", "category", "city", "onboarding_date", "bulk_order_capable", "certifications",
            "price_index", "product_count", "order_count", "gmv", "average_order_value", "average_rating",
            "catalogue_description"]
    prof = vendors[base].merge(metrics.drop(columns=["category", "product_count"]), on="vendor_id", how="left")
    prof = prof.merge(latent.scores.reset_index(), on="vendor_id", how="left")
    for k in ALL_SCORE_KEYS:
        prof[f"{k}_pct"] = (prof[k].rank(pct=True) * 100).round().astype(int)
    prof["cluster"] = prof["vendor_id"].map(seg.labels).astype(int)
    prof["segment"] = "Cluster " + prof["cluster"].astype(str)
    prof["segment_description"] = prof["cluster"].map(seg.descriptions.set_index("cluster")["description"])
    prof["gmv"] = prof["gmv"].fillna(0.0)
    return prof


def build_system(data_dir: Path | None = None, artifact_dir: Path | None = None, regenerate: bool = False,
                 embedding_backend: str | None = None, seed: int = SEED, n_customers: int = 3000) -> MarketplaceSystem:
    data_dir = Path(data_dir or DATA_DIR)
    artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
    if regenerate or not tables_exist(data_dir):
        logger.info("Generating synthetic marketplace data in %s", data_dir)
        generate_marketplace(seed=seed, n_customers=n_customers, out_dir=data_dir)
    tables = clean_tables(load_tables(data_dir))
    for issue in validate_tables(tables):
        logger.warning("Data validation: %s", issue)
    metrics = build_vendor_metrics(tables["vendors"], tables["products"], tables["orders"], tables["customers"])
    latent = run_latent_analysis(metrics, seed=seed)
    seg = run_segmentation(latent.scores, seed=seed)
    profile = build_vendor_profile(tables["vendors"], metrics, latent, seg)
    index = MarketplaceIndex.build(tables["products"], profile, get_encoder(embedding_backend), artifact_dir)
    cat, sub = category_coverage_table(profile, tables["orders"], tables["products"])
    return MarketplaceSystem(tables, metrics, latent, seg, profile, index, cat, sub, load_ground_truth(data_dir),
                             data_dir, artifact_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EarthBased vendor-intelligence pipeline.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data.")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation suite and save artifacts/evaluation.json.")
    parser.add_argument("--backend", default=None, help="Embedding backend: auto | sentence-transformers | tfidf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = build_system(regenerate=args.regenerate, embedding_backend=args.backend)
    t = s.tables
    print(f"\nData: {len(t['vendors'])} vendors, {len(t['products'])} products, {len(t['orders'])} orders, "
          f"{len(t['customers'])} customers")
    L = s.latent
    print(f"KMO={L.kmo_overall:.3f}  Bartlett p={L.bartlett['p_value']:.2e}  {L.retention_note}")
    for r in L.naming_table.itertuples(index=False):
        print(f"  {r.factor} -> {r.label}: {r.top_loadings}")
    print(f"Segmentation: {s.segmentation.selection_note}")
    for r in s.segmentation.descriptions.itertuples(index=False):
        print(f"  {r.label} (n={r.size}): {r.description}")
    print(f"Embedding backend: {s.index.backend_label}")

    s.artifact_dir.mkdir(parents=True, exist_ok=True)
    L.loadings.round(4).to_csv(s.artifact_dir / "factor_loadings.csv")
    s.profile.to_csv(s.artifact_dir / "vendor_scores.csv", index=False)
    s.segmentation.descriptions.to_csv(s.artifact_dir / "cluster_profiles.csv", index=False)

    if args.evaluate:
        from .evaluation import run_full_evaluation, save_evaluation
        ev = run_full_evaluation(s)
        save_evaluation(ev, s.artifact_dir / "evaluation.json")
        print("\nRetrieval:", {k: round(v, 3) for k, v in ev["retrieval_summary"].items()})
        print("Copilot:", {k: v for k, v in ev["rag_summary"].items()})
        if not ev["recovery"].empty:
            print("Factor recovery (corr with hidden traits):\n", ev["recovery"].round(2))
        print("Predictive validity:\n", ev["prediction"][["feature_set", "roc_auc", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
