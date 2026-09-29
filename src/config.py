"""Central configuration: paths, seeds and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("EB_DATA_DIR", str(ROOT_DIR / "data")))
ARTIFACT_DIR = Path(os.getenv("EB_ARTIFACT_DIR", str(ROOT_DIR / "artifacts")))
GROUND_TRUTH_DIR_NAME = "_ground_truth"

SEED = int(os.getenv("EB_SEED", "42"))

WINDOW_START = "2024-01-01"
WINDOW_END = "2024-12-31"
SLA_DAYS = 5  # promised delivery window used for on-time fulfilment

FACTOR_KEYS = [
    "operational_reliability",
    "commercial_potential",
    "catalogue_market_fit",
    "customer_product_fit",
]
STRATEGIC_KEY = "strategic_value"
ALL_SCORE_KEYS = FACTOR_KEYS + [STRATEGIC_KEY]

FACTOR_LABELS = {
    "operational_reliability": "Operational Reliability",
    "commercial_potential": "Commercial Potential",
    "catalogue_market_fit": "Catalogue-Market Fit",
    "customer_product_fit": "Customer/Product Fit",
    "strategic_value": "Marketplace Strategic Value",
}
FACTOR_SHORT = {
    "operational_reliability": "Reliability",
    "commercial_potential": "Commercial",
    "catalogue_market_fit": "Catalogue fit",
    "customer_product_fit": "Customer fit",
    "strategic_value": "Strategic",
}

RAW_TABLES = ["vendors", "products", "orders", "customers", "vendor_metrics"]
