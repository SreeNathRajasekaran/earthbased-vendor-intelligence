"""Vendor prioritisation and semantic product/vendor matching (transparent components, no black-box score)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FACTOR_KEYS, FACTOR_LABELS
from .feature_engineering import DISPLAY_SOURCE, FEATURE_LABELS, format_value
from .latent_analysis import factor_drivers

PCT = {k: f"{k}_pct" for k in FACTOR_KEYS}


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def prioritise_vendors(profile: pd.DataFrame, weights: dict[str, float] | None = None, category: str | list[str] | None = None,
                       city: str | list[str] | None = None, min_percentiles: dict[str, float] | None = None,
                       require_bulk: bool = False, top_n: int = 10) -> tuple[pd.DataFrame, dict]:
    """Rank vendors by a user-weighted mean of factor percentiles; every component is returned."""
    df = profile.copy()
    filters = []
    if category:
        cats = [category] if isinstance(category, str) else list(category)
        df = df[df["category"].isin(cats)]
        filters.append(f"category in {cats}")
    if city:
        cities = [city] if isinstance(city, str) else list(city)
        df = df[df["city"].isin(cities)]
        filters.append(f"city in {cities}")
    for key, floor in (min_percentiles or {}).items():
        df = df[df[PCT[key]] >= floor]
        filters.append(f"{FACTOR_LABELS[key]} ≥ {ordinal(floor)} percentile")
    if require_bulk:
        df = df[df["bulk_order_capable"]]
        filters.append("bulk-capable")
    w = {k: float((weights or {}).get(k, 1.0)) for k in FACTOR_KEYS}
    total = sum(v for v in w.values() if v > 0) or 1.0
    w = {k: max(v, 0.0) / total for k, v in w.items()}
    for k in FACTOR_KEYS:
        df[f"{k}_contribution"] = df[PCT[k]] * w[k]
    df["priority_index"] = df[[f"{k}_contribution" for k in FACTOR_KEYS]].sum(axis=1)
    df = df.sort_values("priority_index", ascending=False)
    df["rationale"] = df.apply(_priority_rationale, axis=1)
    spec = {"weights": w, "filters": filters, "n_candidates": int(len(df))}
    return df.head(top_n).reset_index(drop=True), spec


def _priority_rationale(row: pd.Series) -> str:
    pcts = {k: row[PCT[k]] for k in FACTOR_KEYS}
    ranked = sorted(pcts, key=pcts.get, reverse=True)
    s1, s2, weak = ranked[0], ranked[1], ranked[-1]
    text = (f"Strongest on {FACTOR_LABELS[s1]} ({ordinal(pcts[s1])} pct) and {FACTOR_LABELS[s2]} "
            f"({ordinal(pcts[s2])}); weakest on {FACTOR_LABELS[weak]} ({ordinal(pcts[weak])}).")
    if pcts["operational_reliability"] < 40:
        text += " Review fulfilment history before scaling volume."
    return text


def explain_vendor_factor(system, vendor_id: str, factor_key: str, top_n: int = 4) -> pd.DataFrame:
    """Drivers of one vendor's factor score with values shown on their original scale."""
    drivers = factor_drivers(system.latent, vendor_id, factor_key, top_n)
    m = system.metrics.set_index("vendor_id")
    rows = []
    for r in drivers.itertuples(index=False):
        col = DISPLAY_SOURCE.get(r.feature, r.feature)
        if col in m.columns:
            vendor_val, median_val = format_value(col, m.at[vendor_id, col]), format_value(col, m[col].median())
            label = FEATURE_LABELS.get(r.feature, r.feature)
        else:  # strategic-value inputs
            vendor_val, median_val = f"{r.vendor_z:+.2f} z", "+0.00 z"
            label = FACTOR_LABELS.get(r.feature, r.feature.replace("_", " ").capitalize())
        rows.append({"indicator": label, "vendor_value": vendor_val, "platform_median": median_val,
                     "weight_or_loading": f"{r.weight:+.2f}", "contribution": round(float(r.contribution), 2),
                     "effect": "raises score" if r.contribution > 0 else "lowers score"})
    return pd.DataFrame(rows)


def category_coverage_table(profile: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame):
    """Demand share vs vendor share per category and subcategory."""
    o = orders.merge(products[["product_id", "category", "subcategory"]], on="product_id", how="left")
    t = pd.DataFrame({"orders": o.groupby("category").size(), "vendors": profile.groupby("category").size(),
                      "products": products.groupby("category").size()}).fillna(0)
    t["order_share"] = t["orders"] / t["orders"].sum()
    t["vendor_share"] = t["vendors"] / t["vendors"].sum()
    t["demand_to_supply_ratio"] = t["order_share"] / t["vendor_share"]
    t["orders_per_vendor"] = t["orders"] / t["vendors"]
    t["median_reliability_pct"] = profile.groupby("category")[PCT["operational_reliability"]].median()
    t = t.sort_values("demand_to_supply_ratio", ascending=False).reset_index().rename(columns={"index": "category"})
    sub = pd.DataFrame({"orders": o.groupby(["category", "subcategory"]).size(),
                        "vendors_covering": products.groupby(["category", "subcategory"])["vendor_id"].nunique()}).fillna(0)
    sub["orders_per_covering_vendor"] = sub["orders"] / sub["vendors_covering"].replace(0, np.nan)
    sub = sub.sort_values("orders_per_covering_vendor", ascending=False).reset_index()
    return t, sub


MATCH_COLUMNS = ["vendor_id", "vendor_name", "category", "top_product", "semantic_similarity", "catalogue_similarity",
                 "operational_reliability_pct", "commercial_potential_pct", "catalogue_market_fit_pct",
                 "customer_product_fit_pct", "rank_score", "match_strength", "why_it_matches", "caveat"]


def semantic_match(system, query: str, top_k: int = 5, category: str | None = None, quality_blend: float = 0.0,
                   candidate_pool: int = 80) -> pd.DataFrame:
    """Embed the query, retrieve nearest products, aggregate to vendors and attach factor evidence.

    rank_score = (1 - quality_blend) * relative similarity + quality_blend * mean(reliability pct, customer fit pct) / 100.
    With the default blend of 0, ranking is by semantic similarity alone.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty.")
    cands = system.index.search_products(query, k=candidate_pool)
    if category:
        cands = cands[cands["category"] == category]
    cands = cands[cands["similarity"] > 0].sort_values("similarity", ascending=False)
    if cands.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    top_rows = cands.drop_duplicates("vendor_id").set_index("vendor_id")
    res = top_rows[["product_id", "product_name", "similarity"]].rename(
        columns={"product_id": "top_product_id", "product_name": "top_product", "similarity": "semantic_similarity"})
    res["matched_products"] = cands.groupby("vendor_id")["product_name"].apply(lambda s: "; ".join(s.head(3)))
    res["n_matched_products"] = cands.groupby("vendor_id").size()
    res["catalogue_similarity"] = system.index.vendor_similarity(query).reindex(res.index)
    P = system.profile.set_index("vendor_id")
    keep = ["vendor_name", "category", "city", "segment", "segment_description", "bulk_order_capable", "price_index",
            "gmv", "strategic_value_pct"] + list(PCT.values())
    res = res.join(P[keep])
    b = float(np.clip(quality_blend, 0, 1))
    res["quality_component"] = res[[PCT["operational_reliability"], PCT["customer_product_fit"]]].mean(axis=1) / 100
    res["relative_similarity"] = res["semantic_similarity"] / res["semantic_similarity"].max()
    res["rank_score"] = (1 - b) * res["relative_similarity"] + b * res["quality_component"]
    res = res.sort_values("rank_score", ascending=False).head(top_k).reset_index()
    res["match_strength"] = res["semantic_similarity"].map(system.index.strength_label)
    res["why_it_matches"] = res.apply(_why, axis=1)
    res["caveat"] = res.apply(_caveat, axis=1)
    return res


def _why(r: pd.Series) -> str:
    return (f"Closest listing is '{r.top_product}' (cosine similarity {r.semantic_similarity:.2f}); "
            f"{int(r.n_matched_products)} of this vendor's products are among the nearest catalogue matches. "
            f"Operational reliability is at the {ordinal(r.operational_reliability_pct)} percentile, catalogue-market "
            f"fit at the {ordinal(r.catalogue_market_fit_pct)}, commercial potential at the "
            f"{ordinal(r.commercial_potential_pct)}.")


def _caveat(r: pd.Series) -> str:
    notes = []
    if str(r.match_strength).startswith("Weak"):
        notes.append("Weak textual match: confirm the vendor actually supplies this.")
    if r.operational_reliability_pct < 40:
        notes.append("Operational-risk signals: review cancellations and delivery times before shortlisting.")
    notes.append("Similarity reflects catalogue text, not verified capability or capacity.")
    return " ".join(notes)
