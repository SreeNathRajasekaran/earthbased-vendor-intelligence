"""Evaluation: factor adequacy & recovery, clustering, retrieval, predictive validity and RAG grounding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, make_scorer, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FACTOR_KEYS, FACTOR_LABELS, SEED, WINDOW_START
from .feature_engineering import FEATURE_CONSTRUCTS, build_vendor_metrics
from .latent_analysis import run_latent_analysis
from .rag import VendorCopilot
from .recommendation import semantic_match

TRUTH_COLUMNS = {"operational_reliability": "true_reliability", "commercial_potential": "true_commercial",
                 "catalogue_market_fit": "true_catalogue_fit", "customer_product_fit": "true_customer_fit"}

# Ground-truth proxy: a result is relevant if its subcategory matches. Descriptions share vocabulary with
# these queries, so absolute scores are optimistic; they are most useful for comparing encoders.
EVAL_QUERIES = [
    ("sustainable food packaging for bulk restaurant takeaway orders", ["Food Containers"]),
    ("compostable spoons and forks for catering events", ["Compostable Cutlery"]),
    ("paper cups for a cafe", ["Cups & Lids"]),
    ("mailer boxes for shipping ecommerce orders", ["Shipping & Mailers"]),
    ("plastic free shampoo bar", ["Hair Care"]),
    ("bamboo toothbrush", ["Oral Care"]),
    ("organic millet flour and pulses in bulk", ["Grains & Pulses"]),
    ("healthy snacks for office pantry", ["Snacks"]),
    ("organic cotton t-shirts", ["Clothing"]),
    ("jute tote bags for corporate merchandise", ["Bags"]),
    ("terracotta planters for balcony garden", ["Planters"]),
    ("cloth diapers for newborns", ["Baby Care"]),
    ("wooden toys for toddlers", ["Toys"]),
    ("recycled paper notebooks for corporate gifting", ["Notebooks"]),
    ("beeswax wraps and glass jars for pantry storage", ["Storage"]),
    ("natural dishwash liquid and cleaning scrubs", ["Cleaning Supplies"]),
]

GROUNDING_TESTS = [
    {"question": "Which vendors should we investigate for sustainable packaging?", "intent": "prioritisation", "sufficient": True, "category": "Sustainable Packaging"},
    {"question": "Why was Vendor V023 classified as commercially promising?", "intent": "vendor_explanation", "sufficient": True},
    {"question": "Which vendors have high catalogue fit but weak operational reliability?", "intent": "factor_contrast", "sufficient": True},
    {"question": "What categories have strong demand but limited vendor coverage?", "intent": "category_gaps", "sufficient": True},
    {"question": "Find vendors suitable for a premium customer segment.", "intent": "premium_segment", "sufficient": True},
    {"question": "Which vendors should we prioritise?", "intent": "prioritisation", "sufficient": True},
    {"question": "Why is V012 strategically important?", "intent": "vendor_explanation", "sufficient": True},
    {"question": "Which vendors have strong demand but operational issues?", "intent": "factor_contrast", "sufficient": True},
    {"question": "Find vendors similar to V008.", "intent": "similar_vendors", "sufficient": True},
    {"question": "Which categories need more vendor coverage?", "intent": "category_gaps", "sufficient": True},
    {"question": "Recommend vendors for bamboo toothbrushes", "intent": "semantic_search", "sufficient": True, "category": "Personal Care"},
    {"question": "Which vendors have excellent ratings and strong reliability?", "intent": "factor_contrast", "sufficient": True},
    {"question": "What is the GMV of vendor V999?", "intent": "vendor_explanation", "sufficient": False},
    {"question": "What was EarthBased's actual revenue last year?", "intent": "out_of_scope", "sufficient": False},
    {"question": "Which vendor is guaranteed to succeed on the platform?", "intent": "out_of_scope", "sufficient": False},
    {"question": "Find vendors selling quantum computing hardware", "intent": "semantic_search", "sufficient": False},
]


def factor_summary(system) -> dict:
    L = system.latent
    return {"kmo_overall": L.kmo_overall, "bartlett": L.bartlett, "n_factors": L.n_factors,
            "n_factors_parallel": L.n_factors_parallel, "n_factors_kaiser": L.n_factors_kaiser,
            "retention_note": L.retention_note,
            "variance_explained": float(L.variance_table["proportion_of_variance"].sum()),
            "strategic_pc1_variance": L.strategic_explained_variance,
            "kmo_per_item": L.kmo_per_item.round(3).to_dict()}


def factor_recovery(system) -> pd.DataFrame:
    """Correlation of estimated factor scores with the generator's hidden traits (synthetic-data check only)."""
    gt = system.ground_truth
    if gt is None:
        return pd.DataFrame()
    gt = gt.set_index("vendor_id").loc[system.latent.scores.index]
    rows = {}
    for k in [k for k in FACTOR_KEYS if k in system.latent.scores]:
        rows[FACTOR_LABELS[k]] = {c: float(np.corrcoef(system.latent.scores[k], gt[c])[0, 1]) for c in TRUTH_COLUMNS.values()}
    return pd.DataFrame(rows).T


def clustering_evaluation(system) -> dict:
    S = system.segmentation
    out = {"chosen_k": S.chosen_k, "elbow_k": S.elbow_k, "silhouette": S.silhouette,
           "inertia": float(S.k_grid.set_index("k").at[S.chosen_k, "inertia"]),
           "sizes": S.descriptions[["label", "size", "mean_silhouette"]].to_dict(orient="records")}
    if system.ground_truth is not None:
        gt = system.ground_truth.set_index("vendor_id").loc[S.labels.index, "archetype"]
        out["ari_vs_hidden_archetype"] = float(adjusted_rand_score(gt, S.labels))
        out["crosstab"] = pd.crosstab(S.labels.rename("cluster"), gt.rename("hidden archetype"))
    return out


def retrieval_evaluation(system, k: int = 10):
    products = system.tables["products"]
    rows, dist = [], []
    for query, subs in EVAL_QUERIES:
        res = system.index.search_products(query, k=50)
        rel = res["subcategory"].isin(subs).to_numpy()
        top = rel[:k].astype(float)
        n_rel = int(products["subcategory"].isin(subs).sum())
        hits = np.flatnonzero(top)
        ideal = float(np.sum(1 / np.log2(np.arange(2, min(k, n_rel) + 2)))) if n_rel else 0.0
        dcg = float(np.sum(top / np.log2(np.arange(2, len(top) + 2))))
        rel_vendors = set(products.loc[products["subcategory"].isin(subs), "vendor_id"])
        retrieved = semantic_match(system, query, top_k=10)["vendor_id"].tolist()
        rows.append({"query": query, "relevant_subcategory": ", ".join(subs),
                     "product_P@5": float(top[:5].mean()), "product_P@10": float(top.mean()),
                     "MRR": 1.0 / (hits[0] + 1) if len(hits) else 0.0, "nDCG@10": dcg / ideal if ideal else 0.0,
                     "vendor_P@5": float(np.mean([v in rel_vendors for v in retrieved[:5]])) if retrieved else 0.0,
                     "vendor_R@10": len(set(retrieved[:10]) & rel_vendors) / len(rel_vendors) if rel_vendors else np.nan,
                     "relevant_vendors": len(rel_vendors)})
        dist += [{"query": query, "similarity": float(s), "relevant": bool(r)} for s, r in zip(res["similarity"], rel)]
    per = pd.DataFrame(rows)
    return per, per.mean(numeric_only=True).drop("relevant_vendors").to_dict(), pd.DataFrame(dist)


def predictive_validity(system, seed: int = SEED, positive_share: float = 0.3):
    """Do H1 latent factors predict top-30% H2 GMV better than surface metrics? (cross-validated)"""
    t = system.tables
    vendors, products, customers = t["vendors"], t["products"], t["customers"]
    orders = t["orders"].copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    eligible = pd.Index(vendors.loc[pd.to_datetime(vendors["onboarding_date"]) < pd.Timestamp(WINDOW_START), "vendor_id"])
    m1 = build_vendor_metrics(vendors[vendors["vendor_id"].isin(eligible)], products[products["vendor_id"].isin(eligible)],
                              orders[orders["vendor_id"].isin(eligible)], customers, WINDOW_START, "2024-06-30")
    lat1 = run_latent_analysis(m1, n_factors=len(FEATURE_CONSTRUCTS), seed=seed, n_parallel_iter=50)
    h2 = orders[(orders["order_date"] >= "2024-07-01") & ~orders["cancelled"]]
    h2_gmv = h2.groupby("vendor_id")["order_value"].sum().reindex(eligible, fill_value=0.0)
    y = (h2_gmv >= h2_gmv.quantile(1 - positive_share)).astype(int)
    m1i = m1.set_index("vendor_id").loc[eligible]
    naive = pd.DataFrame({"avg_rating": m1i["avg_rating"],
                          "price_index": vendors.set_index("vendor_id")["price_index"].loc[eligible],
                          "log_product_count": m1i["log_product_count"]})
    latent_x = lat1.scores.loc[eligible, [k for k in FACTOR_KEYS if k in lat1.scores]]
    sets = {"Surface metrics (rating, price, product count)": naive,
            "Surface metrics + H1 GMV": naive.assign(log_gmv_h1=np.log1p(m1i["gmv_window"])),
            "Latent factor scores (H1)": latent_x,
            "Latent factors + surface metrics": pd.concat([latent_x, naive], axis=1)}
    scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
               "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    rows = []
    for name, X in sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        res = cross_validate(model, X.to_numpy(), y.to_numpy(), cv=cv, scoring=scoring)
        row = {"feature_set": name}
        for m in scoring:
            row[m] = float(res[f"test_{m}"].mean())
            row[f"{m}_sd"] = float(res[f"test_{m}"].std())
        rows.append(row)
    meta = {"n_vendors": int(len(eligible)), "n_positive": int(y.sum()),
            "design": "Factors fitted on H1 (Jan-Jun) data of vendors live before 2024; target = top 30% of H2 GMV. "
                      "The factor model is unsupervised (no labels); the classifier is 5x5 repeated stratified CV."}
    return pd.DataFrame(rows), meta


def run_grounding_suite(system, use_llm: bool = False):
    cp = VendorCopilot(system, use_llm=use_llm)
    P = system.profile.set_index("vendor_id")
    rows = []
    for t in GROUNDING_TESTS:
        r = cp.ask(t["question"])
        cat_prec = np.nan
        if t.get("category") and not r.evidence.empty and "vendor_id" in r.evidence.columns:
            cat_prec = float((P.loc[r.evidence["vendor_id"].unique(), "category"] == t["category"]).mean())
        rows.append({"question": t["question"], "expected_intent": t["intent"], "intent": r.intent,
                     "intent_ok": r.intent == t["intent"], "expected_sufficient": t["sufficient"],
                     "sufficient": r.sufficient, "sufficiency_ok": r.sufficient == t["sufficient"],
                     "grounded": r.grounded, "issues": "; ".join(r.grounding_issues), "evidence_rows": len(r.evidence),
                     "category_precision": cat_prec, "mode": r.mode, "latency_ms": r.latency_ms,
                     "tokens": r.input_tokens + r.output_tokens, "cost_usd": r.estimated_cost_usd})
    df = pd.DataFrame(rows)
    summary = {"questions": len(df), "intent_accuracy": float(df["intent_ok"].mean()),
               "sufficiency_accuracy": float(df["sufficiency_ok"].mean()), "grounded_rate": float(df["grounded"].mean()),
               "error_cases": int((~df["grounded"] | ~df["sufficiency_ok"]).sum()),
               "mean_category_precision": float(df["category_precision"].mean()) if df["category_precision"].notna().any() else None,
               "mean_latency_ms": float(df["latency_ms"].mean()), "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
               "total_tokens": int(df["tokens"].sum()), "estimated_cost_usd": float(df["cost_usd"].sum())}
    return df, summary


def run_full_evaluation(system, include_prediction: bool = True) -> dict:
    per_q, r_summary, dist = retrieval_evaluation(system)
    rag_df, rag_summary = run_grounding_suite(system, use_llm=VendorCopilot(system).use_llm)
    out = {"factor": factor_summary(system), "recovery": factor_recovery(system),
           "clustering": clustering_evaluation(system), "retrieval_per_query": per_q, "retrieval_summary": r_summary,
           "similarity_distribution": dist, "rag": rag_df, "rag_summary": rag_summary,
           "embedding_backend": system.index.backend_label}
    if include_prediction:
        out["prediction"], out["prediction_meta"] = predictive_validity(system)
    return out


def _json_default(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_evaluation(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, default=_json_default, indent=2), encoding="utf-8")
