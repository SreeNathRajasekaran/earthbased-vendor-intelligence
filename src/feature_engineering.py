"""Vendor-level feature engineering.

Rates are empirical-Bayes smoothed toward the platform rate so that vendors with
few orders are not assigned extreme values from small samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SLA_DAYS, WINDOW_END, WINDOW_START

# Hypothesised measurement model. Used ONLY to *name* factors after unsupervised
# extraction (see latent_analysis.assign_factor_names); it does not constrain the fit.
FEATURE_CONSTRUCTS: dict[str, dict[str, int]] = {
    "operational_reliability": {"fulfilment_rate": 1, "cancellation_rate": -1, "return_rate": -1,
                                "avg_delivery_days": -1, "delivery_variability": -1,
                                "stock_availability": 1, "log_response_time": -1},
    "commercial_potential": {"log_gmv_per_month": 1, "log_orders_per_month": 1, "conversion_rate": 1},
    "catalogue_market_fit": {"demand_alignment": 1, "search_relevance": 1, "category_coverage": 1,
                             "log_product_count": 1},
    "customer_product_fit": {"avg_rating": 1, "repeat_customer_rate": 1, "recommendation_acceptance_rate": 1,
                             "customer_engagement": 1},
}
MODEL_FEATURES: list[str] = [f for block in FEATURE_CONSTRUCTS.values() for f in block]

FEATURE_LABELS = {
    "fulfilment_rate": "On-time fulfilment rate", "cancellation_rate": "Cancellation rate",
    "return_rate": "Return rate", "avg_delivery_days": "Average delivery days",
    "delivery_variability": "Delivery-time variability", "stock_availability": "Stock availability",
    "log_response_time": "Response time", "log_gmv_per_month": "GMV per active month",
    "log_orders_per_month": "Orders per active month", "conversion_rate": "View-to-order conversion",
    "demand_alignment": "Catalogue demand alignment", "search_relevance": "Search relevance",
    "category_coverage": "Subcategory coverage", "log_product_count": "Catalogue size",
    "avg_rating": "Average rating", "repeat_customer_rate": "Repeat-customer rate",
    "recommendation_acceptance_rate": "Recommendation acceptance", "customer_engagement": "Add-to-cart engagement",
}
# Log features are displayed on their original scale
DISPLAY_SOURCE = {"log_gmv_per_month": "gmv_per_month", "log_orders_per_month": "orders_per_month",
                  "log_product_count": "product_count", "log_response_time": "response_time_hours"}
PERCENT_COLUMNS = {"fulfilment_rate", "cancellation_rate", "return_rate", "stock_availability", "conversion_rate",
                   "customer_engagement", "category_coverage", "repeat_customer_rate",
                   "recommendation_acceptance_rate", "premium_customer_share"}


def format_value(column: str, value: float) -> str:
    """Human-readable formatting for a metric value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if column in PERCENT_COLUMNS:
        return f"{value:.1%}"
    if column == "gmv_per_month":
        return f"₹{value:,.0f}"
    if column in ("avg_delivery_days", "delivery_variability"):
        return f"{value:.1f} days"
    if column == "response_time_hours":
        return f"{value:.1f} h"
    if column == "orders_per_month":
        return f"{value:.1f}"
    if column == "product_count":
        return f"{int(value)}"
    return f"{value:.2f}"


def _smooth(successes, trials, prior: float, strength: float):
    return (successes + strength * prior) / (trials + strength)


def compute_vendor_kpis(vendors: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Unsmoothed descriptive vendor KPIs (written to vendors.csv)."""
    ids = pd.Index(vendors["vendor_id"], name="vendor_id")
    o = orders
    n = o.groupby("vendor_id").size().reindex(ids, fill_value=0)
    n_safe = n.replace(0, np.nan)
    cancels = o.groupby("vendor_id")["cancelled"].sum().reindex(ids, fill_value=0)
    d = o.loc[~o["cancelled"].astype(bool)]
    on_time = (d["delivery_days"] <= SLA_DAYS).groupby(d["vendor_id"]).sum().reindex(ids, fill_value=0)
    gd = d.groupby("vendor_id")
    kpi = pd.DataFrame(index=ids)
    kpi["order_count"] = n
    kpi["fulfilment_rate"] = on_time / n_safe
    kpi["cancellation_rate"] = cancels / n_safe
    kpi["return_rate"] = gd["returned"].mean().reindex(ids)
    kpi["avg_delivery_days"] = gd["delivery_days"].mean().reindex(ids)
    kpi["gmv"] = gd["order_value"].sum().reindex(ids, fill_value=0.0)
    kpi["average_order_value"] = gd["order_value"].mean().reindex(ids)
    kpi["average_rating"] = o.groupby("vendor_id")["rating"].mean().reindex(ids)
    pairs = o.groupby(["vendor_id", "customer_id"]).size()
    kpi["repeat_customer_rate"] = (pairs >= 2).groupby(level="vendor_id").mean().reindex(ids)
    p = products.groupby("vendor_id")
    kpi["product_count"] = p.size().reindex(ids, fill_value=0)
    views = p["views"].sum().reindex(ids)
    kpi["conversion_rate"] = p["orders"].sum().reindex(ids) / views
    kpi["customer_engagement"] = p["add_to_cart"].sum().reindex(ids) / views
    subs_per_cat = products.groupby("category")["subcategory"].nunique()
    vcat = vendors.set_index("vendor_id")["category"].reindex(ids)
    kpi["category_coverage"] = p["subcategory"].nunique().reindex(ids) / vcat.map(subs_per_cat)
    return kpi.reset_index()


def build_vendor_metrics(vendors: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame,
                         customers: pd.DataFrame | None = None, window_start: str = WINDOW_START,
                         window_end: str = WINDOW_END) -> pd.DataFrame:
    """Engineered, smoothed, tenure-adjusted model features for one time window."""
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o["order_date"])
    o = o[(o["order_date"] >= ws) & (o["order_date"] <= we)]
    if o.empty:
        raise ValueError("No orders fall inside the requested window.")
    ids = pd.Index(vendors["vendor_id"], name="vendor_id")
    v = vendors.set_index("vendor_id").reindex(ids)
    onboard = pd.to_datetime(v["onboarding_date"])
    active_start = onboard.where(onboard > ws, ws)
    active_days = ((we - active_start).dt.days + 1).clip(lower=15)
    window_fraction = ((we - ws).days + 1) / 366.0

    cancelled = o["cancelled"].astype(bool)
    d = o[~cancelled]
    n = o.groupby("vendor_id").size().reindex(ids, fill_value=0).astype(float)
    n_cancel = o[cancelled].groupby("vendor_id").size().reindex(ids, fill_value=0)
    n_del = d.groupby("vendor_id").size().reindex(ids, fill_value=0)
    n_ontime = d[d["delivery_days"] <= SLA_DAYS].groupby("vendor_id").size().reindex(ids, fill_value=0)
    n_ret = d[d["returned"].astype(bool)].groupby("vendor_id").size().reindex(ids, fill_value=0)
    sum_days = d.groupby("vendor_id")["delivery_days"].sum().reindex(ids, fill_value=0.0)
    std_days = d.groupby("vendor_id")["delivery_days"].std(ddof=0).reindex(ids)
    rated = o.dropna(subset=["rating"])
    n_rated = rated.groupby("vendor_id").size().reindex(ids, fill_value=0)
    sum_rating = rated.groupby("vendor_id")["rating"].sum().reindex(ids, fill_value=0.0)
    gmv = d.groupby("vendor_id")["order_value"].sum().reindex(ids, fill_value=0.0)
    pairs = o.groupby(["vendor_id", "customer_id"]).size()
    n_cust = pairs.groupby(level="vendor_id").size().reindex(ids, fill_value=0)
    n_rep = (pairs >= 2).groupby(level="vendor_id").sum().reindex(ids, fill_value=0)

    g_fulfil = n_ontime.sum() / n.sum()
    g_cancel = n_cancel.sum() / n.sum()
    g_ret = n_ret.sum() / max(n_del.sum(), 1)
    g_days = d["delivery_days"].mean()
    g_std = d["delivery_days"].std(ddof=0)
    g_rating = rated["rating"].mean()
    g_rep = n_rep.sum() / max(n_cust.sum(), 1)
    K_ORD, K_RAT, K_CUST = 25, 10, 15

    m = pd.DataFrame(index=ids)
    m["active_months"] = (active_days / 30.44).round(2)
    m["order_count_window"] = n.astype(int)
    m["fulfilment_rate"] = _smooth(n_ontime, n, g_fulfil, K_ORD)
    m["cancellation_rate"] = _smooth(n_cancel, n, g_cancel, K_ORD)
    m["return_rate"] = _smooth(n_ret, n_del, g_ret, K_ORD)
    m["avg_delivery_days"] = (sum_days + K_ORD * g_days) / (n_del + K_ORD)
    m["delivery_variability"] = (n_del * std_days.fillna(g_std) + K_ORD * g_std) / (n_del + K_ORD)
    m["stock_availability"] = v["stock_availability"]
    m["response_time_hours"] = v["response_time_hours"]
    m["log_response_time"] = np.log(m["response_time_hours"])
    m["gmv_window"] = gmv
    m["gmv_per_month"] = gmv / m["active_months"]
    m["orders_per_month"] = n / m["active_months"]
    m["log_gmv_per_month"] = np.log1p(m["gmv_per_month"])
    m["log_orders_per_month"] = np.log1p(m["orders_per_month"])

    pg = products.groupby("vendor_id")
    views_w = pg["views"].sum().reindex(ids, fill_value=0) * window_fraction
    atc_w = pg["add_to_cart"].sum().reindex(ids, fill_value=0) * window_fraction
    g_conv = n.sum() / max(views_w.sum(), 1)
    g_eng = atc_w.sum() / max(views_w.sum(), 1)
    m["conversion_rate"] = _smooth(n, views_w, g_conv, 200)
    m["customer_engagement"] = _smooth(atc_w, views_w, g_eng, 200)
    m["product_count"] = pg.size().reindex(ids, fill_value=0)
    m["log_product_count"] = np.log(m["product_count"].clip(lower=1))
    m["demand_alignment"] = pg["subcategory_demand_index"].mean().reindex(ids)
    m["search_relevance"] = pg["search_relevance_score"].mean().reindex(ids)
    subs_per_cat = products.groupby("category")["subcategory"].nunique()
    m["category_coverage"] = pg["subcategory"].nunique().reindex(ids) / v["category"].map(subs_per_cat)
    m["avg_rating"] = (sum_rating + K_RAT * g_rating) / (n_rated + K_RAT)
    m["repeat_customer_rate"] = _smooth(n_rep, n_cust, g_rep, K_CUST)
    m["recommendation_acceptance_rate"] = v["recommendation_acceptance_rate"]
    cat_gmv = gmv.groupby(v["category"]).sum()
    m["category_gmv_share"] = v["category"].map(cat_gmv / cat_gmv.sum())
    m["retention_contribution"] = n_rep / max(n_rep.sum(), 1)
    if customers is not None and "customer_segment" in customers.columns:
        seg = o.merge(customers[["customer_id", "customer_segment"]], on="customer_id", how="left")
        prem = seg[seg["customer_segment"] == "Premium"].groupby("vendor_id").size().reindex(ids, fill_value=0)
        m["premium_customer_share"] = _smooth(prem, n, prem.sum() / n.sum(), 20)
    else:
        m["premium_customer_share"] = np.nan
    m["category"] = v["category"]
    return m.reset_index()
