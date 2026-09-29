"""Column-level documentation for every generated table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# (table, column, type, description)
DATA_DICTIONARY: list[tuple[str, str, str, str]] = [
    # vendors.csv
    ("vendors", "vendor_id", "str", "Synthetic vendor identifier (V001...)."),
    ("vendors", "vendor_name", "str", "Synthetic vendor name; not a real business."),
    ("vendors", "category", "str", "Primary marketplace category."),
    ("vendors", "city", "str", "Vendor dispatch city."),
    ("vendors", "onboarding_date", "date", "Date the vendor went live on the marketplace."),
    ("vendors", "product_count", "int", "Number of listed products."),
    ("vendors", "average_rating", "float", "Mean customer rating across rated orders (unsmoothed, 1-5)."),
    ("vendors", "price_index", "float", "Vendor price level relative to category average (1.0 = average)."),
    ("vendors", "stock_availability", "float", "Share of catalogue days in stock (0-1), from inventory logs."),
    ("vendors", "avg_delivery_days", "float", "Mean days from order to delivery for non-cancelled orders."),
    ("vendors", "fulfilment_rate", "float", "Share of all orders delivered within the 5-day SLA (cancellations count as unfulfilled)."),
    ("vendors", "cancellation_rate", "float", "Share of orders cancelled."),
    ("vendors", "return_rate", "float", "Share of delivered orders returned."),
    ("vendors", "response_time_hours", "float", "Median hours to respond to customer/ops queries."),
    ("vendors", "order_count", "int", "Orders placed in the analysis window (2024)."),
    ("vendors", "gmv", "float", "Gross merchandise value of non-cancelled orders (INR)."),
    ("vendors", "average_order_value", "float", "Mean value of non-cancelled orders (INR)."),
    ("vendors", "repeat_customer_rate", "float", "Share of the vendor's customers who ordered from it at least twice."),
    ("vendors", "conversion_rate", "float", "Orders divided by product-page views."),
    ("vendors", "customer_engagement", "float", "Add-to-cart events divided by product-page views."),
    ("vendors", "category_coverage", "float", "Share of the category's subcategories the vendor lists products in."),
    ("vendors", "recommendation_acceptance_rate", "float", "Click-through on the vendor's products when shown in recommendation modules."),
    ("vendors", "bulk_order_capable", "bool", "Vendor accepts bulk/B2B order quantities."),
    ("vendors", "certifications", "str", "Declared certifications (semicolon-separated)."),
    ("vendors", "catalogue_description", "str", "Generated summary of the vendor catalogue, used for semantic search."),
    # products.csv
    ("products", "product_id", "str", "Synthetic product identifier (P0001...)."),
    ("products", "vendor_id", "str", "Owning vendor."),
    ("products", "product_name", "str", "Generated product name."),
    ("products", "category", "str", "Marketplace category."),
    ("products", "subcategory", "str", "Marketplace subcategory."),
    ("products", "price", "float", "List price (INR)."),
    ("products", "rating", "float", "Mean rating from orders of this product (NaN if unrated)."),
    ("products", "review_count", "int", "Number of rated orders."),
    ("products", "orders", "int", "Orders placed for the product in 2024."),
    ("products", "returns", "int", "Returned orders."),
    ("products", "views", "int", "Product-page views in 2024."),
    ("products", "add_to_cart", "int", "Add-to-cart events in 2024."),
    ("products", "conversion_rate", "float", "orders / views."),
    ("products", "stock_availability", "float", "Share of days in stock."),
    ("products", "subcategory_demand_index", "float", "Platform search-demand index for the subcategory (1.0 = average)."),
    ("products", "search_relevance_score", "float", "Search click-through quality score for the listing (0-1)."),
    ("products", "description", "str", "Listing description used for semantic matching."),
    # orders.csv
    ("orders", "order_id", "str", "Synthetic order identifier."),
    ("orders", "customer_id", "str", "Ordering customer."),
    ("orders", "vendor_id", "str", "Fulfilling vendor."),
    ("orders", "product_id", "str", "Ordered product."),
    ("orders", "order_date", "date", "Order date (2024)."),
    ("orders", "quantity", "int", "Units ordered; bulk orders are 5-30 units."),
    ("orders", "order_value", "float", "Order value (INR); bulk orders of 10+ units carry a 7% discount."),
    ("orders", "delivery_days", "float", "Days to delivery; NaN if cancelled."),
    ("orders", "returned", "bool", "Order was returned."),
    ("orders", "cancelled", "bool", "Order was cancelled."),
    ("orders", "rating", "float", "Customer rating 1-5; NaN if unrated or cancelled."),
    # customers.csv
    ("customers", "customer_id", "str", "Synthetic customer identifier."),
    ("customers", "location", "str", "Customer city."),
    ("customers", "orders_count", "int", "Orders placed in 2024."),
    ("customers", "avg_order_value", "float", "Mean value of non-cancelled orders (INR)."),
    ("customers", "repeat_rate", "float", "Share of the customer's orders placed with a vendor they had used before."),
    ("customers", "category_preference", "str", "Most-browsed category."),
    ("customers", "engagement_score", "int", "Session-engagement score (0-100)."),
    ("customers", "customer_segment", "str", "Value / Core / Premium tertile of the average price index of vendors purchased from."),
    # vendor_metrics.csv
    ("vendor_metrics", "vendor_id", "str", "Vendor identifier."),
    ("vendor_metrics", "active_months", "float", "Months live within the analysis window (min 0.5)."),
    ("vendor_metrics", "order_count_window", "int", "Orders within the analysis window."),
    ("vendor_metrics", "fulfilment_rate", "float", "On-time fulfilment rate, empirical-Bayes smoothed toward the platform rate (25 pseudo-orders)."),
    ("vendor_metrics", "cancellation_rate", "float", "Smoothed cancellation rate."),
    ("vendor_metrics", "return_rate", "float", "Smoothed return rate among delivered orders."),
    ("vendor_metrics", "avg_delivery_days", "float", "Smoothed mean delivery days."),
    ("vendor_metrics", "delivery_variability", "float", "Smoothed standard deviation of delivery days (consistency)."),
    ("vendor_metrics", "stock_availability", "float", "Copied from vendor inventory logs."),
    ("vendor_metrics", "response_time_hours", "float", "Copied from vendor support logs."),
    ("vendor_metrics", "log_response_time", "float", "Natural log of response_time_hours."),
    ("vendor_metrics", "gmv_window", "float", "GMV in the window (INR)."),
    ("vendor_metrics", "gmv_per_month", "float", "GMV per active month (tenure-adjusted)."),
    ("vendor_metrics", "orders_per_month", "float", "Orders per active month."),
    ("vendor_metrics", "log_gmv_per_month", "float", "log1p(gmv_per_month)."),
    ("vendor_metrics", "log_orders_per_month", "float", "log1p(orders_per_month)."),
    ("vendor_metrics", "conversion_rate", "float", "Smoothed orders / views (views pro-rated to the window)."),
    ("vendor_metrics", "customer_engagement", "float", "Smoothed add-to-cart / views."),
    ("vendor_metrics", "product_count", "int", "Listed products."),
    ("vendor_metrics", "log_product_count", "float", "Natural log of product_count."),
    ("vendor_metrics", "demand_alignment", "float", "Mean subcategory demand index across the vendor's products."),
    ("vendor_metrics", "search_relevance", "float", "Mean search relevance score across products."),
    ("vendor_metrics", "category_coverage", "float", "Share of the category's subcategories covered."),
    ("vendor_metrics", "avg_rating", "float", "Bayesian-average rating (10 pseudo-ratings at the platform mean)."),
    ("vendor_metrics", "repeat_customer_rate", "float", "Smoothed share of customers with 2+ orders at the vendor."),
    ("vendor_metrics", "recommendation_acceptance_rate", "float", "Copied from recommendation logs."),
    ("vendor_metrics", "category_gmv_share", "float", "Share of platform GMV captured by the vendor's category (category importance)."),
    ("vendor_metrics", "retention_contribution", "float", "Vendor's share of all platform repeat-customer relationships."),
    ("vendor_metrics", "premium_customer_share", "float", "Smoothed share of the vendor's orders from Premium-segment customers."),
    ("vendor_metrics", "category", "str", "Vendor category (for grouping)."),
]


def data_dictionary_frame() -> pd.DataFrame:
    """Return the data dictionary as a DataFrame."""
    return pd.DataFrame(DATA_DICTIONARY, columns=["table", "column", "type", "description"])


def write_data_dictionary(out_dir: Path) -> None:
    """Write the data dictionary as CSV and Markdown into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = data_dictionary_frame()
    df.to_csv(out_dir / "data_dictionary.csv", index=False)
    lines = ["# Data dictionary", "", "All data is synthetic. See README for the generation logic.", ""]
    for table, grp in df.groupby("table", sort=False):
        lines += [f"## {table}.csv", "", "| Column | Type | Description |", "|---|---|---|"]
        lines += [f"| `{r.column}` | {r.type} | {r.description} |" for r in grp.itertuples()]
        lines.append("")
    (out_dir / "DATA_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")
