import numpy as np

from src.preprocessing import validate_tables


def test_scale(system):
    t = system.tables
    assert 80 <= len(t["vendors"]) <= 150
    assert 800 <= len(t["products"]) <= 1500
    assert 8000 <= len(t["orders"]) <= 15000
    assert 2000 <= len(t["customers"]) <= 4000


def test_integrity(system):
    assert validate_tables(system.tables) == []


def test_rates_bounded(system):
    m = system.metrics
    for c in ["fulfilment_rate", "cancellation_rate", "return_rate", "repeat_customer_rate", "conversion_rate",
              "customer_engagement", "category_coverage"]:
        assert m[c].between(0, 1).all(), c


def test_cancelled_orders_have_no_delivery(system):
    o = system.tables["orders"]
    assert o.loc[o["cancelled"], "delivery_days"].isna().all()


def test_realistic_relationships(system):
    gt = system.ground_truth.set_index("vendor_id")
    m = system.metrics.set_index("vendor_id").loc[gt.index]
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    assert corr(gt["true_reliability"], m["cancellation_rate"]) < -0.3
    assert corr(gt["true_reliability"], m["fulfilment_rate"]) > 0.3
    assert corr(gt["true_commercial"], m["log_gmv_per_month"]) > 0.4
    assert corr(gt["true_customer_fit"], m["avg_rating"]) > 0.3
    assert corr(gt["true_catalogue_fit"], m["search_relevance"]) > 0.3
