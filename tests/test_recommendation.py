import numpy as np

from src.config import FACTOR_KEYS
from src.evaluation import retrieval_evaluation
from src.recommendation import MATCH_COLUMNS, prioritise_vendors, semantic_match


def test_prioritise_filters_and_components(system):
    res, spec = prioritise_vendors(system.profile, category="Sustainable Packaging",
                                   min_percentiles={"operational_reliability": 30})
    assert (res["category"] == "Sustainable Packaging").all()
    assert (res["operational_reliability_pct"] >= 30).all()
    contrib = res[[f"{k}_contribution" for k in FACTOR_KEYS]].sum(axis=1)
    assert np.allclose(contrib, res["priority_index"])
    assert res["priority_index"].is_monotonic_decreasing
    assert abs(sum(spec["weights"].values()) - 1) < 1e-9


def test_weights_change_ranking(system):
    res, _ = prioritise_vendors(system.profile, weights={"commercial_potential": 1, "operational_reliability": 0,
                                                         "catalogue_market_fit": 0, "customer_product_fit": 0})
    assert res.iloc[0]["commercial_potential_pct"] == system.profile["commercial_potential_pct"].max()


def test_semantic_match_packaging(system):
    res = semantic_match(system, "sustainable food packaging for bulk restaurant orders", top_k=5)
    assert len(res) > 0
    assert set(MATCH_COLUMNS) <= set(res.columns)
    assert res.iloc[0]["category"] == "Sustainable Packaging"
    assert res["why_it_matches"].str.len().gt(0).all()


def test_retrieval_quality(system):
    _, summary, _ = retrieval_evaluation(system)
    assert summary["product_P@5"] >= 0.5
