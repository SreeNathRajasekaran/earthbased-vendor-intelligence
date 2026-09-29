import numpy as np

from src.config import FACTOR_KEYS
from src.evaluation import factor_recovery
from src.latent_analysis import kaiser_meyer_olkin


def test_adequacy(system):
    assert system.latent.kmo_overall > 0.6
    assert system.latent.bartlett["p_value"] < 1e-3


def test_four_named_factors(system):
    assert set(FACTOR_KEYS) <= set(system.latent.loadings.columns)


def test_signature_loadings(system):
    L = system.latent.loadings
    assert L.at["fulfilment_rate", "operational_reliability"] > 0.4
    assert L.at["cancellation_rate", "operational_reliability"] < -0.3
    assert L.at["log_gmv_per_month", "commercial_potential"] > 0.5
    assert L.at["search_relevance", "catalogue_market_fit"] > 0.4
    assert L.at["avg_rating", "customer_product_fit"] > 0.3


def test_factor_recovery(system):
    rec = factor_recovery(system)
    assert (np.diag(rec.to_numpy()) > 0.6).all(), rec.round(2)


def test_scores_standardised(system):
    s = system.latent.scores[FACTOR_KEYS]
    assert np.allclose(s.mean(), 0, atol=1e-6) and np.allclose(s.std(ddof=0), 1, atol=1e-6)


def test_kmo_on_uncorrelated_data():
    X = np.random.default_rng(0).standard_normal((500, 8))
    assert 0.3 < kaiser_meyer_olkin(X)[0] < 0.7


def test_segmentation_reasoned(system):
    S = system.segmentation
    assert S.chosen_k in set(S.k_grid["k"])
    assert S.silhouette > 0.15
    assert (S.descriptions["size"] >= 3).all()
    assert not S.descriptions["description"].str.contains("bad", case=False).any()
