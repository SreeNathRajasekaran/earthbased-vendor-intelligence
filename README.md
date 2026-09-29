# EarthBased Vendor Intelligence & Recommendation Engine

A retrospective analytics and AI prototype that infers hidden vendor characteristics from observable marketplace
behaviour. It uses those characteristics for vendor prioritisation, assortment analysis and product/vendor matching.

> This prototype was inspired by vendor-selection and assortment challenges encountered while working on a
> multi-vendor e-commerce platform. The dataset used in this repository is synthetic and does not contain
> proprietary EarthBased data. The system was not deployed in production, and no business results are claimed.

## 1. Project overview
The pipeline generates a realistic synthetic marketplace (120 vendors, ~1,000 products, ~12,000 orders,
3,000 customers). It then engineers vendor features and discovers four latent vendor dimensions with factor
analysis. Vendors are segmented with K-Means. A transparent recommendation layer and a grounded "Vendor Copilot"
sit on top, and everything is surfaced in an eight-page Streamlit dashboard.

## 2. Business problem
As a marketplace scales, vendor evaluation stops fitting in one person's head. Rating, price, product count and
sales each tell part of the story, and they often disagree. The qualities that matter are not recorded directly:
operational reliability, commercial potential, catalogue-market fit, customer/product fit and strategic value.

**Research question:** *Can latent vendor characteristics be inferred from observable marketplace behaviour to
improve vendor prioritisation, assortment decisions and product/vendor matching?*

## 3. Why latent variables?
Reliability is never logged. Cancellations, late deliveries, returns, stock-outs and slow responses are. If these
move together across vendors, one hidden quality is a more parsimonious and more stable explanation than any
single metric. Factor analysis estimates that quality from the shared variance, so the weights come from the data
rather than from a hand-built scorecard.

## 4. Product-management context
| Historical experience (real) | This repository (retrospective prototype) |
|---|---|
| Owned the 0-to-1 launch of a multi-vendor e-commerce platform | Synthetic marketplace modelled on that type of platform |
| Designed and scaled vendor onboarding; onboarded 80+ vendors | Vendor-prioritisation and segmentation engine |
| Translated vendor feedback into platform processes | Evidence panels explaining each vendor score |
| Used analytics and experimentation for growth decisions; managed Google Ads (6x ROAS) | Evaluation suite and proposed production KPIs |

## 5. System architecture
```
Raw data -> cleaning/validation -> feature engineering (smoothed, tenure-adjusted)
  -> EDA & correlation -> standardisation -> factor analysis (+PCA comparison)
  -> K-Means segmentation -> embeddings + FAISS index
  -> prioritisation & semantic matching -> grounded Copilot -> Streamlit
```
| Module | Responsibility |
|---|---|
| `src/data_generator.py` | Structural simulation of vendors, products, customers, orders |
| `src/preprocessing.py` | Loading, validation, cleaning |
| `src/feature_engineering.py` | Smoothed vendor metrics, construct hypotheses |
| `src/latent_analysis.py` | KMO, Bartlett, parallel analysis, FA/PCA, naming, Strategic Value |
| `src/segmentation.py` | K selection, clustering, neutral cluster descriptions |
| `src/embeddings.py`, `src/retrieval.py` | Encoders, cache, vector index |
| `src/recommendation.py` | Prioritisation, semantic matching, category gaps |
| `src/rag.py` | Intent routing, retrieval, grounded answers, grounding checks |
| `src/evaluation.py` | All evaluation metrics |
| `src/pipeline.py` | Orchestration and CLI |

## 6. Dataset
Five CSVs in `data/`, documented column by column in `data/DATA_DICTIONARY.md`. Each vendor receives four hidden
traits drawn around one of five hidden archetypes. Customers then place orders over 2024:
- Vendor choice depends on commercial pull, catalogue fit and price fit.
- Cancellations, delivery times and returns depend on reliability.
- Ratings and repeat purchases depend on customer fit and satisfaction.

Vendor metrics are aggregated from those simulated orders. The hidden traits live in `data/_ground_truth/`. They
are used only to check factor recovery and clustering, and nothing user-facing reads them.

## 7. ML methodology
Features are standardised. Rates are empirical-Bayes smoothed, for example 25 pseudo-orders at the platform rate,
so small vendors don't get extreme values. GMV and orders are divided by active months so newer vendors aren't
penalised for tenure.

## 8. Latent-variable methodology
- **Adequacy:** KMO and Bartlett's test of sphericity.
- **Retention:** scree plot, Kaiser rule and parallel analysis (95th-percentile random eigenvalues).
- **Model:** maximum-likelihood factor analysis with varimax rotation over 18 indicators, compared with unrotated
  and rotated PCA on variance explained, simple structure and construct alignment.
- **Naming:** indicator groups were written *before* fitting. Each extracted factor is matched to a group by
  Hungarian assignment on mean sign-aligned loadings. The dashboard shows the loadings and the runner-up group
  behind every name.
- **Strategic Value:** first principal component of the four factor scores plus category importance and retention
  contribution, with its weights and explained variance reported. Varimax factors are near-orthogonal, so this is
  treated as a data-weighted composite rather than a fifth latent trait.

## 9. Vendor segmentation
K-Means on factor scores for k = 2–8. k is chosen by silhouette, with the elbow reported alongside. If k = 2 wins,
the smallest k ≥ 3 within 0.02 silhouette is used, and the rule is displayed in the app. Clusters are first
labelled "Cluster 0..n", then described from their centroids using neutral terms such as "Operational-risk
signals" or "Limited commercial traction".

## 10. Recommendation engine
- **Prioritisation:** a user-weighted mean of factor percentiles after category, city, reliability and bulk
  filters. Each factor's contribution is shown.
- **Semantic matching:** product text and vendor catalogue text are embedded with Sentence Transformers
  (`all-MiniLM-L6-v2`), falling back to TF-IDF if unavailable, and searched with FAISS inner product (cosine).
  Results are aggregated to vendors with similarity, factor percentiles, an explanation and a caveat. Ranking is by
  similarity unless the user blends in a visible quality component.

## 11. RAG architecture
```
Question -> intent router (rules) -> structured + vector retrieval -> evidence table
  -> deterministic answer composer -> [optional LLM rewrite] -> grounding check -> answer + evidence + caveat
```
The grounding check rejects any vendor ID or number not present in the evidence, and any certainty language. If
an LLM answer fails the check, the deterministic answer is shown instead. Without `ANTHROPIC_API_KEY` the Copilot
runs fully deterministic. A 16-question suite includes unanswerable questions: an unknown vendor, company revenue,
"guaranteed" success and an unrelated product.

## 12. Evaluation
| Area | Metrics |
|---|---|
| Factor analysis | KMO, Bartlett, variance explained, loadings, recovery of hidden traits |
| Clustering | Silhouette, inertia, sizes, ARI vs hidden archetypes |
| Recommendation | Precision@5/10, MRR, nDCG@10, vendor Recall@10, similarity distributions |
| Prediction | Do H1 factors predict top-30% H2 GMV? Accuracy, precision, recall, F1, ROC-AUC vs surface-metric baselines |
| RAG | Intent accuracy, sufficiency, groundedness, error cases, latency, tokens/cost |

Results are computed at runtime and saved to `artifacts/evaluation.json`. No numbers are hard-coded in this README.
Retrieval relevance labels come from generator subcategories, so absolute scores are optimistic.

## 13. Screenshots
Run `python scripts/capture_screenshots.py` to populate `docs/screenshots/`.

| Overview | Latent factors | Matching | Copilot |
|---|---|---|---|
| ![](docs/screenshots/overview.png) | ![](docs/screenshots/factors.png) | ![](docs/screenshots/matching.png) | ![](docs/screenshots/copilot.png) |

## 14. Installation
Python 3.10–3.12.
```bash
git clone https://github.com/<your-username>/earthbased-vendor-intelligence.git
cd earthbased-vendor-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
To avoid the large GPU build of PyTorch, first run
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 15. Running locally
```bash
python -m src.pipeline --regenerate --evaluate   # generate data, fit models, save artifacts
pytest                                            # tests (TF-IDF backend, no downloads)
streamlit run app.py                              # dashboard at http://localhost:8501
python scripts/build_notebooks.py --execute       # optional: rebuild notebooks with outputs
```
Optional LLM layer:
```bash
export ANTHROPIC_API_KEY=...        # never commit keys
export EB_LLM_MODEL=<model-name>    # optional override
```
Other settings: `EB_EMBEDDING_BACKEND=auto|sentence-transformers|tfidf`, `EB_SEED`.

## 16. Deployment
Streamlit Community Cloud: push the repo, set `app.py` as the entry point and add any API key under
*Secrets*. The generated CSVs are committed, so the app starts without regenerating data. The first run downloads
the embedding model (~90 MB) and caches embeddings in `artifacts/`.

## 17. Limitations
- The data is synthetic. Relationships reflect the simulator's assumptions, not measured marketplace behaviour.
- With 120 vendors, factor solutions are sensitive to sampling noise; confirmatory analysis on real data would be
  required.
- Retrieval labels share vocabulary with product descriptions, so retrieval scores are optimistic.
- Intent routing is rule-based and will miss unusual phrasings.
- Percentiles are relative. A top-decile vendor in a weak pool is not necessarily a strong vendor.

## 18. Future improvements
Confirmatory factor analysis and measurement invariance across categories. Oblique rotation if factors correlate
on real data. Time-varying scores. Learning-to-rank from category-manager decisions. Cross-encoder re-ranking.
An LLM intent classifier with fallback. A pilot measuring the proposed production KPIs against a holdout.

## 19. Ethical and data disclaimer
All vendors, products, customers and orders are fictitious. Scores are decision support, not automated decisions.
Low scores describe observed signals in a simulation and are not judgements of any real business. Any real
deployment would need vendor transparency about how scores are computed, a route to contest them, and monitoring
for bias against newer or smaller vendors.

## License
MIT
