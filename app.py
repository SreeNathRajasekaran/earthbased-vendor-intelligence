"""EarthBased Vendor Intelligence: Streamlit dashboard (retrospective prototype on synthetic data)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ALL_SCORE_KEYS, FACTOR_KEYS, FACTOR_LABELS, STRATEGIC_KEY
from src.data_dictionary import data_dictionary_frame
from src.evaluation import run_full_evaluation
from src.feature_engineering import FEATURE_LABELS, format_value
from src.pipeline import build_system
from src.rag import VendorCopilot
from src.recommendation import explain_vendor_factor, ordinal, prioritise_vendors, semantic_match

st.set_page_config(page_title="EarthBased Vendor Intelligence", layout="wide", initial_sidebar_state="expanded")

PALETTE = ["#2F5D50", "#B08D3C", "#4E6E8E", "#A0522D", "#6B7B3A", "#7A6A8C", "#5E8C7F", "#8C6A5E"]
INK, MUTED, GRID = "#1E2B32", "#56646B", "#E3E7E1"
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stMarkdown, .stDataFrame, .stButton button, .stTextArea textarea { font-family: 'IBM Plex Sans', sans-serif; }
h1 { font-weight: 600; font-size: 1.75rem; color: #1E2B32; }
h2, h3 { font-weight: 600; color: #1E2B32; }
.kpi { border-left: 3px solid #2F5D50; background: #F1F3EF; padding: .65rem .9rem; border-radius: 2px; min-height: 4.4rem; }
.kpi-label { font-size: .8rem; color: #56646B; }
.kpi-value { font-size: 1.4rem; font-weight: 600; color: #1E2B32; }
.note { font-size: .85rem; color: #56646B; border-top: 1px solid #E1E5DF; padding-top: .4rem; margin-top: .2rem; }
</style>
"""
PAGES = {"overview": "Executive overview", "vendor": "Vendor intelligence", "factors": "Latent factor analysis",
         "segments": "Vendor segmentation", "matching": "Product and vendor matching", "copilot": "Vendor Copilot",
         "evaluation": "Model evaluation", "method": "Data and methodology"}
EXAMPLE_QUERIES = ["Premium sustainable packaging with reliable fulfilment for restaurants.",
                   "Need sustainable food packaging suitable for bulk restaurant orders.",
                   "Plastic-free personal care products for a zero-waste store.",
                   "Organic millets and pulses for a cloud kitchen, bulk supply."]
SUGGESTED = ["Which vendors should we prioritise?", "Why is V012 strategically important?",
             "Which vendors have strong demand but operational issues?", "Find vendors similar to V008.",
             "Which categories need more vendor coverage?", "Find vendors suitable for a premium customer segment."]


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Building the marketplace model (first run only)...")
def get_system():
    return build_system()


@st.cache_resource(show_spinner=False)
def get_copilot():
    return VendorCopilot(get_system())


@st.cache_data(show_spinner="Running the evaluation suite (about a minute on first load)...")
def get_evaluation(version: str = "v1") -> dict:
    return run_full_evaluation(get_system())


@st.cache_data(show_spinner=False)
def cached_match(query: str, top_k: int, category: str | None, blend: float) -> pd.DataFrame:
    return semantic_match(get_system(), query, top_k=top_k, category=category, quality_blend=blend)


# ---------------------------------------------------------------- helpers
def style(fig, height: int = 380, title: str | None = None):
    fig.update_layout(template="simple_white", height=height, title=title,
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
                      margin=dict(l=10, r=10, t=45 if title else 15, b=10), legend=dict(orientation="h", y=-0.18),
                      title_font=dict(size=15))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


def kpi(col, label: str, value: str, help_text: str = "") -> None:
    col.markdown(f"<div class='kpi' title='{help_text}'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def segment_colors(profile: pd.DataFrame) -> dict[str, str]:
    segs = sorted(profile["segment"].unique(), key=lambda s: int(s.split()[-1]))
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(segs)}


def pct_cols(keys=ALL_SCORE_KEYS) -> list[str]:
    return [f"{k}_pct" for k in keys]


# ---------------------------------------------------------------- pages
def page_overview(s) -> None:
    t, P = s.tables, s.profile
    v, o = t["vendors"], t["orders"]
    st.title("Executive overview")
    st.caption("Retrospective prototype on synthetic, anonymised data. Figures describe the simulated marketplace only.")
    c = st.columns(6)
    gmv = o.loc[~o["cancelled"], "order_value"].sum()
    kpi(c[0], "Vendors", f"{len(v):,}")
    kpi(c[1], "Products", f"{len(t['products']):,}")
    kpi(c[2], "Orders", f"{len(o):,}")
    kpi(c[3], "GMV", f"₹{gmv / 1e7:,.2f} Cr", "Non-cancelled order value, 2024")
    kpi(c[4], "Avg vendor rating", f"{v['average_rating'].mean():.2f}")
    kpi(c[5], "Avg on-time fulfilment", f"{v['fulfilment_rate'].mean():.1%}", "Delivered within 5 days")

    st.subheader("Vendor landscape")
    colors = segment_colors(P)
    fig = px.scatter(P, x="commercial_potential", y="operational_reliability", size="gmv", color="segment",
                     color_discrete_map=colors, category_orders={"segment": list(colors)}, hover_name="vendor_name",
                     hover_data={"vendor_id": True, "category": True, "segment_description": True, "gmv": ":,.0f",
                                 "commercial_potential": ":.2f", "operational_reliability": ":.2f", "segment": False},
                     size_max=36, labels={"commercial_potential": "Commercial Potential (z-score)",
                                          "operational_reliability": "Operational Reliability (z-score)"})
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
    st.plotly_chart(style(fig, 480), use_container_width=True)
    note("Bubble size is GMV. Axes are standardised factor scores (0 = platform average); the dotted lines mark the "
         "average, not a decision threshold.")

    c1, c2, c3 = st.columns(3)
    vt = s.latent.variance_table.reset_index().rename(columns={"index": "factor"})
    vt["label"] = vt["factor"].map(lambda k: FACTOR_LABELS.get(k, k))
    fig = px.bar(vt, x="proportion_of_variance", y="label", orientation="h", color_discrete_sequence=[PALETTE[0]],
                 labels={"proportion_of_variance": "Share of indicator variance", "label": ""})
    fig.update_xaxes(tickformat=".0%")
    c1.plotly_chart(style(fig, 320, "Latent factors by variance explained"), use_container_width=True)
    seg = P.groupby(["segment", "segment_description"]).size().reset_index(name="vendors")
    fig = px.bar(seg, x="segment", y="vendors", color="segment", color_discrete_map=colors,
                 hover_data={"segment_description": True, "segment": False}, labels={"segment": ""})
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style(fig, 320, "Vendor segments"), use_container_width=True)
    ct = s.category_table.melt(id_vars="category", value_vars=["order_share", "vendor_share"], var_name="share",
                               value_name="value")
    ct["share"] = ct["share"].map({"order_share": "Share of orders", "vendor_share": "Share of vendors"})
    fig = px.bar(ct, x="value", y="category", color="share", barmode="group", orientation="h",
                 color_discrete_sequence=[PALETTE[0], PALETTE[1]], labels={"value": "", "category": ""})
    fig.update_xaxes(tickformat=".0%")
    c3.plotly_chart(style(fig, 320, "Category opportunity"), use_container_width=True)
    note("Categories where the share of orders exceeds the share of vendors are candidates for targeted onboarding.")


def page_vendor(s) -> None:
    P = s.profile.set_index("vendor_id")
    st.title("Vendor intelligence")
    vid = st.selectbox("Vendor", P.index.tolist(),
                       format_func=lambda x: f"{x}  {P.at[x, 'vendor_name']} ({P.at[x, 'category']})")
    r = P.loc[vid]
    st.markdown(f"**{r.vendor_name}** is based in {r.city}, live since {r.onboarding_date}, and sits in "
                f"**{r.segment}**: {r.segment_description}.")
    c = st.columns(8)
    kpi(c[0], "Products", f"{int(r.product_count)}")
    kpi(c[1], "GMV (2024)", f"₹{r.gmv:,.0f}")
    kpi(c[2], "Rating", f"{r.avg_rating:.2f}")
    kpi(c[3], "On-time fulfilment", format_value("fulfilment_rate", r.fulfilment_rate))
    kpi(c[4], "Returns", format_value("return_rate", r.return_rate))
    kpi(c[5], "Cancellations", format_value("cancellation_rate", r.cancellation_rate))
    kpi(c[6], "Repeat customers", format_value("repeat_customer_rate", r.repeat_customer_rate))
    kpi(c[7], "Price index", f"{r.price_index:.2f}")
    note("Rates are smoothed toward the platform average so that vendors with few orders do not show extreme values.")

    left, right = st.columns(2)
    cluster = s.profile[s.profile["cluster"] == r.cluster][pct_cols()].mean()
    theta = [FACTOR_LABELS[k] for k in ALL_SCORE_KEYS]
    vals = [r[f"{k}_pct"] for k in ALL_SCORE_KEYS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=theta + theta[:1], name=vid, fill="toself",
                                  line_color=PALETTE[0]))
    fig.add_trace(go.Scatterpolar(r=list(cluster.values) + [cluster.values[0]], theta=theta + theta[:1],
                                  name=f"{r.segment} average", line=dict(color=PALETTE[1], dash="dash")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10))))
    left.plotly_chart(style(fig, 420, "Factor percentiles vs segment average"), use_container_width=True)
    table = pd.DataFrame({"Score": theta, "Percentile": vals,
                          "z-score": [round(r[k], 2) for k in ALL_SCORE_KEYS],
                          "Segment average percentile": cluster.round(0).astype(int).to_numpy()})
    right.markdown("**Latent scores**")
    right.dataframe(table, hide_index=True, use_container_width=True)
    right.caption("Percentiles rank this vendor against all vendors. Strategic Value is a composite of the four "
                  "factors, category importance and retention contribution.")

    st.subheader("What drives each score")
    fk = st.selectbox("Score to explain", ALL_SCORE_KEYS, format_func=FACTOR_LABELS.get)
    st.dataframe(explain_vendor_factor(s, vid, fk, top_n=5), hide_index=True, use_container_width=True)
    note("Contribution = standardised indicator value x loading (or composite weight). It shows which evidence pushes "
         "the score up or down; it is a descriptive decomposition, not a causal one.")

    st.subheader("Comparison with segment and platform")
    cols = ["fulfilment_rate", "cancellation_rate", "return_rate", "avg_delivery_days", "gmv_per_month",
            "conversion_rate", "avg_rating", "repeat_customer_rate", "customer_engagement", "category_coverage"]
    seg_med = s.profile[s.profile["cluster"] == r.cluster][cols].median()
    plat_med = s.profile[cols].median()
    comp = pd.DataFrame({"Metric": [FEATURE_LABELS.get(c, c.replace("_", " ").capitalize()) for c in cols],
                         vid: [format_value(c, r[c]) for c in cols],
                         "Segment median": [format_value(c, seg_med[c]) for c in cols],
                         "Platform median": [format_value(c, plat_med[c]) for c in cols]})
    st.dataframe(comp, hide_index=True, use_container_width=True)


def page_factors(s) -> None:
    L = s.latent
    st.title("Latent factor analysis")
    with st.expander("What is a latent variable?", expanded=True):
        st.markdown(
            "Some vendor qualities, such as *operational reliability*, are never recorded directly. We only see their "
            "traces: cancellations, late deliveries, returns, slow responses. When several observed metrics rise and "
            "fall together across vendors, a simpler explanation is that one hidden quality drives them. Factor "
            "analysis estimates those hidden qualities from the shared movement (correlation) in the metrics.\n\n"
            "The model does **not** know the business names. It finds groups of co-moving metrics; we then name each "
            "factor by checking which metrics load on it most strongly, against a list of indicators written down "
            "before the model was fitted.")
    c = st.columns(4)
    kpi(c[0], "KMO sampling adequacy", f"{L.kmo_overall:.2f}", "Above 0.6 is generally acceptable for factor analysis")
    kpi(c[1], "Bartlett's test p-value", f"{L.bartlett['p_value']:.1e}", "Small p means the metrics are correlated enough to factor")
    kpi(c[2], "Factors retained", f"{L.n_factors} (parallel analysis: {L.n_factors_parallel})")
    kpi(c[3], "Indicator variance explained", f"{L.variance_table['proportion_of_variance'].sum():.0%}")
    note(L.retention_note)

    c1, c2 = st.columns(2)
    idx = list(range(1, len(L.eigenvalues) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=L.eigenvalues, mode="lines+markers", name="Observed eigenvalues", line_color=PALETTE[0]))
    fig.add_trace(go.Scatter(x=idx, y=L.parallel_thresholds, mode="lines", name="Parallel analysis (95th pct random)",
                             line=dict(color=PALETTE[1], dash="dash")))
    fig.add_hline(y=1, line_dash="dot", line_color=MUTED, annotation_text="Kaiser = 1")
    fig.update_xaxes(title="Component")
    fig.update_yaxes(title="Eigenvalue")
    c1.plotly_chart(style(fig, 380, "Scree plot"), use_container_width=True)
    vt = L.variance_table.copy()
    vt.index = [FACTOR_LABELS.get(i, i) for i in vt.index]
    c2.markdown("**Variance explained per factor**")
    c2.dataframe(vt.style.format({"ss_loadings": "{:.2f}", "proportion_of_variance": "{:.1%}", "cumulative": "{:.1%}"}),
                 use_container_width=True)
    c2.caption("Factors retained where observed eigenvalues exceed what random data of the same size produces.")

    st.subheader("Factor loadings")
    load = L.loadings.copy()
    load.index = [FEATURE_LABELS.get(f, f) for f in load.index]
    load.columns = [FACTOR_LABELS.get(c, c) for c in load.columns]
    fig = px.imshow(load, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(style(fig, 620), use_container_width=True)
    note("Loadings are correlations between each metric and each factor after varimax rotation. Signs are aligned so a "
         "higher factor score always means a more favourable position (e.g. cancellation rate loads negatively on "
         "reliability).")

    st.subheader("Why each factor has its name")
    st.dataframe(L.naming_table[["factor", "label", "top_loadings", "alignment", "runner_up", "runner_up_alignment",
                                 "interpretation"]], hide_index=True, use_container_width=True)

    st.subheader("Factor Analysis vs PCA")
    st.dataframe(L.method_comparison.style.format({"variance_explained": "{:.1%}", "simple_structure_share": "{:.0%}",
                                                   "mean_construct_alignment": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    note("PCA explains more total variance because it models all variance, including measurement noise. Factor "
         "analysis models only the shared variance, which is what a latent trait should explain. The simple-structure "
         "share (variables loading clearly on one factor) is the interpretability comparison.")

    c1, c2 = st.columns(2)
    dist = s.profile.melt(id_vars="vendor_id", value_vars=FACTOR_KEYS, var_name="factor", value_name="score")
    dist["factor"] = dist["factor"].map(FACTOR_LABELS)
    fig = px.box(dist, x="factor", y="score", points="all", color_discrete_sequence=[PALETTE[0]],
                 labels={"factor": "", "score": "Factor score (z)"})
    c1.plotly_chart(style(fig, 380, "Vendor factor-score distribution"), use_container_width=True)
    w = L.strategic_weights.rename(lambda k: FACTOR_LABELS.get(k, k.replace("_", " ").capitalize())).reset_index()
    w.columns = ["component", "weight"]
    fig = px.bar(w, x="weight", y="component", orientation="h", color_discrete_sequence=[PALETTE[2]],
                 labels={"component": ""})
    c2.plotly_chart(style(fig, 380, "Strategic Value composite weights"), use_container_width=True)
    c2.caption(f"First principal component of the inputs; it explains {L.strategic_explained_variance:.0%} of their "
               "variance. The four factors are near-uncorrelated by construction, so this is a data-weighted index "
               "rather than a separate latent trait.")


def page_segments(s) -> None:
    S, P = s.segmentation, s.profile
    st.title("Vendor segmentation")
    c1, c2 = st.columns(2)
    fig = px.line(S.k_grid, x="k", y="inertia", markers=True, color_discrete_sequence=[PALETTE[0]])
    fig.add_vline(x=S.elbow_k, line_dash="dot", line_color=MUTED, annotation_text=f"elbow k={S.elbow_k}")
    c1.plotly_chart(style(fig, 320, "Elbow method (within-cluster inertia)"), use_container_width=True)
    fig = px.line(S.k_grid, x="k", y="silhouette", markers=True, color_discrete_sequence=[PALETTE[1]])
    fig.add_vline(x=S.chosen_k, line_dash="dot", line_color=MUTED, annotation_text=f"chosen k={S.chosen_k}")
    c2.plotly_chart(style(fig, 320, "Silhouette score"), use_container_width=True)
    st.info(S.selection_note)

    st.subheader("Cluster characteristics")
    st.dataframe(S.descriptions.style.format({"mean_silhouette": "{:.2f}"}), hide_index=True, use_container_width=True)
    cent = S.centroids.copy()
    cent.index = [f"Cluster {i}" for i in cent.index]
    cent.columns = [FACTOR_LABELS[c] for c in cent.columns]
    fig = px.imshow(cent, text_auto=".2f", zmin=-1.5, zmax=1.5, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(style(fig, 300, "Mean factor score by cluster (z)"), use_container_width=True)
    note("Descriptions are generated from the centroids: a factor is mentioned when the cluster mean is at least 0.35 "
         "standard deviations above or below the platform average.")

    st.subheader("Latent space")
    f = st.columns(3)
    cats = f[0].multiselect("Category", sorted(P["category"].unique()))
    cities = f[1].multiselect("City", sorted(P["city"].unique()))
    segs = f[2].multiselect("Segment", sorted(P["segment"].unique()))
    df = P.merge(S.embedding_2d, on="vendor_id")
    if cats:
        df = df[df["category"].isin(cats)]
    if cities:
        df = df[df["city"].isin(cities)]
    if segs:
        df = df[df["segment"].isin(segs)]
    if df.empty:
        st.warning("No vendors match these filters. Clear a filter to see the latent space.")
        return
    colors = segment_colors(P)
    fig = px.scatter(df, x="component_1", y="component_2", color="segment", color_discrete_map=colors,
                     hover_name="vendor_name", hover_data={"vendor_id": True, "category": True, "city": True},
                     labels={"component_1": "Latent component 1", "component_2": "Latent component 2"})
    st.plotly_chart(style(fig, 480), use_container_width=True)
    note(f"{len(df)} vendors shown. Two-dimensional PCA projection of the four factor scores, for visualisation only; "
         "clustering uses all four dimensions.")
    st.dataframe(df[["vendor_id", "vendor_name", "category", "city", "segment"] + pct_cols()],
                 hide_index=True, use_container_width=True)


def page_matching(s) -> None:
    st.title("Product and vendor matching")
    st.caption(f"Embedding backend: {s.index.backend_label}")
    if s.index.encoder.fallback_reason:
        st.warning("Sentence Transformers could not be loaded, so matching uses the TF-IDF fallback, which matches "
                   f"words rather than meaning. Reason: {s.index.encoder.fallback_reason}")
    st.session_state.setdefault("match_query", EXAMPLE_QUERIES[0])
    ex = st.columns(len(EXAMPLE_QUERIES))
    for i, q in enumerate(EXAMPLE_QUERIES):
        ex[i].button(q, key=f"ex{i}", use_container_width=True,
                     on_click=lambda q=q: st.session_state.__setitem__("match_query", q))
    with st.form("match_form"):
        query = st.text_area("Describe what type of vendor or product you are looking for.", key="match_query", height=90)
        c1, c2, c3 = st.columns(3)
        cat = c1.selectbox("Category filter", ["All"] + sorted(s.profile["category"].unique()))
        k = c2.slider("Results", 3, 10, 5)
        blend = c3.slider("Weight on vendor quality", 0.0, 1.0, 0.0, 0.1,
                          help="0 ranks by text similarity only. Higher values blend in the mean of reliability and "
                               "customer-fit percentiles.")
        st.form_submit_button("Find matches")
    if not query.strip():
        st.info("Enter a requirement or pick an example to see matches.")
        return
    res = cached_match(query, k, None if cat == "All" else cat, blend)
    if res.empty:
        st.warning("No catalogue listing shares vocabulary with this request. Try naming a product type or material.")
        return
    for r in res.itertuples(index=False):
        with st.container(border=True):
            a, b = st.columns([3, 2])
            a.markdown(f"**{r.vendor_id} {r.vendor_name}** ({r.category}, {r.city})  \n{r.segment}: {r.segment_description}")
            a.markdown(f"Relevant product: *{r.top_product}*")
            a.caption(r.why_it_matches)
            a.caption(f"Caveat: {r.caveat}")
            m = b.columns(2)
            m[0].metric("Semantic similarity", f"{r.semantic_similarity:.2f}", help=r.match_strength)
            m[1].metric("Operational reliability", ordinal(r.operational_reliability_pct) + " pct")
            m[0].metric("Commercial potential", ordinal(r.commercial_potential_pct) + " pct")
            m[1].metric("Catalogue-market fit", ordinal(r.catalogue_market_fit_pct) + " pct")
    st.subheader("Evidence table")
    ev = res[["vendor_id", "vendor_name", "top_product", "semantic_similarity", "catalogue_similarity",
              "quality_component", "rank_score"] + pct_cols(FACTOR_KEYS) + ["match_strength"]]
    st.dataframe(ev.round(3), hide_index=True, use_container_width=True)
    st.download_button("Download evidence (CSV)", ev.to_csv(index=False), "match_evidence.csv", "text/csv")
    note(f"rank score = (1 - {blend:.1f}) x similarity relative to the best match + {blend:.1f} x quality component. "
         "Similarity measures how closely catalogue text matches the request, not verified capability.")
    cands = s.index.search_products(query, k=50)
    fig = px.histogram(cands, x="similarity", nbins=25, color_discrete_sequence=[PALETTE[2]])
    fig.add_vline(x=s.index.encoder.weak_threshold, line_dash="dot", annotation_text="weak-match threshold")
    st.plotly_chart(style(fig, 260, "Similarity of the 50 nearest listings"), use_container_width=True)


def render_response(resp) -> None:
    with st.chat_message("user"):
        st.markdown(resp.question)
    with st.chat_message("assistant"):
        st.markdown(resp.answer)
        with st.expander(f"Evidence ({len(resp.evidence)} records)"):
            if resp.evidence.empty:
                st.write("No records matched, so the answer reports insufficient evidence.")
            else:
                st.dataframe(resp.evidence, hide_index=True, use_container_width=True)
            for n in resp.notes:
                st.caption(n)
        meta = (f"Intent: {resp.intent} | mode: {resp.mode} | grounding check: "
                f"{'passed' if resp.grounded else 'failed'} | {resp.latency_ms:.0f} ms")
        if resp.input_tokens:
            meta += f" | tokens {resp.input_tokens}+{resp.output_tokens} (approx. ${resp.estimated_cost_usd:.4f})"
        st.caption(meta)
        if resp.grounding_issues:
            st.warning("Grounding issues: " + "; ".join(resp.grounding_issues))
        if resp.llm_error:
            st.caption(f"LLM note: {resp.llm_error}")


def page_copilot(s) -> None:
    cp = get_copilot()
    st.title("Vendor Copilot")
    mode = ("LLM answers checked against retrieved evidence" if cp.use_llm
            else "evidence-templated answers (no ANTHROPIC_API_KEY found; the LLM layer is off)")
    st.caption(f"Answer mode: {mode}. Every answer lists the records it used and states when evidence is insufficient.")
    st.session_state.setdefault("chat", [])
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        cols[i % 3].button(q, key=f"sugg{i}", use_container_width=True,
                           on_click=lambda q=q: st.session_state.__setitem__("pending_q", q))
    for resp in st.session_state["chat"]:
        render_response(resp)
    typed = st.chat_input("Ask about vendors, factors, segments or category coverage")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        resp = cp.ask(question)
        st.session_state["chat"].append(resp)
        render_response(resp)
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()


def page_evaluation(s) -> None:
    st.title("Model evaluation")
    ev = get_evaluation()
    tabs = st.tabs(["Factor analysis", "Clustering", "Recommendation", "Prediction", "Copilot", "Proposed production KPIs"])
    with tabs[0]:
        f = ev["factor"]
        c = st.columns(4)
        kpi(c[0], "KMO", f"{f['kmo_overall']:.2f}")
        kpi(c[1], "Bartlett chi-square", f"{f['bartlett']['chi_square']:,.0f}")
        kpi(c[2], "Factors (PA / Kaiser)", f"{f['n_factors_parallel']} / {f['n_factors_kaiser']}")
        kpi(c[3], "Variance explained", f"{f['variance_explained']:.0%}")
        kmo = pd.Series(f["kmo_per_item"]).rename(lambda x: FEATURE_LABELS.get(x, x)).reset_index()
        kmo.columns = ["indicator", "KMO"]
        st.plotly_chart(style(px.bar(kmo, x="KMO", y="indicator", orientation="h",
                                     color_discrete_sequence=[PALETTE[0]]), 460, "Per-indicator KMO"),
                        use_container_width=True)
        if not ev["recovery"].empty:
            st.markdown("**Recovery of hidden generator traits**")
            st.plotly_chart(style(px.imshow(ev["recovery"], text_auto=".2f", zmin=-1, zmax=1,
                                            color_continuous_scale="RdBu", aspect="auto"), 320),
                            use_container_width=True)
            note("Correlation between estimated factor scores and the generator's hidden traits. This check is only "
                 "possible because the data is synthetic; a real deployment would validate against outcomes instead.")
    with tabs[1]:
        cl = ev["clustering"]
        c = st.columns(4)
        kpi(c[0], "Clusters", str(cl["chosen_k"]))
        kpi(c[1], "Silhouette", f"{cl['silhouette']:.2f}")
        kpi(c[2], "Inertia", f"{cl['inertia']:.0f}")
        kpi(c[3], "ARI vs hidden archetype", f"{cl.get('ari_vs_hidden_archetype', float('nan')):.2f}")
        st.dataframe(pd.DataFrame(cl["sizes"]), hide_index=True, use_container_width=True)
        if "crosstab" in cl:
            st.markdown("**Clusters vs hidden archetypes (synthetic check)**")
            st.dataframe(cl["crosstab"], use_container_width=True)
    with tabs[2]:
        rs = ev["retrieval_summary"]
        c = st.columns(5)
        for col, key in zip(c, ["product_P@5", "product_P@10", "MRR", "nDCG@10", "vendor_R@10"]):
            kpi(col, key, f"{rs[key]:.2f}")
        st.caption(f"Backend: {ev['embedding_backend']}")
        st.dataframe(ev["retrieval_per_query"].round(3), hide_index=True, use_container_width=True)
        fig = px.histogram(ev["similarity_distribution"], x="similarity", color="relevant", barmode="overlay",
                           nbins=40, color_discrete_sequence=[PALETTE[3], PALETTE[0]])
        st.plotly_chart(style(fig, 300, "Cosine similarity: relevant vs non-relevant results"), use_container_width=True)
        note("Relevance proxy = product subcategory matches the query's intended subcategory. Product descriptions share "
             "vocabulary with these test queries, so absolute scores are optimistic. The numbers are most useful for "
             "comparing encoders or settings, not as a production accuracy claim.")
    with tabs[3]:
        st.dataframe(ev["prediction"].round(3), hide_index=True, use_container_width=True)
        st.caption(f"{ev['prediction_meta']['design']} Vendors: {ev['prediction_meta']['n_vendors']}, "
                   f"positives: {ev['prediction_meta']['n_positive']}.")
        note("This tests the research question directly: do latent factors from the first half-year carry information "
             "about second-half commercial outcomes beyond surface metrics? Read the results as reported, including "
             "when the latent model does not win. First-half GMV is a strong baseline because commercial momentum "
             "persists.")
    with tabs[4]:
        rsum = ev["rag_summary"]
        c = st.columns(4)
        kpi(c[0], "Intent accuracy", f"{rsum['intent_accuracy']:.0%}")
        kpi(c[1], "Sufficiency judged correctly", f"{rsum['sufficiency_accuracy']:.0%}")
        kpi(c[2], "Grounded answers", f"{rsum['grounded_rate']:.0%}")
        kpi(c[3], "Mean latency", f"{rsum['mean_latency_ms']:.0f} ms")
        st.dataframe(ev["rag"], hide_index=True, use_container_width=True)
        note("Grounded = every vendor ID and number in the answer appears in the retrieved evidence, and no certainty "
             "language is used. The deterministic composer builds answers from that evidence, so its pass rate mostly "
             "verifies the pipeline. The check matters most when an LLM is enabled: failing LLM answers are replaced "
             "by the deterministic answer. Tokens and cost are zero without an API key.")
    with tabs[5]:
        st.markdown("None of these have been measured. They describe how the prototype would be evaluated if piloted.")
        st.dataframe(pd.DataFrame([
            ("Vendor evaluation time", "Minutes from vendor application to decision", "Time-stamp workflow before vs after"),
            ("Vendor shortlisting time", "Time to produce a shortlist for a sourcing request", "Timed tasks, A/B by team"),
            ("Recommendation acceptance", "Share of suggested vendors that category managers shortlist", "Log accept/reject"),
            ("Catalogue coverage", "Share of high-demand subcategories with 3+ reliable vendors", "Monthly snapshot"),
            ("Supplier response rate", "Share of contacted vendors who respond within 48 h", "Outreach CRM"),
            ("Repeat purchase rate", "Share of customers reordering within 90 days", "Cohort analysis"),
            ("Conversion rate", "Orders per product-page view for newly onboarded vendors", "Holdout comparison"),
            ("Operational issue rate", "Cancellations + late deliveries + returns per 100 orders", "Ops dashboard"),
        ], columns=["Proposed production KPI", "Definition", "Measurement approach"]), hide_index=True,
            use_container_width=True)


def page_method(s) -> None:
    st.title("Data and methodology")
    st.markdown(
        "**Context.** This prototype is inspired by vendor-selection and assortment challenges encountered while "
        "working on a multi-vendor e-commerce platform. The dataset is synthetic and contains no proprietary "
        "EarthBased data. It was not deployed in production.")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=10, color="#2F5D50"];
      raw [label="Synthetic raw data"]; clean [label="Cleaning &\\nvalidation"]; fe [label="Feature engineering\\n(smoothed rates)"];
      eda [label="EDA &\\ncorrelation"]; std [label="Standardisation"]; fa [label="Factor analysis\\n(+ PCA comparison)"];
      km [label="K-Means\\nsegmentation"]; emb [label="Embeddings +\\nvector index"]; rec [label="Prioritisation &\\nmatching"];
      rag [label="Grounded\\nCopilot"]; ui [label="Dashboard"];
      raw -> clean -> fe -> eda -> std -> fa -> km -> rec -> rag -> ui; emb -> rec; fa -> rec; }""")
    st.markdown(
        "**Generation.** Each vendor has four hidden traits drawn around one of five hidden archetypes. Orders are "
        "simulated customer by customer: vendor choice depends on commercial pull, catalogue fit and price fit; "
        "cancellations, delivery times and returns depend on reliability; ratings and repeat purchases depend on "
        "customer fit and satisfaction. Vendor metrics are aggregated from these simulated orders.\n\n"
        "**Features.** Rates use empirical-Bayes smoothing (e.g. 25 pseudo-orders at the platform rate). GMV and orders "
        "are divided by active months so newer vendors are not penalised for tenure.\n\n"
        "**Factors.** Maximum-likelihood factor analysis with varimax rotation on 18 standardised indicators. "
        "Retention uses parallel analysis, the scree plot and the Kaiser rule. Factors are named by Hungarian matching "
        "against indicator groups written before fitting, and every name is shown with its loadings.\n\n"
        "**Segments.** K-Means on the four factor scores. k is chosen by silhouette, with the elbow reported alongside.\n\n"
        "**Matching.** Product and catalogue text is embedded with Sentence Transformers (TF-IDF fallback) and searched "
        "with FAISS inner product (cosine on normalised vectors).\n\n"
        "**Copilot.** A rule-based intent router, structured and vector retrieval, and a deterministic composer. An "
        "optional LLM rewrites answers, and a grounding check validates every answer.")
    st.subheader("Data dictionary")
    dd = data_dictionary_frame()
    tbl = st.selectbox("Table", dd["table"].unique())
    st.dataframe(dd[dd["table"] == tbl].drop(columns="table"), hide_index=True, use_container_width=True)
    st.subheader("Preview and download")
    name = st.selectbox("Dataset", list(s.tables))
    st.dataframe(s.tables[name].head(200), use_container_width=True)
    st.download_button(f"Download {name}.csv", s.tables[name].to_csv(index=False), f"{name}.csv", "text/csv")


RENDER = {"overview": page_overview, "vendor": page_vendor, "factors": page_factors, "segments": page_segments,
          "matching": page_matching, "copilot": page_copilot, "evaluation": page_evaluation, "method": page_method}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    keys = list(PAGES)
    default = st.query_params.get("page", "overview")
    with st.sidebar:
        st.markdown("### EarthBased Vendor Intelligence")
        page = st.radio("Section", keys, index=keys.index(default) if default in keys else 0,
                        format_func=PAGES.get, label_visibility="collapsed")
        note("Retrospective prototype on synthetic data. Not a production system; contains no EarthBased data.")
    st.query_params["page"] = page
    try:
        system = get_system()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The data pipeline failed: {exc}. Run `python -m src.pipeline --regenerate` and check the log output.")
        st.stop()
    RENDER[page](system)


if __name__ == "__main__":
    main()
