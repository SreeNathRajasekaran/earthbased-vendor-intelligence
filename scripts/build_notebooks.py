"""Generate (and optionally execute) the five analysis notebooks, which call the src/ modules."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
SETUP = ("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport pandas as pd\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)\n"
         "from src.pipeline import build_system\nsystem = build_system()")

NOTEBOOKS = {
    "01_data_generation.ipynb": [
        ("md", "# 01 Data generation\nStructural simulation: hidden vendor traits drive observable behaviour. "
               "All data is synthetic."),
        ("code", "import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nfrom src.data_generator import generate_marketplace, ARCHETYPES\n"
                 "tables = generate_marketplace(out_dir=__import__('pathlib').Path('../data'))\n"
                 "{k: v.shape for k, v in tables.items()}"),
        ("code", "tables['vendors'].head()"),
        ("code", "tables['orders'].describe(include='all').T.head(12)"),
        ("md", "Hidden archetypes (used only for validation):"),
        ("code", "pd = __import__('pandas')\npd.DataFrame(ARCHETYPES).T"),
    ],
    "02_eda.ipynb": [
        ("md", "# 02 Exploratory data analysis"),
        ("code", SETUP),
        ("code", "system.metrics.describe().T.round(3)"),
        ("code", "import plotly.express as px\nfrom src.feature_engineering import MODEL_FEATURES\n"
                 "px.imshow(system.metrics[MODEL_FEATURES].corr(), zmin=-1, zmax=1, color_continuous_scale='RdBu', height=650)"),
        ("code", "system.category_table.round(3)"),
        ("code", "px.histogram(system.tables['orders'], x='delivery_days', nbins=15)"),
    ],
    "03_latent_variable_analysis.ipynb": [
        ("md", "# 03 Latent-variable analysis\nAdequacy tests, factor retention, FA vs PCA, loadings and naming."),
        ("code", SETUP + "\nL = system.latent"),
        ("code", "print('KMO', round(L.kmo_overall, 3)); print('Bartlett', L.bartlett); print(L.retention_note)"),
        ("code", "L.loadings.round(2)"),
        ("code", "L.naming_table[['factor', 'label', 'top_loadings', 'alignment', 'runner_up_alignment']]"),
        ("code", "L.method_comparison"),
        ("code", "from src.evaluation import factor_recovery\nfactor_recovery(system).round(2)"),
    ],
    "04_vendor_segmentation.ipynb": [
        ("md", "# 04 Vendor segmentation"),
        ("code", SETUP + "\nS = system.segmentation"),
        ("code", "S.k_grid"),
        ("code", "print(S.selection_note)\nS.descriptions"),
        ("code", "S.centroids.round(2)"),
        ("code", "from src.evaluation import clustering_evaluation\nclustering_evaluation(system)['crosstab']"),
    ],
    "05_recommendation_engine.ipynb": [
        ("md", "# 05 Recommendation engine and Copilot"),
        ("code", SETUP),
        ("code", "from src.recommendation import prioritise_vendors, semantic_match\n"
                 "res, spec = prioritise_vendors(system.profile, category='Sustainable Packaging')\nprint(spec)\n"
                 "res[['vendor_id', 'vendor_name', 'priority_index', 'rationale']]"),
        ("code", "semantic_match(system, 'Need sustainable food packaging suitable for bulk restaurant orders.')"
                 "[['vendor_id', 'top_product', 'semantic_similarity', 'operational_reliability_pct', 'why_it_matches']]"),
        ("code", "from src.evaluation import retrieval_evaluation, run_grounding_suite, predictive_validity\n"
                 "per_q, summary, _ = retrieval_evaluation(system)\nsummary"),
        ("code", "df, s = run_grounding_suite(system)\nprint(s)\ndf[['question', 'intent', 'sufficient', 'grounded']]"),
        ("code", "pred, meta = predictive_validity(system)\nprint(meta['design'])\npred.round(3)"),
    ],
}


def build(execute: bool) -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        if execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbf.write(nb, NB_DIR / name)
        print(f"wrote {name}{' (executed)' if execute else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Execute notebooks so outputs are saved.")
    build(ap.parse_args().execute)
