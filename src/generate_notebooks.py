"""
Generate reproducible notebooks for the Tox21 project.

The notebooks are intentionally lightweight: they document and validate each
phase using the artifacts produced by src/pipeline.py instead of retraining all
models during notebook execution.
"""

from pathlib import Path

import nbformat as nbf


BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

KERNEL = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = code(
    r"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
REPORTS = BASE_DIR / "reports"
FIGURES = REPORTS / "figures"
MODELS = BASE_DIR / "models"

print("Project:", BASE_DIR)
print("Artifacts present:", REPORTS.exists(), FIGURES.exists(), MODELS.exists())
"""
)


def write_notebook(filename: str, title: str, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = KERNEL
    nb["metadata"]["language_info"] = {"name": "python"}
    nb["cells"] = [md(f"# {title}"), SETUP] + cells
    path = NOTEBOOKS_DIR / filename
    nbf.write(nb, path)
    print(f"Wrote {path}")


def main():
    write_notebook(
        "01_dataset_download.ipynb",
        "01 - Dataset Download",
        [
            md("Load or download the official Tox21 dataset and verify the raw file."),
            code(
                r"""
from data_loader import download_tox21, TOX21_TASKS
df = download_tox21()
print(df.shape)
display(df.head())
print("Tasks present:", [t for t in TOX21_TASKS if t in df.columns])
print("Raw CSV exists:", (DATA_RAW / "tox21.csv").exists())
"""
            ),
        ],
    )

    write_notebook(
        "02_data_understanding.ipynb",
        "02 - Data Understanding",
        [
            md("Inspect dimensions, label availability, missingness, and class balance."),
            code(
                r"""
df = pd.read_csv(DATA_RAW / "tox21.csv")
tasks = [c for c in df.columns if c.startswith(("NR-", "SR-"))]
summary = pd.DataFrame({
    "labeled": df[tasks].notna().sum(),
    "positive": df[tasks].sum(),
    "missing_pct": df[tasks].isna().mean() * 100,
    "positive_pct_labeled": df[tasks].sum() / df[tasks].notna().sum() * 100,
}).round(2)
print("Shape:", df.shape)
display(summary.sort_values("positive_pct_labeled", ascending=False))
"""
            ),
        ],
    )

    write_notebook(
        "03_eda.ipynb",
        "03 - Exploratory Data Analysis",
        [
            md("Review the generated exploratory figures and confirm that they exist."),
            code(
                r"""
eda_figures = [
    "class_distribution.png",
    "missing_values.png",
    "descriptor_distributions.png",
    "descriptor_boxplots.png",
    "correlation_matrix.png",
]
for name in eda_figures:
    path = FIGURES / name
    print(f"{name:35s}", path.exists(), path)
"""
            ),
            code(
                r"""
from IPython.display import Image, display
display(Image(filename=str(FIGURES / "class_distribution.png")))
display(Image(filename=str(FIGURES / "missing_values.png")))
"""
            ),
        ],
    )

    write_notebook(
        "04_feature_engineering.ipynb",
        "04 - Feature Engineering",
        [
            md("Validate RDKit descriptors and molecular fingerprints."),
            code(
                r"""
from molecular_features import smiles_to_mol, compute_descriptors, morgan_fingerprint, maccs_fingerprint
mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
desc = compute_descriptors(mol)
print("Aspirin MolWt:", round(desc["MolWt"], 2))
print("Aspirin LogP:", round(desc["LogP"], 2))
print("Morgan shape:", morgan_fingerprint(mol).shape)
print("MACCS shape:", maccs_fingerprint(mol).shape)
"""
            ),
            code(
                r"""
desc_df = pd.read_csv(DATA_PROCESSED / "descriptors.csv")
morgan = np.load(DATA_PROCESSED / "morgan_fp.npy")
maccs = np.load(DATA_PROCESSED / "maccs_fp.npy")
rdkit_fp = np.load(DATA_PROCESSED / "rdkit_fp.npy")
print("Descriptors:", desc_df.shape)
print("Morgan:", morgan.shape)
print("MACCS:", maccs.shape)
print("RDKit:", rdkit_fp.shape)
display(desc_df.head())
"""
            ),
        ],
    )

    write_notebook(
        "05_machine_learning.ipynb",
        "05 - Machine Learning",
        [
            md("Load saved model metrics and compare the classical ML models."),
            code(
                r"""
metrics_path = REPORTS / "NR_AhR_morgan_metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)
rows = []
for model, vals in metrics.items():
    rows.append({k: vals.get(k) for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]} | {"model": model})
ml = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
display(ml)
print("Best ROC-AUC:", ml.iloc[0]["model"], ml.iloc[0]["roc_auc"])
"""
            ),
            code(
                r"""
from IPython.display import Image, display
display(Image(filename=str(FIGURES / "model_comparison_NR_AhR.png")))
display(Image(filename=str(FIGURES / "roc_curves_NR_AhR.png")))
"""
            ),
        ],
    )

    write_notebook(
        "06_hyperparameter_tuning.ipynb",
        "06 - Hyperparameter Tuning",
        [
            md("Summarize the class-weighted and cross-validated training strategy used by the pipeline."),
            code(
                r"""
df = pd.read_csv(DATA_RAW / "tox21.csv")
y = df["NR-AhR"].dropna().astype(int).to_numpy()
n_pos = int(y.sum())
n_neg = int((y == 0).sum())
class_weights = {
    0: len(y) / (2 * n_neg),
    1: len(y) / (2 * n_pos),
}
print("NR-AhR labeled:", len(y))
print("Positive:", n_pos, "Negative:", n_neg)
print("Balanced class weights:", class_weights)
print("Pipeline uses stratified splitting and 5-fold CV for primary Morgan models.")
"""
            ),
            code(
                r"""
with open(REPORTS / "NR_AhR_morgan_metrics.json") as f:
    metrics = json.load(f)
cv_rows = []
for name, m in metrics.items():
    cv_rows.append({
        "model": name,
        "cv_f1_mean": m.get("cv_f1_mean"),
        "cv_f1_std": m.get("cv_f1_std"),
        "cv_roc_auc_mean": m.get("cv_roc_auc_mean"),
        "cv_roc_auc_std": m.get("cv_roc_auc_std"),
    })
display(pd.DataFrame(cv_rows).dropna(how="all", subset=["cv_f1_mean", "cv_roc_auc_mean"]))
"""
            ),
        ],
    )

    write_notebook(
        "07_deep_learning.ipynb",
        "07 - Deep Learning",
        [
            md("Inspect the ToxNet artifact and the training history figure."),
            code(
                r"""
toxnet_path = MODELS / "NR_AhR_morgan_ToxNet.pt"
print("ToxNet exists:", toxnet_path.exists(), toxnet_path)
final = pd.read_csv(REPORTS / "final_model_comparison.csv")
display(final[final["model"] == "ToxNet"])
"""
            ),
            code(
                r"""
from IPython.display import Image, display
display(Image(filename=str(FIGURES / "training_history_NR_AhR.png")))
"""
            ),
        ],
    )

    write_notebook(
        "08_explainability.ipynb",
        "08 - Explainability",
        [
            md("Review SHAP global and local explanations generated by the pipeline."),
            code(
                r"""
shap_files = sorted(FIGURES.glob("shap_*.png"))
for path in shap_files:
    print(path.name)
top_path = REPORTS / "shap_top_features_NR_AhR.csv"
print("Top feature CSV exists:", top_path.exists())
if top_path.exists():
    display(pd.read_csv(top_path).head(15))
"""
            ),
            code(
                r"""
from IPython.display import Image, display
display(Image(filename=str(FIGURES / "shap_bar_NR_AhR_desc_XGBoost.png")))
display(Image(filename=str(FIGURES / "shap_waterfall_NR_AhR_desc_XGBoost_s0.png")))
"""
            ),
        ],
    )

    write_notebook(
        "09_final_evaluation.ipynb",
        "09 - Final Evaluation",
        [
            md("Collect final metrics, deliverables, and automated test status."),
            code(
                r"""
summary = pd.read_csv(REPORTS / "final_model_comparison.csv")
display(summary)
print("Best model by ROC-AUC:", summary.iloc[0]["model"], summary.iloc[0]["roc_auc"])
deliverables = {
    "raw_data": (DATA_RAW / "tox21.csv").exists(),
    "processed_features": (DATA_PROCESSED / "morgan_fp.npy").exists(),
    "models": len(list(MODELS.glob("*"))) > 0,
    "figures": len(list(FIGURES.glob("*.png"))),
    "api": (BASE_DIR / "api" / "main.py").exists(),
    "streamlit": (BASE_DIR / "webapp" / "app.py").exists(),
    "latex": (BASE_DIR / "latex" / "rapport_tox21.tex").exists(),
    "presentation": (REPORTS / "Tox21_Presentation.pptx").exists(),
}
display(pd.Series(deliverables, name="status"))
"""
            ),
            code(
                r"""
import subprocess, sys
result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=BASE_DIR, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
assert result.returncode == 0
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
