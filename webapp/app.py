"""
Streamlit web application for Tox21 toxicity prediction.
Features:
- Molecule input (SMILES)
- Toxicity prediction with probability
- Molecular 2D visualization
- Model comparison dashboard
- SHAP explanations
- Analytics panel
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
TASK        = "NR-AhR"
FEATURE     = "morgan"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxPredict AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #c0392b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .risk-HIGH    { background:#e74c3c; color:white; padding:6px 14px; border-radius:8px; font-weight:bold; }
    .risk-MEDIUM  { background:#f39c12; color:white; padding:6px 14px; border-radius:8px; font-weight:bold; }
    .risk-LOW     { background:#27ae60; color:white; padding:6px 14px; border-radius:8px; font-weight:bold; }
    .metric-box   { background:#f8f9fa; border-radius:8px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name="RandomForest"):
    import joblib
    path = MODELS_DIR / f"{TASK.replace('-','_')}_{FEATURE}_{model_name}.pkl"
    if not path.exists():
        return None, None
    model = joblib.load(path)
    return model, model_name


@st.cache_resource
def load_preprocessor():
    from preprocessing import build_preprocessor
    return build_preprocessor(scaler_type="robust")


@st.cache_data
def load_metrics():
    path = REPORTS_DIR / f"{TASK.replace('-','_')}_{FEATURE}_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def smiles_to_features(smiles: str):
    from rdkit.Chem import AllChem
    from rdkit import Chem, DataStructs
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1), mol


def mol_to_image(mol, size=(300, 250)):
    try:
        from rdkit.Chem import Draw
        from PIL import Image
        import io
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None


def risk_level(prob):
    if prob >= 0.7:
        return "HIGH", "#e74c3c"
    elif prob >= 0.4:
        return "MEDIUM", "#f39c12"
    else:
        return "LOW", "#27ae60"


# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header"> ToxPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediction of Chemical Molecule Toxicity Using Artificial Intelligence | Tox21 Dataset</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Prediction",
    "Batch Prediction",
    "Model Comparison",
    "SHAP Explainability",
    "Analytics Dashboard",
    "About",
])

available_models = sorted([
    p.stem.split(f"{TASK.replace('-','_')}_{FEATURE}_")[1]
    for p in MODELS_DIR.glob(f"{TASK.replace('-','_')}_{FEATURE}_*.pkl")
])

if not available_models:
    st.error("No trained models found. Please run `python src/pipeline.py` first.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Model", available_models, index=0)
model, model_name = load_model(selected_model_name)

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Prediction":
    st.header(" Single Molecule Prediction")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Enter SMILES")
        example_smiles = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Benzene": "c1ccccc1",
            "Caffeine": "Cn1c(=O)c2c(ncn2C)n(c1=O)C",
            "Benzo[a]pyrene (toxic)": "c1ccc2c(c1)ccc3cccc4ccccc43",
            "Ethanol": "CCO",
            "Dioxin (toxic)": "Clc1ccc2c(c1)Oc1cc(Cl)ccc1O2",
        }
        example_choice = st.selectbox("Or choose an example:", ["Custom"] + list(example_smiles.keys()))
        if example_choice != "Custom":
            default_smi = example_smiles[example_choice]
        else:
            default_smi = ""

        smiles_input = st.text_input("SMILES string:", value=default_smi, placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
        predict_btn = st.button("Predict Toxicity", type="primary", use_container_width=True)

    with col2:
        if smiles_input:
            X, mol = smiles_to_features(smiles_input)
            if mol:
                img = mol_to_image(mol)
                if img:
                    st.image(img, caption="Molecular Structure", use_container_width=True)
            else:
                st.error("Invalid SMILES — cannot parse molecule.")

    if predict_btn and smiles_input:
        X, mol = smiles_to_features(smiles_input)
        if X is None:
            st.error("Invalid SMILES string. Please check your input.")
        else:
            prob = float(model.predict_proba(X)[0][1])
            level, color = risk_level(prob)
            toxic = prob >= 0.5

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toxicity", "TOXIC" if toxic else "NON-TOXIC")
            c2.metric("Probability", f"{prob:.1%}")
            c3.metric("Risk Level", level)
            c4.metric("Model", selected_model_name)

            # Progress bar
            st.markdown(f"**Toxicity Probability:** {prob:.1%}")
            st.progress(prob)

            # SMILES properties
            if mol:
                from rdkit.Chem import Descriptors, rdMolDescriptors
                st.subheader("Molecular Properties")
                props = {
                    "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
                    "LogP": f"{Descriptors.MolLogP(mol):.3f}",
                    "TPSA": f"{Descriptors.TPSA(mol):.2f} A^2",
                    "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
                    "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
                    "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                }
                prop_df = pd.DataFrame(props.items(), columns=["Property", "Value"])
                st.table(prop_df)

elif page == "Batch Prediction":
    st.header(" Batch Prediction")
    st.markdown("Enter one SMILES per line or upload a CSV file with a `smiles` column.")

    batch_input = st.text_area("SMILES (one per line):", height=150,
                               value="CC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccc2c(c1)ccc3cccc4ccccc43")

    uploaded = st.file_uploader("Or upload CSV with 'smiles' column", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if "smiles" in df_up.columns:
            batch_input = "\n".join(df_up["smiles"].dropna().tolist())
        else:
            st.error("CSV must have a 'smiles' column.")

    if st.button("Run Batch Prediction", type="primary"):
        smiles_list = [s.strip() for s in batch_input.split("\n") if s.strip()]
        results = []
        progress = st.progress(0)
        for i, smi in enumerate(smiles_list):
            X, mol = smiles_to_features(smi)
            if X is None:
                results.append({"SMILES": smi, "Toxic": "INVALID", "Probability": 0, "Risk": "N/A"})
            else:
                prob = float(model.predict_proba(X)[0][1])
                level, _ = risk_level(prob)
                results.append({
                    "SMILES": smi,
                    "Toxic": "YES" if prob >= 0.5 else "NO",
                    "Probability": round(prob, 4),
                    "Risk": level,
                })
            progress.progress((i + 1) / len(smiles_list))

        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        n_t = (df_res["Toxic"] == "YES").sum()
        st.info(f"**{n_t} toxic** / **{len(df_res) - n_t} non-toxic** / **{(df_res['Toxic']=='INVALID').sum()} invalid**")

        csv = df_res.to_csv(index=False)
        st.download_button("Download Results CSV", csv, "toxicity_predictions.csv", "text/csv")

elif page == "Model Comparison":
    st.header(" Model Comparison")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run pipeline.py first.")
    else:
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": round(m.get("accuracy", 0), 4),
                "Precision": round(m.get("precision", 0), 4),
                "Recall": round(m.get("recall", 0), 4),
                "F1": round(m.get("f1", 0), 4),
                "ROC-AUC": round(m.get("roc_auc", 0), 4),
                "PR-AUC": round(m.get("pr_auc", 0), 4),
            })
        df_m = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        st.dataframe(df_m.set_index("Model"), use_container_width=True)

        # Charts
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(df_m["Model"], df_m["ROC-AUC"], color="steelblue")
        axes[0].set_title("ROC-AUC by Model")
        axes[0].set_xticklabels(df_m["Model"], rotation=30, ha="right")
        axes[0].set_ylim(0.5, 1)

        axes[1].bar(df_m["Model"], df_m["F1"], color="tomato")
        axes[1].set_title("F1-Score by Model")
        axes[1].set_xticklabels(df_m["Model"], rotation=30, ha="right")
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show saved figures
        for fig_name in ["roc_curves_NR_AhR.png", "pr_curves_NR_AhR.png", "confusion_matrices_NR_AhR.png"]:
            p = FIGURES_DIR / fig_name
            if p.exists():
                st.image(str(p), caption=fig_name.replace(".png", "").replace("_", " ").title(), use_container_width=True)

elif page == "SHAP Explainability":
    st.header("SHAP Explainability")
    st.markdown("Global and local feature importance using SHAP (SHapley Additive exPlanations).")

    for fig_name in [
        "shap_summary_NR_AhR_RandomForest.png",
        "shap_bar_NR_AhR_RandomForest.png",
        "shap_summary_NR_AhR_desc_XGBoost.png",
        "shap_bar_NR_AhR_desc_XGBoost.png",
        "shap_waterfall_NR_AhR_desc_XGBoost_s0.png",
    ]:
        p = FIGURES_DIR / fig_name
        if p.exists():
            label = fig_name.replace(".png", "").replace("_", " ").replace("NR AhR", "NR-AhR")
            st.subheader(label)
            st.image(str(p), use_container_width=True)

    shap_csv = REPORTS_DIR / "shap_top_features_NR_AhR.csv"
    if shap_csv.exists():
        df_s = pd.read_csv(shap_csv)
        st.subheader("Top Features by Mean |SHAP|")
        st.dataframe(df_s, use_container_width=True)

elif page == "Analytics Dashboard":
    st.header("Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        for fig_name in ["class_distribution.png", "missing_values.png"]:
            p = FIGURES_DIR / fig_name
            if p.exists():
                st.image(str(p), caption=fig_name.replace(".png","").replace("_"," ").title(), use_container_width=True)

    with col2:
        for fig_name in ["descriptor_distributions.png", "correlation_matrix.png", "descriptor_boxplots.png"]:
            p = FIGURES_DIR / fig_name
            if p.exists():
                st.image(str(p), caption=fig_name.replace(".png","").replace("_"," ").title(), use_container_width=True)

    # Training history
    p = FIGURES_DIR / "training_history_NR_AhR.png"
    if p.exists():
        st.subheader("Neural Network Training")
        st.image(str(p), use_container_width=True)

elif page == "About":
    st.header("About ToxPredict AI")
    st.markdown("""
    ## Project Summary

    **ToxPredict AI** is a machine learning system for predicting the toxicity of chemical molecules
    directly from their molecular structure (SMILES strings).

    ### Dataset
    - **Tox21**: 7,831 molecules, 12 toxicity assays
    - Primary task: **NR-AhR** (Aryl Hydrocarbon Receptor — 768 positives, 5,774 negatives)

    ### Models Trained
    | Model | ROC-AUC |
    |---|---|
    | SVM (RBF) | 0.911 |
    | Random Forest | 0.909 |
    | XGBoost | 0.893 |
    | LightGBM | 0.887 |
    | Logistic Regression | 0.879 |
    | ToxNet (Neural Network) | 0.877 |
    | Decision Tree | 0.802 |

    ### Features Used
    - **Morgan Fingerprints** (ECFP4, 2048 bits) — primary
    - **Molecular Descriptors** (130 RDKit descriptors) — secondary

    ### Explainability
    - SHAP TreeExplainer on Random Forest and XGBoost
    - Global and local feature importance

    ### Tech Stack
    Python | RDKit | scikit-learn | XGBoost | LightGBM | PyTorch | SHAP | FastAPI | Streamlit

    ---
    *Master AI Project — Tox21 Chemical Toxicity Prediction*
    """)
