"""
pipeline.py
===========
Master pipeline: download → EDA → features → ML → DL → SHAP → report.
Run this script directly to execute the full project.
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import io

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SRC_DIR.parent / "reports" / "pipeline.log", mode="w", encoding="utf-8"),
    ]
)
# Safe console handler (replaces non-ASCII)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
class _SafeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg.encode("ascii", "replace").decode("ascii") + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
_safe_ch = _SafeHandler(sys.stdout)
_safe_ch.setLevel(logging.INFO)
_safe_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(_safe_ch)
logger = logging.getLogger(__name__)

BASE_DIR      = SRC_DIR.parent
FIGURES_DIR   = BASE_DIR / "reports" / "figures"
REPORTS_DIR   = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

# The primary task to run full ML pipeline on
PRIMARY_TASK = "NR-AhR"  # Most populated task in Tox21
FEATURE_TYPE = "morgan"  # Use Morgan FP as primary features (most informative)


def run():
    logger.info("=" * 70)
    logger.info("TOX21 TOXICITY PREDICTION - MASTER PIPELINE")
    logger.info("=" * 70)

    # ── PHASE 1: Download Dataset ─────────────────────────────────────────
    logger.info("\n[PHASE 1] Downloading Tox21 dataset...")
    from data_loader import download_tox21, TOX21_TASKS
    df_raw = download_tox21()
    logger.info(f"Raw dataset: {df_raw.shape}")

    # ── PHASE 2: EDA & Visualization ─────────────────────────────────────
    logger.info("\n[PHASE 2] Exploratory Data Analysis...")
    from visualization import (
        plot_class_distribution, plot_missing_values,
        plot_descriptor_distributions, plot_correlation_matrix,
        plot_descriptor_boxplots,
    )

    plot_class_distribution(df_raw)
    plot_missing_values(df_raw)

    # ── PHASE 3: Molecular Processing ────────────────────────────────────
    logger.info("\n[PHASE 3] Generating molecular descriptors and fingerprints...")
    from molecular_features import process_dataframe

    smiles_col = "smiles" if "smiles" in df_raw.columns else df_raw.columns[0]
    desc_df, morgan_arr, maccs_arr, rdkit_arr, valid_mask = process_dataframe(
        df_raw, smiles_col=smiles_col
    )

    # EDA on descriptors
    plot_descriptor_distributions(desc_df)
    plot_correlation_matrix(desc_df)
    plot_descriptor_boxplots(desc_df)

    # Save processed data
    desc_df.to_csv(PROCESSED_DIR / "descriptors.csv", index=False)
    np.save(PROCESSED_DIR / "morgan_fp.npy",   morgan_arr)
    np.save(PROCESSED_DIR / "maccs_fp.npy",    maccs_arr)
    np.save(PROCESSED_DIR / "rdkit_fp.npy",    rdkit_arr)
    np.save(PROCESSED_DIR / "valid_mask.npy",  valid_mask)
    logger.info("All feature arrays saved to data/processed/")

    # ── PHASE 4: Preprocessing ───────────────────────────────────────────
    logger.info("\n[PHASE 4] Preprocessing...")
    from preprocessing import (
        remove_duplicates, filter_valid_smiles,
        build_preprocessor, split_data, save_processed,
        describe_imbalance, get_class_weight,
    )

    df_valid = df_raw[valid_mask].reset_index(drop=True)
    # Align arrays with cleaned df (no duplicates in this dataset so keep all)
    df_clean = remove_duplicates(df_valid, smiles_col=smiles_col)
    # Build positional keep mask from df_valid after dedup
    keep_mask = df_valid[smiles_col].duplicated(keep="first") == False
    keep_positions = np.where(keep_mask.values)[0]

    desc_clean   = desc_df.iloc[keep_positions].reset_index(drop=True)
    morgan_clean = morgan_arr[keep_positions]
    maccs_clean  = maccs_arr[keep_positions]
    rdkit_clean  = rdkit_arr[keep_positions]
    df_clean = df_clean.reset_index(drop=True)

    logger.info(f"Clean dataset: {df_clean.shape}")

    # ── PHASE 5: ML Training (Primary Task) ──────────────────────────────
    logger.info(f"\n[PHASE 5] Training ML models | Task={PRIMARY_TASK}")
    from models import train_all_models, get_best_model
    from visualization import (
        plot_roc_curves, plot_pr_curves,
        plot_confusion_matrices, plot_model_comparison,
    )

    task = PRIMARY_TASK
    if task not in df_clean.columns:
        # fallback to first available task
        task = [t for t in TOX21_TASKS if t in df_clean.columns][0]
        logger.warning(f"PRIMARY_TASK not found, using: {task}")

    y_series = df_clean[task]

    # --- Morgan fingerprints ---
    X_fp = morgan_clean
    labeled_mask = ~y_series.isna().values
    X_labeled = X_fp[labeled_mask]
    y_labeled = y_series[labeled_mask].values.astype(int)

    describe_imbalance(y_labeled, task)

    # Preprocessor (FP don't need scaling but we do imputer+variance filter)
    preprocessor = build_preprocessor(scaler_type="robust")
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X_labeled, y_labeled))
    X_train_raw, X_test_raw = X_labeled[train_idx], X_labeled[test_idx]
    y_train, y_test = y_labeled[train_idx], y_labeled[test_idx]

    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_test  = preprocessor.transform(X_test_raw)

    save_processed(X_train, X_test, y_train, y_test, task, feature_type="morgan")

    trained_models, all_metrics = train_all_models(
        X_train, X_test, y_train, y_test,
        task=task, feature_type="morgan", do_cv=True
    )

    # Plots
    plot_roc_curves(all_metrics, task)
    plot_pr_curves(all_metrics, task)
    plot_confusion_matrices(all_metrics, task)
    plot_model_comparison(all_metrics, task)

    best_name = get_best_model(all_metrics, metric="roc_auc")

    # ── PHASE 6: Descriptor-based training ───────────────────────────────
    logger.info(f"\n[PHASE 5b] Training ML models with descriptors | Task={task}")
    desc_arr = desc_clean.values.astype(float)
    # Replace inf with nan so imputer handles them
    desc_arr = np.where(np.isinf(desc_arr), np.nan, desc_arr)
    X_desc_labeled = desc_arr[labeled_mask]

    X_desc_train_raw = X_desc_labeled[train_idx]
    X_desc_test_raw  = X_desc_labeled[test_idx]

    pre_desc = build_preprocessor(scaler_type="robust")
    X_desc_train = pre_desc.fit_transform(X_desc_train_raw, y_train)
    X_desc_test  = pre_desc.transform(X_desc_test_raw)

    save_processed(X_desc_train, X_desc_test, y_train, y_test, task, feature_type="descriptors")

    trained_desc, metrics_desc = train_all_models(
        X_desc_train, X_desc_test, y_train, y_test,
        task=task, feature_type="descriptors", do_cv=False
    )

    plot_model_comparison(metrics_desc, task=f"{task}_descriptors")

    # ── PHASE 7: Deep Learning ────────────────────────────────────────────
    logger.info(f"\n[PHASE 6] Training ToxNet (Deep Learning) | Task={task}")
    from deep_learning import train_toxnet
    from visualization import plot_training_history

    nn_model, nn_metrics = train_toxnet(
        X_train, y_train, X_test, y_test,
        task=task, feature_type="morgan",
        epochs=80, batch_size=256, lr=1e-3, patience=12,
    )
    plot_training_history(nn_metrics["history"], task)
    all_metrics["ToxNet"] = nn_metrics

    # Final combined comparison
    plot_roc_curves(all_metrics, task=f"{task}_all")
    plot_model_comparison(all_metrics, task=f"{task}_all")

    # ── PHASE 8: SHAP Explainability ─────────────────────────────────────
    logger.info(f"\n[PHASE 8] SHAP Explainability | Task={task}")
    from explainability import (
        compute_shap_values, plot_shap_summary,
        plot_shap_bar, plot_shap_waterfall, get_top_features,
    )

    # Use best tree model for SHAP (avoid slow KernelExplainer on SVM)
    shap_model_name = None
    for candidate in ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"]:
        if candidate in trained_models:
            shap_model_name = candidate
            break

    if shap_model_name:
        shap_model = trained_models[shap_model_name]
        feature_names = [f"morgan_{i}" for i in range(X_train.shape[1])]
        explainer, sv, X_sub = compute_shap_values(
            shap_model, X_train, X_test,
            feature_names=feature_names,
            model_name=shap_model_name, task=task,
        )
        if sv is not None:
            plot_shap_summary(sv, X_sub, feature_names, shap_model_name, task)
            plot_shap_bar(sv, feature_names, shap_model_name, task)
            plot_shap_waterfall(explainer, sv, X_sub, feature_names, shap_model_name, task)
            top_feats = get_top_features(sv, feature_names)
            top_feats.to_csv(REPORTS_DIR / f"shap_top_features_{task.replace('-','_')}.csv", index=False)

    # SHAP on descriptors (most interpretable — named features)
    desc_feature_names_all = list(desc_clean.columns)
    best_desc_name = get_best_model(metrics_desc, metric="roc_auc")
    # Prefer tree models for SHAP
    for candidate in ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression"]:
        if candidate in trained_desc:
            best_desc_name = candidate
            break
    best_desc_model = trained_desc.get(best_desc_name)
    if best_desc_model is not None:
        # feature names after variance filter — use all desc names (129 after imputer)
        n_desc_features = X_desc_train.shape[1]
        fn2 = desc_feature_names_all[:n_desc_features]
        exp2, sv2, X_sub2 = compute_shap_values(
            best_desc_model, X_desc_train, X_desc_test,
            feature_names=fn2,
            model_name=best_desc_name, task=f"{task}_desc",
        )
        if sv2 is not None:
            plot_shap_summary(sv2, X_sub2, fn2, best_desc_name, f"{task}_desc")
            plot_shap_bar(sv2, fn2, best_desc_name, f"{task}_desc")
            plot_shap_waterfall(exp2, sv2, X_sub2, fn2, best_desc_name, f"{task}_desc")

    # ── PHASE 9: Final Metrics Summary ───────────────────────────────────
    logger.info("\n[PHASE 9] Final Results Summary")
    summary_rows = []
    for name, m in all_metrics.items():
        summary_rows.append({
            "model":     name,
            "accuracy":  round(m.get("accuracy", 0), 4),
            "precision": round(m.get("precision", 0), 4),
            "recall":    round(m.get("recall", 0), 4),
            "f1":        round(m.get("f1", 0), 4),
            "roc_auc":   round(m.get("roc_auc", 0), 4),
            "pr_auc":    round(m.get("pr_auc", 0), 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
    summary_csv = REPORTS_DIR / "final_model_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nFINAL MODEL COMPARISON:\n{summary_df.to_string(index=False)}")
    logger.info(f"\nSaved → {summary_csv}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model: {best_name}")
    logger.info(f"All figures saved to: {FIGURES_DIR}")
    logger.info(f"All models saved to:  {MODELS_DIR}")
    logger.info(f"Summary CSV:          {summary_csv}")
    logger.info("=" * 70)

    return {
        "task": task,
        "best_model": best_name,
        "all_metrics": all_metrics,
        "summary_df": summary_df,
        "trained_models": trained_models,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "desc_clean": desc_clean,
        "df_clean": df_clean,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    run()
