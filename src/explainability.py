"""
explainability.py
=================
SHAP-based explainability: global importance, local explanations,
summary plots, waterfall plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional

import shap
import joblib

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    feature_names: list,
    model_name: str = "model",
    task: str = "task",
    max_samples: int = 500,
):
    """
    Compute SHAP values using tree or kernel explainer.
    Returns (explainer, shap_values).
    """
    logger.info(f"Computing SHAP values for {model_name} | Task={task}")

    # Subsample background for speed
    n_bg = min(100, len(X_train))
    bg = shap.sample(X_train, n_bg)

    model_type = type(model).__name__
    try:
        if model_type in ("RandomForestClassifier", "XGBClassifier",
                          "LGBMClassifier", "DecisionTreeClassifier",
                          "GradientBoostingClassifier"):
            explainer = shap.TreeExplainer(model, bg)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, bg, link="logit"
            )

        # Subsample test set for speed
        n_test = min(max_samples, len(X_test))
        X_sub = X_test[:n_test]

        sv = explainer.shap_values(X_sub)

        # Handle list output (binary classification → class 1)
        if isinstance(sv, list):
            sv = sv[1]

        # Handle 3D array (n_samples, n_features, n_classes) → take class 1
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]

        logger.info(f"SHAP values computed: {sv.shape}")
        return explainer, sv, X_sub

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return None, None, None


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sub: np.ndarray,
    feature_names: list,
    model_name: str,
    task: str,
    top_n: int = 20,
):
    """Generate and save SHAP summary (beeswarm) plot."""
    if shap_values is None:
        return

    fname = FIGURES_DIR / f"shap_summary_{task.replace('-','_')}_{model_name}.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sub,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {model_name} | {task}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved → {fname.name}")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list,
    model_name: str,
    task: str,
    top_n: int = 20,
):
    """Generate and save global SHAP bar chart (mean |SHAP|)."""
    if shap_values is None:
        return

    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values   = mean_abs[idx]

    fname = FIGURES_DIR / f"shap_bar_{task.replace('-','_')}_{model_name}.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features[::-1], top_values[::-1], color="#e05252")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Global Feature Importance — {model_name} | {task}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP bar plot saved → {fname.name}")


def plot_shap_waterfall(
    explainer,
    shap_values: np.ndarray,
    X_sub: np.ndarray,
    feature_names: list,
    model_name: str,
    task: str,
    sample_idx: int = 0,
):
    """Generate and save SHAP waterfall plot for a single prediction."""
    if shap_values is None or explainer is None:
        return

    fname = FIGURES_DIR / f"shap_waterfall_{task.replace('-','_')}_{model_name}_s{sample_idx}.png"
    try:
        expl_obj = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
            data=X_sub[sample_idx],
            feature_names=feature_names,
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(expl_obj, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {model_name} | {task} | Sample {sample_idx}", fontsize=12)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP waterfall saved → {fname.name}")
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return DataFrame of top features sorted by mean |SHAP|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":        feature_names,
        "mean_abs_shap":  mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df
