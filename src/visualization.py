"""
visualization.py
================
All EDA and model evaluation plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """Bar chart of positive rates for all Tox21 tasks."""
    tasks_present = [t for t in TOX21_TASKS if t in df.columns]
    pos_rates = []
    for t in tasks_present:
        col = df[t].dropna()
        pos_rates.append({"task": t, "positive_rate": col.mean(),
                          "n_labeled": len(col), "n_positive": int(col.sum())})
    pr_df = pd.DataFrame(pos_rates)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Positive rate
    sns.barplot(data=pr_df, x="task", y="positive_rate", ax=axes[0], palette="Reds_d")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Tox21 Positive Rate per Assay", fontsize=13)
    axes[0].set_ylabel("Proportion Positive")
    axes[0].set_xlabel("")

    # Sample counts
    pr_df_melt = pr_df.melt(id_vars="task", value_vars=["n_labeled", "n_positive"])
    sns.barplot(data=pr_df_melt, x="task", y="value", hue="variable", ax=axes[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("Labeled vs. Positive Counts per Assay", fontsize=13)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "class_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_missing_values(df: pd.DataFrame, save: bool = True):
    """Heatmap of missing values for Tox21 task columns."""
    tasks_present = [t for t in TOX21_TASKS if t in df.columns]
    miss = df[tasks_present].isnull()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    miss_pct = miss.mean() * 100
    sns.barplot(x=miss_pct.index, y=miss_pct.values, ax=axes[0], palette="Blues_d")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Missing Value % per Tox21 Task", fontsize=13)
    axes[0].set_ylabel("Missing %")

    # Subsample for heatmap visibility
    sample = miss.sample(min(300, len(miss)), random_state=42)
    sns.heatmap(sample.T, cbar=False, ax=axes[1], cmap="YlOrRd", yticklabels=True)
    axes[1].set_title("Missing Value Pattern (sample)", fontsize=13)
    axes[1].set_xlabel("Molecules")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "missing_values.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_descriptor_distributions(desc_df: pd.DataFrame, top_n: int = 12, save: bool = True):
    """Histograms of top molecular descriptors."""
    key_descs = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
                 "RingCount", "AromaticRings", "HeavyAtomCount",
                 "FractionCSP3", "QED", "BertzCT"]
    cols = [c for c in key_descs if c in desc_df.columns][:top_n]
    if not cols:
        cols = list(desc_df.select_dtypes(include=np.number).columns[:top_n])

    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = desc_df[col].dropna()
        q1, q99 = data.quantile(0.01), data.quantile(0.99)
        data_clipped = data.clip(q1, q99)
        axes[i].hist(data_clipped, bins=40, color="#4e8cc2", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution of Molecular Descriptors", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "descriptor_distributions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_correlation_matrix(desc_df: pd.DataFrame, top_n: int = 25, save: bool = True):
    """Correlation heatmap of molecular descriptors."""
    num_cols = desc_df.select_dtypes(include=np.number).columns
    df_num = desc_df[num_cols].dropna(axis=1, how="all")

    # Select most variable features
    top_cols = df_num.std().nlargest(top_n).index
    corr = df_num[top_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, ax=ax, square=True, linewidths=0.3)
    ax.set_title("Molecular Descriptor Correlation Matrix (Top Variants)", fontsize=13)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "correlation_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_roc_curves(all_metrics: dict, task: str, save: bool = True):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

    for (name, m), color in zip(all_metrics.items(), colors):
        fpr = m.get("roc_fpr")
        tpr = m.get("roc_tpr")
        auc = m.get("roc_auc", 0)
        if fpr is not None:
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {task}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"roc_curves_{task.replace('-','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_pr_curves(all_metrics: dict, task: str, save: bool = True):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

    for (name, m), color in zip(all_metrics.items(), colors):
        prec = m.get("pr_precision")
        rec  = m.get("pr_recall")
        prauc = m.get("pr_auc", 0)
        if prec is not None:
            ax.plot(rec, prec, label=f"{name} (PR-AUC={prauc:.3f})", color=color, lw=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {task}", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"pr_curves_{task.replace('-','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_confusion_matrices(all_metrics: dict, task: str, save: bool = True):
    """Grid of confusion matrices for all models."""
    names = list(all_metrics.keys())
    n = len(names)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, name in enumerate(names):
        cm = np.array(all_metrics[name].get("confusion_matrix", [[0, 0], [0, 0]]))
        r, c = divmod(idx, n_cols)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=axes[r][c], cbar=False)
        axes[r][c].set_title(name, fontsize=10)
        axes[r][c].set_xlabel("Predicted")
        axes[r][c].set_ylabel("Actual")

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.suptitle(f"Confusion Matrices — {task}", fontsize=13)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"confusion_matrices_{task.replace('-','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_model_comparison(all_metrics: dict, task: str, save: bool = True):
    """Bar chart comparing models across all metrics."""
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for name, m in all_metrics.items():
        row = {"model": name}
        for k in metrics_keys:
            row[k] = m.get(k, 0)
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    palette = sns.color_palette("husl", len(df))

    for i, metric in enumerate(metrics_keys):
        ax = axes[i]
        bars = ax.bar(df["model"], df[metric], color=palette)
        ax.set_title(metric.upper().replace("_", "-"), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(df["model"], rotation=35, ha="right", fontsize=9)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6)

    plt.suptitle(f"Model Comparison — {task}", fontsize=14)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"model_comparison_{task.replace('-','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_training_history(history: dict, task: str, save: bool = True):
    """Plot neural network training loss and validation ROC-AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], color="#4e8cc2", lw=2)
    axes[0].set_title("ToxNet Training Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")

    axes[1].plot(history["val_roc_auc"], color="#e05252", lw=2)
    axes[1].set_title("ToxNet Validation ROC-AUC", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].axhline(0.5, color="gray", linestyle="--", lw=0.8)

    plt.suptitle(f"ToxNet Training — {task}", fontsize=13)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"training_history_{task.replace('-','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()


def plot_descriptor_boxplots(desc_df: pd.DataFrame, save: bool = True):
    """Boxplots of key descriptors to detect outliers."""
    key_descs = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
                 "RingCount", "AromaticRings", "HeavyAtomCount", "QED"]
    cols = [c for c in key_descs if c in desc_df.columns]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols[:10]):
        data = desc_df[col].dropna()
        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#add8e6", color="#2c6fad"))
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xticks([])
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Molecular Descriptor Boxplots (Outlier Detection)", fontsize=13)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "descriptor_boxplots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path.name}")
    plt.close()
