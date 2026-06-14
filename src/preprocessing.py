"""
preprocessing.py
================
Data cleaning, imputation, scaling, feature selection, and train/test split.
All operations are pipeline-safe to prevent data leakage.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def remove_duplicates(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Remove duplicate SMILES entries, keeping first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[smiles_col], keep="first").reset_index(drop=True)
    after = len(df)
    logger.info(f"Duplicate removal: {before:,} → {after:,} ({before - after:,} removed)")
    return df


def filter_valid_smiles(df: pd.DataFrame, valid_mask: np.ndarray) -> pd.DataFrame:
    """Keep only rows corresponding to valid molecules."""
    df_valid = df[valid_mask].reset_index(drop=True)
    logger.info(f"Valid molecule filter: {valid_mask.sum():,} / {len(valid_mask):,} kept")
    return df_valid


def prepare_target(
    df: pd.DataFrame,
    task: str,
    min_samples: int = 50
) -> Optional[pd.Series]:
    """
    Prepare binary target for a single Tox21 task.
    Returns None if too few labeled samples.
    """
    if task not in df.columns:
        logger.warning(f"Task '{task}' not found in DataFrame.")
        return None
    y = df[task].dropna()
    if len(y) < min_samples:
        logger.warning(f"Task '{task}' has only {len(y)} labeled samples. Skipping.")
        return None
    return df[task]


def build_preprocessor(scaler_type: str = "robust") -> Pipeline:
    """
    Build a sklearn pipeline for descriptor-based features:
    Impute → Remove zero-variance → Scale
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    pipe = Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("var_filter", VarianceThreshold(threshold=0.0)),
        ("scaler",   scaler),
    ])
    return pipe


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split on labeled samples only."""
    # Keep only labeled rows
    labeled_mask = ~np.isnan(y.astype(float))
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask].astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_labeled, y_labeled))

    X_train = X_labeled[train_idx]
    X_test  = X_labeled[test_idx]
    y_train = y_labeled[train_idx]
    y_test  = y_labeled[test_idx]

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train pos rate: {y_train.mean():.3f} | Test pos rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def get_class_weight(y: np.ndarray) -> dict:
    """Compute balanced class weights for imbalanced targets."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def select_features_kbest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 100,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Select top-k features by ANOVA F-score."""
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    logger.info(f"Feature selection: {X_train.shape[1]} → {X_train_sel.shape[1]}")
    return X_train_sel, X_test_sel, selector


def save_processed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    feature_type: str = "descriptors",
):
    """Save processed arrays to disk."""
    prefix = PROCESSED_DIR / f"{task.replace('-','_')}_{feature_type}"
    np.save(f"{prefix}_X_train.npy", X_train)
    np.save(f"{prefix}_X_test.npy",  X_test)
    np.save(f"{prefix}_y_train.npy", y_train)
    np.save(f"{prefix}_y_test.npy",  y_test)
    logger.info(f"Saved processed arrays for task={task}, features={feature_type}")


def load_processed(task: str, feature_type: str = "descriptors"):
    """Load previously saved processed arrays."""
    prefix = PROCESSED_DIR / f"{task.replace('-','_')}_{feature_type}"
    X_train = np.load(f"{prefix}_X_train.npy")
    X_test  = np.load(f"{prefix}_X_test.npy")
    y_train = np.load(f"{prefix}_y_train.npy")
    y_test  = np.load(f"{prefix}_y_test.npy")
    return X_train, X_test, y_train, y_test


def describe_imbalance(y: np.ndarray, task: str = ""):
    """Log class distribution statistics."""
    y_clean = y[~np.isnan(y.astype(float))].astype(int)
    n_pos = int(y_clean.sum())
    n_neg = int((y_clean == 0).sum())
    ratio = n_pos / max(n_neg, 1)
    logger.info(
        f"Task={task} | Positive={n_pos} ({ratio:.2%}) | "
        f"Negative={n_neg} | Total={len(y_clean)}"
    )
    return {"positive": n_pos, "negative": n_neg, "ratio": ratio, "total": len(y_clean)}
