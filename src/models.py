"""
models.py
=========
Train, evaluate, and save classical ML models on Tox21 tasks.
Models: Logistic Regression, Decision Tree, Random Forest,
        XGBoost, LightGBM, SVM.
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_configs(class_weight: dict = None, scale_pos_weight: float = 1.0) -> dict:
    """Return all model configurations."""
    cw = class_weight or {0: 1, 1: 1}
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight=cw,
            solver="lbfgs", random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, class_weight=cw, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight=cw,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            class_weight=cw, random_state=42, n_jobs=-1,
            verbose=-1
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight=cw, random_state=42
        ),
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a fitted model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "pr_auc":    float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics["roc_fpr"] = fpr.tolist()
    metrics["roc_tpr"] = tpr.tolist()

    # PR curve data
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    metrics["pr_precision"] = prec.tolist()
    metrics["pr_recall"]    = rec.tolist()

    logger.info(
        f"{model_name:20s} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    model_name: str = "",
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc", "average_precision"]
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    cv_results = {
        "cv_accuracy_mean":  float(scores["test_accuracy"].mean()),
        "cv_accuracy_std":   float(scores["test_accuracy"].std()),
        "cv_f1_mean":        float(scores["test_f1"].mean()),
        "cv_f1_std":         float(scores["test_f1"].std()),
        "cv_roc_auc_mean":   float(scores["test_roc_auc"].mean()),
        "cv_roc_auc_std":    float(scores["test_roc_auc"].std()),
        "cv_pr_auc_mean":    float(scores["test_average_precision"].mean()),
        "cv_pr_auc_std":     float(scores["test_average_precision"].std()),
    }
    logger.info(
        f"CV {model_name:20s} | "
        f"F1={cv_results['cv_f1_mean']:.4f}±{cv_results['cv_f1_std']:.4f} | "
        f"ROC-AUC={cv_results['cv_roc_auc_mean']:.4f}±{cv_results['cv_roc_auc_std']:.4f}"
    )
    return cv_results


def train_all_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    task:    str,
    feature_type: str = "descriptors",
    do_cv:   bool = True,
) -> Tuple[Dict, Dict]:
    """
    Train all models for a given task and feature set.
    Returns dicts: {model_name: trained_model}, {model_name: metrics}
    """
    from preprocessing import get_class_weight
    cw = get_class_weight(y_train)
    spw = cw.get(0, 1.0) / max(cw.get(1, 1.0), 1e-6)

    configs   = get_model_configs(class_weight=cw, scale_pos_weight=spw)
    all_metrics = {}
    trained   = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Training models | Task={task} | Features={feature_type}")
    logger.info(f"Train={X_train.shape} | Test={X_test.shape}")
    logger.info(f"Class weights: {cw}")
    logger.info(f"{'='*60}")

    for name, model in configs.items():
        logger.info(f"\n→ Training {name}...")
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            metrics = evaluate_model(model, X_test, y_test, model_name=name)
            metrics["train_time_sec"] = round(elapsed, 2)

            if do_cv and len(y_train) >= 200:
                cv_res = cross_validate_model(model, X_train, y_train, cv=5, model_name=name)
                metrics.update(cv_res)

            all_metrics[name] = metrics
            trained[name] = model

            # Save model
            model_path = MODELS_DIR / f"{task.replace('-','_')}_{feature_type}_{name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  Saved model → {model_path.name}")

        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    # Save metrics JSON
    metrics_path = REPORTS_DIR / f"{task.replace('-','_')}_{feature_type}_metrics.json"
    _metrics_serializable = {}
    for k, v in all_metrics.items():
        v2 = {kk: vv for kk, vv in v.items()
              if kk not in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall", "confusion_matrix")}
        _metrics_serializable[k] = v2

    with open(metrics_path, "w") as f:
        json.dump(_metrics_serializable, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path.name}")

    return trained, all_metrics


def get_best_model(all_metrics: dict, metric: str = "roc_auc") -> str:
    """Return name of best model by given metric."""
    best = max(all_metrics.keys(), key=lambda k: all_metrics[k].get(metric, 0))
    logger.info(f"Best model by {metric}: {best} = {all_metrics[best].get(metric, 0):.4f}")
    return best
