"""
deep_learning.py
================
PyTorch feedforward neural network for Tox21 toxicity prediction.
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ToxNet(nn.Module):
    """
    Feedforward neural network for binary toxicity prediction.
    Architecture: Input → BN → [512→256→128→64] → Output
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super(ToxNet, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """Create a sampler to handle class imbalance."""
    classes, counts = np.unique(y_train, return_counts=True)
    weights_per_class = 1.0 / counts
    sample_weights = np.array([weights_per_class[int(yi)] for yi in y_train])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train_toxnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    task:    str,
    feature_type: str = "descriptors",
    epochs:  int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[nn.Module, Dict]:
    """Train ToxNet with early stopping. Returns (model, metrics)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ToxNet on {device} | Task={task} | Input dim={X_train.shape[1]}")

    # Tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train.astype(float)).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.FloatTensor(y_test.astype(float)).to(device)

    # DataLoader with weighted sampling
    sampler = make_weighted_sampler(y_train)
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Model
    model = ToxNet(input_dim=X_train.shape[1]).to(device)

    # Pos weight for BCEWithLogitsLoss
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_roc_auc = 0.0
    best_state   = None
    no_improve   = 0
    history      = {"train_loss": [], "val_roc_auc": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_te)
            proba_val  = torch.sigmoid(logits_val).cpu().numpy()

        val_roc = roc_auc_score(y_test, proba_val)
        scheduler.step(1 - val_roc)

        history["train_loss"].append(epoch_loss)
        history["val_roc_auc"].append(val_roc)

        if val_roc > best_roc_auc:
            best_roc_auc = val_roc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | Loss={epoch_loss:.4f} | "
                f"Val ROC-AUC={val_roc:.4f} | Best={best_roc_auc:.4f}"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits_final = model(X_te)
        proba_final  = torch.sigmoid(logits_final).cpu().numpy()
        pred_final   = (proba_final >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    metrics = {
        "model":       "ToxNet",
        "accuracy":    float(accuracy_score(y_test, pred_final)),
        "precision":   float(precision_score(y_test, pred_final, zero_division=0)),
        "recall":      float(recall_score(y_test, pred_final, zero_division=0)),
        "f1":          float(f1_score(y_test, pred_final, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_test, proba_final)),
        "pr_auc":      float(average_precision_score(y_test, proba_final)),
        "train_time_sec": round(time.time() - t0, 2),
        "history":     history,
    }

    logger.info(
        f"ToxNet Final | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | PR-AUC={metrics['pr_auc']:.4f}"
    )

    # Save model
    model_path = MODELS_DIR / f"{task.replace('-','_')}_{feature_type}_ToxNet.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": X_train.shape[1]}, model_path)
    logger.info(f"ToxNet saved → {model_path.name}")

    return model, metrics


def load_toxnet(task: str, feature_type: str = "descriptors") -> Optional[nn.Module]:
    """Load a saved ToxNet model."""
    model_path = MODELS_DIR / f"{task.replace('-','_')}_{feature_type}_ToxNet.pt"
    if not model_path.exists():
        logger.warning(f"ToxNet model not found: {model_path}")
        return None
    ckpt = torch.load(model_path, map_location="cpu")
    model = ToxNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_toxnet(
    model: nn.Module,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference. Returns (y_pred, y_proba)."""
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        proba  = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= threshold).astype(int)
    return pred, proba
