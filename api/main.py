"""
FastAPI application for Tox21 toxicity prediction.
Endpoints: GET /health, POST /predict, POST /batch_predict,
           GET /model_info, GET /metrics
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import joblib
from rdkit import Chem
from molecular_features import process_dataframe
import pandas as pd

app = FastAPI(
    title="Tox21 Toxicity Prediction API",
    description="Predict chemical toxicity from SMILES strings using ML models trained on Tox21.",
    version="1.0.0",
)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
TASK        = "NR-AhR"
FEATURE     = "morgan"

# ── Model loading ─────────────────────────────────────────────────────────────
_model = None
_preprocessor = None
_metrics = None


def load_model():
    global _model, _preprocessor
    # Load best model (RandomForest by default as good balance of metrics)
    for model_name in ["RandomForest", "LightGBM", "LogisticRegression"]:
        model_path = MODELS_DIR / f"{TASK.replace('-','_')}_{FEATURE}_{model_name}.pkl"
        if model_path.exists():
            _model = joblib.load(model_path)
            print(f"Loaded model: {model_name} from {model_path.name}")
            return model_name
    # Try to find any matching model
    matches = list(MODELS_DIR.glob(f"{TASK.replace('-','_')}_{FEATURE}_*.pkl"))
    if matches:
        _model = joblib.load(matches[0])
        name = matches[0].stem.split(f"{TASK.replace('-','_')}_{FEATURE}_")[1]
        print(f"Loaded fallback model: {name}")
        return name
    raise FileNotFoundError("No trained model found. Run pipeline.py first.")


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        from preprocessing import build_preprocessor
        # Re-fit a basic preprocessor (identity for binary FP)
        _preprocessor = build_preprocessor(scaler_type="robust")
    return _preprocessor


def load_metrics():
    global _metrics
    metrics_path = REPORTS_DIR / f"{TASK.replace('-','_')}_{FEATURE}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics = json.load(f)
    return _metrics


# Load on startup
_loaded_model_name = None


@app.on_event("startup")
def startup_event():
    global _loaded_model_name
    try:
        _loaded_model_name = load_model()
        load_metrics()
        print("API startup complete.")
    except Exception as e:
        print(f"Warning: {e}")


# ── Request / Response Models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str
    model_name: Optional[str] = None  # optional model override

class PredictResponse(BaseModel):
    smiles: str
    toxic: bool
    probability: float
    risk_level: str
    model_used: str
    task: str

class BatchPredictRequest(BaseModel):
    smiles_list: List[str]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    n_toxic: int
    n_nontoxic: int


def smiles_to_features(smiles: str) -> np.ndarray:
    """Convert a SMILES string to Morgan fingerprint features."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def predict_single(smiles: str, model=None):
    if model is None:
        model = _model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    X = smiles_to_features(smiles)
    prob = float(model.predict_proba(X)[0][1])
    toxic = prob >= 0.5
    return prob, toxic


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_name": _loaded_model_name,
        "task": TASK,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        prob, toxic = predict_single(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        smiles=request.smiles,
        toxic=toxic,
        probability=round(prob, 4),
        risk_level=risk_label(prob),
        model_used=_loaded_model_name or "unknown",
        task=TASK,
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    results = []
    for smi in request.smiles_list:
        try:
            prob, toxic = predict_single(smi)
            results.append(PredictResponse(
                smiles=smi,
                toxic=toxic,
                probability=round(prob, 4),
                risk_level=risk_label(prob),
                model_used=_loaded_model_name or "unknown",
                task=TASK,
            ))
        except ValueError:
            results.append(PredictResponse(
                smiles=smi,
                toxic=False,
                probability=0.0,
                risk_level="INVALID_SMILES",
                model_used="none",
                task=TASK,
            ))

    n_toxic    = sum(1 for r in results if r.toxic)
    n_nontoxic = len(results) - n_toxic
    return BatchPredictResponse(results=results, n_toxic=n_toxic, n_nontoxic=n_nontoxic)


@app.get("/model_info")
def model_info():
    models_available = [
        p.name for p in MODELS_DIR.glob(f"{TASK.replace('-','_')}_{FEATURE}_*.pkl")
    ]
    return {
        "task": TASK,
        "feature_type": FEATURE,
        "model_loaded": _loaded_model_name,
        "available_models": models_available,
        "input": "SMILES string",
        "output": "toxicity probability (0–1) for task NR-AhR",
    }


@app.get("/metrics")
def get_metrics():
    if _metrics is None:
        raise HTTPException(status_code=404, detail="Metrics not found. Run pipeline.py first.")
    # Return simplified metrics (exclude large curve data)
    simplified = {}
    for model_name, m in _metrics.items():
        simplified[model_name] = {k: v for k, v in m.items()
                                  if k not in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall")}
    return simplified
