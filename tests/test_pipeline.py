"""
Automated test suite for the Tox21 ML pipeline.
Tests: data loading, RDKit, feature engineering, model loading, API, prediction.
"""

import sys
import json
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Data Loading ──────────────────────────────────────────────────────────────

def test_raw_data_exists():
    csv = BASE_DIR / "data" / "raw" / "tox21.csv"
    assert csv.exists(), "Raw dataset not found. Run pipeline.py first."

def test_raw_data_shape():
    import pandas as pd
    df = pd.read_csv(BASE_DIR / "data" / "raw" / "tox21.csv")
    assert len(df) > 7000, "Dataset should have > 7000 rows"
    assert "smiles" in df.columns, "Dataset must have 'smiles' column"

def test_tasks_present():
    import pandas as pd
    TOX21_TASKS = ["NR-AR", "NR-AhR", "SR-ARE", "SR-MMP"]
    df = pd.read_csv(BASE_DIR / "data" / "raw" / "tox21.csv")
    for task in TOX21_TASKS:
        assert task in df.columns, f"Task {task} missing"

# ── RDKit Molecular Processing ───────────────────────────────────────────────

def test_rdkit_import():
    from rdkit import Chem
    assert Chem is not None

def test_valid_smiles():
    from molecular_features import smiles_to_mol
    mol = smiles_to_mol("CCO")  # ethanol
    assert mol is not None

def test_invalid_smiles():
    from molecular_features import smiles_to_mol
    mol = smiles_to_mol("INVALIDSMILES###")
    assert mol is None

def test_descriptor_computation():
    from molecular_features import smiles_to_mol, compute_descriptors
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    desc = compute_descriptors(mol)
    assert "MolWt" in desc
    assert desc["MolWt"] > 100  # aspirin MW ~ 180
    assert "LogP" in desc

def test_morgan_fingerprint():
    from molecular_features import smiles_to_mol, morgan_fingerprint
    mol = smiles_to_mol("c1ccccc1")  # benzene
    fp = morgan_fingerprint(mol)
    assert fp.shape == (2048,)
    assert fp.sum() > 0

def test_maccs_fingerprint():
    from molecular_features import smiles_to_mol, maccs_fingerprint
    mol = smiles_to_mol("CCO")
    fp = maccs_fingerprint(mol)
    assert fp.shape == (167,)

# ── Feature Engineering ───────────────────────────────────────────────────────

def test_process_dataframe():
    import pandas as pd
    from molecular_features import process_dataframe
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O"]})
    desc_df, morgan, maccs, rdkit_fp, valid_mask = process_dataframe(df)
    assert len(desc_df) == 3
    assert morgan.shape == (3, 2048)
    assert maccs.shape == (3, 167)
    assert valid_mask.sum() == 3

def test_invalid_smiles_filtered():
    import pandas as pd
    from molecular_features import process_dataframe
    df = pd.DataFrame({"smiles": ["CCO", "INVALID###", "c1ccccc1"]})
    desc_df, morgan, maccs, rdkit_fp, valid_mask = process_dataframe(df)
    assert valid_mask.sum() == 2  # only 2 valid

# ── Model Loading ─────────────────────────────────────────────────────────────

def test_models_exist():
    models_dir = BASE_DIR / "models"
    pkl_files = list(models_dir.glob("*.pkl"))
    assert len(pkl_files) > 0, "No trained models found."

def test_load_best_model():
    import joblib
    models_dir = BASE_DIR / "models"
    model_path = models_dir / "NR_AhR_morgan_RandomForest.pkl"
    assert model_path.exists(), "RandomForest model not found"
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba")

def test_model_prediction_shape():
    import joblib
    model_path = BASE_DIR / "models" / "NR_AhR_morgan_RandomForest.pkl"
    model = joblib.load(model_path)
    X = np.random.rand(5, 2048)
    proba = model.predict_proba(X)
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_model_prediction_range():
    import joblib
    model_path = BASE_DIR / "models" / "NR_AhR_morgan_RandomForest.pkl"
    model = joblib.load(model_path)
    X = np.random.rand(10, 2048)
    proba = model.predict_proba(X)[:, 1]
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_preprocessor_fit_transform():
    from preprocessing import build_preprocessor
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 2, 100)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert Xt.shape[0] == 100

def test_class_weight_computation():
    from preprocessing import get_class_weight
    y = np.array([0, 0, 0, 1])
    cw = get_class_weight(y)
    assert 1 in cw
    assert cw[1] > cw[0]  # minority class should have higher weight

# ── Metrics Files ─────────────────────────────────────────────────────────────

def test_metrics_file_exists():
    metrics_path = BASE_DIR / "reports" / "NR_AhR_morgan_metrics.json"
    assert metrics_path.exists(), "Metrics JSON not found"

def test_metrics_content():
    metrics_path = BASE_DIR / "reports" / "NR_AhR_morgan_metrics.json"
    with open(metrics_path) as f:
        m = json.load(f)
    assert "RandomForest" in m or "LogisticRegression" in m
    for model_name, vals in m.items():
        assert "roc_auc" in vals
        assert 0 <= vals["roc_auc"] <= 1

# ── Figures ───────────────────────────────────────────────────────────────────

def test_figures_generated():
    figs_dir = BASE_DIR / "reports" / "figures"
    pngs = list(figs_dir.glob("*.png"))
    assert len(pngs) >= 5, f"Expected >= 5 figures, got {len(pngs)}"

def test_roc_curve_figure():
    fig = BASE_DIR / "reports" / "figures" / "roc_curves_NR_AhR.png"
    assert fig.exists()

def test_shap_figure():
    fig = BASE_DIR / "reports" / "figures" / "shap_bar_NR_AhR_RandomForest.png"
    assert fig.exists()

# ── Prediction Pipeline (end-to-end) ─────────────────────────────────────────

def test_end_to_end_prediction():
    """Full pipeline: SMILES -> features -> model -> probability"""
    import joblib
    from molecular_features import smiles_to_mol, morgan_fingerprint
    from rdkit import DataStructs
    import numpy as np

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = smiles_to_mol(smiles)
    assert mol is not None

    fp = morgan_fingerprint(mol)
    assert fp.shape == (2048,)

    model = joblib.load(BASE_DIR / "models" / "NR_AhR_morgan_RandomForest.pkl")
    X = fp.reshape(1, -1).astype(float)
    prob = model.predict_proba(X)[0][1]
    assert 0 <= prob <= 1

def test_known_toxic_compound():
    """Benzo[a]pyrene is a known AhR agonist (toxic)"""
    import joblib
    from molecular_features import smiles_to_mol, morgan_fingerprint

    # Benzo[a]pyrene - strong AhR agonist
    smiles = "c1ccc2c(c1)ccc3cccc4ccccc43"
    mol = smiles_to_mol(smiles)
    if mol is None:
        pytest.skip("SMILES parsing failed")

    fp = morgan_fingerprint(mol).reshape(1, -1).astype(float)
    model = joblib.load(BASE_DIR / "models" / "NR_AhR_morgan_RandomForest.pkl")
    prob = model.predict_proba(fp)[0][1]
    # Benzo[a]pyrene should be predicted as toxic
    assert prob > 0.3, f"Expected high probability for known toxic compound, got {prob:.3f}"

# ── API Tests (with test client) ──────────────────────────────────────────────

def test_api_health():
    try:
        import httpx
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(BASE_DIR / "api"))
        from main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    except ImportError:
        pytest.skip("httpx not available")

def test_api_predict():
    try:
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(BASE_DIR / "api"))
        from main import app, load_model
        load_model()  # ensure model loaded in test context
        client = TestClient(app)
        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 200
        data = response.json()
        assert "toxic" in data
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
    except ImportError:
        pytest.skip("httpx not available")

def test_api_predict_invalid():
    try:
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(BASE_DIR / "api"))
        from main import app, load_model
        load_model()
        client = TestClient(app)
        response = client.post("/predict", json={"smiles": "INVALID###"})
        assert response.status_code in (422, 200)  # either validation error or handled gracefully
    except ImportError:
        pytest.skip("httpx not available")

def test_api_metrics():
    try:
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(BASE_DIR / "api"))
        from main import app, load_metrics
        load_metrics()
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
    except ImportError:
        pytest.skip("httpx not available")

def test_api_model_info():
    try:
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(BASE_DIR / "api"))
        from main import app
        client = TestClient(app)
        response = client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "task" in data
        assert "available_models" in data
    except ImportError:
        pytest.skip("httpx not available")
