"""
data_loader.py
==============
Download and load the Tox21 dataset from MoleculeNet / DeepChem CDN.
Stores raw CSV files in data/raw/.
"""

import os
import hashlib
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# MoleculeNet / DeepChem CDN URLs for Tox21
URLS = {
    "tox21_train": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}

FALLBACK_URLS = {
    "tox21_train": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz",
}

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _download(url: str, dest: Path) -> bool:
    """Download a file from url to dest. Returns True on success."""
    try:
        logger.info(f"Downloading from {url}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        logger.info(f"Saved {downloaded:,} bytes to {dest}")
        return True
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return False


def download_tox21() -> pd.DataFrame:
    """Download Tox21 dataset, returning a DataFrame."""
    gz_path = RAW_DIR / "tox21.csv.gz"
    csv_path = RAW_DIR / "tox21.csv"

    if csv_path.exists():
        logger.info(f"Dataset already exists at {csv_path}. Loading cached version.")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
        return df

    # Try primary then fallback
    success = False
    for name, urls in [("primary", URLS), ("fallback", FALLBACK_URLS)]:
        url = urls["tox21_train"]
        if _download(url, gz_path):
            success = True
            break
        logger.warning(f"{name} URL failed, trying next...")

    if not success:
        raise RuntimeError("All download URLs failed. Check your internet connection.")

    # Read compressed file
    try:
        df = pd.read_csv(gz_path, compression="gzip")
    except Exception:
        df = pd.read_csv(gz_path)

    # Save uncompressed
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved uncompressed CSV: {csv_path}")
    logger.info(f"Dataset shape: {df.shape}")

    # Log basic info
    _log_dataset_info(df)
    return df


def _log_dataset_info(df: pd.DataFrame):
    """Log dataset summary."""
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Shape        : {df.shape}")
    logger.info(f"Columns      : {list(df.columns)}")
    logger.info(f"SMILES col   : {'smiles' in df.columns}")
    logger.info(f"Missing vals : {df.isnull().sum().sum():,}")

    tasks_present = [t for t in TOX21_TASKS if t in df.columns]
    logger.info(f"Tasks found  : {len(tasks_present)}/{len(TOX21_TASKS)}")
    for t in tasks_present:
        pos = int(df[t].sum())
        total_labeled = int(df[t].notna().sum())
        logger.info(f"  {t:20s}: {pos:4d} positive / {total_labeled:5d} labeled")
    logger.info("=" * 60)


def load_raw() -> pd.DataFrame:
    """Load the raw Tox21 CSV. Download if not present."""
    csv_path = RAW_DIR / "tox21.csv"
    if not csv_path.exists():
        return download_tox21()
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded raw data: {df.shape}")
    return df


if __name__ == "__main__":
    df = download_tox21()
    print(df.head())
    print(df.shape)
