"""Heart Disease UCI loader.

Downloads the Cleveland processed file once, caches under ./data/, and
returns a cleaned binary-classification dataset.

Target: presence of heart disease (column ``num`` > 0 -> 1, else 0).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

# allow `from shared.utils import ...` when running `python train.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def _download(dest: Path) -> None:
    print(f"[data] downloading -> {dest}")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def load_raw() -> pd.DataFrame:
    cache = data_cache_dir(__file__) / "processed.cleveland.data"
    if not cache.exists():
        _download(cache)
    df = pd.read_csv(cache, header=None, names=COLUMNS, na_values="?")
    return df


def load_clean() -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with NaNs dropped and y binarised."""
    df = load_raw().dropna().reset_index(drop=True)
    y = (df["num"] > 0).astype(int).rename("target")
    X = df.drop(columns=["num"]).astype(float)
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print("X shape:", X.shape, " positive rate:", float(y.mean()))
    print(X.head())
