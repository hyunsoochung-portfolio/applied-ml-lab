"""Re-use the Heart Disease loader from `cardio_screener`."""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import pandas as pd

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


def load_clean():
    cache = data_cache_dir(__file__) / "processed.cleveland.data"
    if not cache.exists():
        print(f"[data] downloading -> {cache}")
        r = requests.get(URL, timeout=60); r.raise_for_status()
        cache.write_bytes(r.content)
    df = pd.read_csv(cache, header=None, names=COLUMNS, na_values="?").dropna()
    y = (df["num"] > 0).astype(int).rename("target")
    X = df.drop(columns=["num"]).astype(float).reset_index(drop=True)
    return X, y.reset_index(drop=True)


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, float(y.mean()))
