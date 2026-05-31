"""Wholesale Customers (UCI) loader.

Predict the ``Channel`` column (1=Horeca/Restaurant, 2=Retail) -> binary
target {0,1} from annual spend across 6 product categories. The features
span very different magnitudes, which is ideal for showing scaling
effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00292/Wholesale%20customers%20data.csv"
)


def load_clean():
    cache = data_cache_dir(__file__) / "wholesale_customers.csv"
    if not cache.exists():
        print(f"[data] downloading -> {cache}")
        r = requests.get(URL, timeout=60); r.raise_for_status()
        cache.write_bytes(r.content)
    df = pd.read_csv(cache)
    y = (df["Channel"] == 2).astype(int).rename("target")
    X = df.drop(columns=["Channel", "Region"]).astype(float)
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, " positive rate:", float(y.mean()))
    print(X.describe().T[["mean", "std", "min", "max"]])
