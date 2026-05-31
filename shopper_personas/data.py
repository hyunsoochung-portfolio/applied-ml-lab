"""Mall Customers loader from a public mirror."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/"
    "master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20"
    "Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
)


def _download(dest):
    print(f"[data] downloading -> {dest}")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def load_clean():
    cache = data_cache_dir(__file__) / "Mall_Customers.csv"
    if not cache.exists():
        _download(cache)
    df = pd.read_csv(cache)
    # Use the canonical 2-D clustering features for the headline plot,
    # plus age + gender (encoded) for higher-d experiments.
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    extra = ["Age"]
    gender_col = "Genre" if "Genre" in df.columns else "Gender"
    df["Gender_male"] = (df[gender_col] == "Male").astype(int)
    X = df[features + extra + ["Gender_male"]].astype(float)
    return X, features


if __name__ == "__main__":
    X, headline = load_clean()
    print(X.shape, headline)
