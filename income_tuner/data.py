"""Adult Income loader via UCI direct URL (OpenML mirror has been unreliable)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_FILE = DATA_DIR / "adult.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "class",
]


def _download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return
    print(f"[data] downloading -> {DATA_FILE}")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    DATA_FILE.write_bytes(r.content)


def load_clean():
    _download()
    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values=["?"],
    )
    # target: binary income > 50K
    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).rename("target")
    X = df.drop(columns=["class"])
    mask = X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, float(y.mean()))
