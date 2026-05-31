"""Telco Customer Churn loader (IBM mirror, falls back to OpenML)."""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


CSV_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def _download_csv() -> pd.DataFrame:
    cache = data_cache_dir(__file__) / "telco_churn.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print(f"[data] downloading -> {cache}")
    r = requests.get(CSV_URL, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return pd.read_csv(StringIO(r.text))


def load_raw() -> pd.DataFrame:
    try:
        return _download_csv()
    except Exception as e:
        print(f"[data] CSV mirror failed ({e!r}); falling back to OpenML")
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml("telco-customer-churn", as_frame=True)
        return pd.concat([bunch.data, bunch.target.rename("Churn")], axis=1)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 'TotalCharges' arrives as string with blanks for new customers
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df = df.dropna().reset_index(drop=True)
    return df


def load_binary():
    """Return (X, y) for binary churn prediction with one-hot features."""
    df = _prepare(load_raw())
    y = (df["Churn"].astype(str).str.lower() == "yes").astype(int).rename("target")
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y


def load_multiclass():
    """3-class tenure-bucket target derived from `tenure` (short/mid/long).

    Buckets at quantiles 1/3 and 2/3 of `tenure`. Drops `tenure` and
    `Churn` from features so the task isn't trivially leaky.
    """
    df = _prepare(load_raw())
    q1, q2 = df["tenure"].quantile([1 / 3, 2 / 3])
    bucket = pd.cut(
        df["tenure"], bins=[-0.1, q1, q2, df["tenure"].max() + 1],
        labels=[0, 1, 2],
    ).astype(int).rename("target")
    X = df.drop(columns=["tenure", "Churn"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, bucket


if __name__ == "__main__":
    X, y = load_binary()
    print("binary :", X.shape, " positive rate:", float(y.mean()))
    Xm, ym = load_multiclass()
    print("multi  :", Xm.shape, " class counts:", ym.value_counts().to_dict())
