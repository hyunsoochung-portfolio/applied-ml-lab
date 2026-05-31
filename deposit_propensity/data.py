"""UCI Bank Marketing loader.

Downloads the `bank.zip` archive and extracts `bank-full.csv` (the larger
of the two simple-form datasets in that zip). Optionally falls back to
`bank-additional-full.csv` from a separate archive if the first URL is
unreachable. Binary target `y` (yes/no -> 1/0); strong class imbalance.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


URL_BANK = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
URL_BANK_ADDITIONAL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
)


def _download_and_extract(url: str, target_name: str, out_dir: Path) -> Path:
    print(f"[data] downloading -> {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path = out_dir / target_name
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # the file we want may sit at the root or in a nested directory
        match = next(
            (n for n in zf.namelist() if n.endswith(target_name)), None
        )
        if match is None:
            raise FileNotFoundError(f"{target_name} not in {url}")
        with zf.open(match) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def load_raw() -> pd.DataFrame:
    cache_dir = data_cache_dir(__file__)
    bank_full = cache_dir / "bank-full.csv"
    if bank_full.exists():
        return pd.read_csv(bank_full, sep=";")
    try:
        path = _download_and_extract(URL_BANK, "bank-full.csv", cache_dir)
        return pd.read_csv(path, sep=";")
    except Exception as e:
        print(f"[data] primary archive failed ({e!r}); trying bank-additional")
        path = _download_and_extract(
            URL_BANK_ADDITIONAL, "bank-additional-full.csv", cache_dir
        )
        return pd.read_csv(path, sep=";")


def load_clean():
    df = load_raw()
    y = (df["y"].astype(str).str.strip() == "yes").astype(int).rename("target")
    X = df.drop(columns=["y"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, " positive rate:", float(y.mean()))
