"""UCI Mushroom loader (all-categorical, binary edible/poisonous)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import data_cache_dir  # noqa: E402


URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mushroom/agaricus-lepiota.data"
)

COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]


def load_clean():
    cache = data_cache_dir(__file__) / "agaricus-lepiota.data"
    if not cache.exists():
        print(f"[data] downloading -> {cache}")
        r = requests.get(URL, timeout=60); r.raise_for_status()
        cache.write_bytes(r.content)
    df = pd.read_csv(cache, header=None, names=COLUMNS, na_values="?")
    # 'stalk-root' has '?' missing entries; drop them
    df = df.dropna().reset_index(drop=True)
    y = (df["class"] == "p").astype(int).rename("target")  # poisonous=1
    X = df.drop(columns=["class"])
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, " poisonous rate:", float(y.mean()))
