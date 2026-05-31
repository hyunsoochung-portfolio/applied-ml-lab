"""Eval-only: report test R² and MAE for a chosen k."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_knn  # noqa: E402


def main(k: int = 20) -> None:
    set_seed(42)
    X, y = load_clean()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = build_knn(k).fit(Xtr, ytr).predict(Xte)
    print(f"k={k}  R2={r2_score(yte, pred):.4f}  MAE={mean_absolute_error(yte, pred):.4f}")


if __name__ == "__main__":
    main()
