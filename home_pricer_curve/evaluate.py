"""Eval-only: print test R² for the three polynomial degrees."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_poly  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_clean()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    for d in (1, 2, 3):
        r2 = r2_score(yte, build_poly(d).fit(Xtr, ytr).predict(Xte))
        print(f"degree={d}: test R²={r2:.4f}")


if __name__ == "__main__":
    main()
