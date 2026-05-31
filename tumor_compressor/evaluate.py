"""Eval-only: PCA(5) + LogisticRegression accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_pca_logreg  # noqa: E402


def main(n: int = 5) -> None:
    set_seed(42)
    X, y = load_clean()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipe = build_pca_logreg(n).fit(Xtr, ytr)
    print(f"n_components={n}  test acc={accuracy_score(yte, pipe.predict(Xte)):.4f}")


if __name__ == "__main__":
    main()
