"""Eval-only: re-run cross_validate with extra scorers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_classifier  # noqa: E402


def main() -> None:
    set_seed(0)
    X, y = load_clean()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    res = cross_validate(
        build_classifier(), X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
        return_train_score=True,
    )
    for k, v in res.items():
        if k.startswith("test_") or k.startswith("train_"):
            print(f"{k:>22s}: mean={np.mean(v):.4f}  std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
