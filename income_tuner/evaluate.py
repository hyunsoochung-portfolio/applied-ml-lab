"""Eval-only: 3-fold cross_validate ROC-AUC for a RF with sensible defaults."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_rf  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_clean()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    res = cross_validate(
        build_rf(n_estimators=200, max_depth=15, min_samples_leaf=4),
        X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    print(f"roc_auc = {res['test_score'].mean():.4f} ± {res['test_score'].std():.4f}")


if __name__ == "__main__":
    main()
