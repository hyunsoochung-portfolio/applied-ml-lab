"""Eval-only: rerun the cross_validate benchmark and print sorted results."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_models  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_clean()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for name, mdl in build_models().items():
        res = cross_validate(mdl, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        rows.append((name, res["test_score"].mean()))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, score in rows:
        print(f"{name:<22s}  AUC={score:.4f}")


if __name__ == "__main__":
    main()
