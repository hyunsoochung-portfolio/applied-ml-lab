"""Eval-only: binary churn accuracy and ROC-AUC."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_binary  # noqa: E402
from model import build_binary  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_binary()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = build_binary().fit(Xtr, ytr)
    print(f"accuracy = {accuracy_score(yte, pipe.predict(Xte)):.4f}")
    print(f"roc_auc  = {roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]):.4f}")


if __name__ == "__main__":
    main()
