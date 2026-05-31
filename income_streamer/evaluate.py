"""Eval-only: one-shot SGD log-loss accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_sgd  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_clean()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = build_sgd("log_loss", max_iter=100).fit(Xtr, ytr)
    print(f"test acc = {accuracy_score(yte, pipe.predict(Xte)):.4f}")


if __name__ == "__main__":
    main()
