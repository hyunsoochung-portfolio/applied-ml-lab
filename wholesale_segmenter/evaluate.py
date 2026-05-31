"""Re-run accuracy table only."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build, scalers  # noqa: E402


def main() -> None:
    set_seed(42)
    X, y = load_clean()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    for name, scaler in scalers().items():
        acc = accuracy_score(yte, build(scaler).fit(Xtr, ytr).predict(Xte))
        print(f"{name:>8s}: {acc:.4f}")


if __name__ == "__main__":
    main()
