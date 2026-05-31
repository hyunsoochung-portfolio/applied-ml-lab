"""Eval-only entry: retrains the best k quickly and reports test metrics."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_knn  # noqa: E402


def main(k: int = 11) -> None:
    set_seed(42)
    X, y = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipe = build_knn(n_neighbors=k).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"=== kNN (k={k}) evaluation ===")
    print("confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main()
