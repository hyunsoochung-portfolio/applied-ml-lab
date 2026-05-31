"""Breast Cancer Wisconsin loader (built-in sklearn dataset)."""
from __future__ import annotations

from sklearn.datasets import load_breast_cancer


def load_clean():
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data.astype(float), bunch.target.astype(int)


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, " positive rate:", float(y.mean()))
