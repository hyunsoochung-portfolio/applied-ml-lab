"""California Housing loader via sklearn (cached by sklearn itself)."""
from __future__ import annotations

from sklearn.datasets import fetch_california_housing


def load_clean():
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.astype(float)
    y = bunch.target.astype(float)
    return X, y


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, y.describe())
