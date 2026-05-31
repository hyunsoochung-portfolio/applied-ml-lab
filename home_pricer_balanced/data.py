"""California Housing loader."""
from __future__ import annotations

from sklearn.datasets import fetch_california_housing


def load_clean():
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data.astype(float), bunch.target.astype(float)


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape)
