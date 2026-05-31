"""California Housing loader (shared with `home_pricer_neighbor` but locally re-fetched)."""
from __future__ import annotations

from sklearn.datasets import fetch_california_housing


def load_clean():
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data.astype(float), bunch.target.astype(float)


if __name__ == "__main__":
    X, y = load_clean()
    print(X.shape, " y mean:", float(y.mean()))
