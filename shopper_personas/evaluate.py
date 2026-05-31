"""Eval-only: print inertia and silhouette for a chosen k."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import set_seed  # noqa: E402

from data import load_clean  # noqa: E402


def main(k: int = 5) -> None:
    set_seed(42)
    X, _ = load_clean()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    print(f"k={k}  inertia={km.inertia_:.2f}  silhouette={silhouette_score(Xs, km.labels_):.4f}")


if __name__ == "__main__":
    main()
