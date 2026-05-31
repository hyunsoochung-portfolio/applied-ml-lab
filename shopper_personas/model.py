"""KMeans factory with sensible defaults."""
from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_kmeans(n_clusters: int, random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("km", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
        ]
    )
