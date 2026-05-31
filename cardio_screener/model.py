"""kNN classifier factory for the Heart Disease task."""
from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn(n_neighbors: int = 5) -> Pipeline:
    """Standard scaler + kNN classifier pipeline.

    Scaling matters here because features (age, chol, thalach, ...) live
    on very different ranges and Euclidean distance otherwise gets
    dominated by the highest-magnitude column.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
