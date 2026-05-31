"""Scaled / unscaled kNN regressor pipelines."""
from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn(
    n_neighbors: int = 5,
    *,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    scaled: bool = True,
) -> Pipeline:
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors, weights=weights, metric=metric, p=p,
    )
    steps = []
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("knn", knn))
    return Pipeline(steps=steps)
