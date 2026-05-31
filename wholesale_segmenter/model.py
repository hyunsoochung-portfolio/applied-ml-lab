"""Pipeline factories used by the scaling × algorithm matrix."""
from __future__ import annotations

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC


def build(scaler: TransformerMixin | None, k: int = 5) -> Pipeline:
    """Backwards-compatible kNN pipeline with optional scaler."""
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k)))
    return Pipeline(steps=steps)


def scalers():
    """Mapping of scaler name -> fresh instance (or ``None``)."""
    return {
        "none": None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
    }


def algorithms(seed: int = 0):
    """Mapping of algorithm name -> unfitted estimator."""
    return {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "LinearSVC": LinearSVC(max_iter=5000, dual="auto", random_state=seed),
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1
        ),
    }


def build_for(scaler_name: str, algo_name: str, *, seed: int = 0) -> Pipeline:
    """Return ``scaler -> algo`` pipeline. ``scaler_name='none'`` skips scaling."""
    sc = scalers()[scaler_name]
    algo = algorithms(seed=seed)[algo_name]
    steps = []
    if sc is not None:
        steps.append(("scaler", sc))
    steps.append(("clf", algo))
    return Pipeline(steps=steps)
