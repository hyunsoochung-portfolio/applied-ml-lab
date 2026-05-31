"""Ridge / Lasso / ElasticNet / LinearRegression pipelines with poly features."""
from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _poly(degree: int, interaction_only: bool) -> PolynomialFeatures:
    return PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False,
    )


def build_linear(degree: int = 2, interaction_only: bool = False) -> Pipeline:
    return Pipeline([
        ("poly", _poly(degree, interaction_only)),
        ("scaler", StandardScaler()),
        ("lin", LinearRegression()),
    ])


def build_ridge(alpha: float, degree: int = 2, interaction_only: bool = False) -> Pipeline:
    return Pipeline([
        ("poly", _poly(degree, interaction_only)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_lasso(alpha: float, degree: int = 2, interaction_only: bool = False) -> Pipeline:
    return Pipeline([
        ("poly", _poly(degree, interaction_only)),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=20000)),
    ])


def build_enet(alpha: float, l1_ratio: float, degree: int = 2,
               interaction_only: bool = False) -> Pipeline:
    return Pipeline([
        ("poly", _poly(degree, interaction_only)),
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)),
    ])
