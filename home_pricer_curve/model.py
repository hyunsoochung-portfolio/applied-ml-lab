"""Linear / polynomial regression factories."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_linear() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("lin", LinearRegression())])


def build_poly(degree: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin", LinearRegression()),
        ]
    )
