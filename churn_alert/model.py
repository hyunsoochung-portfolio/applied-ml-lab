"""Logistic regression pipelines (binary and multiclass) with solver knob."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_binary(
    *, C: float = 1.0, solver: str = "lbfgs",
    class_weight=None, penalty: str = "l2",
) -> Pipeline:
    kw = {}
    if solver == "saga":
        kw["max_iter"] = 4000
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            class_weight=class_weight, max_iter=kw.get("max_iter", 2000),
        )),
    ])


def build_multinomial(solver: str = "lbfgs") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=4000, solver=solver)),
    ])


def build_ovr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=4000, solver="lbfgs")
        )),
    ])
