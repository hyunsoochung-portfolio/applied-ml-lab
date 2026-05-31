"""PCA / KernelPCA + LogisticRegression pipelines."""
from __future__ import annotations

from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pca(n_components: int, *, whiten: bool = False) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42, whiten=whiten)),
    ])


def build_pca_logreg(n_components: int, *, whiten: bool = False) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42, whiten=whiten)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def build_kpca_logreg(n_components: int, kernel: str = "rbf",
                      gamma: float | None = None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kpca", KernelPCA(n_components=n_components, kernel=kernel,
                           gamma=gamma, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
