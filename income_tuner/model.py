"""RandomForest estimator factory used by the hyperparam search."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_rf(**kw) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, n_jobs=-1, **kw)
