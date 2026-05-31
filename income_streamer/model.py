"""SGDClassifier pipelines with different losses."""
from __future__ import annotations

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_sgd(loss: str = "log_loss", **kw) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),  # works with sparse-like
            ("clf", SGDClassifier(loss=loss, random_state=42, **kw)),
        ]
    )
