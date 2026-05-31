"""Decision tree pipeline with one-hot encoding for all categorical inputs."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def build_tree(
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), slice(0, None))
        ]
    )
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
