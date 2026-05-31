"""Cross-validation R&D study on Heart Disease.

Experiments:
1. Splitter variance across 10 seeds: random hold-out, stratified
   hold-out, KFold, StratifiedKFold, RepeatedStratifiedKFold.
2. Fold-count sensitivity (K ∈ {3,5,10,20}).
3. Nested CV (outer 5-fold for model selection, inner 3-fold for
   hyperparameter selection over LogisticRegression `C`).
4. Leakage demo: scaling *before* splitting (leaks) vs *inside* the
   pipeline (clean) — quantifies the test-score inflation.

Outputs go to ``artifacts/``: ``splitter_variance.csv``,
``fold_count.csv``, ``nested_cv.csv``, ``leakage.csv`` + matching PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_classifier  # noqa: E402


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1) Splitter variance across seeds
# ---------------------------------------------------------------------------

def _holdout_score(X, y, stratify: bool, seed: int) -> float:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25,
        stratify=y if stratify else None, random_state=seed,
    )
    pipe = build_classifier().fit(Xtr, ytr)
    return float(accuracy_score(yte, pipe.predict(Xte)))


def _cv_scores(X, y, splitter) -> list[float]:
    pipe = build_classifier()
    return cross_val_score(pipe, X, y, cv=splitter, scoring="accuracy").tolist()


def splitter_variance(X, y, *, seeds) -> pd.DataFrame:
    print("\n== splitter variance across seeds ==")
    rows = []
    for s in seeds:
        rows.append({"splitter": "holdout_random", "seed": int(s),
                     "fold": 0, "score": _holdout_score(X, y, False, s)})
        rows.append({"splitter": "holdout_stratified", "seed": int(s),
                     "fold": 0, "score": _holdout_score(X, y, True, s)})
        for i, sc in enumerate(_cv_scores(
                X, y, KFold(n_splits=5, shuffle=True, random_state=s))):
            rows.append({"splitter": "KFold5", "seed": int(s),
                         "fold": i, "score": sc})
        for i, sc in enumerate(_cv_scores(
                X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=s))):
            rows.append({"splitter": "StratifiedKFold5", "seed": int(s),
                         "fold": i, "score": sc})
        for i, sc in enumerate(_cv_scores(
                X, y, RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=2, random_state=s))):
            rows.append({"splitter": "RepeatedSKF_5x2", "seed": int(s),
                         "fold": i, "score": sc})
    df = pd.DataFrame(rows)
    summary = (df.groupby("splitter")["score"]
                 .agg(["mean", "std", "min", "max"])
                 .round(4))
    print(summary.to_string())
    return df


def plot_variance(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = df.groupby("splitter")["score"].apply(list)
    try:
        ax.boxplot(groups.values, tick_labels=groups.index)
    except TypeError:
        # older matplotlib (<3.9) used `labels=` instead of `tick_labels=`
        ax.boxplot(groups.values, labels=groups.index)
    ax.set_ylabel("accuracy")
    ax.set_title("Splitter score variance across seeds")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_plot(fig, out_path)


# ---------------------------------------------------------------------------
# 2) Fold-count sensitivity
# ---------------------------------------------------------------------------

def fold_count(X, y, *, seeds, ks=(3, 5, 10, 20)) -> pd.DataFrame:
    print("\n== fold-count sensitivity ==")
    rows = []
    for k in ks:
        means, stds = [], []
        for s in seeds:
            sc = _cv_scores(X, y,
                            StratifiedKFold(n_splits=k, shuffle=True,
                                            random_state=s))
            rows.append({"k_folds": k, "seed": int(s),
                         "mean_score": float(np.mean(sc)),
                         "std_score": float(np.std(sc)),
                         "raw_scores": ";".join(f"{x:.4f}" for x in sc)})
            means.append(np.mean(sc)); stds.append(np.std(sc))
        print(f"  K={k:>3d}  across-seed mean={np.mean(means):.4f}  "
              f"across-seed std={np.std(means):.4f}  "
              f"avg within-CV std={np.mean(stds):.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Nested CV
# ---------------------------------------------------------------------------

def nested_cv(X, y, *, seed: int) -> pd.DataFrame:
    print("\n== nested CV (outer=5, inner=3) ==")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=1)
    res = cross_validate(gs, X, y, cv=outer, scoring="accuracy",
                         return_estimator=True, n_jobs=1)
    rows = []
    for i, (sc, est) in enumerate(zip(res["test_score"], res["estimator"])):
        rows.append({"outer_fold": i, "outer_score": float(sc),
                     "best_C": float(est.best_params_["clf__C"]),
                     "inner_best": float(est.best_score_)})
        print(f"  outer fold {i}: score={sc:.4f}  "
              f"chosen C={est.best_params_['clf__C']}  "
              f"inner best={est.best_score_:.4f}")
    print(f"  outer mean={res['test_score'].mean():.4f}  "
          f"std={res['test_score'].std():.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) Leakage demo
# ---------------------------------------------------------------------------

def leakage_demo(X, y, *, seeds) -> pd.DataFrame:
    print("\n== leakage demo: scale BEFORE vs INSIDE pipeline ==")
    rows = []
    for s in seeds:
        # LEAKY: scale on the full X before any split
        scaler = StandardScaler().fit(X)
        X_pre = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        leaky = cross_val_score(
            LogisticRegression(max_iter=2000), X_pre, y,
            cv=cv, scoring="accuracy",
        ).mean()
        # CLEAN: scaling lives inside the pipeline
        clean = cross_val_score(
            build_classifier(), X, y, cv=cv, scoring="accuracy",
        ).mean()
        rows.append({"seed": int(s), "leaky_mean": float(leaky),
                     "clean_mean": float(clean),
                     "inflation": float(leaky - clean)})
        print(f"  seed={s:>2d}  leaky={leaky:.4f}  clean={clean:.4f}  "
              f"inflation={leaky - clean:+.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    X, y = load_clean()
    print(f"[data] X={X.shape}  positive rate={y.mean():.3f}")

    df_var = splitter_variance(X, y, seeds=seeds)
    save_csv(df_var.to_dict(orient="records"), out_dir / "splitter_variance.csv")
    plot_variance(df_var, out_dir / "splitter_variance.png")

    df_k = fold_count(X, y, seeds=seeds)
    save_csv(df_k.to_dict(orient="records"), out_dir / "fold_count.csv")

    df_nested = nested_cv(X, y, seed=args.seed)
    save_csv(df_nested.to_dict(orient="records"), out_dir / "nested_cv.csv")

    df_leak = leakage_demo(X, y, seeds=seeds[:5])
    save_csv(df_leak.to_dict(orient="records"), out_dir / "leakage.csv")


if __name__ == "__main__":
    main()
