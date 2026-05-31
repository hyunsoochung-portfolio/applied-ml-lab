"""Logistic regression R&D study on Telco Churn.

Experiments:
1. Solver comparison on the same problem: `lbfgs / saga / liblinear`.
2. C regularization sweep on a log grid (5-fold CV ROC-AUC).
3. `class_weight ∈ {None, 'balanced'}` ablation on minority recall.
4. Calibration curve + Brier score for the best CV C.
5. ROC + Precision-Recall curves for the best CV C.
6. OvR vs multinomial comparison on the synthetic 3-class tenure target.

Outputs under ``artifacts/``: ``solver_comparison.csv``, ``c_sweep.csv``,
``c_sweep.png``, ``class_weight_ablation.csv``, ``calibration.png``,
``roc_pr_curves.png``, ``multiclass_ovr_vs_softmax.csv``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_binary, load_multiclass  # noqa: E402
from model import build_binary, build_multinomial, build_ovr  # noqa: E402


HERE = Path(__file__).resolve().parent
C_GRID = np.logspace(-3, 2, 11)
SOLVERS = ["lbfgs", "saga", "liblinear"]


def solver_comparison(X, y, *, seed) -> pd.DataFrame:
    print("\n== solver comparison (5-fold CV ROC-AUC) ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for s in SOLVERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sc = cross_val_score(build_binary(solver=s), X, y,
                                     cv=cv, scoring="roc_auc", n_jobs=1)
                rows.append({"solver": s,
                             "cv_auc_mean": float(sc.mean()),
                             "cv_auc_std": float(sc.std())})
                print(f"  {s:>10s}  AUC={sc.mean():.4f}±{sc.std():.4f}")
            except Exception as e:
                print(f"  {s:>10s}  FAILED ({e!r})")
                rows.append({"solver": s, "cv_auc_mean": float("nan"),
                             "cv_auc_std": float("nan")})
    return pd.DataFrame(rows)


def c_sweep(X, y, *, seed) -> pd.DataFrame:
    print("\n== C regularization sweep ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for C in C_GRID:
        sc = cross_val_score(build_binary(C=float(C)), X, y,
                             cv=cv, scoring="roc_auc", n_jobs=1)
        rows.append({"C": float(C),
                     "cv_auc_mean": float(sc.mean()),
                     "cv_auc_std": float(sc.std())})
        print(f"  C={C:>9.4g}  AUC={sc.mean():.4f}±{sc.std():.4f}")
    return pd.DataFrame(rows)


def plot_c_sweep(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df["C"], df["cv_auc_mean"], yerr=df["cv_auc_std"],
                marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("LogisticRegression: AUC vs C")
    ax.grid(alpha=0.3)
    save_plot(fig, out_path)


def class_weight_ablation(X, y, *, seed) -> pd.DataFrame:
    print("\n== class_weight ablation on minority recall ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )
    rows = []
    for cw in (None, "balanced"):
        pipe = build_binary(class_weight=cw).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:, 1]
        rows.append({
            "class_weight": str(cw),
            "accuracy": float(accuracy_score(yte, pred)),
            "roc_auc": float(roc_auc_score(yte, prob)),
            "minority_recall": float(recall_score(yte, pred)),
        })
        print(f"  class_weight={str(cw):>9s}  acc={rows[-1]['accuracy']:.4f}  "
              f"AUC={rows[-1]['roc_auc']:.4f}  "
              f"minority recall={rows[-1]['minority_recall']:.4f}")
    return pd.DataFrame(rows)


def calibration_and_curves(X, y, best_C: float, *, seed) -> None:
    print(f"\n== calibration + ROC/PR (best C={best_C:g}) ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )
    pipe = build_binary(C=best_C).fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    brier = brier_score_loss(yte, prob)
    print(f"  Brier score = {brier:.4f}  (lower is better)")

    frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-",
            label=f"LogReg (Brier={brier:.4f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction positive")
    ax.set_title("Calibration curve (10 quantile bins)")
    ax.grid(alpha=0.3); ax.legend()
    save_plot(fig, HERE / "artifacts" / "calibration.png")

    fpr, tpr, _ = roc_curve(yte, prob)
    auc = roc_auc_score(yte, prob)
    prec, rec, _ = precision_recall_curve(yte, prob)
    ap = average_precision_score(yte, prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fpr, tpr, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k:", label="chance")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(rec, prec, label=f"AP={ap:.4f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall"); axes[1].grid(alpha=0.3); axes[1].legend()
    save_plot(fig, HERE / "artifacts" / "roc_pr_curves.png")


def multiclass_demo(*, seed) -> pd.DataFrame:
    print("\n== OvR vs multinomial on synthetic 3-class tenure target ==")
    X, y = load_multiclass()
    print(f"[data] X={X.shape}  class counts={y.value_counts().to_dict()}")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    rows = []
    for name, pipe in (
        ("ovr", build_ovr()), ("multinomial", build_multinomial()),
    ):
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)
        rows.append({
            "strategy": name,
            "accuracy": float(accuracy_score(yte, pred)),
            "proba_rowsum_mean": float(proba.sum(axis=1).mean()),
        })
        print(f"  {name:>11s}  acc={rows[-1]['accuracy']:.4f}  "
              f"proba rowsum mean={rows[-1]['proba_rowsum_mean']:.4f}")
    pipe = build_multinomial().fit(Xtr, ytr)
    print("\n  multinomial classification report:")
    print(classification_report(yte, pipe.predict(Xte), digits=4))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_binary()
    print(f"[data] X={X.shape}  positive rate={y.mean():.3f}")

    sol_df = solver_comparison(X, y, seed=args.seed)
    save_csv(sol_df.to_dict(orient="records"), out_dir / "solver_comparison.csv")

    c_df = c_sweep(X, y, seed=args.seed)
    save_csv(c_df.to_dict(orient="records"), out_dir / "c_sweep.csv")
    plot_c_sweep(c_df, out_dir / "c_sweep.png")

    cw_df = class_weight_ablation(X, y, seed=args.seed)
    save_csv(cw_df.to_dict(orient="records"),
             out_dir / "class_weight_ablation.csv")

    best_C = float(c_df.loc[c_df["cv_auc_mean"].idxmax(), "C"])
    calibration_and_curves(X, y, best_C, seed=args.seed)

    mc_df = multiclass_demo(seed=args.seed)
    save_csv(mc_df.to_dict(orient="records"),
             out_dir / "multiclass_ovr_vs_softmax.csv")


if __name__ == "__main__":
    main()
