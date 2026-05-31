"""Tree-ensemble R&D study on UCI Bank Marketing.

Experiments:
1. 5-seed × 6-ensemble benchmark: mean ± std for ROC-AUC, F1, accuracy.
2. Permutation importance on the best model (test split, ROC-AUC scoring).
3. Feature-importance agreement matrix across ensembles
   (Spearman correlation of `feature_importances_` vectors).
4. Calibration curves overlaid for each ensemble.
5. Out-of-fold (`cross_val_predict`, 5-fold) probabilities saved per
   model so downstream comparisons are possible.

Artifacts:
- ``ensemble_benchmark.csv``
- ``perm_importance.csv`` + ``perm_importance.png``
- ``importance_agreement.csv`` + ``importance_agreement.png``
- ``calibration_overlay.png``
- ``oof_predictions.csv`` (one column per ensemble)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_models  # noqa: E402


HERE = Path(__file__).resolve().parent


def benchmark(X, y, *, seeds) -> pd.DataFrame:
    print("\n== 6-ensemble × 5-seed benchmark ==")
    rows = []
    for s in seeds:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=s)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=s,
        )
        for name, mdl in build_models().items():
            try:
                t0 = time.perf_counter()
                mdl.fit(Xtr, ytr)
                dt = time.perf_counter() - t0
                prob = mdl.predict_proba(Xte)[:, 1]
                pred = mdl.predict(Xte)
                auc = float(roc_auc_score(yte, prob))
                f1 = float(f1_score(yte, pred))
                acc = float(accuracy_score(yte, pred))
                rows.append({"model": name, "seed": int(s),
                             "roc_auc": auc, "f1": f1,
                             "accuracy": acc,
                             "fit_time_s": float(dt)})
            except Exception as e:
                print(f"  {name} seed={s} failed: {e!r}")
        print(f"  seed={s} done")
    df = pd.DataFrame(rows)
    summary = (df.groupby("model")[["roc_auc", "f1", "accuracy", "fit_time_s"]]
                 .agg(["mean", "std"]).round(4))
    print(summary.to_string())
    return df


def perm_importance(X, y, best_name: str, seed: int) -> pd.DataFrame:
    print(f"\n== permutation importance — {best_name} ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    mdl = build_models()[best_name].fit(Xtr, ytr)
    pi = permutation_importance(
        mdl, Xte, yte, n_repeats=5, random_state=seed,
        n_jobs=1, scoring="roc_auc",
    )
    feat = X.columns.to_numpy()
    rows = [{"feature": str(feat[i]),
             "perm_mean": float(pi.importances_mean[i]),
             "perm_std": float(pi.importances_std[i])}
            for i in range(len(feat))]
    df = pd.DataFrame(rows).sort_values("perm_mean", ascending=False)

    top = df.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["perm_mean"][::-1],
            xerr=top["perm_std"][::-1])
    ax.set_xlabel("mean drop in ROC-AUC (permutation)")
    ax.set_title(f"Permutation importance — {best_name}")
    save_plot(fig, HERE / "artifacts" / "perm_importance.png")
    return df


def importance_agreement(X, y, *, seed) -> pd.DataFrame:
    print("\n== feature-importance Spearman agreement matrix ==")
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    importances = {}
    for name, mdl in build_models().items():
        try:
            mdl.fit(Xtr, ytr)
            if hasattr(mdl, "feature_importances_"):
                importances[name] = np.asarray(mdl.feature_importances_)
        except Exception as e:
            print(f"  skip {name}: {e!r}")
    names = list(importances.keys())
    n = len(names)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearmanr(importances[names[i]], importances[names[j]]).statistic
            corr[i, j] = corr[j, i] = float(r)
    df = pd.DataFrame(corr, index=names, columns=names)
    print(df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center",
                    va="center", fontsize=8)
    ax.set_title("Spearman ρ of feature_importances_")
    fig.colorbar(im, ax=ax)
    save_plot(fig, HERE / "artifacts" / "importance_agreement.png")
    return df


def calibration_overlay(X, y, *, seed) -> None:
    print("\n== calibration overlay ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    for name, mdl in build_models().items():
        try:
            mdl.fit(Xtr, ytr)
            prob = mdl.predict_proba(Xte)[:, 1]
            fp, mp = calibration_curve(yte, prob, n_bins=10, strategy="quantile")
            brier = brier_score_loss(yte, prob)
            ax.plot(mp, fp, "o-", label=f"{name} (Brier={brier:.3f})")
        except Exception as e:
            print(f"  skip {name}: {e!r}")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction positive")
    ax.set_title("Calibration overlay (10 quantile bins)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save_plot(fig, HERE / "artifacts" / "calibration_overlay.png")


def oof_predictions(X, y, *, seed) -> pd.DataFrame:
    print("\n== out-of-fold predictions (5-fold cross_val_predict) ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cols = {"y_true": np.asarray(y)}
    for name, mdl in build_models().items():
        try:
            oof = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba",
                                    n_jobs=1)[:, 1]
            cols[f"oof_{name}"] = oof
            auc = roc_auc_score(y, oof)
            print(f"  {name:>22s}  OOF AUC={auc:.4f}")
        except Exception as e:
            print(f"  skip {name}: {e!r}")
    return pd.DataFrame(cols)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--output-dir", type=str, default=str(HERE / "artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    X, y = load_clean()
    print(f"[data] X={X.shape}  positive rate={y.mean():.3f}")

    bench = benchmark(X, y, seeds=seeds)
    save_csv(bench.to_dict(orient="records"), out_dir / "ensemble_benchmark.csv")

    best = (bench.groupby("model")["roc_auc"].mean().idxmax())
    perm = perm_importance(X, y, best, seed=args.seed)
    save_csv(perm.to_dict(orient="records"), out_dir / "perm_importance.csv")

    agree = importance_agreement(X, y, seed=args.seed)
    agree.to_csv(out_dir / "importance_agreement.csv")
    print(f"[csv] wrote agreement matrix -> {out_dir / 'importance_agreement.csv'}")

    calibration_overlay(X, y, seed=args.seed)

    oof = oof_predictions(X, y, seed=args.seed)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"[csv] wrote OOF predictions -> {out_dir / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
