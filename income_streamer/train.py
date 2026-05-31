"""SGDClassifier R&D study on Adult Income.

Experiments:
1. Loss × learning-rate-schedule grid.
2. `partial_fit` batch-size study (128 / 512 / 2048) with
   train/test/wall-clock per batch size.
3. Early stopping vs full epochs comparison.
4. Train/val accuracy curve over epochs across 5 seeds, plotted as mean
   ± shaded std.

Artifacts:
- ``loss_schedule_grid.csv``
- ``batch_size_study.csv``
- ``early_stopping.csv``
- ``epoch_curve_multiseed.csv``
- ``epoch_curve.png``
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_sgd  # noqa: E402


HERE = Path(__file__).resolve().parent
LOSSES = ["log_loss", "hinge", "modified_huber", "squared_hinge"]
SCHEDULES = ["constant", "optimal", "invscaling", "adaptive"]
BATCH_SIZES = [128, 512, 2048]


def _split_scaled(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y.reset_index(drop=True),
        test_size=0.2, stratify=y, random_state=seed,
    )
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    scaler = StandardScaler(with_mean=False).fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def loss_schedule_grid(X, y, *, seed) -> pd.DataFrame:
    print("\n== loss × learning_rate grid ==")
    Xtr, Xte, ytr, yte = _split_scaled(X, y, seed)
    rows = []
    for loss in LOSSES:
        for sched in SCHEDULES:
            kw = dict(loss=loss, learning_rate=sched, random_state=seed,
                      max_iter=50, tol=1e-4)
            # `constant` / `invscaling` / `adaptive` require eta0
            if sched in ("constant", "invscaling", "adaptive"):
                kw["eta0"] = 0.01
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    clf = SGDClassifier(**kw).fit(Xtr, ytr)
                    acc = float(accuracy_score(yte, clf.predict(Xte)))
                except Exception as e:
                    print(f"  loss={loss} sched={sched} failed: {e!r}")
                    acc = float("nan")
            rows.append({"loss": loss, "schedule": sched, "test_acc": acc})
            print(f"  loss={loss:>15s}  sched={sched:>10s}  acc={acc:.4f}")
    return pd.DataFrame(rows)


def batch_size_study(X, y, *, seed, epochs: int = 8) -> pd.DataFrame:
    print("\n== partial_fit batch-size study ==")
    Xtr, Xte, ytr, yte = _split_scaled(X, y, seed)
    classes = np.unique(ytr)
    n = Xtr.shape[0]
    rows = []
    for batch in BATCH_SIZES:
        rng = np.random.default_rng(seed)
        clf = SGDClassifier(loss="log_loss", random_state=seed)
        t0 = time.perf_counter()
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                sl = order[start:start + batch]
                clf.partial_fit(Xtr[sl], ytr.iloc[sl], classes=classes)
        dt = time.perf_counter() - t0
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        rows.append({"batch_size": batch, "epochs": epochs,
                     "test_acc": acc, "train_time_s": float(dt)})
        print(f"  batch={batch:>4d}  test_acc={acc:.4f}  time={dt:.2f}s")
    return pd.DataFrame(rows)


def early_stopping_comparison(X, y, *, seed, full_epochs: int = 30) -> pd.DataFrame:
    print("\n== early stopping vs full epochs ==")
    Xtr, Xte, ytr, yte = _split_scaled(X, y, seed)

    t0 = time.perf_counter()
    full = SGDClassifier(loss="log_loss", random_state=seed,
                         max_iter=full_epochs, tol=None,
                         early_stopping=False).fit(Xtr, ytr)
    full_time = time.perf_counter() - t0
    full_acc = float(accuracy_score(yte, full.predict(Xte)))

    t0 = time.perf_counter()
    es = SGDClassifier(loss="log_loss", random_state=seed,
                       max_iter=full_epochs, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=5,
                       tol=1e-4).fit(Xtr, ytr)
    es_time = time.perf_counter() - t0
    es_acc = float(accuracy_score(yte, es.predict(Xte)))

    rows = [
        {"mode": "full_epochs", "test_acc": full_acc,
         "n_iter": int(full.n_iter_), "train_time_s": float(full_time)},
        {"mode": "early_stopping", "test_acc": es_acc,
         "n_iter": int(es.n_iter_), "train_time_s": float(es_time)},
    ]
    for r in rows:
        print(f"  {r['mode']:>15s}  acc={r['test_acc']:.4f}  "
              f"n_iter={r['n_iter']}  time={r['train_time_s']:.2f}s")
    return pd.DataFrame(rows)


def multiseed_epoch_curve(X, y, *, seeds, epochs: int = 25,
                          batch: int = 1024) -> tuple[pd.DataFrame, dict]:
    print(f"\n== multi-seed epoch curve ({len(seeds)} seeds) ==")
    rows = []
    per_seed = {"train": [], "test": []}
    for s in seeds:
        Xtr, Xte, ytr, yte = _split_scaled(X, y, s)
        classes = np.unique(ytr)
        rng = np.random.default_rng(s)
        clf = SGDClassifier(loss="log_loss", random_state=s)
        n = Xtr.shape[0]
        tr_curve, te_curve = [], []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                sl = order[start:start + batch]
                clf.partial_fit(Xtr[sl], ytr.iloc[sl], classes=classes)
            ta = float(accuracy_score(ytr, clf.predict(Xtr)))
            va = float(accuracy_score(yte, clf.predict(Xte)))
            tr_curve.append(ta); te_curve.append(va)
            rows.append({"seed": int(s), "epoch": epoch,
                         "train_acc": ta, "test_acc": va})
        per_seed["train"].append(tr_curve)
        per_seed["test"].append(te_curve)
        print(f"  seed={s}: final test acc={te_curve[-1]:.4f}")
    return pd.DataFrame(rows), per_seed


def plot_epoch_curve(per_seed, epochs, out_path: Path) -> None:
    tr = np.asarray(per_seed["train"])  # (seeds, epochs)
    te = np.asarray(per_seed["test"])
    xs = np.arange(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, tr.mean(axis=0), color="C0", label="train mean")
    ax.fill_between(xs, tr.mean(0) - tr.std(0), tr.mean(0) + tr.std(0),
                    color="C0", alpha=0.2)
    ax.plot(xs, te.mean(axis=0), color="C1", label="test mean")
    ax.fill_between(xs, te.mean(0) - te.std(0), te.mean(0) + te.std(0),
                    color="C1", alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title(f"SGD epoch curve — mean ± std over {tr.shape[0]} seeds")
    ax.grid(alpha=0.3); ax.legend()
    save_plot(fig, out_path)


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

    grid_df = loss_schedule_grid(X, y, seed=args.seed)
    save_csv(grid_df.to_dict(orient="records"), out_dir / "loss_schedule_grid.csv")

    bs_df = batch_size_study(X, y, seed=args.seed)
    save_csv(bs_df.to_dict(orient="records"), out_dir / "batch_size_study.csv")

    es_df = early_stopping_comparison(X, y, seed=args.seed)
    save_csv(es_df.to_dict(orient="records"), out_dir / "early_stopping.csv")

    epoch_df, per_seed = multiseed_epoch_curve(X, y, seeds=seeds)
    save_csv(epoch_df.to_dict(orient="records"),
             out_dir / "epoch_curve_multiseed.csv")
    plot_epoch_curve(per_seed, 25, out_dir / "epoch_curve.png")


if __name__ == "__main__":
    main()
