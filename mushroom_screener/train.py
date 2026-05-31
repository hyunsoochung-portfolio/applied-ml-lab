"""Decision-tree R&D study on UCI Mushroom (all-categorical, binary).

Experiments:
1. `criterion × max_depth × min_samples_leaf` grid (single hold-out).
2. Cost-complexity post-pruning path via
   `cost_complexity_pruning_path` — sweep `ccp_alpha` and re-score.
3. `feature_importances_` vs `permutation_importance` side-by-side on
   the best pruned model.
4. Tree visualisations: a shallow `max_depth=4` plus the best pruned
   model.

Artifacts: ``grid_search.csv``, ``ccp_path.csv``, ``ccp_path.png``,
``importance_compare.csv``, ``importance_compare.png``,
``tree_shallow.png``, ``tree_best.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import save_csv, save_plot, set_seed  # noqa: E402

from data import load_clean  # noqa: E402
from model import build_tree  # noqa: E402


HERE = Path(__file__).resolve().parent
CRITERIA = ["gini", "entropy"]
MAX_DEPTHS = [2, 4, 6, 8, 12, None]
MIN_LEAVES = [1, 5, 20]


def grid_search(Xtr, Xte, ytr, yte) -> pd.DataFrame:
    print("\n== criterion × max_depth × min_samples_leaf grid ==")
    rows = []
    for crit in CRITERIA:
        for d in MAX_DEPTHS:
            for ml in MIN_LEAVES:
                pipe = build_tree(criterion=crit, max_depth=d,
                                  min_samples_split=2)
                # set min_samples_leaf on the tree step
                pipe.set_params(clf__min_samples_leaf=ml)
                pipe.fit(Xtr, ytr)
                acc = accuracy_score(yte, pipe.predict(Xte))
                rows.append({"criterion": crit, "max_depth": str(d),
                             "min_samples_leaf": int(ml),
                             "test_acc": float(acc)})
                print(f"  crit={crit:>7s}  depth={str(d):>5s}  "
                      f"leaf={ml:>3d}  acc={acc:.4f}")
    return pd.DataFrame(rows)


def ccp_path_experiment(Xtr, Xte, ytr, yte):
    print("\n== cost-complexity post-pruning path ==")
    # build a base pipeline + fit a deep tree, then ask for the ccp_alpha
    # sequence
    pipe = build_tree(max_depth=None).fit(Xtr, ytr)
    pre = pipe.named_steps["pre"]
    tree = pipe.named_steps["clf"]
    Xtr_enc = pre.transform(Xtr)
    Xte_enc = pre.transform(Xte)

    path = tree.cost_complexity_pruning_path(Xtr_enc, ytr)
    alphas = path.ccp_alphas
    # keep a manageable subset
    alphas = alphas[::max(1, len(alphas) // 30)]

    rows = []
    best_pipe = pipe
    best_acc = -1.0
    for a in alphas:
        t = DecisionTreeClassifier(ccp_alpha=float(a), random_state=42)
        t.fit(Xtr_enc, ytr)
        ta = accuracy_score(ytr, t.predict(Xtr_enc))
        te = accuracy_score(yte, t.predict(Xte_enc))
        rows.append({"ccp_alpha": float(a), "train_acc": float(ta),
                     "test_acc": float(te),
                     "n_nodes": int(t.tree_.node_count),
                     "depth": int(t.get_depth())})
        if te > best_acc:
            best_acc = te
            # build a fresh pipeline wrapping the pruned tree
            best_pipe = build_tree(max_depth=None)
            best_pipe.set_params(clf__ccp_alpha=float(a))
            best_pipe.fit(Xtr, ytr)
    df = pd.DataFrame(rows)
    print(df.tail(10).to_string(index=False))
    return df, best_pipe


def plot_ccp_path(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["ccp_alpha"], df["train_acc"], "o-",
             color="C0", label="train acc")
    ax1.plot(df["ccp_alpha"], df["test_acc"], "s-",
             color="C1", label="test acc")
    ax1.set_xscale("log")
    ax1.set_xlabel("ccp_alpha (log)")
    ax1.set_ylabel("accuracy")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower left")
    ax2 = ax1.twinx()
    ax2.plot(df["ccp_alpha"], df["n_nodes"], "k:", alpha=0.6,
             label="n_nodes")
    ax2.set_ylabel("number of nodes (dotted)")
    ax2.legend(loc="upper right")
    ax1.set_title("Cost-complexity pruning path")
    save_plot(fig, out_path)


def importance_compare(pipe, Xtr, Xte, ytr, yte) -> pd.DataFrame:
    print("\n== feature_importances_ vs permutation_importance ==")
    fi = pipe.named_steps["clf"].feature_importances_
    feat_names = pipe.named_steps["pre"].get_feature_names_out()

    pi = permutation_importance(
        pipe, Xte, yte, n_repeats=5, random_state=0, n_jobs=1,
        scoring="accuracy",
    )
    pi_mean = pi.importances_mean
    pi_std = pi.importances_std

    rows = []
    for n, f, pm, ps in zip(feat_names, fi, pi_mean, pi_std):
        rows.append({"feature": str(n), "tree_importance": float(f),
                     "perm_mean": float(pm), "perm_std": float(ps)})
    df = pd.DataFrame(rows)
    top = df.assign(score=df["tree_importance"] + df["perm_mean"]) \
            .sort_values("score", ascending=False).head(15)
    print(top[["feature", "tree_importance", "perm_mean", "perm_std"]]
          .to_string(index=False))
    return df


def plot_importance(df: pd.DataFrame, out_path: Path) -> None:
    top = (df.assign(score=df["tree_importance"] + df["perm_mean"])
             .sort_values("score", ascending=False).head(12))
    y_pos = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.4
    ax.barh(y_pos + width / 2, top["tree_importance"],
            height=width, label="feature_importances_")
    ax.barh(y_pos - width / 2, top["perm_mean"], height=width,
            xerr=top["perm_std"], label="permutation_importance")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].str.replace("onehot__", "",
                                                  regex=False))
    ax.set_xlabel("importance")
    ax.set_title("Top features — tree vs permutation importance")
    ax.legend()
    save_plot(fig, out_path)


def visualise(pipe, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        pipe.named_steps["clf"],
        feature_names=pipe.named_steps["pre"].get_feature_names_out(),
        class_names=["edible", "poisonous"],
        filled=True, rounded=True, fontsize=7, ax=ax, max_depth=4,
    )
    ax.set_title(title)
    save_plot(fig, out_path)


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

    X, y = load_clean()
    print(f"[data] X={X.shape}  poisonous rate={y.mean():.3f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )

    grid_df = grid_search(Xtr, Xte, ytr, yte)
    save_csv(grid_df.to_dict(orient="records"), out_dir / "grid_search.csv")

    ccp_df, best_pipe = ccp_path_experiment(Xtr, Xte, ytr, yte)
    save_csv(ccp_df.to_dict(orient="records"), out_dir / "ccp_path.csv")
    plot_ccp_path(ccp_df, out_dir / "ccp_path.png")

    imp_df = importance_compare(best_pipe, Xtr, Xte, ytr, yte)
    save_csv(imp_df.to_dict(orient="records"),
             out_dir / "importance_compare.csv")
    plot_importance(imp_df, out_dir / "importance_compare.png")

    shallow = build_tree(max_depth=4).fit(Xtr, ytr)
    visualise(shallow, out_dir / "tree_shallow.png",
              title="Decision tree (max_depth=4)")
    visualise(best_pipe, out_dir / "tree_best.png",
              title="Best pruned tree (showing top 4 depth levels)")


if __name__ == "__main__":
    main()
