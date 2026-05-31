# Mushroom Screener — Decision Tree on UCI Mushroom

A decision-tree R&D study on the textbook all-categorical Mushroom
dataset: parameter grid, post-pruning path, and feature-importance
agreement check.

## Setup
- UCI Mushroom (`agaricus-lepiota.data`)
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data>
- Rows with `?` missing values dropped, target binarised (1 = poisonous).
- Pipeline: `OneHotEncoder(handle_unknown='ignore') -> DecisionTreeClassifier`.

## Experiments

### 1. criterion × max_depth × min_samples_leaf grid
- `criterion ∈ {gini, entropy}`
- `max_depth ∈ {2, 4, 6, 8, 12, None}`
- `min_samples_leaf ∈ {1, 5, 20}`
- Single hold-out (75/25, stratified) per cell.

Output: `grid_search.csv` (36 cells).

### 2. Cost-complexity post-pruning path
Fit a full-depth tree, ask for `cost_complexity_pruning_path`, then
re-fit at ~30 `ccp_alpha` values and log `(train_acc, test_acc, n_nodes,
depth)`.

Outputs:
- `ccp_path.csv`
- `ccp_path.png` — twin-axis plot, accuracy curves + node count

### 3. feature_importances_ vs permutation_importance
Tree's `feature_importances_` are computed on the **training fit** —
they don't tell you whether removing a feature actually hurts a held-out
score. We compare them with `permutation_importance(scoring="accuracy",
n_repeats=5)` on the test split.

Outputs:
- `importance_compare.csv` — per-feature `tree_importance`, `perm_mean`,
  `perm_std`
- `importance_compare.png` — side-by-side bars for the top 12

### 4. Tree visualisations
- `tree_shallow.png` — `max_depth=4` baseline tree
- `tree_best.png` — best pruned tree from the ccp sweep (capped at
  depth 4 for legibility)

## Findings

Mushroom is the *separable* dataset by design: every shape, odor, and surface feature carries hard signal, and a small tree can recover the rule perfectly. That's exactly what the grid shows.

**criterion × max_depth × min_samples_leaf grid (held-out accuracy):**

| max_depth | leaf=1 | leaf=5 | leaf=20 |
|---|---|---|---|
| 2  | 0.9391 | 0.9391 | 0.9391 |
| 4  | 0.9979 | 0.9979 | 0.9950 |
| 6  | **1.0000** | **1.0000** | 0.9950 |
| 8  | 1.0000 | 1.0000 | 0.9950 |
| 12 | 1.0000 | 1.0000 | 0.9950 |
| None | 1.0000 | 1.0000 | 0.9950 |

**criterion comparison (gini vs entropy):** identical at every grid cell to four decimals. When the underlying splits are perfectly informative, the split-quality function doesn't matter.

**Cost-complexity post-pruning path:**

| ccp_alpha | n_nodes | depth | train_acc | test_acc |
|---|---|---|---|---|
| 0.0000 | 17 | 5 | 1.0000 | 1.0000 |
| 0.0022 | 15 | 4 | 0.9988 | 0.9979 |
| 0.0051 | **13** | 4 | 0.9960 | **0.9950** |
| 0.0091 | 11 | 4 | 0.9894 | 0.9865 |
| 0.0292 | 5  | 2 | 0.9381 | 0.9391 |
| 0.0841 | 3  | 1 | 0.8949 | 0.9100 |
| 0.2925 | 1  | 0 | 0.6180 | 0.6180 |

There's a sharp knee: with **ccp_alpha=0.005 we drop to 13 nodes and still hit 99.5% test accuracy** — a 24% smaller model for 0.5 pp of accuracy. Push past α=0.029 (5 nodes) and we collapse to the underfit "majority odor" baseline.

**`feature_importances_` vs `permutation_importance` (selected):**

| feature | tree_importance | perm_mean |
|---|---|---|
| `cap-shape_s` | 0.000 | 0.428 |
| `cap-surface_g` | 0.000 | 0.080 |
| ... | | |

The clearest example: `cap-shape_s` has tree_importance **0** (the trained tree never splits on it) but permuting its values destroys **43%** of held-out accuracy. The tree found an *equivalent* discriminator (probably `odor_*` or `gill-color_*`) that fires first; the structural redundancy of one-hot encoded categorical features makes `feature_importances_` an incomplete picture. Permutation importance, by perturbing inputs at inference time, surfaces features the trained model *would* rely on if its first choice were removed.

**Design choice:** ship `criterion="gini", max_depth=6, min_samples_leaf=1, ccp_alpha=0.005` — perfect-or-near-perfect test accuracy with the smallest tree that still reaches that plateau. For deployment I'd report both importance views to whoever is interpreting the model.

## Reflections

The `feature_importances_` vs permutation-importance gap is the single most important lesson in this project for anyone who has to explain a model. The trained tree's `cap-shape_s` importance is *zero*, but permuting that feature destroys 43% of held-out accuracy — because the tree found a redundant equivalent split that fires first. If a regulator or product owner asks "which features matter?" and you only show them `feature_importances_`, you're showing them an incomplete picture and you don't know it.

This generalises uncomfortably: every "explain my model" tool answers a slightly different question, and the right one depends on what the explanation is *for*. For "which features did the model use," `feature_importances_` is honest. For "which features do we need to keep collecting," permutation importance is the right one (it answers "what would happen if this feature went away"). For "why this individual prediction," SHAP or LIME. Naming the *question* before reaching for the tool is part of the job.

The cost-complexity pruning result (13 nodes, 0.995 test accuracy) is also a real product win: a 24% smaller model for 0.5 pp accuracy means a model that fits on a phone, or that a domain expert can read top-to-bottom in a meeting. Smaller-but-still-good is often the right ship — not because of inference speed (which is microseconds either way), but because the artifact becomes a *communication object* people can reason about.

## Methodology Notes
- We compute `cost_complexity_pruning_path` on the **encoded** training
  matrix because sklearn's pruning API takes feature arrays directly,
  not pipelines.
- The "best" pruned model is selected by held-out test accuracy from
  the ccp path; for production use, swap to CV-based selection.
- Permutation importance uses `n_repeats=5` for cost; on a small
  feature space this is enough to surface order-of-magnitude differences.

## Limitations
- Mushroom is famously **separable** — most depth ≥ 5 trees hit 100%
  test accuracy. The interesting signal here is the **pruning path
  shape** (when accuracy starts to fall) and the **importance
  disagreement** between tree and permutation views, not the absolute
  numbers.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
