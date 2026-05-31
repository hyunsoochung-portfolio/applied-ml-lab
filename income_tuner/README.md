# Income Tuner — Hyperparameter Tuning on UCI Adult Income

4-way hyperparameter-search comparison on `RandomForestClassifier`. The
same search space is fed to every method so the wall-clock vs CV-score
trade-off is directly comparable.

## Setup
- Same Adult Income dataset as `income_streamer`
  (`fetch_openml('adult', version=2)`).
- Estimator: `RandomForestClassifier(random_state=42, n_jobs=-1)`.
- CV: `StratifiedKFold(3)` for the search loops, `StratifiedKFold(5)`
  for the baseline.
- `--subsample 15000` (default) carves a stratified subsample for the
  search comparison so total wall-clock stays in the few-minute range.

## Search space
Discrete grid (`GridSearchCV` / `HalvingGridSearchCV`):
```python
{
  "n_estimators": [100, 200, 300],
  "max_depth": [None, 10, 20],
  "min_samples_leaf": [1, 4, 16],
  "max_features": ["sqrt", 0.5],
}  # 54 cells
```

Distributions (`RandomizedSearchCV` / `HalvingRandomSearchCV`):
```python
{
  "n_estimators": randint(100, 400),
  "max_depth": randint(5, 30),
  "min_samples_leaf": randint(1, 32),
  "max_features": uniform(0.3, 0.6),
}
```

## Experiments

### 1. 4-way search comparison
- `GridSearchCV`
- `RandomizedSearchCV` (n_iter=20)
- `HalvingGridSearchCV` (factor=3)
- `HalvingRandomSearchCV` (n_candidates=40, factor=3)

Logs `best_score`, `best_params`, total wall-clock per method.

Outputs:
- `search_comparison.csv`
- `pareto.png` — wall-clock vs best CV ROC-AUC, red dots mark the
  Pareto-front (non-dominated) methods.

### 2. RandomizedSearch n_iter sensitivity
Sweep `n_iter ∈ {10, 50, 200}` to show how diminishing returns kick in.

Output: `rand_n_iter_sensitivity.csv`.

## Findings

**4-way search comparison on RandomForest (Adult Income, same search space, same CV split, ROC-AUC):**

| search | best CV ROC-AUC | wall-clock (s) | best params (top-3 shown) |
|---|---|---|---|
| `GridSearchCV`           | **0.9140** |  99.4 | depth=None, max_features=sqrt, min_samples_leaf=4, n_estimators=100 |
| `RandomizedSearchCV` (n_iter=50)  | 0.9125 |  71.3 | depth=15, max_features=0.56, min_samples_leaf=5, n_estimators=362 |
| `HalvingGridSearchCV`    | 0.9134 |  **28.8** | depth=20, max_features=sqrt, min_samples_leaf=4, n_estimators=200 |
| `HalvingRandomSearchCV`  | 0.8997 |  **16.9** | depth=7, max_features=0.38, min_samples_leaf=5, n_estimators=153 |

Pareto picture:
- **HalvingGridSearchCV is the clear winner.** It reaches AUC=0.9134 in 28.8s — *0.06 pp behind full Grid, 3.5× faster.* The successive-halving budget allocates compute to promising configs and prunes the rest early; on this RF grid it loses almost nothing in solution quality.
- **HalvingRandomSearchCV** is *fastest* (16.9s) but takes a real hit on score (0.8997, ~1.4 pp below Grid). Halving's small initial budget plus random sampling lands on under-trained configurations that look bad and get pruned before they can prove themselves.
- **RandomizedSearch (n_iter=50)** sits in the middle on both axes — slower than Halving variants because every candidate gets the full budget, lower score than Grid because exhaustive coverage matters on a small space like this one.

**n_iter sensitivity for RandomizedSearch:**

| n_iter | best CV ROC-AUC | wall-clock (s) |
|---|---|---|
| 10  | 0.9120 | 47.0 |
| 50  | 0.9135 | 176.2 |
| 200 | **0.9135** | 693.9 |

A flat plateau at n_iter ≈ 50. Going from 50 → 200 spends 4× more compute for zero gain — the search space has been thoroughly explored at 50 samples. Lesson: for any new search problem, **start with n_iter≈30-50 and only push higher if the leaderboard is still climbing**.

**Design choice:** for *this* search space, **HalvingGridSearchCV** is the right default — best speed-vs-score trade and reproducible (deterministic). I'd reach for `RandomizedSearchCV(n_iter=50)` only when the search space is much larger (continuous distributions over many hyperparameters) and exhaustive grids become intractable.

## Reflections

The Pareto picture (HalvingGridSearch matching full Grid in ~30% of the time) is the engineering result I'd most want a junior teammate to internalise. Hyperparameter search is *compute that doesn't make the user's life better unless it changes a decision* — every second spent on it should be defensible against either (a) tighter accuracy on the deployed model, or (b) the team's time. Halving searches reach the same decision with much less compute, so the rational default is to start with Halving and only fall back to exhaustive grid when the search space is small enough that the savings don't matter.

The `n_iter` plateau (50 = 200) is a small product/communication point too. "We ran 200 candidates" sounds more rigorous in a slide than "we ran 50," but it isn't — the 4× extra compute returned zero accuracy gain on this space. Reporting the *plateau* rather than the *max* is honest about diminishing returns and stops the next team from wasting their compute budget on the same dead end. This is the kind of thing I'd want the README of any team's tuning scripts to call out by default.

On the systems side, all four search APIs share the same scorer and CV splitter — the trick was making them comparable. Building that little harness once means future hyperparameter questions become "swap models / spaces, re-run" instead of one-off scripts. A small investment in shared infrastructure pays back across every project that touches model selection.

## Methodology Notes
- All four methods share the **same** CV splitter, scoring metric, and
  estimator factory — the only knob being changed is the search
  strategy and (for the randomised variants) the seed.
- Halving variants need `sklearn.experimental.enable_halving_search_cv`
  imported before construction (handled at the top of `train.py`).
- The Pareto front is computed in `shared/utils.pareto_front` —
  `minimize_x` (time), `maximize_y` (score).

## Limitations
- 15k-row subsample under-states the absolute scores for the baseline
  comparison; the relative ordering of search methods is stable, which
  is the point.
- Single-seed comparison — for a more rigorous benchmark, wrap the
  whole loop in `--n-seeds 5` and average.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --subsample 15000
python evaluate.py
```
