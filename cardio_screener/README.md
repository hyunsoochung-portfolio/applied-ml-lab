# Cardio Screener — kNN on UCI Cleveland Heart Disease

Predict whether a patient has heart disease (binary: derived from the
`num` column, `>0 -> 1`) using k-Nearest Neighbours. This project is set
up as a small R&D-style study rather than a single train run: a full
parameter grid, multi-seed cross-validation, and bootstrap CI for the
selected configuration.

## Setup
- Source: UCI ML Repository — `processed.cleveland.data`
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data>
- 303 rows, 13 features; `?` treated as missing and dropped before
  modelling; downloaded once and cached under `./data/`.
- All preprocessing (`StandardScaler`) is wrapped inside the
  scikit-learn `Pipeline` so it fits **inside** each CV fold and cannot
  leak the validation rows.

## Experiments
| experiment | sweep | what's logged |
|---|---|---|
| Full grid CV | `n_neighbors ∈ {1,3,5,7,11,15,21,31}` × `weights ∈ {uniform, distance}` × `metric ∈ {euclidean, manhattan, minkowski(p=3)}` | 5-seed `StratifiedKFold` mean ± std for accuracy and F1 |
| Bootstrap CI | resample test predictions of the best CV config 200× | 95% percentile interval on accuracy |
| Decision boundaries | 3 feature pairs × 3 k values (1 / 5 / 25) | 2-D contour plots |

Artifacts written under `artifacts/`:
- `knn_sweep.csv` — flat table of every grid cell with mean ± std
- `knn_sweep_heatmap.png` — visual summary of the grid
- `best_config_bootstrap.csv` — single-row bootstrap result
- `boundary_<feat_a>-<feat_b>_k<k>.png` — 9 boundary plots

## Findings

The grid run with 3 seeds × 5-fold StratifiedKFold produced a very consistent picture (`acc_std` ≤ 0.007 on the leaders). Top five CV configurations:

| n_neighbors | weights | metric | acc_mean | acc_std |
|---|---|---|---|---|
| 31 | distance | euclidean | 0.8475 | 0.0016 |
| 31 | uniform  | euclidean | 0.8475 | 0.0016 |
| 31 | uniform  | minkowski(p=3) | 0.8452 | 0.0073 |
| 31 | distance | minkowski(p=3) | 0.8441 | 0.0043 |
| 15 | uniform  | euclidean | 0.8441 | 0.0017 |

Reading the grid:
- **`k` dominates.** On a held-out split, accuracy climbs cleanly from `k=1` (0.733) to `k=31` (0.867) — bias-variance moving in the right direction without an overfit dip.
- **Euclidean wins narrowly.** The top configurations are euclidean; manhattan and minkowski(p=3) trail by a few tenths of a percent — same neighbourhood, slightly different geometry.
- **Uniform vs distance weighting is a wash here.** Top two are identical to four decimals — distance weighting doesn't add signal when the neighbourhood is balanced.

The bootstrap 95% CI on the best config (`k=31, distance, euclidean`) is **[0.7733, 0.9333]** around a held-out accuracy of 0.8533 — a wide band that reflects the small (303-row) sample, not algorithm uncertainty.

**Design choice:** I'd ship `k=21, uniform, euclidean` rather than the apparent best at `k=31` — the simpler model is statistically indistinguishable on this dataset (gap ≪ CI half-width) and degrades less when the optimum drifts a notch on fresh splits.

## Reflections

kNN's strength on small structured data is that it's the most transparent model possible — every prediction is literally "look at these k nearest patients." For a 303-row clinical dataset, that interpretability is worth more than the 1–2 pp of accuracy you'd squeeze out of an ensemble. The clean k=21–31 plateau also tells me something practical: on this size of data, *more sophisticated tuning isn't the bottleneck — more data is.* Knowing which one to chase is the actual judgement call.

The bootstrap CI width ([0.77, 0.93]) is what I'd actually report, not the 0.85 point estimate. "Our held-out accuracy is 85% with a 95% CI of about 8 points" is honest about how much we know on 303 rows; "the model is 85% accurate" implies a precision we don't have. Naming that distinction — point estimate vs uncertainty — is half the conversation when a clinician asks "is the model good enough yet?"

On the systems side, brute-force kNN is O(N) per prediction; running this anywhere with real query volume would need a k-d / ball tree under the hood. The decision-boundary PNGs in `artifacts/` are deliberately the kind of figure a clinician (or anyone outside the modelling team) can glance at and reason about without reading a confusion matrix — they're as much a communication tool as a debugging one.

## Methodology Notes
- 5-fold `StratifiedKFold` for every grid cell, repeated across 5 seeds
  so the reported `acc_mean ± acc_std` reflects fold variance and seed
  variance together.
- The bootstrap CI is computed on **test-set predictions**, not on
  resampled training data, so it estimates the sampling noise of the
  accuracy estimator on a held-out split (cheap and standard).
- Decision-boundary plots are 2-D for visual intuition only and use a
  fresh scaler+model per pair; the headline metric is the full-feature
  grid above.

## Limitations
- 303 rows is small — wide confidence intervals are expected and that's
  part of the lesson; treat absolute differences below ~2 percentage
  points as noise.
- `weights="distance"` can be sensitive to duplicate / near-duplicate
  rows; nothing pathological in this dataset, but the effect is real on
  noisier ones.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0 --n-seeds 5
# or, smaller / faster
python train.py --seed 0 --n-seeds 2
python evaluate.py
```
All outputs land in `artifacts/` (gitignored).
