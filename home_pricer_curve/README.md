# Home Pricer (Curve) — Linear & Polynomial Regression on California Housing

Linear baseline plus a polynomial-degree study with a numerical-stability
side channel: condition number per design matrix.

## Setup
- `sklearn.datasets.fetch_california_housing` (20 640 rows × 8 features,
  target = `MedHouseVal`).
- Pipeline: `StandardScaler -> PolynomialFeatures(degree, interaction_only)
  -> LinearRegression`.

## Experiments

### 1. Degree sweep (1..6) with CV R² + MAE
For each `degree ∈ {1..6}` and each `interaction_only ∈ {False, True}`
we run `cross_validate` (`KFold(5)`) and log mean ± std of R² and MAE.

Outputs:
- `poly_degree_sweep.csv` — `(degree, interaction_only, cv_r2_mean,
  cv_r2_std, cv_mae_mean, cv_mae_std)`
- `poly_degree_curve.png` — error-bar plot for both metrics

### 2. Condition-number diagnostic per design matrix
For each `(degree, interaction_only)` we build the standardised
polynomial design matrix and compute `cond(Z) = σ_max / σ_min`. Catches
the moment OLS becomes numerically dicey — the curve typically rises
several orders of magnitude as degree grows, particularly with
`interaction_only=False`.

Output: `condition_number.csv`.

### 3. Interaction-only vs full polynomial
Same sweep as (1) but with `interaction_only=True` overlayed — strips
the per-feature higher-order terms (no x², x³, ...) and keeps only
cross-products. Useful when monomials blow up the condition number but
cross terms still help.

### 4. 1-D synthetic extrapolation
kNN flatlines past the training range; `LinearRegression` keeps the
trend slope. Static plot in `extrapolation_demo.png`.

## Findings

**LinearRegression baseline (held-out):** train R² = 0.6089, **test R² = 0.5943**, test MAE = 0.5351 — a clean linear fit that explains roughly 60% of variance in California Housing.

**Polynomial degree sweep (5-fold CV R²):**

| degree | full polynomial | interaction-only |
|---|---|---|
| 1 | 0.6037 ± 0.0079 | 0.6037 ± 0.0079 |
| 2 | -0.3260 ± 0.8763 | -0.0516 ± 0.8492 |
| 3 | -54,701 ± 108,929 | -11.79 ± 17.82 |
| 4 | -8.9 × 10⁹ ± 1.7 × 10¹⁰ | -757 ± 1,223 |
| 5 | -4.6 × 10¹⁵ ± 9.2 × 10¹⁵ | -1,437 ± 1,902 |
| 6 | -4.4 × 10¹⁶ ± 8.8 × 10¹⁶ | -1,679 ± 2,314 |

Polynomial expansion without regularization is a footgun on this dataset. The expanded design matrix is wildly ill-conditioned and OLS coefficients explode:

**Design matrix condition number per degree:**

| degree | n_features (full) | cond (full) | n_features (interaction-only) | cond (interaction-only) |
|---|---|---|---|---|
| 1 | 8    | 6.5 × 10⁰  | 8   | 6.5 × 10⁰ |
| 2 | 44   | 8.8 × 10³  | 36  | 1.7 × 10³ |
| 3 | 164  | 1.1 × 10⁷  | 92  | 1.3 × 10⁴ |
| 4 | 494  | 1.4 × 10¹⁰ | 162 | 5.7 × 10⁴ |
| 5 | 1286 | 3.0 × 10¹³ | 218 | 2.1 × 10⁵ |
| 6 | 3002 | 1.6 × 10¹⁷ | 246 | 4.8 × 10⁵ |

`interaction_only=True` keeps the condition number 6–8 orders of magnitude smaller at high degrees because it suppresses pure power terms (x², x³, ...) that are nearly collinear with the originals after standardisation. It buys you a few more degrees before the model explodes, but it does **not** rescue you — the right answer is to add regularization, which is exactly the next module.

**1-D extrapolation demo** (kNN vs LinearRegression on `y = 2x + noise`, trained on x ∈ [0, 10]):

| x (test point) | kNN(k=5) | LinearRegression | true (slope 2) |
|---|---|---|---|
| 18 | 18.33 | 33.78 | 36 |

kNN clips at the maximum training neighbourhood (~9.17 on average → 18.33 after the +noise structure) and refuses to extrapolate. LinearRegression respects the linear structure and extrapolates correctly. This is the structural argument for linear models when the underlying generative process is parametric and you might query out-of-range inputs.

## Reflections

This project is the cleanest "more complexity, less performance" story in classical ML. Polynomial expansion sounds like a free upgrade ("we can fit non-linear shapes now!") but on California Housing's correlated features the design-matrix condition number jumps 16 orders of magnitude going from degree 1 to 6, and OLS coefficients explode. The general principle — *capacity without regularization is a footgun on correlated features* — is one I keep coming back to, including in deep nets.

This is also the kind of result you have to explain carefully. Anyone hearing "we tried a more powerful model and it got worse" will assume something went wrong. The honest framing is "we removed a constraint that was doing useful work without realising it, and the next module — regularization — puts a smarter version of that constraint back." Phrasing "fit got worse because we removed structure" in plain language matters because the easy mistake is for someone to conclude "more complex models are bad" rather than "complexity without constraint is bad."

The 1-D extrapolation demo is small but it sums up *when* to reach for linear models in a deployed system: if your production inputs can drift outside the training range (price spikes, sensor failures, post-launch usage patterns), kNN and tree models silently clip to the training distribution. Linear models extrapolate honestly — sometimes badly, but at least *visibly* badly, which is the property an oncall engineer actually wants.

## Methodology Notes
- The CV is done on the **full** dataset; the held-out hold-out split
  reported at the top of the run is for a quick eyeball check only.
- Condition number is computed on a 4000-row subsample (random) so
  large degree matrices fit in memory; SVD cost is the dominant term.
- All polynomial expansion happens **after** scaling so the bias term
  isn't dragged around by raw column magnitudes.

## Limitations
- California Housing has a clipped target (`MedHouseVal` caps at 5.0)
  which limits the gain from high-degree expansions — they fit the
  bulk of the distribution but lose the censored tail. Visible as a
  diagonal stripe in any residual plot of `MedHouseVal ≈ 5`.
- Degrees ≥ 5 generate hundreds of features on 8 inputs — `cond(Z)`
  often exceeds `1e10` and OLS coefficients become essentially
  uninterpretable. `home_pricer_balanced` (Ridge / Lasso / ElasticNet) is the
  natural follow-up.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
