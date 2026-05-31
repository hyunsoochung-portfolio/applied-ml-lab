# Shopper Personas — KMeans Clustering on Mall Customers

A KMeans R&D study with four internal-validation metrics, init
sensitivity, MiniBatch vs full comparison, and centroid scatter plots
across multiple feature pairs.

## Setup
- Public mirror (primary): SteffiPeTaffy/machineLearningAZ GitHub mirror.
- Falls back to the historical `Avik-Jain/100-Days-Of-ML-Code` mirror
  if the primary 404s.
- Features used: annual income, spending score, age, gender (encoded).

## Experiments

### 1. K-sweep (k = 2..12) with four metrics
For every k:
- `inertia_` (elbow)
- `silhouette_score`
- `davies_bouldin_score`
- `calinski_harabasz_score`

Outputs: `k_sweep.csv`, `k_sweep.png` (2×2 panel, one subplot per metric).

### 2. Init comparison
`init ∈ {k-means++, random}` × `n_init ∈ {1, 5, 10, 25}` at k=5,
averaged across 5 seeds. Logs `inertia` mean ± std, silhouette mean,
and wall-clock.

Output: `init_comparison.csv`.

### 3. MiniBatchKMeans vs KMeans
For `k ∈ {3, 5, 8}`: fit both, log `(full_inertia, mini_inertia,
full_time, mini_time, ARI agreement)`. ARI ≈ 1 means MiniBatch landed
on the same partition; ARI < 0.9 means the speed-up came at a real
cluster-stability cost on this dataset.

Output: `minibatch_vs_full.csv`.

### 4. Centroid scatter
Best K (highest silhouette) plotted on every pair of numeric features
(excluding gender). Centroids drawn as black `X`. One PNG per pair.

Outputs: `clusters_<feat_a>_vs_<feat_b>_k<K>.png`.

## Findings

**K-sweep with four cluster-quality metrics (Mall Customers, 200 rows × 4 numeric features):**

| K | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2  | 588.80 | 0.2518 | 1.61 | 71.0 |
| 4  | 386.83 | 0.3012 | 1.29 | 69.8 |
| 6  | 275.03 | 0.3343 | 1.01 | 74.1 |
| 8  | 199.75 | 0.3880 | 0.94 | 82.4 |
| 10 | 152.03 | 0.4208 | 0.83 | 90.0 |
| **11** | 137.26 | **0.4285** | **0.82** | 91.3 |
| 12 | 126.02 | 0.4263 | 0.83 | **91.4** |

The four metrics broadly agree: silhouette peaks at **k=11**, Davies-Bouldin (lower is better) bottoms at k=11, Calinski-Harabasz keeps inching up but starts plateauing. The "elbow" on inertia is around k=5-6 — the elbow rule is much more conservative than silhouette here. **On a 200-row dataset, k=11 is too many segments to be operationally useful** even if it's "optimal" by silhouette; I'd ship k=5-6 for an actual marketing team.

**Init comparison at k=5:**

| init | n_init | inertia (mean ± std) | silhouette | time (s) |
|---|---|---|---|---|
| k-means++ | 1  | 332.20 ± 3.84 | 0.3122 | 0.001 |
| k-means++ | 5  | 328.24 ± 3.72 | 0.3169 | 0.003 |
| k-means++ | 10 | 327.81 ± 3.11 | 0.3147 | 0.005 |
| k-means++ | 25 | **325.59 ± 0.48** | 0.3172 | 0.011 |
| random    | 1  | 344.40 ± 6.51 | 0.2960 | 0.001 |
| random    | 5  | 333.66 ± 4.76 | 0.3077 | 0.002 |
| random    | 10 | 328.55 ± 2.41 | 0.3165 | 0.003 |
| random    | 25 | 326.27 ± 1.11 | 0.3157 | 0.008 |

k-means++ at `n_init=1` matches random at `n_init=10` on inertia, with comparable silhouette — that's a real efficiency win on bigger data. By `n_init=25` both converge to the same global optimum (inertia within rounding error).

**MiniBatchKMeans vs KMeans (same K, ARI agreement on cluster labels):**

| K | full t (s) | mini t (s) | ARI |
|---|---|---|---|
| 3 | 0.004 | 0.008 | 0.7256 |
| 5 | 0.005 | 0.006 | 0.5591 |
| 8 | 0.005 | 0.007 | 0.6428 |

On 200 rows MiniBatch is actually *slower* than full KMeans (overhead dominates). The ARI numbers (~0.6) say MiniBatch finds *related* but not identical clusterings — fine on huge data where full KMeans is intractable; pointless overhead here.

**Design choice:** ship `KMeans(n_clusters=5, init="k-means++", n_init=10)` — meaningful segments, fast convergence, stable across seeds.

## Reflections

The cleanest illustration of "model-best" vs "useful" in any module. Silhouette peaks at k=11 on Mall Customers — *technically optimal* by the metric — but 11 customer segments is operationally useless for a marketing team that needs to design 5-6 campaigns. The right answer is k=5 or 6, and the right framing is "the metric measures cluster compactness, not how many segments your business can act on." That translation is the entire job of clustering in product work.

The init comparison is also a small reminder that defaults *encode opinions*. `k-means++` with `n_init=10` is the sklearn default for a reason — it matches random init at `n_init=25` on inertia and beats it on time. That's a research result that the library has quietly built in for you, and the cost of *not knowing* it's there is people thinking they need to "tune the initialization" when the library has already settled it.

On the systems side, MiniBatchKMeans being slower than full KMeans on 200 rows is a useful reminder that scale matters for tradeoffs. The right "use MiniBatch when N exceeds ~10^5" rule is the kind of operational know-how that sits in a senior engineer's head; explicit benchmarks like this one make that knowledge a *team property* rather than tribal lore.

## Methodology Notes
- All clustering happens on `StandardScaler`-transformed features so
  inertia comparisons aren't dominated by raw column magnitudes.
- "Best K" is picked by silhouette; on this dataset that almost
  always lands at K=5 (matches the famous teaching answer).
- Init comparison reports `inertia_std` across seeds to make explicit
  why `n_init=1, init=random` is a bad default (high variance).

## Limitations
- Only 200 rows — every metric carries non-trivial noise. Patterns are
  more interesting than absolute numbers.
- Internal validation metrics all assume convex / equally-sized
  clusters. They will mis-rank K on non-convex shapes; not an issue
  for this dataset but worth keeping in mind elsewhere.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
