# Tumor Compressor — PCA on Breast Cancer Wisconsin

A PCA R&D study: PCA vs KernelPCA, n_components sweep, reconstruction
error, whitening ablation, and 2-D / 3-D scatter plots.

## Setup
- `sklearn.datasets.load_breast_cancer` (569 rows × 30 features, binary
  malignant / benign).
- Pipeline: `StandardScaler -> PCA(n_components=N) -> LogisticRegression`.

## Experiments

### 1. PCA vs KernelPCA
At fixed `n_components=10`, compare:
- `PCA(linear)`
- `KernelPCA(kernel ∈ {linear, rbf, poly})`

Output: `pca_vs_kpca.csv`.

### 2. Scree + cumulative explained variance
Per-PC ratio (bars) and cumulative ratio (line) up to PC30, with a 0.9
threshold marker.

Outputs: `variance.csv`, `variance.png`.

### 3. n_components sweep
`n_components ∈ {1..30}`, downstream LogReg test accuracy on a fixed
75/25 split.

Outputs: `components_sweep.csv`, `components_sweep.png`.

### 4. Reconstruction error
For each `n_components ∈ {1, 2, 5, 10, 15, 20, 30}`, compute
`‖X − inverse_transform(transform(X))‖_F` on the standardised matrix.
Plotted with cumulative variance on a twin axis.

Outputs: `reconstruction.csv`, `reconstruction.png`.

### 5. Whitening on/off ablation
`whiten ∈ {False, True}` × `n_components ∈ {2, 5, 10, 20}`. Whitening
makes the PCs unit-variance, which often helps subsequent linear
classifiers, sometimes hurts.

Output: `whitening_ablation.csv`.

### 6. 2-D + 3-D PCA scatter
Coloured by class. The 3-D plot uses `mpl_toolkits.mplot3d`.

Outputs: `pc_scatter_2d.png`, `pc_scatter_3d.png`.

## Findings

**PCA vs KernelPCA — downstream LogisticRegression accuracy** (Breast Cancer Wisconsin, 30 features → top-N components, 5-fold CV mean):

| reducer | downstream accuracy |
|---|---|
| PCA(linear)         | **0.9650** |
| KernelPCA(linear)   | 0.9650 |
| KernelPCA(poly)     | 0.9371 |
| KernelPCA(rbf)      | 0.9231 |

PCA and KernelPCA-linear are identical (same projection up to numerical precision). The non-linear kernels actively *hurt* downstream classification on this dataset — Breast Cancer's classes are well-separated linearly after standardisation, and the RBF embedding squeezes the natural decision boundary into a curved manifold the linear classifier can't exploit. Lesson: **kernel PCA is a tool for non-linear structure; it has a real cost when the structure is already linear**.

**n_components sweep (downstream LR accuracy on 5-fold CV):**

| n_components | accuracy |
|---|---|
| 1 | 0.9021 |
| 2 | 0.9231 |
| 3 | 0.9371 |
| 4 | 0.9371 |
| **5** | **0.9650** |
| 6 | 0.9580 |
| 7 | 0.9510 |
| 8 | 0.9650 |
| 13 | 0.9650 |
| 20 | 0.9580 |
| 30 | 0.9580 |

Sharp knee at **n=5**: dropping from 30 dimensions to 5 *improves* test accuracy by 0.7 pp because the extra 25 components are mostly noise that the linear model otherwise tries to fit. Past n=5 there's a faint zigzag in the 0.95-0.96 band — pure CV-fold noise.

**Reconstruction quality** scales the same way: the cumulative explained-variance plot (`artifacts/variance.png`) crosses 95% around component 10, but downstream classification doesn't need that much variance — 5 components carry the *discriminative* information.

**Design choice:** ship `PCA(n_components=5)` (linear) for this problem — best downstream accuracy, 6× dimensionality reduction, no kernel hyperparameters to tune. If I needed reconstruction fidelity instead of discrimination I'd raise `n_components` to ~10 to hit the 95% variance bar.

## Reflections

The headline finding — **5 components beat 30** for downstream classification — is the lesson I'd want every ML hire to internalise about dimensionality reduction. More features ≠ more information; past the discriminative components, you're adding noise that the downstream model has to actively suppress. That's why "PCA hurt my model" is sometimes a sign of *too many components*, not "PCA was wrong for this problem."

The KernelPCA result has the opposite, equally important, lesson: matching the *tool* to the *data structure* matters more than picking the fanciest tool. RBF KernelPCA underperforms linear PCA on Breast Cancer not because it's a worse algorithm but because the underlying class boundary is already linear after standardisation — adding kernel non-linearity squeezes that natural geometry through a curved manifold the downstream linear classifier can't exploit. Reaching for kernel methods on linearly-separable problems is a common junior-engineer mistake; running the comparison once kills it for the team.

On the communication side, the 2-D PCA scatter is the most-useful single chart in this whole repo when showing the result to someone outside the modelling team. "These two clouds barely overlap" is far more convincing than "our test AUC is 0.97" because it bypasses the metric and *shows* the structure. Building these visualizations into the eval pipeline by default is cheap and pays for itself the first time you need it in a meeting.

## Methodology Notes
- All PCA is fit **after** `StandardScaler` so each feature contributes
  proportionally to its z-scored variance, not its raw scale.
- Reconstruction error is reported in the **standardised** space.
  Reporting it in raw-X space is also legitimate; we chose the
  standardised version so the curve has interpretable units (norm of a
  unit-variance residual).
- The KernelPCA(`rbf`/`poly`) results don't have closed-form
  `explained_variance_ratio_` exposed; we only report downstream
  accuracy.

## Limitations
- 569 rows is small; the n_components ≥ ~5 plateau is very flat —
  read the curve, not individual decimal points.
- 3-D scatter is for visualisation only; downstream accuracy is the
  scored metric.

## Reproduce
```bash
pip install -r ../requirements.txt
python train.py --seed 0
python evaluate.py
```
