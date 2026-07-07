# Mewon: implementation guide

This guide explains how Mewon, a persistent-basis, noise-aware, low-rank spectral optimizer, is implemented and tested. It is inspired by:
- Spectral Wasserstein / Muon geometry
- cached whitening / historical basis methods
- TurboQuant-style rotate-then-scalar-compress ideas

It is organized in three layers:
1. deterministic theorem-validation experiments on matrices;
2. synthetic optimization tasks;
3. small language-model training, first in a minimal PyTorch setup and then in NanoGPT/modded-nanogpt.

---

## 1. What to implement first

Implement three optimizer variants.

### A. Exact soft-Muon (full-rank, theorem-faithful)
For a matrix gradient or momentum block `M`:
1. compute SVD `M = U diag(s) V^T`
2. transform singular values with
   `d_i = min(s_i / lambda, rho)`
3. update with
   `Delta = - U diag(d) V^T`

This is the exact projection of `-M/lambda` onto the spectral-norm ball.
Use it for theorem validation and as a clean baseline.

### B. Cached diagonal noise-aware rule (theorem-faithful persistent basis)
Maintain orthonormal cached bases `U, V`.
At each step:
1. form `C = U^T M V`
2. extract `c = diag(C)`
3. estimate per-coordinate noise weights `a_i > 0`
4. set `d_i = clip(c_i / (a_i + eps), rho)`
5. use `Delta = - U diag(d) V^T`

This is the exact optimizer implied by the diagonal-restricted mean-plus-variance surrogate.
Use it for theorems about drift, leakage, and compression.

### C. Smooth practical variant (recommended for actual NanoGPT runs)
Use the same cached bases, but keep a full `r x r` core and smooth shrinkage:
1. `C = U^T M V`
2. maintain `Vcore = EMA(C ⊙ C)`
3. transform with
   `Cupd = C / sqrt(C^2 + alpha * Vcore + tau^2)`
4. reconstruct
   `Delta = - U Cupd V^T`
5. optionally add a residual path outside the tracked basis.

This is not the exact theorem operator, but it is smoother and usually easier to train with.

---

## 2. Parameter grouping in transformers

Use the spectral optimizer only on matrix-valued hidden-layer weights.
Keep the following on AdamW unless you have a very good reason not to:
- token embeddings
- position embeddings
- layer norms / RMSNorm scales
- biases
- tied output head / embedding share
- very small matrices where orthogonalization overhead dominates

Good default split:
- spectral optimizer: 2D weights in attention projections and MLP projections
- AdamW: everything else

Optional ablation:
- include untied `lm_head` in the spectral optimizer only if it is not weight-tied
- split packed QKV matrices into separate blocks vs treat them jointly

---

## 3. State you must maintain per spectral parameter

For each 2D parameter `W`:
- `M`: full momentum buffer, fp32, shape like `W`
- `U`: left basis, shape `(m, r)`
- `V`: right basis, shape `(n, r)`
- `var_diag`: length-`r` diagonal variance estimate if using diagonal rule
- or `var_core`: `(r, r)` variance estimate if using the smooth full-core rule
- step counter
- optional quantized stored states for compression experiments

Recommended dtype policy:
- parameters: bf16/fp16 for training if supported
- optimizer state: fp32
- quantized state experiments: quantize only after the fp32 path is already stable

---

## 4. Basis refresh

### Warm-started block power iteration
This is the simplest robust refresh.
Given full momentum `M` and previous `U, V`:
1. `Y = M @ V`; orthonormalize columns -> new `U`
2. `Z = M.T @ U`; orthonormalize columns -> new `V`
3. repeat 1-2 a small number of times

Use previous `U, V` as the warm start.

### Orthonormalization choices
For research code:
- QR or eigen-based inverse square root on the small Gram matrix is fine.

For production / GPU-friendliness:
- replace the small-kernel orthogonalization with Newton-Schulz / Polar-Express / related matmul-heavy kernels.

### Refresh schedule
Start with periodic refresh every `k` steps.
Sweep:
- `k in {1, 2, 4, 8, 16, 32}`

Also test trigger-based refresh:
- refresh if energy capture drops below a threshold
- refresh if off-diagonal leakage exceeds a threshold
- refresh if principal-angle drift estimate exceeds a threshold

---

## 5. Noise estimation

You need two distinct paths because the exact theorem and best practical optimizer are not identical.

### Theorem-faithful diagonal noise estimate
In cached coordinates, form `c_t = diag(U^T g_t V)` or `diag(U^T M_t V)` and update
`a_t = beta2 * a_{t-1} + (1-beta2) * c_t^2`
Then use
`d_t = clip(c_t / (a_t + eps), rho)`.

### Practical Adam-like estimate
More likely to train well:
`v_t = beta2 * v_{t-1} + (1-beta2) * c_t^2`
`d_t = clip(c_t / (sqrt(v_t) + eps), rho)`
This is not the exact theorem object, but it is a strong practical ablation because it separates:
- persistent basis
- variance adaptation
- saturation / orthogonalization

### Full-core estimate
If keeping a full `r x r` core:
`Vcore_t = beta2 * Vcore_{t-1} + (1-beta2) * (C_t ⊙ C_t)`
then use the smooth soft-polar rule.

---

## 6. Residual handling outside the tracked basis

If `r < min(m, n)`, decide what to do with the residual
`R = M - U (U^T M V) V^T`.

Test all four:
1. `drop`: ignore residual entirely
2. `rms`: add `alpha * R / rms(R)`
3. `adam-like`: apply a scalar adaptive update to the residual
4. `error feedback`: store residual compression error and add it back next step

For theorem-validation experiments, start with `drop`.
For real training, `rms` or `error feedback` are much more likely to help.

---

## 7. Quantization path

Implement quantization only after the unquantized optimizer is stable.

### What to quantize first
The safest order is:
1. quantize cached diagonal/core states, not the parameter tensor
2. keep `U, V` in fp16/bf16 or fp32 initially
3. later quantize `U, V` or store them less frequently if needed

### Best first quantization target
Quantize `D = diag(U^T M V)` or the full small core `C = U^T M V`.
This is the direct compression experiment predicted by the theory.

### Quantization schemes to compare
- uniform scalar quantization per coordinate
- blockwise linear quantization
- logarithmic quantization
- TurboQuant-style random rotation followed by scalar quantization
- persistent-basis coordinates followed by scalar quantization
- fresh-SVD coordinates followed by scalar quantization

The main question is not just “which compresses best?” but:
- does the learned persistent basis beat random rotations?
- how close is it to fresh optimal spectral coordinates?

---

## 8. Exact theorem-validation experiments

These are cheap and should be done before any language-model training.

### Experiment A: soft-Muon equals projection onto spectral-norm ball
For random matrices `G`:
1. compute theorem solution by SVD
2. solve the constrained quadratic program with CVX/CVXPY for tiny matrices
3. verify equality of objective values and solutions

Metrics:
- Frobenius difference between closed form and solver solution
- objective gap

### Experiment B: Lipschitz / firm nonexpansiveness
For random `G, H`:
1. compute `T(G), T(H)`
2. verify numerically
   `||T(G)-T(H)||_F <= ||G-H||_F / lambda`
3. verify the stronger inner-product inequality

Metrics:
- worst observed ratio
- histogram over random trials

### Experiment C: condition number monotonicity
For matrices with known singular spectra, verify
`kappa(T(G)) <= kappa(G)`
for soft-Muon.

### Experiment D: threshold stability
Construct matrices with singular values near and far from `lambda * rho`.
Perturb by noise `E` and test whether the active saturated set changes only when `||E||_op` crosses the proven margin.

### Experiment E: diagonal restriction price
For a fixed cached basis, verify exactly that
`J(diag-best) - J(full-best) = ||offdiag(C)||_F^2 / (2a)`
for the scalar-quadratic surrogate.

### Experiment F: compression decomposition identity
For arbitrary `X`, fixed `U, V`, and diagonal code `Dhat`, verify exactly that
`||X - U Dhat V^T||_F^2 = tail^2 + leakage^2 + scalar_error^2`.

These experiments should pass to numerical precision.

---

## 9. Synthetic optimization suite

Use synthetic tasks because they let you control condition number, noise, and subspace drift.

### Task 1: ill-conditioned least squares
Train `W` in
`f(W) = 0.5 ||A W B - Y||_F^2`
where `A, B` have prescribed singular spectra.

Sweep condition numbers:
- `1e1, 1e2, 1e3, 1e4`

Use additive minibatch-like gradient noise to test robustness.

Record:
- objective vs steps
- objective vs wall-clock
- update condition number
- spectral saturation count
- basis drift and gap statistics

### Task 2: low-rank matrix factorization
Use
`f(U, V) = 0.5 ||UV^T - M*||_F^2`
with noisy observations.
This tests whether a low-rank persistent basis tracks the true signal subspace.

### Task 3: heavy-tailed noise
Inject Student-t or Pareto-tailed gradient noise.
This is important because large singular directions and outliers are where saturation should help most.

### Task 4: drifting subspace task
Construct time-varying targets whose optimal singular subspace rotates slowly or quickly.
This directly tests when basis reuse stops being valid.

---

## 10. Reproduce Peyré’s Section 7 first

The paper’s own experiments compare static spectral couplings for `p = 1, 2, infinity` and MMD gradient flows using the corresponding explicit selectors. Reproducing those experiments gives you a geometry sanity check before touching transformers.

### Static coupling experiment
Use two point clouds with equal weights and solve
`min_P gamma_p(sum_ij P_ij (y_j - x_i)(y_j - x_i)^T)`
for `p = 1, 2, infinity`.

Implementation options:
- CVXPY for `p = 2` and `p = infinity`
- assignment / OT solver for `p = 1`

Outputs:
- coupling matrix
- displacement covariance
- value of `gamma_p`
- visualization of extracted permutation / coupling lines

### MMD flow experiment
Empirical measures:
- anisotropic Gaussian source cloud
- farther-away Gaussian-mixture target cloud

Use the energy-distance kernel with smoothing.
Implement explicit Euler updates with the exact Schatten selector `Xi_p` for:
- `p = 1`: identity / W2-like
- `p = 2`: intermediate
- `p = infinity`: Muon-like

Outputs:
- all trajectories
- final MMD
- covariance of update field over time

---

## 11. Small neural-network sanity checks

Before NanoGPT, run two quick neural experiments.

### MLP on MNIST / CIFAR-like toy setup
Use a 2- or 3-layer MLP.
Apply spectral optimizer only to hidden weight matrices.

Why this helps:
- much cheaper than GPT
- still gives real minibatch noise
- easy to test small-batch robustness

### Tiny transformer on Shakespeare or TinyStories subset
Use a very small GPT:
- 6-12 layers
- width 256-512
- seq len 128-256

This is where you verify that the optimizer actually integrates cleanly with autoregressive LM training.

---

## 12. NanoGPT / modded-nanogpt implementation path

### Which repo to use
Use two tracks.

#### Track A: original karpathy/nanoGPT
Use this if you want the simplest hackable code path.
It is old, but the code is compact and easy to modify.

#### Track B: modded-nanogpt
Use this if you want a Muon-native / speedrun-oriented baseline.
This is where Muon comparisons make the most sense.

### File changes
1. `from mewon.optim import Mewon`
2. modify optimizer construction in `train.py`
3. split parameters into spectral vs auxiliary groups
4. add logging hooks for basis diagnostics
5. optionally add checkpoint save/load of `U, V, var_*`

### Minimal parameter split
Pseudo-code:

```python
def splitparams(model):
    spec,aux=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim==2 and 'wte' not in name and 'wpe' not in name and 'head' not in name: spec.append(p)
        else: aux.append(p)
    return spec,aux
```

### Training-step structure
1. zero both optimizers
2. forward
3. backward
4. unscale if using AMP
5. optional global grad clip
6. step spectral optimizer
7. step AdamW optimizer
8. scheduler step(s)
9. log diagnostics

### Good starting hyperparameter ranges
For a first sweep on a small GPT:
- spectral lr: `0.01, 0.02, 0.03, 0.05`
- AdamW lr for aux params: `2e-4, 4e-4, 6e-4`
- beta1: `0.9, 0.95`
- beta2 or variance EMA: `0.95, 0.98, 0.995`
- rank: `8, 16, 32, 64`
- basis refresh: `1, 4, 8, 16`
- residual coeff: `0.0, 0.1, 0.25`
- clip/saturation `rho`: `0.25, 0.5, 1.0, 2.0`
- weight decay on spectral params: start with `0.0`, then test small values
- weight decay on AdamW params: standard model-tuned values

### Two recommended implementations

#### Implementation 1: exact, research-clean
- full SVD soft-Muon on every spectral parameter
- no low-rank truncation
- no quantization

Use this only on tiny models or infrequent refreshes.
It is the cleanest baseline.

#### Implementation 2: persistent low-rank practical
- full fp32 momentum buffer
- rank-`r` cached basis
- periodic warm-started basis refresh
- full `r x r` core or diagonal-only update
- optional residual correction
- optional quantized core storage

This is the main experiment.

---

## 13. What to log in every NanoGPT run

Per training step or every `N` steps:
- train loss
- val loss
- tokens/sec
- step time
- optimizer wall-clock share
- peak memory

Per spectral parameter group / selected layers:
- energy capture: `||U^T M V||_F^2 / ||M||_F^2`
- off-diagonal leakage: `||offdiag(U^T M V)||_F / ||U^T M V||_F`
- residual ratio: `||M - U(U^T M V)V^T||_F / ||M||_F`
- principal angle drift after refresh
- singular gap proxy `sigma_r - sigma_{r+1}` if computable on sampled layers
- update condition number
- fraction of saturated coordinates
- quantization error in cached coordinates if quantized

These are not optional. Without them you will not know why a run won or failed.

---

## 14. Core ablations to run

### Geometry / optimizer ablations
- AdamW only
- hard Muon
- soft-Muon
- cached diagonal theorem-faithful rule
- cached full-core smooth rule
- cached full-core + residual

### Basis ablations
- full-rank vs low-rank
- random basis vs fresh SVD basis vs persistent basis
- basis refresh every step vs periodic refresh

### Noise ablations
- no variance adaptation
- diagonal variance adaptation
- full-core variance adaptation
- small batch vs large batch
- explicit injected noise

### Compression ablations
- no quantization
- 8-bit, 6-bit, 4-bit, 3-bit core/diag quantization
- random rotation + scalar quantization
- persistent basis + scalar quantization
- fresh SVD basis + scalar quantization

### Systems ablations
- QR/eigh basis orthonormalization vs Newton-Schulz / Polar-Express variant on the small core
- with and without residual path
- with and without error feedback

---

## 15. The three flagship hypothesis tests

### Hypothesis 1: persistent coordinates exist
Evidence would be:
- principal-angle drift remains small over many steps
- a low-rank basis captures a large fraction of momentum energy
- off-diagonal leakage in cached coordinates is small

Primary plots:
- drift vs step
- leakage vs step
- energy capture vs rank

### Hypothesis 2: one basis helps optimization and compression at once
Evidence would be:
- the same persistent basis that gives a good update also gives low quantization distortion
- persistent basis beats random rotations for equal bit budget
- persistent basis approaches fresh-SVD performance with fewer refreshes

Primary plots:
- val loss vs bits
- distortion vs bits
- val loss vs refresh interval

### Hypothesis 3: variance adaptation is the missing ingredient
Evidence would be:
- persistent-basis + variance adaptation beats persistent-basis sign-only
- gains grow as batch size shrinks or noise rises

Primary plots:
- val loss vs steps under batch sweep
- training stability / divergence rate
- update SNR in cached coordinates

---

## 16. Recommended experiment order

### Phase 0: theorem verification (1 day)
Run exact matrix tests A-F.
If any identity fails, stop and fix the implementation.

### Phase 1: synthetic optimization (1-3 days)
Run least-squares, factorization, noise, and drifting-subspace tasks.
Choose the best two variants.

### Phase 2: tiny transformer (1-3 days)
Run small GPT on Shakespeare or TinyStories subset.
Tune only a narrow band of learning rates and ranks.

### Phase 3: NanoGPT baseline (3-7 days)
Compare against AdamW and Muon on a reproducible small/medium setup.
Collect all geometry diagnostics.

### Phase 4: compression study (2-5 days)
Turn on core/diag quantization and compare against random-rotation baselines.

### Phase 5: large run (optional)
Only after the small runs show:
- slow drift
- low leakage
- small compression penalty
- stable training curves

---

## 17. Failure modes to watch for

- basis collapse or NaNs during orthonormalization
- stale basis causing leakage spikes
- too-small rank causing unstable residuals
- quantizing too early before optimizer is stable
- tying `lm_head` and embedding but accidentally optimizing them with different rules
- comparing wall-clock without measuring orthogonalization overhead
- comparing perplexity/loss with unfairly untuned learning rates

---

## 18. Practical success criteria

A result is genuinely interesting if you get at least one of these:

1. **Scientific**: low-rank persistent coordinates really exist for important GPT layers; drift and leakage stay small.
2. **Algorithmic**: cached-basis + variance adaptation matches or beats Muon at lower memory.
3. **Systems**: compressed cached-core states keep validation loss essentially unchanged at 8-bit or lower.
4. **Theoretical-experimental bridge**: the measured leakage term predicts the actual diagonal restriction penalty and quantization penalty.

---

## 19. Code map

- `mewon/optim/core.py` — `Mewon`, `MewonP`, `MewonR`, `Muon`, `ExactSoftMuon`, `updatebasis`
- `mewon/optim/ops.py` — `softmuon`, `hardmuon`, `schatten`, `nsorth`, `polarmuon`
- `mewon/experiments/` — theorem regressions, synthetic optim, compression, mmd, transport, lm, probe, kernels, vision
- `mewon/diag.py`, `mewon/quant.py` — basis diagnostics and rotate-then-compress
- `mewon/integrations/nanogpt.py` — guarded train.py patcher

