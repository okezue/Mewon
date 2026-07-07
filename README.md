# Mewon

Persistent low-rank spectral optimizer and a research suite around it: noise-aware whitening, Muon/soft-Muon baselines, spectral-Wasserstein selectors, and rotate-then-scalar-compress experiments.

`Mewon` is the flagship optimizer. It keeps a cached rank-`r` basis `U,V` per 2D weight, whitens the momentum core in those coordinates, and refreshes the basis with warm-started power iteration. `Muon` and `ExactSoftMuon` are the spectral baselines.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

## Run

```bash
mewon validate --out runs/validate
mewon suite --config configs/quick.yaml --out runs/quick
mewon report --runs runs/quick --out runs/quick_report
mewon dashboard --runs runs/quick --out runs/quick_dash
```

`validate`/`quick` run CPU-friendly versions of every experiment. The full suite is GPU-oriented:

```bash
bash scripts/fullsuite.sh
```

FineWeb-scale runs need your own compute and cache:

```bash
RUN_FINEWEB=1 PREPARE_FINEWEB=1 FINEWEB_TOKENS=1000000000 bash scripts/fullsuite.sh
```

## Optimizers

```python
from mewon.optim import Mewon
opt = Mewon(params, lr=0.02, lam=1.0, clip=1.5, zeta=0.05)
```

`Mewon` is the flagship: a bulk-aware spike-polar optimizer that treats momentum as a spiked random matrix rather than a set of scalar singular channels. Gradients pass a bounded-influence gate (`clip` times an EMA of clipped norms). The gradient-noise scale is estimated from the median singular value of the innovation `g - m` against the precomputed Marchenko-Pastur median for the layer's aspect ratio; the momentum noise scale follows from the exact EMA attenuation `(1-b)/(1+b)*(1+b^t)/(1-b^t)`. Only singular values above the MP bulk edge `sigma*(1+sqrt(phi))` (plus a `k^(-2/3)` Tracy-Widom margin) are treated as signal; those are de-biased through the BBP spike inversion `theta^2=(A+sqrt(A^2-4*phi*sigma^4))/2`, attenuated by the Benaych-Georges-Nadakuditi singular-vector alignment, and gated by `theta/sqrt(theta^2+lam*sigma^2)`. Below-edge structure is served by a small rms-normalized residual path that fades as spikes explain the step. Pure-noise momentum produces exactly zero spectral update — where Muon takes a full polar step in a random direction. Exact rectangular SVD via Householder QR; update scaled by `sqrt(dout/din)`. State beyond momentum: two scalars and one cached constant.

```python
from mewon.optim import MewonR
opt = MewonR(params, lr=0.01, lam=1.0, aspect=True)
```

`MewonR` is the scalar-channel predecessor: per-direction noise from the dispersion of the raw gradient around momentum in the singular frame, `nu=EMA((diag(U^T g V)-s)^2)` floored at its median, Wiener gate `d=s/sqrt(s^2+lam*nu+tau^2)`, same clip and aspect scaling. `lam=0, aspect=False` reproduces Muon's polar direction exactly.

```python
from mewon.optim import MewonP
opt = MewonP(params, lr=0.03, rank=16, freq=8, mode='softpolar', resid=0.05)
```

`MewonP` is the persistent low-rank variant. Modes: `diag` (cached diagonal whitening), `core` (full r×r core whitening), `softpolar` (`C/sqrt(C^2+nw*Var+tau^2)`), `exactsoft` (exact soft-Muon on full momentum).

## Results

Tiny-Shakespeare char-LM (0.8M-param 4-layer GPT, held-out val loss, 1200 steps, 2 seeds, per-optimizer tuned lr at interior optima): Mewon 1.720±0.008, MewonR 1.715±0.002, AdamW 1.792±0.007, soft-Muon 2.460, Muon 2.523. Ill-conditioned LS (cond 1e4, clean objective): MewonR 5.1e-9, Mewon 5.0e-8, AdamW 2.6e-8, soft-Muon 6.3e-5; with fresh gradient noise MewonR 3.3e-6, AdamW 4.3e-6, Mewon 5.6e-6. Heavy-tailed data x~t(2) (rank-1 gradient outliers): Mewon 40/7.1/98 and MewonR 44/3.6/50 vs AdamW 115/88/84. Elementwise-iid heavy noise (the MP bulk itself): Mewon 5.2/5.3 beats AdamW 7.4/7.7 — the only optimizer here that does; MewonR trails ~3x (coordinate-aligned noise is Adam's matched basis; the bulk edge is its spectral counter). Pure-noise momentum yields exactly zero spectral update under Mewon (regression-tested).

## What's inside

- `mewon/optim` — `Mewon`, `Muon`, `ExactSoftMuon`, Schatten selectors, hard/soft Muon, Newton-Schulz, cached whitening.
- `mewon/experiments` — theorem regressions, ill-conditioned least squares, drifting subspace, compression, MMD flows, static transport, tiny-GPT training, model probes, kernel timings, vision.
- `mewon/analysis` — JSONL/CSV collection, matplotlib report, plotly dashboard.
- `mewon/integrations` — guarded, idempotent nanoGPT/modded-nanogpt patchers.
- `aws/terraform` — launch scaffold; credentials and buckets are yours to supply.
