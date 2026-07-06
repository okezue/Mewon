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
from mewon.optim import MewonR
opt = MewonR(params, lr=0.01, lam=1.0, aspect=True)
```

`MewonR` is the rectangular-native flagship. Gradients pass a bounded-influence gate (`clip` times an EMA of clipped norms — outliers cannot corrupt momentum or scale estimates, per clipped-SGD theory for alpha-stable noise). Householder QR reduces the momentum to its small square factor and an exact thin SVD there replaces Newton-Schulz (which is 20-70% off the true polar factor on ill-conditioned rectangular matrices). Per-direction noise is estimated honestly from the dispersion of the raw gradient around the momentum in the singular frame, `nu=EMA((diag(U^T g V)-s)^2)`, floored at its median across directions (permutation-immune, outlier-robust); singular values are then Wiener-gated by `d=s/sqrt(s^2+lam*nu+tau^2)` and the update scaled by `sqrt(dout/din)`. High SNR gives `d≈1` (exact polar, conditioning-proof); momentum attenuates noise power by `(1-b)/(1+b)` but not signal, so pure-noise directions land in the linear regime `d∝s` (whitened momentum descent, noise-proof) — a self-gating trust region. `lam=0, aspect=False` reproduces Muon's polar direction exactly. State beyond momentum: one length-`min(m,n)` vector and a scalar.

```python
from mewon.optim import Mewon
opt = Mewon(params, lr=0.03, rank=16, freq=8, mode='softpolar', resid=0.05)
```

`Mewon` is the persistent low-rank variant. Modes: `diag` (cached diagonal whitening), `core` (full r×r core whitening), `softpolar` (`C/sqrt(C^2+nw*Var+tau^2)`), `exactsoft` (exact soft-Muon on full momentum).

## What's inside

- `mewon/optim` — `Mewon`, `Muon`, `ExactSoftMuon`, Schatten selectors, hard/soft Muon, Newton-Schulz, cached whitening.
- `mewon/experiments` — theorem regressions, ill-conditioned least squares, drifting subspace, compression, MMD flows, static transport, tiny-GPT training, model probes, kernel timings, vision.
- `mewon/analysis` — JSONL/CSV collection, matplotlib report, plotly dashboard.
- `mewon/integrations` — guarded, idempotent nanoGPT/modded-nanogpt patchers.
- `aws/terraform` — launch scaffold; credentials and buckets are yours to supply.
