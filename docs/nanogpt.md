# Mewon: NanoGPT / modded-nanogpt notes

Smallest practical way to try `MewonP` in a NanoGPT-style loop. The patcher in `mewon/integrations/nanogpt.py` does steps 1-2 automatically and is guarded: it backs up `train.py`, aborts if the optimizer anchor is missing, and is idempotent.

## 1) Import

```python
from mewon.optim import MewonP
```

## 2) Split parameters

Mewon on hidden matrix weights only, AdamW for everything else.

```python
def splitparams(model):
    spec,aux=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim==2 and 'wte' not in name and 'wpe' not in name and 'head' not in name: spec.append(p)
        else: aux.append(p)
    return spec,aux

spec,aux=splitparams(model)
ospec=MewonP(spec,lr=0.03,betas=(0.95,0.98),rank=16,freq=8,piters=1,nw=1.0,tau=1e-3,resid=0.1,wd=0.0)
oaux=torch.optim.AdamW(aux,lr=adamw_lr,betas=(0.9,0.95),weight_decay=wd)
```

## 3) Step

```python
ospec.zero_grad(set_to_none=True); oaux.zero_grad(set_to_none=True)
with ctx: logits,loss=model(X,Y)
scaler.scale(loss).backward()
scaler.unscale_(ospec); scaler.unscale_(oaux)
torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
scaler.step(ospec); scaler.step(oaux); scaler.update()
```

Without AMP just call `ospec.step(); oaux.step()`.

## 4) Log per spectral layer

`MewonP.diagstate()` returns energy capture and off-diagonal leakage per tracked parameter. Track those, residual ratio `||M-U(U^T M V)V^T||/||M||`, principal-angle drift after refresh, and optimizer-step wallclock.

## 5) First ablations

`rank in {8,16,32}`, `freq in {4,8,16}`, `nw in {0.5,1.0,2.0}`, `resid in {0.0,0.1,0.25}`, against AdamW and full-rank Muon.

## 6) Evidence for the hypothesis

Cached bases drift slowly, a low-rank basis captures most momentum energy, transformed coordinates are less correlated than raw, updates stay stable across lr sweeps, and low-bit state compression in cached coordinates barely hurts loss.
