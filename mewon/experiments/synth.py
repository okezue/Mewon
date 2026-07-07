import torch
from mewon.data.synthetic import lsprob,driftmat
from mewon.optim import Mewon,MewonP,ExactSoftMuon,Muon
from mewon.tracking import RunLogger
from mewon.utils import setseed

def makeopt(name,params,lr):
    if name=='adamw': return torch.optim.AdamW(params,lr=lr)
    if name=='muon': return Muon(params,lr=lr,scale='rms')
    if name=='softmuon': return ExactSoftMuon(params,lr=lr,lam=1.0,rho=1.0)
    if name=='mewondiag': return MewonP(params,lr=lr,mode='diag',rank=8,freq=4)
    if name=='mewonp': return MewonP(params,lr=lr,mode='softpolar',rank=8,freq=4,resid=0.05)
    return Mewon(params,lr=lr)

def runls(outdir,seed=0,steps=100,opt='mewon',cond=1e4,dev='cpu'):
    setseed(seed); dev=torch.device(dev); A,B,Y,W=lsprob(32,32,cond,dev=dev)
    P=torch.nn.Parameter(torch.zeros_like(W)); o=makeopt(opt,[P],lr=0.03 if opt!='adamw' else 0.01)
    log=RunLogger(outdir,f'ls-{opt}-seed{seed}',{'cond':cond,'optimizer':opt})
    for step in range(steps):
        o.zero_grad(set_to_none=True); loss=0.5*((A@P@B-Y)**2).mean(); loss.backward(); o.step()
        log.log({'loss':float(loss.detach().cpu()),'w_err':float((P-W).norm().detach().cpu())},step=step)
    log.close(); return {'loss':float(loss.detach().cpu())}

def rundrift(outdir,seed=0,steps=80,dev='cpu'):
    setseed(seed); dev=torch.device(dev); from mewon.optim.core import updatebasis
    U=V=None; log=RunLogger(outdir,f'drift-seed{seed}',{'steps':steps})
    for t in range(steps):
        M=driftmat(t,dev=dev); U,V=updatebasis(M,U,V,rank=8,piters=1)
        C=U.T@M@V; off=C-torch.diag(torch.diagonal(C)); en=float((C.norm()**2/(M.norm()**2+1e-12)).cpu()); lk=float((off.norm()/(C.norm()+1e-12)).cpu())
        log.log({'energy_capture':en,'offdiag_leakage':lk},step=t)
    log.close(); return {'ok':True}
