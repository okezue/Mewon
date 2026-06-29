import torch

def energycap(M,U,V,eps=1e-12):
    C=U.T@M.float()@V
    return float((C.norm()**2/(M.float().norm()**2+eps)).detach().cpu())

def leakage(C,eps=1e-12):
    off=C-torch.diag(torch.diagonal(C))
    return float((off.norm()/(C.norm()+eps)).detach().cpu())

def condnum(M,eps=1e-12):
    s=torch.linalg.svdvals(M.float()); s=s[s>eps]
    if s.numel()==0: return 0.0
    return float((s.max()/s.min()).detach().cpu())

def specgap(M,r):
    s=torch.linalg.svdvals(M.float())
    if s.numel()<=r: return float(s[-1].detach().cpu()) if s.numel() else 0.0
    return float((s[r-1]-s[r]).detach().cpu())

def projdist(U,Uprev):
    P=U@U.T; Q=Uprev@Uprev.T
    return float((P-Q).norm().detach().cpu())

def rotstats(M,U,V):
    C=U.T@M.float()@V
    return {'energy_capture':energycap(M,U,V),'offdiag_leakage':leakage(C),'core_norm':float(C.norm().detach().cpu()),'diag_norm':float(torch.diagonal(C).norm().detach().cpu()),'matrix_norm':float(M.float().norm().detach().cpu())}
