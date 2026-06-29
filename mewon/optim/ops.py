import math
import torch
from torch import Tensor

def fmat(x):
    return x.float() if x.dtype in (torch.float16,torch.bfloat16) else x

def svd(x,full=False):
    return torch.linalg.svd(fmat(x),full_matrices=full)

def hardmuon(g,eps=1e-12):
    if g.ndim!=2: raise ValueError('hardmuon expects a matrix')
    U,S,Vh=svd(g)
    return -(S.sum().clamp_min(eps)*(U@Vh)).to(g.dtype)

def softmuon(g,lam=1.0,rho=1.0):
    if g.ndim!=2: raise ValueError('softmuon expects a matrix')
    if lam<=0 or rho<=0: raise ValueError('lam and rho must be positive')
    U,S,Vh=svd(g)
    d=torch.clamp(S/lam,max=rho)
    return -(U*d.unsqueeze(0)@Vh).to(g.dtype)

def specclip(g,rho=1.0):
    U,S,Vh=svd(g)
    d=torch.clamp(S,max=rho)
    return (U*d.unsqueeze(0)@Vh).to(g.dtype)

def schatten(g,p):
    if p==1: return -g
    if math.isinf(p): return hardmuon(g)
    r=2.0*p; q=r/(r-1.0)
    U,S,Vh=svd(g)
    nq=torch.linalg.vector_norm(S,ord=q).clamp_min(1e-12)
    d=(nq**(2.0-q))*(S.clamp_min(0)**(q-1.0))
    return -(U*d.unsqueeze(0)@Vh).to(g.dtype)

def invsqrt(a,eps=1e-6):
    n=a.shape[-1]
    I=torch.eye(n,device=a.device,dtype=a.dtype)
    sym=0.5*(a+a.transpose(-1,-2))+eps*I
    v,Q=torch.linalg.eigh(sym)
    v=v.clamp_min(eps).rsqrt()
    return (Q*v.unsqueeze(-2))@Q.transpose(-1,-2)

def orthon(y,eps=1e-6):
    return y@invsqrt(y.T@y,eps=eps)

def nsorth(g,steps=5,eps=1e-7):
    if g.ndim!=2: raise ValueError('nsorth expects a matrix')
    x=fmat(g); x=x/x.norm().clamp_min(eps)
    tr=False
    if x.shape[0]<x.shape[1]: x=x.T; tr=True
    I=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
    for _ in range(steps):
        x=0.5*x@(3*I-x.T@x)
    if tr: x=x.T
    return x.to(g.dtype)

def polarmuon(g,scale='nuclear',steps=5):
    q=nsorth(g,steps=steps)
    if scale=='nuclear': s=torch.linalg.svdvals(fmat(g)).sum()
    elif scale=='rms': s=g.norm()/math.sqrt(max(1,g.numel()))
    else: s=torch.tensor(1.0,device=g.device,dtype=torch.float32)
    return -(s*q).to(g.dtype)
