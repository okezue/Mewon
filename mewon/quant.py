import math
import torch

def uquant(x,bits=8,stoch=False,eps=1e-12):
    if bits<=0: raise ValueError('bits must be positive')
    xmin,xmax=x.min(),x.max()
    if float((xmax-xmin).abs())<eps:
        return x.clone(),{'scale':0.0,'zero':float(xmin)}
    levels=2**bits-1; scale=(xmax-xmin)/levels; y=(x-xmin)/scale
    if stoch:
        lo=y.floor(); yq=lo+torch.bernoulli((y-lo).clamp(0,1))
    else:
        yq=y.round()
    yq=yq.clamp(0,levels); out=yq*scale+xmin
    return out,{'scale':float(scale.detach().cpu()),'zero':float(xmin.detach().cpu()),'bits':bits}

def randorth(n,dev=None,dtype=torch.float32):
    Q,_=torch.linalg.qr(torch.randn(n,n,device=dev,dtype=dtype)); return Q

def hadamard(n,dev=None,dtype=torch.float32):
    if n&(n-1)!=0: raise ValueError('Hadamard requires power-of-two n')
    H=torch.tensor([[1.0]],device=dev,dtype=dtype)
    while H.shape[0]<n:
        H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H/math.sqrt(n)

def compdecomp(X,U,V,Dhat):
    C=U.T@X.float()@V; D=torch.diag(torch.diagonal(C)); E=C-D
    tail=X.float()-U@C@V.T
    lhs=(X.float()-U@Dhat.float()@V.T).norm().pow(2)
    rhs=tail.norm().pow(2)+E.norm().pow(2)+(D-Dhat.float()).norm().pow(2)
    return {'lhs':float(lhs.cpu()),'rhs':float(rhs.cpu()),'abs_err':float((lhs-rhs).abs().cpu()),'tail':float(tail.norm().pow(2).cpu()),'leakage':float(E.norm().pow(2).cpu()),'scalar_error':float((D-Dhat.float()).norm().pow(2).cpu())}

def quantbasis(X,U,V,bits=8):
    C=U.T@X.float()@V; d=torch.diagonal(C)
    dq,meta=uquant(d,bits=bits); Dhat=torch.diag(dq)
    rec=U@Dhat@V.T; stats=compdecomp(X,U,V,Dhat)
    return rec,stats|{'bits':bits,**meta}
