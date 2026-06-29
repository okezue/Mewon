import torch
from mewon.quant import compdecomp,uquant

def testcompdecomp():
    X=torch.randn(20,18); U,_,Vh=torch.linalg.svd(X,full_matrices=False); r=6; U=U[:,:r]; V=Vh[:r,:].T
    C=U.T@X@V; dq,_=uquant(torch.diagonal(C),bits=4); stats=compdecomp(X,U,V,torch.diag(dq))
    assert stats['abs_err']<1e-4
