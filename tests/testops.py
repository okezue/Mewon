import torch
from mewon.optim.ops import softmuon,hardmuon

def testprojection():
    G=torch.randn(12,9); lam=1.3; rho=.7
    T=softmuon(G,lam,rho)
    U,S,Vh=torch.linalg.svd(G,full_matrices=False)
    prox=(U*((S-lam*rho).clamp_min(0)).unsqueeze(0))@Vh
    assert torch.allclose(G+lam*T,prox,atol=1e-4,rtol=1e-4)

def testdescent():
    G=torch.randn(10,7); T=hardmuon(G)
    assert (G*T).sum()<0
