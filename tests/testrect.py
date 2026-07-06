import torch
from mewon.optim.ops import rectsvd,qrpolar
from mewon.optim import MewonR

def polar(M):
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); return U@Vh

def testrectsvd():
    for sh in [(50,12),(12,50),(20,20)]:
        M=torch.randn(*sh); U,s,V=rectsvd(M)
        assert float((U@torch.diag(s)@V.T-M).norm()/M.norm())<1e-4
        assert float((U.T@U-torch.eye(U.shape[1])).norm())<1e-4
        assert float((V.T@V-torch.eye(V.shape[1])).norm())<1e-4

def testqrpolar():
    for sh in [(50,12),(12,50)]:
        M=torch.randn(*sh)
        assert float((qrpolar(M)-polar(M)).norm()/polar(M).norm())<1e-4

def testmuonlimit():
    p=torch.nn.Parameter(torch.randn(50,12)); g=torch.randn(50,12); p.grad=g.clone()
    o=MewonR([p],lr=1.0,momentum=0.0,lam=0.0,tau=1e-9,aspect=False)
    b=p.detach().clone(); o.step()
    assert float(((b-p.detach())-polar(g)).norm()/polar(g).norm())<1e-4

def testaspect():
    p=torch.nn.Parameter(torch.randn(64,16)); p.grad=torch.randn(64,16)
    o=MewonR([p],lr=1.0,momentum=0.0,lam=0.0,tau=1e-9,aspect=True)
    b=p.detach().clone(); o.step()
    assert abs(float(torch.linalg.matrix_norm(b-p.detach(),ord=2))-2.0)<1e-3
