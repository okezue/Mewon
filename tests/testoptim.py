import torch
from mewon.optim import MewonP

def teststep():
    p=torch.nn.Parameter(torch.randn(16,12)); o=MewonP([p],lr=1e-3,rank=4)
    loss=(p**2).mean(); loss.backward(); before=p.detach().clone(); o.step()
    assert not torch.allclose(before,p.detach())
    assert o.diagstate()
