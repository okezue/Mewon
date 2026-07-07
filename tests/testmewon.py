import math
import torch
from mewon.optim import Mewon,despike,align,mpmed

def testdespike():
    phi=4.0; sig=1.0
    yedge=sig*(1+math.sqrt(phi))
    th2=despike(torch.tensor([yedge*1.001]),sig,phi)
    assert abs(float(th2.sqrt())-sig*phi**0.25)<0.1
    th=5.0
    y2=th*th+sig*sig*(1+phi)+phi*sig**4/(th*th)
    th2=despike(torch.tensor([math.sqrt(y2)]),sig,phi)
    assert abs(float(th2.sqrt())-th)<1e-4

def testalign():
    phi=2.0; sig=1.0
    r=align(torch.tensor([sig*sig*math.sqrt(phi)]),sig,phi)
    assert float(r)<1e-3
    r=align(torch.tensor([100.0]),sig,phi)
    assert float(r)>0.95

def testpurenoise():
    torch.manual_seed(0)
    p=torch.nn.Parameter(torch.zeros(200,50))
    o=Mewon([p],lr=1.0,momentum=0.9,zeta=0.0,clip=0.0)
    for _ in range(30):
        p.grad=torch.randn(200,50); o.step()
    st=o.state[p]['metrics']
    assert st['nspikes']==0
    assert float(p.detach().norm())<1e-4

def testspikerecovery():
    torch.manual_seed(0)
    u=torch.zeros(200,1); u[0]=1; v=torch.zeros(50,1); v[0]=1
    S=8.0*math.sqrt(50)*(u@v.T)
    p=torch.nn.Parameter(torch.zeros(200,50))
    o=Mewon([p],lr=1.0,momentum=0.9,zeta=0.0,clip=0.0,aspect=False)
    for _ in range(60):
        p.grad=S+torch.randn(200,50); o.step()
    st=o.state[p]['metrics']
    assert st['nspikes']==1
    d=p.detach()
    assert float(d.norm())>0.5
    assert float(torch.abs(u.T@d@v)/d.norm())>0.98
