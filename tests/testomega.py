import math
import torch
from mewon.optim import MewonO,Mewon

def testpurenoise():
    torch.manual_seed(0)
    p=torch.nn.Parameter(torch.zeros(120,40))
    o=MewonO([p],lr=1.0,momentum=0.9,zeta=0.0,clip=0.0)
    for _ in range(200):
        p.views=[torch.randn(120,40),torch.randn(120,40)]; o.step()
    st=o.state[p]['metrics']
    assert st['mature']==0
    assert float(p.detach().norm())<0.5

def testsubedge():
    torch.manual_seed(0)
    m,n=96,32
    u=torch.linalg.qr(torch.randn(m,1))[0]; v=torch.linalg.qr(torch.randn(n,1))[0]
    def run(cls,**kw):
        torch.manual_seed(1)
        p=torch.nn.Parameter(torch.zeros(m,n))
        o=cls([p],lr=0.05,momentum=0.9,zeta=0.0,clip=0.0,**kw)
        sig=3.0; pa=0.0; mid=None
        for t in range(400):
            amp=8.0 if t<80 else 0.15
            S=amp*sig*math.sqrt(n)*(u@v.T)
            g1=S+sig*torch.randn(m,n); g2=S+sig*torch.randn(m,n)
            if cls is MewonO: p.views=[g1,g2]
            else: p.grad=0.5*(g1+g2)
            o.step(); p.grad=None
            if t==79: pa=float(torch.abs(u.T@p.detach()@v))
            if t==250: mid=dict(o.state[p]['metrics'])
        pb=float(torch.abs(u.T@p.detach()@v))-pa
        return pb,mid
    pm,mm=run(Mewon)
    po,mo=run(MewonO)
    assert mm['nspikes']==0
    assert mo['mature']>=1
    assert po>1.5*pm
