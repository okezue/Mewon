import torch
from mewon.optim.ops import schatten
from mewon.tracking import RunLogger
from mewon.utils import setseed

def dist(x,y,eps=1e-2): return ((x[:,None,:]-y[None,:,:]).square().sum(-1)+eps*eps).sqrt()
def mmdenergy(X,Y): return -dist(X,X).mean()-dist(Y,Y).mean()+2*dist(X,Y).mean()

def run(outdir,seed=0,n=80,steps=100,p='inf',dev='cpu'):
    setseed(seed); dev=torch.device(dev)
    X=torch.randn(n,2,device=dev)@torch.diag(torch.tensor([2.5,0.4],device=dev))+torch.tensor([2.0,1.0],device=dev)
    Y=torch.cat([torch.randn(n//2,2,device=dev)*0.35+torch.tensor([-2.0,-2.0],device=dev),torch.randn(n-n//2,2,device=dev)*0.35+torch.tensor([2.0,3.0],device=dev)],0)
    pp=float('inf') if p=='inf' else float(p); log=RunLogger(outdir,f'mmd-p{p}-seed{seed}',{'p':p})
    X=torch.nn.Parameter(X)
    for step in range(steps):
        loss=mmdenergy(X,Y); loss.backward(); G=X.grad.detach(); upd=schatten(G,pp); X.data.add_(upd,alpha=0.05); X.grad=None
        log.log({'mmd':float(loss.detach().cpu()),'x_norm':float(X.detach().norm().cpu())},step=step)
    log.close(); return {'mmd':float(loss.detach().cpu())}
