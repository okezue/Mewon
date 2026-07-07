import math
import torch
from torch.optim import Optimizer
from .ops import rectsvd

def mpmed(m,n,draws=3):
    k=min(m,n); meds=[]
    for _ in range(draws):
        meds.append(float(torch.linalg.svdvals(torch.randn(m,n)/math.sqrt(k)).median()))
    return sum(meds)/len(meds)

def despike(y,sig,phi):
    A=y*y-sig*sig*(1+phi)
    disc=A*A-4*phi*sig**4
    return torch.where((disc>0)&(A>0),(A+disc.clamp_min(0).sqrt())/2,torch.zeros_like(y))

def align(th2,sig,phi,eps=1e-30):
    rho=(sig*sig)/th2.clamp_min(eps)
    num=(1-phi*rho*rho).clamp_min(0)
    return (num/(1+phi*rho)*(num/(1+rho))).sqrt()

class Bask(Optimizer):
    def __init__(self,params,lr=1e-3,momentum=0.95,beta2=0.98,lam=1.0,tau=1e-8,clip=1.5,kedge=1.5,alpha=1.0,zeta=0.05,aspect=True,wd=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,beta2=beta2,lam=lam,tau=tau,clip=clip,kedge=kedge,alpha=alpha,zeta=zeta,aspect=aspect,wd=wd))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            b=gr['momentum']
            for p in gr['params']:
                if p.grad is None: continue
                if p.ndim!=2: raise ValueError('Bask only supports 2D params')
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                st=self.state[p]
                if 'm' not in st:
                    st['m']=torch.zeros_like(p,dtype=torch.float32); st['rbar']=0.0; st['sg']=0.0; st['step']=0
                    st['qmed']=mpmed(*p.shape); st['metrics']={}
                g=p.grad.float()
                if not torch.isfinite(g).all(): continue
                st['step']+=1
                if gr['clip']:
                    gn=float(g.norm())
                    if st['rbar']==0.0: st['rbar']=gn
                    th=gr['clip']*st['rbar']
                    if gn>th: g=g*(th/gn)
                    st['rbar']=gr['beta2']*st['rbar']+(1-gr['beta2'])*min(gn,th)
                k=min(p.shape); phi=max(p.shape)/k; rk=math.sqrt(k)
                bc=1-b**st['step']
                I=g-st['m']/bc if st['step']>1 else g
                inorm=float(I.norm())
                sgnow=float(torch.linalg.svdvals(I/inorm).median())*inorm/rk/st['qmed'] if inorm>1e-30 else 0.0
                st['sg']=sgnow if st['sg']==0.0 else gr['beta2']*st['sg']+(1-gr['beta2'])*sgnow
                M=st['m'].mul_(b).add_(g,alpha=1-b)
                mnorm=float(M.norm())
                if mnorm<1e-30: continue
                U,s,V=rectsvd(M/mnorm)
                y=s*(mnorm/bc)/rk
                sig=math.sqrt((1-b)/(1+b)*(1+b**st['step'])/bc)*st['sg']
                edge=sig*(1+math.sqrt(phi))+gr['kedge']*sig*k**(-2/3)
                th2=despike(y,sig,phi)
                th2=torch.where(y>edge,th2,torch.zeros_like(th2))
                r=align(th2,sig,phi)
                gt=r**gr['alpha']*th2.sqrt()/(th2+gr['lam']*sig*sig+gr['tau']**2).sqrt()
                upd=U*gt.unsqueeze(0)@V.T
                if gr['zeta']:
                    act=gt>0
                    if act.any():
                        Ua=U[:,act]; Va=V[:,act]
                        gres=g-Ua@(Ua.T@g@Va).diagonal().diag()@Va.T
                    else: gres=g
                    zt=gr['zeta']*(1-float(gt.square().mean()))
                    upd=upd+zt*gres/(gres.norm()/math.sqrt(p.numel())+1e-12)
                a=(p.shape[0]/p.shape[1])**0.5 if gr['aspect'] else 1.0
                st['metrics']={'nspikes':int((gt>0).sum()),'y1_over_edge':float(y.max()/(edge+1e-20)),'gsum':float(gt.square().sum()),'sig':sig}
                p.add_((-a*upd).to(p.dtype),alpha=gr['lr'])
        return loss
    def diagstate(self):
        return [self.state[p]['metrics']|{'shape':tuple(p.shape)} for gr in self.param_groups for p in gr['params'] if p in self.state and 'metrics' in self.state[p]]
