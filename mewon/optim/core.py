import math
import torch
from torch.optim import Optimizer
from .ops import softmuon,orthon,nsorth,rectsvd

def rms(x,eps=1e-12):
    return x.norm()/math.sqrt(max(1,x.numel()))+eps

def randbasis(m,n,r,dev,dtype,eps):
    r=min(r,m,n)
    U=torch.randn(m,r,device=dev,dtype=torch.float32)
    V=torch.randn(n,r,device=dev,dtype=torch.float32)
    return orthon(U,eps),orthon(V,eps)

@torch.no_grad()
def updatebasis(mat,U,V,rank,piters=1,eps=1e-6,align=True):
    m,n=mat.shape; r=min(rank,m,n); M=mat.float()
    if U is None or V is None or tuple(U.shape)!=(m,r) or tuple(V.shape)!=(n,r) or not torch.isfinite(U).all() or not torch.isfinite(V).all():
        U,V=randbasis(m,n,r,M.device,M.dtype,eps)
    for _ in range(max(1,piters)):
        Y=M@V
        if not torch.isfinite(Y).all() or float(Y.norm())<=eps:
            U,V=randbasis(m,n,r,M.device,M.dtype,eps); Y=M@V
        U=orthon(Y,eps)
        Z=M.T@U
        if not torch.isfinite(Z).all() or float(Z.norm())<=eps:
            U,V=randbasis(m,n,r,M.device,M.dtype,eps); Z=M.T@U
        V=orthon(Z,eps)
    if align:
        C=U.T@M@V
        if torch.isfinite(C).all():
            Uc,_,Vch=torch.linalg.svd(C); U=U@Uc; V=V@Vch.T
    return U,V

class ExactSoftMuon(Optimizer):
    def __init__(self,params,lr=1e-3,lam=1.0,rho=1.0,momentum=0.95,wd=0.0):
        super().__init__(params,dict(lr=lr,lam=lam,rho=rho,momentum=momentum,wd=wd))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            for p in gr['params']:
                if p.grad is None: continue
                if p.ndim!=2: raise ValueError('ExactSoftMuon only supports 2D params')
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                st=self.state[p]
                if 'm' not in st: st['m']=torch.zeros_like(p,dtype=torch.float32)
                st['m'].mul_(gr['momentum']).add_(p.grad.float(),alpha=1-gr['momentum'])
                upd=softmuon(st['m'],gr['lam'],gr['rho'])
                p.add_(upd.to(p.dtype),alpha=gr['lr'])
        return loss

class Muon(Optimizer):
    def __init__(self,params,lr=1e-3,momentum=0.95,ns=5,wd=0.0,scale='rms'):
        super().__init__(params,dict(lr=lr,momentum=momentum,ns=ns,wd=wd,scale=scale))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            for p in gr['params']:
                if p.grad is None: continue
                if p.ndim!=2: raise ValueError('Muon only supports 2D params')
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                st=self.state[p]
                if 'm' not in st: st['m']=torch.zeros_like(p,dtype=torch.float32)
                st['m'].mul_(gr['momentum']).add_(p.grad.float(),alpha=1-gr['momentum'])
                q=nsorth(st['m'],steps=gr['ns'])
                if gr['scale']=='rms': q=q*rms(st['m'])
                p.add_((-q).to(p.dtype),alpha=gr['lr'])
        return loss

class Mewon(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.95,0.98),rank=16,freq=8,piters=1,mode='softpolar',rho=1.0,lam=1.0,nw=1.0,tau=1e-3,resid=0.0,norm=True,wd=0.0,eps=1e-8):
        if mode not in {'diag','core','softpolar','exactsoft'}: raise ValueError(f'unknown mode {mode}')
        super().__init__(params,dict(lr=lr,betas=betas,rank=rank,freq=freq,piters=piters,mode=mode,rho=rho,lam=lam,nw=nw,tau=tau,resid=resid,norm=norm,wd=wd,eps=eps))
    @torch.no_grad()
    def initstate(self,p,gr):
        st=self.state[p]
        if st: return st
        if p.ndim!=2: raise ValueError('Mewon only supports 2D params')
        m,n=p.shape; r=min(gr['rank'],m,n)
        U,V=randbasis(m,n,r,p.device,torch.float32,gr['eps'])
        st.update(step=0,m=torch.zeros_like(p,dtype=torch.float32),U=U,V=V,vd=torch.zeros(r,device=p.device),vc=torch.zeros(r,r,device=p.device),metrics={})
        return st
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            b1,b2=gr['betas']
            for p in gr['params']:
                if p.grad is None: continue
                st=self.initstate(p,gr); st['step']+=1
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                g=p.grad.float()
                M=st['m'].mul_(b1).add_(g,alpha=1-b1)
                if gr['mode']=='exactsoft':
                    upd=softmuon(M,gr['lam'],gr['rho']).float()
                else:
                    if st['step']==1 or st['step']%gr['freq']==0:
                        st['U'],st['V']=updatebasis(M,st.get('U'),st.get('V'),gr['rank'],gr['piters'],gr['eps'],align=gr['mode']!='diag')
                    U,V=st['U'],st['V']
                    C=U.T@M@V; diag=torch.diagonal(C)
                    st['vd'].mul_(b2).add_(diag.square(),alpha=1-b2)
                    st['vc'].mul_(b2).add_(C.square(),alpha=1-b2)
                    if gr['mode']=='diag':
                        a=st['vd'].sqrt().add(gr['tau'])
                        core=torch.diag(torch.clamp(diag/a,min=-gr['rho'],max=gr['rho']))
                    elif gr['mode']=='core':
                        den=st['vc'].sqrt().mul(gr['nw']).add(gr['tau'])
                        core=torch.clamp(C/den,min=-gr['rho'],max=gr['rho'])
                    else:
                        den=(C.square()+gr['nw']*st['vc']+gr['tau']**2).sqrt()
                        core=torch.clamp(C/den,min=-gr['rho'],max=gr['rho'])
                    upd=-(U@core@V.T)
                    if gr['resid']:
                        resid=M-U@C@V.T
                        upd=upd-gr['resid']*resid/rms(resid,gr['eps'])*rms(upd,gr['eps'])
                    en=float((C.norm()**2/(M.norm()**2+gr['eps'])).detach().cpu()) if M.norm()>0 else 0.0
                    off=C-torch.diag(diag)
                    lk=float((off.norm()/(C.norm()+gr['eps'])).detach().cpu()) if C.norm()>0 else 0.0
                    st['metrics']={'energy_capture':en,'offdiag_leakage':lk,'basis_rank':int(C.shape[0])}
                if gr['norm']: upd=upd/rms(upd,gr['eps'])
                p.add_(upd.to(p.dtype),alpha=gr['lr'])
        return loss
    def diagstate(self):
        rows=[]
        for gr in self.param_groups:
            for p in gr['params']:
                st=self.state.get(p,{})
                if st and 'metrics' in st:
                    rows.append(st['metrics']|{'shape':tuple(p.shape),'step':st.get('step',0)})
        return rows

class MewonR(Optimizer):
    def __init__(self,params,lr=1e-3,momentum=0.95,beta2=0.98,lam=1.0,tau=1e-8,clip=1.5,aspect=True,wd=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,beta2=beta2,lam=lam,tau=tau,clip=clip,aspect=aspect,wd=wd))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            for p in gr['params']:
                if p.grad is None: continue
                if p.ndim!=2: raise ValueError('MewonR only supports 2D params')
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                st=self.state[p]
                if 'm' not in st:
                    st['m']=torch.zeros_like(p,dtype=torch.float32); st['nu']=torch.zeros(min(p.shape),device=p.device); st['rbar']=0.0; st['metrics']={}
                g=p.grad.float()
                if gr['clip']:
                    gn=float(g.norm())
                    if st['rbar']==0.0: st['rbar']=gn
                    th=gr['clip']*st['rbar']
                    if gn>th: g=g*(th/gn)
                    st['rbar']=gr['beta2']*st['rbar']+(1-gr['beta2'])*min(gn,th)
                M=st['m'].mul_(gr['momentum']).add_(g,alpha=1-gr['momentum'])
                U,s,V=rectsvd(M)
                c=torch.diagonal(U.T@g@V)
                st['nu'].mul_(gr['beta2']).add_((c-s).square(),alpha=1-gr['beta2'])
                nu=torch.maximum(st['nu'],st['nu'].median())
                d=s/(s.square()+gr['lam']*nu+gr['tau']**2).sqrt()
                D=(U*d.unsqueeze(0))@V.T
                a=(p.shape[0]/p.shape[1])**0.5 if gr['aspect'] else 1.0
                st['metrics']={'sv_max':float(s.max()),'shaped_min':float(d.min()),'shaped_max':float(d.max()),'aspect':a}
                p.add_((-a*D).to(p.dtype),alpha=gr['lr'])
        return loss
    def diagstate(self):
        return [self.state[p]['metrics']|{'shape':tuple(p.shape)} for gr in self.param_groups for p in gr['params'] if p in self.state and 'metrics' in self.state[p]]
