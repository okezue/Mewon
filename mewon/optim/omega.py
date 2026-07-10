import math
import torch
from torch.optim import Optimizer
from .ops import rectsvd
from .core import mpmed,despike,align

class MewonO(Optimizer):
    def __init__(self,params,lr=1e-3,momentum=0.95,beta2=0.98,alpha=0.05,lam=1.0,tau=1e-8,clip=1.5,kedge=1.5,zeta=0.05,delta=1e-3,amin=3,dmax=5,smin=0.5,rmax=32,eta=2.0,aspect=True,wd=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,beta2=beta2,alpha=alpha,lam=lam,tau=tau,clip=clip,kedge=kedge,zeta=zeta,delta=delta,amin=amin,dmax=dmax,smin=smin,rmax=rmax,eta=eta,aspect=aspect,wd=wd))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for gr in self.param_groups:
            b=gr['momentum']; al=gr['alpha']
            for p in gr['params']:
                views=getattr(p,'views',None)
                if views is None:
                    if p.grad is None: continue
                    views=[p.grad,p.grad]
                if p.ndim!=2: raise ValueError('MewonO only supports 2D params')
                g1,g2=views[0].float(),views[1].float()
                if not(torch.isfinite(g1).all() and torch.isfinite(g2).all()): continue
                if gr['wd']: p.mul_(1-gr['lr']*gr['wd'])
                st=self.state[p]
                if 'm' not in st:
                    st.update(m=torch.zeros_like(p,dtype=torch.float32),rbar=0.0,sg=0.0,step=0,qmed=mpmed(*p.shape),
                              Ut=None,Vt=None,mh=None,nu=None,S2=None,age=None,fails=None,
                              A=torch.zeros(3),prevZ=None,metrics={})
                st['step']+=1
                if gr['clip']:
                    n1,n2=float(g1.norm()),float(g2.norm())
                    if st['rbar']==0.0: st['rbar']=0.5*(n1+n2)
                    th=gr['clip']*st['rbar']
                    if n1>th: g1=g1*(th/n1)
                    if n2>th: g2=g2*(th/n2)
                    st['rbar']=gr['beta2']*st['rbar']+(1-gr['beta2'])*0.5*(min(n1,th)+min(n2,th))
                    R=th
                else: R=float(max(g1.norm(),g2.norm()))
                gb=0.5*(g1+g2)
                if st['prevZ'] is not None:
                    for j,Z in enumerate(st['prevZ']):
                        zn=float(Z.norm()) if Z is not None else 0.0
                        l=-float((gb*Z).sum())/(float(gb.norm())*zn+1e-12) if zn>0 else 0.0
                        st['A'][j]=0.9*st['A'][j]+0.1*l
                w=torch.softmax(-gr['eta']*st['A'],0)
                k=min(p.shape); phi=max(p.shape)/k; rk=math.sqrt(k)
                bc=1-b**st['step']
                I=gb-st['m']/bc if st['step']>1 else gb
                inorm=float(I.norm())
                sgnow=float(torch.linalg.svdvals(I/inorm).median())*inorm/rk/st['qmed'] if inorm>1e-30 else 0.0
                st['sg']=sgnow if st['sg']==0.0 else gr['beta2']*st['sg']+(1-gr['beta2'])*sgnow
                crad=math.sqrt(2*math.log(2/gr['delta']))
                if st['Ut'] is not None and st['Ut'].shape[1]>0:
                    Ut,Vt=st['Ut'],st['Vt']
                    z1=torch.diagonal(Ut.T@g1@Vt); z2=torch.diagonal(Ut.T@g2@Vt)
                    zb=0.5*(z1+z2); nui=0.5*(z1-z2).square()
                    st['mh']=(1-al)*st['mh']+al*zb
                    st['nu']=(1-al)*st['nu']+al*nui
                    st['S2']=(1-al)**2*st['S2']+al*al
                    st['age']=st['age']+1
                    c=crad*(st['S2']*st['nu']/2).sqrt()
                    ok=(st['mh'].abs()>c)|(st['age']<gr['amin'])
                    st['fails']=torch.where(ok,torch.zeros_like(st['fails']),st['fails']+1)
                    keep=st['fails']<gr['dmax']
                    if not keep.all():
                        for key in ['mh','nu','S2','age','fails']: st[key]=st[key][keep]
                        st['Ut']=Ut[:,keep]; st['Vt']=Vt[:,keep]
                M=st['m'].mul_(b).add_(gb,alpha=1-b)
                mnorm=float(M.norm())
                if mnorm<1e-30: continue
                U,s,V=rectsvd(M/mnorm)
                y=s*(mnorm/bc)/rk
                sig=math.sqrt((1-b)/(1+b)*(1+b**st['step'])/bc)*st['sg']
                edge=sig*(1+math.sqrt(phi))+gr['kedge']*sig*k**(-2/3)
                th2=despike(y,sig,phi)
                th2=torch.where(y>edge,th2,torch.zeros_like(th2))
                r=align(th2,sig,phi)
                gt=r*th2.sqrt()/(th2+gr['lam']*sig*sig+gr['tau']**2).sqrt()
                fresh=(gt>0).nonzero().flatten()
                if len(fresh):
                    Uf=U[:,fresh]; Vf=V[:,fresh]
                    if st['Ut'] is not None and st['Ut'].shape[1]>0:
                        ov=(Uf.T@st['Ut']).abs()*(Vf.T@st['Vt']).abs()
                        best,arg=ov.max(1)
                        for i in range(len(fresh)):
                            if float(best[i])>gr['smin']:
                                j=int(arg[i])
                                rho=float(r[fresh[i]])
                                if st['age'][j]<gr['amin'] or rho>0.9:
                                    su=torch.sign(Uf[:,i]@st['Ut'][:,j]); sv=torch.sign(Vf[:,i]@st['Vt'][:,j])
                                    st['Ut'][:,j]=su*Uf[:,i]; st['Vt'][:,j]=sv*Vf[:,i]
                                st['fails'][j]=0
                            elif st['Ut'].shape[1]<gr['rmax']:
                                u=Uf[:,i:i+1]; v=Vf[:,i:i+1]
                                z1n=float(u.T@g1@v); z2n=float(u.T@g2@v)
                                st['Ut']=torch.cat([st['Ut'],u],1); st['Vt']=torch.cat([st['Vt'],v],1)
                                dev=p.device
                                st['mh']=torch.cat([st['mh'],torch.tensor([0.5*(z1n+z2n)],device=dev)])
                                st['nu']=torch.cat([st['nu'],torch.tensor([0.5*(z1n-z2n)**2],device=dev)])
                                st['S2']=torch.cat([st['S2'],torch.tensor([al*al],device=dev)])
                                st['age']=torch.cat([st['age'],torch.zeros(1,device=dev)])
                                st['fails']=torch.cat([st['fails'],torch.zeros(1,device=dev,dtype=torch.long)])
                    else:
                        dev=p.device
                        z1n=torch.diagonal(Uf.T@g1@Vf); z2n=torch.diagonal(Uf.T@g2@Vf)
                        st['Ut']=Uf.clone(); st['Vt']=Vf.clone()
                        st['mh']=0.5*(z1n+z2n); st['nu']=0.5*(z1n-z2n).square()
                        st['S2']=torch.full((len(fresh),),al*al,device=dev)
                        st['age']=torch.zeros(len(fresh),device=dev)
                        st['fails']=torch.zeros(len(fresh),device=dev,dtype=torch.long)
                J=0 if st['Ut'] is None else st['Ut'].shape[1]
                if J>0:
                    c=crad*(st['S2']*st['nu']/2).sqrt()
                    a=(st['mh'].abs()-c).clamp_min(0)
                    d=a/(a.square()+gr['lam']*st['nu']+gr['tau']**2).sqrt()
                    Zom=-(st['Ut']*(torch.sign(st['mh'])*d).unsqueeze(0))@st['Vt'].T
                    mature=int(((st['age']>=gr['amin'])&(a>0)).sum())
                else:
                    Zom=torch.zeros_like(p,dtype=torch.float32); mature=0
                Zfr=-(U*gt.unsqueeze(0))@V.T
                if gr['zeta']:
                    if len(fresh):
                        gres=gb-U[:,fresh]@(U[:,fresh].T@gb@V[:,fresh]).diagonal().diag()@V[:,fresh].T
                    else: gres=gb
                    zt=gr['zeta']*(1-float(gt.square().mean()))
                    Zfr=Zfr+(-zt)*gres/(gres.norm()/math.sqrt(p.numel())+1e-12)
                Z=w[0]*Zom+w[1]*Zfr
                a2=(p.shape[0]/p.shape[1])**0.5 if gr['aspect'] else 1.0
                st['prevZ']=[Zom,Zfr,torch.zeros_like(Zfr)]
                st['metrics']={'natoms':J,'mature':mature,'nfresh':int(len(fresh)),'w_om':float(w[0]),'w_fr':float(w[1]),'w_zero':float(w[2]),'sig':sig}
                p.add_((a2*Z).to(p.dtype),alpha=gr['lr'])
        return loss
    def diagstate(self):
        return [self.state[p]['metrics']|{'shape':tuple(p.shape)} for gr in self.param_groups for p in gr['params'] if p in self.state and 'metrics' in self.state[p]]
