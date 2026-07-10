import torch
from mewon.data.char import CharDataset,corpus,splittext
from mewon.models.gpt import GPT
from mewon.optim import Mewon,MewonP,MewonR,MewonO,Muon,ExactSoftMuon
from mewon.tracking import RunLogger
from mewon.utils import setseed,splitparams,getdev

def buildopt(model,name,lr,rank=8,freq=4,piters=1):
    if name=='adamw': return [torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)]
    spec,aux=splitparams(model)
    ao=torch.optim.AdamW(aux,lr=min(lr,2e-3),betas=(0.9,0.95),weight_decay=0.01) if aux else None
    if name=='muon': so=Muon(spec,lr=lr,scale='rms')
    elif name=='mewonr': so=MewonR(spec,lr=lr)
    elif name=='mewonr0': so=MewonR(spec,lr=lr,lam=0.0)
    elif name=='softmuon': so=ExactSoftMuon(spec,lr=lr,lam=1.0,rho=1.0)
    elif name=='mewondiag': so=MewonP(spec,lr=lr,mode='diag',rank=rank,freq=freq,piters=piters)
    elif name=='mewoncore': so=MewonP(spec,lr=lr,mode='core',rank=rank,freq=freq,piters=piters)
    elif name=='mewonp': so=MewonP(spec,lr=lr,mode='softpolar',rank=rank,freq=freq,piters=piters,resid=0.05)
    elif name=='mewono': so=MewonO(spec,lr=lr)
    else: so=Mewon(spec,lr=lr)
    return [o for o in [so,ao] if o is not None]

@torch.no_grad()
def evaluate(model,data,dev,batches=10):
    model.eval(); losses=[]
    for _ in range(batches):
        x,y=data.getbatch(8,dev); _,loss=model(x,y); losses.append(float(loss.cpu()))
    model.train(); return sum(losses)/len(losses)

def run(outdir,seed=0,opt='mewon',steps=100,dev=None,nlayer=2,nhead=2,nembd=64,blk=64,lr=None,rank=8,freq=4,piters=1,datapath=None,bs=8):
    setseed(seed); dev=getdev(dev)
    if datapath: data,val=splittext(datapath,blk)
    else: data=val=CharDataset(corpus(120),blk)
    model=GPT(data.vocab,blk,nlayer=nlayer,nhead=nhead,nembd=nembd).to(dev)
    if lr is None: lr=0.03 if opt!='adamw' else 2e-3
    opts=buildopt(model,opt,lr,rank,freq,piters)
    log=RunLogger(outdir,f'lm-{opt}-seed{seed}',{'optimizer':opt,'steps':steps})
    twoview=opt=='mewono'
    for step in range(steps):
        x,y=data.getbatch(bs,dev)
        for o in opts: o.zero_grad(set_to_none=True)
        if twoview:
            h=bs//2
            _,l1=model(x[:h],y[:h]); l1.backward()
            gs={p:p.grad.detach().clone() for p in model.parameters() if p.grad is not None}
            for p in model.parameters(): p.grad=None
            _,l2=model(x[h:],y[h:]); l2.backward()
            loss=0.5*(l1.detach()+l2.detach())
            for p in model.parameters():
                if p.grad is not None and p in gs:
                    g2=p.grad.detach().clone()
                    if p.ndim==2: p.views=[gs[p],g2]
                    p.grad=0.5*(gs[p]+g2)
        else:
            _,loss=model(x,y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        for o in opts: o.step()
        if twoview:
            for p in model.parameters():
                if hasattr(p,'views'): del p.views
        if step%10==0 or step==steps-1:
            metrics={'train_loss':float(loss.detach().cpu()),'eval_loss':evaluate(model,val,dev,batches=4)}
            for o in opts:
                if hasattr(o,'diagstate'):
                    rows=o.diagstate()
                    if rows:
                        metrics['mean_energy_capture']=sum(r.get('energy_capture',0) for r in rows)/len(rows)
                        metrics['mean_offdiag_leakage']=sum(r.get('offdiag_leakage',0) for r in rows)/len(rows)
            log.log(metrics,step=step)
    log.close(); return {'eval_loss':metrics['eval_loss']}
