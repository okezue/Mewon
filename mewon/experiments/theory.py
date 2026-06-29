import torch
from mewon.optim.ops import softmuon
from mewon.quant import compdecomp,uquant
from mewon.tracking import RunLogger
from mewon.utils import setseed

def run(outdir,seed=0,n=24,m=20,trials=20,dev='cpu'):
    setseed(seed); dev=torch.device(dev); log=RunLogger(outdir,f'theory-seed{seed}',{'n':n,'m':m,'trials':trials})
    mp=ml=mc=mk=0; lam=1.7; rho=0.8
    for t in range(trials):
        G=torch.randn(m,n,device=dev); H=torch.randn(m,n,device=dev)
        T=softmuon(G,lam,rho)
        U,S,Vh=torch.linalg.svd(G,full_matrices=False); prox=(U*((S-lam*rho).clamp_min(0)).unsqueeze(0))@Vh
        pe=(G+lam*T-prox).norm()/(G.norm()+1e-12); mp=max(mp,float(pe))
        lip=(softmuon(G,lam,rho)-softmuon(H,lam,rho)).norm()/(G-H).norm().clamp_min(1e-12); ml=max(ml,float(lip))
        sg=torch.linalg.svdvals(G); st=torch.linalg.svdvals(-T); sg=sg[sg>1e-9]; st=st[st>1e-9]
        if sg.numel()>1 and st.numel()>1: mk=max(mk,float((st.max()/st.min())/(sg.max()/sg.min()+1e-12)))
        U2,_,Vh2=torch.linalg.svd(G,full_matrices=False); r=min(8,U2.shape[1]); Uc=U2[:,:r]; Vc=Vh2[:r,:].T
        C=Uc.T@G@Vc; d=torch.diagonal(C); dq,_=uquant(d,bits=4); stats=compdecomp(G,Uc,Vc,torch.diag(dq)); mc=max(mc,stats['abs_err'])
        log.log({'projection_rel_err':float(pe),'lipschitz_ratio':float(lip),'compression_abs_err':stats['abs_err']},step=t)
    summary={'max_projection_rel_err':mp,'max_lipschitz_ratio':ml,'max_compression_abs_err':mc,'max_condition_ratio':mk}
    log.artifact('summary.json',summary); log.close(); return summary
