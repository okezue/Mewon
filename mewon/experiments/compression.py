import torch
from mewon.quant import randorth,hadamard,quantbasis
from mewon.tracking import RunLogger
from mewon.utils import setseed

def run(outdir,seed=0,m=64,n=64,rank=16,bits=4,trials=20,dev='cpu'):
    setseed(seed); dev=torch.device(dev); log=RunLogger(outdir,f'compression-seed{seed}',{'rank':rank,'bits':bits})
    methods=['identity','haar','svd']
    if m==n and m&(m-1)==0: methods.append('hadamard')
    for t in range(trials):
        X=torch.randn(m,n,device=dev); Us,S,Vhs=torch.linalg.svd(X,full_matrices=False)
        for me in methods:
            if me=='identity': U=torch.eye(m,rank,device=dev); V=torch.eye(n,rank,device=dev)
            elif me=='haar': U=randorth(m,dev)[:,:rank]; V=randorth(n,dev)[:,:rank]
            elif me=='hadamard': U=hadamard(m,dev)[:,:rank]; V=hadamard(n,dev)[:,:rank]
            else: U=Us[:,:rank]; V=Vhs[:rank,:].T
            _,stats=quantbasis(X,U,V,bits=bits); log.log({f'{me}/{k}':v for k,v in stats.items()},step=t)
    log.close(); return {'ok':True}
