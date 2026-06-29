import time
import torch
from mewon.optim.ops import hardmuon,softmuon,nsorth
from mewon.tracking import RunLogger

def run(outdir,seed=0,size=256,trials=10,dev='cpu'):
    torch.manual_seed(seed); dev=torch.device(dev); G=torch.randn(size,size,device=dev); log=RunLogger(outdir,f'kernels-seed{seed}',{'size':size})
    for name,fn in [('svd_hard',hardmuon),('soft',lambda x:softmuon(x,1,1)),('ns',nsorth)]:
        if dev.type=='cuda': torch.cuda.synchronize()
        t0=time.time()
        for _ in range(trials): y=fn(G)
        if dev.type=='cuda': torch.cuda.synchronize()
        log.log({f'{name}_seconds_per_call':(time.time()-t0)/trials,f'{name}_norm':float(y.norm().cpu())})
    log.close(); return {'ok':True}
