import torch
from mewon.models.vision import ViT
from mewon.optim import Mewon
from mewon.utils import splitparams,setseed
from mewon.tracking import RunLogger

def run(outdir,seed=0,steps=20,dev='cpu'):
    setseed(seed); dev=torch.device(dev); model=ViT().to(dev); spec,aux=splitparams(model)
    o1=Mewon(spec,lr=0.03,rank=8); o2=torch.optim.AdamW(aux,lr=1e-3)
    log=RunLogger(outdir,f'vision-seed{seed}',{})
    for step in range(steps):
        x=torch.randn(8,3,32,32,device=dev); y=torch.randint(0,10,(8,),device=dev)
        o1.zero_grad(set_to_none=True); o2.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(model(x),y); loss.backward(); o1.step(); o2.step(); log.log({'loss':float(loss.detach().cpu())},step=step)
    log.close(); return {'ok':True}
