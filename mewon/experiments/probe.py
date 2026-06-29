import torch
from mewon.models.gpt import GPT
from mewon.data.char import CharDataset,corpus
from mewon.diag import condnum,specgap
from mewon.tracking import RunLogger
from mewon.utils import setseed

def run(outdir,seed=0,dev='cpu'):
    setseed(seed); dev=torch.device(dev); data=CharDataset(corpus(20),32); model=GPT(data.vocab,32).to(dev)
    x,y=data.getbatch(4,dev); _,loss=model(x,y); loss.backward(); log=RunLogger(outdir,f'probe-seed{seed}',{})
    for name,p in model.named_parameters():
        if p.grad is not None and p.grad.ndim==2:
            log.log({'cond':condnum(p.grad),'gap4':specgap(p.grad,4),'rows':p.grad.shape[0],'cols':p.grad.shape[1]},context={'param':name})
    log.close(); return {'ok':True}
