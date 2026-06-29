import json,random,time
from dataclasses import asdict,is_dataclass
from pathlib import Path
import numpy as np
import torch

def setseed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def getdev(dev=None):
    if dev: return torch.device(dev)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensuredir(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p

def jsonable(x):
    if is_dataclass(x): return asdict(x)
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if isinstance(x,Path): return str(x)
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,torch.Tensor):
        return float(x.detach().cpu().item()) if x.numel()==1 else x.detach().cpu().tolist()
    if isinstance(x,np.ndarray): return x.tolist()
    return str(x)

def readyaml(path):
    import yaml
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def writejson(path,data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(jsonable(data),indent=2,sort_keys=True),encoding='utf-8')

def nowid(prefix='run'):
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"

def nparams(model):
    return sum(p.numel() for p in model.parameters())

def splitparams(model,emb=False):
    spec,aux=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lo=name.lower()
        isemb='emb' in lo or 'wte' in lo or 'wpe' in lo
        isaux=p.ndim<2 or 'norm' in lo or 'ln' in lo or lo.endswith('bias')
        if p.ndim==2 and not isaux and (emb or not isemb) and 'lm_head' not in lo: spec.append(p)
        else: aux.append(p)
    return spec,aux

def membytes(x):
    return x.numel()*x.element_size()
