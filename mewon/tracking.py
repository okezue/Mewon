import csv,json,time
from pathlib import Path
from .utils import ensuredir,jsonable,writejson

class RunLogger:
    def __init__(self,outdir,name,config=None,aim=False,tb=False):
        self.outdir=ensuredir(outdir)/name; ensuredir(self.outdir)
        self.name=name
        self.mpath=self.outdir/'metrics.jsonl'
        self.cpath=self.outdir/'metrics.csv'
        self.fields=None; self.aim=None; self.tb=None
        self.config=config or {}
        writejson(self.outdir/'meta.json',{'run_name':name,'config':self.config,'start_time':time.time()})
        if aim:
            try:
                import aim as A
                self.aim=A.Run(repo=str(Path(outdir)/'.aim')); self.aim['hparams']=jsonable(self.config)
            except Exception as e:
                (self.outdir/'aim_unavailable.txt').write_text(str(e),encoding='utf-8')
        if tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb=SummaryWriter(str(self.outdir/'tb'))
            except Exception as e:
                (self.outdir/'tb_unavailable.txt').write_text(str(e),encoding='utf-8')
    def log(self,metrics,step=None,context=None):
        rec={'time':time.time(),'run_name':self.name}
        if step is not None: rec['step']=int(step)
        if context: rec.update({f'ctx/{k}':v for k,v in context.items()})
        rec.update(metrics); rec=jsonable(rec)
        with open(self.mpath,'a',encoding='utf-8') as f:
            f.write(json.dumps(rec,sort_keys=True)+'\n')
        flat={k:v for k,v in rec.items() if isinstance(v,(str,int,float,bool)) or v is None}
        if self.fields is None:
            self.fields=list(flat.keys())
            with open(self.cpath,'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=self.fields); w.writeheader(); w.writerow(flat)
        else:
            for k in flat:
                if k not in self.fields: self.fields.append(k)
            with open(self.cpath,'a',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=self.fields); w.writerow({k:flat.get(k) for k in self.fields})
        if self.aim is not None:
            for k,v in flat.items():
                if isinstance(v,(int,float)):
                    try: self.aim.track(v,name=k,step=step)
                    except Exception: pass
        if self.tb is not None:
            for k,v in flat.items():
                if isinstance(v,(int,float)): self.tb.add_scalar(k,v,step or 0)
    def artifact(self,name,data):
        path=self.outdir/name
        if isinstance(data,(dict,list)): writejson(path,data)
        else: path.write_text(str(data),encoding='utf-8')
        return path
    def close(self):
        if self.tb is not None: self.tb.close()
        if self.aim is not None:
            try: self.aim.close()
            except Exception: pass
