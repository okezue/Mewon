import argparse,json
from pathlib import Path
from .utils import readyaml,writejson

def main(argv=None):
    p=argparse.ArgumentParser(prog='mewon')
    sub=p.add_subparsers(dest='cmd',required=True)
    v=sub.add_parser('validate'); v.add_argument('--out',default='runs/validate'); v.add_argument('--device',default='cpu')
    s=sub.add_parser('suite'); s.add_argument('--config',default='configs/quick.yaml'); s.add_argument('--out',default='runs/quick'); s.add_argument('--device',default=None)
    r=sub.add_parser('report'); r.add_argument('--runs',required=True); r.add_argument('--out',required=True)
    d=sub.add_parser('dashboard'); d.add_argument('--runs',required=True); d.add_argument('--out',required=True)
    pa=sub.add_parser('patch-nanogpt'); pa.add_argument('repo'); pa.add_argument('--dry-run',action='store_true'); pa.add_argument('--modded',action='store_true')
    a=p.parse_args(argv)
    if a.cmd=='validate':
        from mewon.experiments.runner import runsuite
        res=runsuite(a.out,seed=0,dev=a.device); writejson(Path(a.out)/'validation_summary.json',res); print(json.dumps(res,indent=2))
    elif a.cmd=='suite':
        cfg=readyaml(a.config); from mewon.experiments.runner import runsuite
        res=runsuite(a.out,seed=int(cfg.get('seed',0)),dev=a.device or cfg.get('device','cpu')); writejson(Path(a.out)/'suite_summary.json',res); print(json.dumps(res,indent=2))
    elif a.cmd=='report':
        from mewon.analysis.report import writereport; print(writereport(a.runs,a.out))
    elif a.cmd=='dashboard':
        from mewon.analysis.dashboard import writedash; print(writedash(a.runs,a.out))
    elif a.cmd=='patch-nanogpt':
        mod='mewon.integrations.modded' if a.modded else 'mewon.integrations.nanogpt'
        from importlib import import_module; print(json.dumps(import_module(mod).patch(a.repo,a.dry_run),indent=2))

if __name__=='__main__': main()
