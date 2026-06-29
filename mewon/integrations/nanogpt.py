from pathlib import Path
import json,shutil

def patch(repo,dry=False):
    root=Path(repo); train=root/'train.py'; report={'repo':str(root),'train_exists':train.exists(),'changed':False,'errors':[]}
    if not train.exists(): report['errors'].append('train.py not found'); return report
    text=train.read_text(encoding='utf-8')
    anchor='optimizer = model.configure_optimizers'
    if 'Mewon' in text: return report|{'changed':False,'already_patched':True}
    if anchor not in text: report['errors'].append('optimizer anchor missing'); return report
    ins='\ntry:\n    from mewon.optim import Mewon,Muon,ExactSoftMuon\nexcept Exception:\n    Mewon=None\n'
    new=text.replace('import torch\n','import torch\n'+ins,1)
    if dry: report['would_change']=True; return report
    shutil.copy2(train,train.with_suffix('.py.mewon.bak'))
    train.write_text(new,encoding='utf-8'); report['changed']=True; return report

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('repo'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args(); print(json.dumps(patch(a.repo,a.dry_run),indent=2))
