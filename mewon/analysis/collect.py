import json
from pathlib import Path

def loadrecords(runs):
    rows=[]
    for p in Path(runs).rglob('metrics.jsonl'):
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec=json.loads(line); rec['_file']=str(p); rows.append(rec)
    return rows
