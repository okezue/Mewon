from mewon.analysis.collect import loadrecords
from mewon.utils import ensuredir,writejson

def writereport(runs,out):
    op=ensuredir(out); rows=loadrecords(runs); writejson(op/'records.json',rows)
    try:
        import pandas as pd,matplotlib.pyplot as plt
        df=pd.DataFrame(rows); figs=[]
        if not df.empty:
            for col in [c for c in df.columns if c not in {'time','step'} and df[c].dtype.kind in 'if'][:20]:
                fig=op/f'{col.replace("/","_")}.png'; plt.figure()
                if 'step' in df: plt.scatter(df.get('step',range(len(df))),df[col],s=8)
                else: plt.plot(df[col].values)
                plt.title(col); plt.tight_layout(); plt.savefig(fig); plt.close(); figs.append(fig.name)
        body='<h1>Mewon Report</h1><p>records: %d</p>'%len(rows)+''.join(f'<h2>{f}</h2><img src="{f}" style="max-width:900px">' for f in figs)
    except Exception as e:
        body=f'<h1>Mewon Report</h1><p>records: {len(rows)}</p><pre>{e}</pre>'
    (op/'report.html').write_text(body,encoding='utf-8')
    return op/'report.html'
