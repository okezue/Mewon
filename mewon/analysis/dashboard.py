from mewon.analysis.collect import loadrecords
from mewon.utils import ensuredir

def writedash(runs,out):
    op=ensuredir(out); rows=loadrecords(runs)
    try:
        import pandas as pd,plotly.express as px
        df=pd.DataFrame(rows)
        nums=[c for c in df.columns if c not in {'time','step'} and df[c].dtype.kind in 'if'] if not df.empty else []
        html=['<h1>Mewon Dashboard</h1>']
        for col in nums[:12]:
            fig=px.scatter(df,x='step' if 'step' in df else df.index,y=col,color='run_name' if 'run_name' in df else None,title=col)
            html.append(fig.to_html(include_plotlyjs='cdn',full_html=False))
        (op/'dashboard.html').write_text('\n'.join(html),encoding='utf-8')
    except Exception as e:
        (op/'dashboard.html').write_text(f'<h1>Dashboard unavailable</h1><pre>{e}</pre>',encoding='utf-8')
    return op/'dashboard.html'
