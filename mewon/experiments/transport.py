import numpy as np
from mewon.tracking import RunLogger
from mewon.utils import setseed

def runcoupling(outdir,seed=0,n=20,p='inf'):
    setseed(seed); rng=np.random.default_rng(seed)
    X=np.c_[rng.normal(2,1.5,n),rng.normal(1,0.4,n)]
    Y=np.r_[rng.normal([-2,-2],0.4,(n//2,2)),rng.normal([2,3],0.4,(n-n//2,2))]
    log=RunLogger(outdir,f'static-transport-p{p}-seed{seed}',{'n':n,'p':p})
    try:
        import cvxpy as cp
        P=cp.Variable((n,n),nonneg=True); S=0
        for i in range(n):
            for j in range(n):
                d=(Y[j]-X[i]).reshape(2,1); S=S+P[i,j]*(d@d.T)
        cons=[P@np.ones(n)==np.ones(n)/n,P.T@np.ones(n)==np.ones(n)/n]
        if p=='1': obj=cp.trace(S)
        elif p=='2': obj=cp.norm(S,'fro')
        else: obj=cp.lambda_max(S)
        prob=cp.Problem(cp.Minimize(obj),cons); prob.solve(solver='SCS',verbose=False,max_iters=3000)
        val=float(prob.value); status=prob.status
    except Exception as e:
        val=float('nan'); status=f'skipped:{type(e).__name__}:{e}'
    log.log({'objective':val,'status':status},step=0); log.close(); return {'objective':val,'status':status}
