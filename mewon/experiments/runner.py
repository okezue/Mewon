from mewon.utils import ensuredir

def runsuite(outdir,seed=0,dev='cpu'):
    out=ensuredir(outdir); res={}
    from . import theory,synth,compression,mmd,transport,lm,probe,kernels,vision
    res['theory']=theory.run(out,seed=seed,dev=dev,trials=5)
    res['least_squares']=synth.runls(out,seed=seed,steps=10,opt='mewon',dev=dev)
    res['drift']=synth.rundrift(out,seed=seed,steps=10,dev=dev)
    res['compression']=compression.run(out,seed=seed,trials=3,dev=dev)
    res['mmd']=mmd.run(out,seed=seed,steps=10,p='inf',dev=dev)
    res['transport']=transport.runcoupling(out,seed=seed,n=8,p='inf')
    res['lm']=lm.run(out,seed=seed,steps=12,dev=dev,nembd=32,blk=32)
    res['probe']=probe.run(out,seed=seed,dev=dev)
    res['kernels']=kernels.run(out,seed=seed,size=32,trials=2,dev=dev)
    res['vision']=vision.run(out,seed=seed,steps=2,dev=dev)
    return res
