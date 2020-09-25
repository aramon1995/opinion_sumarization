import numpy as np


def jensen_shannon_divergence(text, summary):
    cwt = []
    cws = []
    cw = {}
    N = 0
    delta = 0.0005
    for sentence in text:
        for token in sentence['tokens']:
            if token in cw.keys():
                cw[token][0] += 1
            else:
                cw[token] = [1,0]
    
    for sentence in summary:
        for token in sentence['tokens']:
            if token in cw.keys():
                cw[token][1] += 1
            else:
                cw[token] = [0,1]
    
    cw = np.array(list(cw.values()))
    cwt = cw[:,0]
    cws = cw[:,1]
    N = np.sum(cwt)
    Ns = np.sum(cws)
    p = np.where(cwt == 0,(cwt+delta)/(N+delta*len(cwt)),cwt/N)
    q = np.where(cws == 0,(cwt+delta)/(N+delta*len(cwt)),cws/Ns)
    a = (p + q)/2
    Dpa = np.sum(np.log2(p/a)*p)
    Dqa = np.sum(np.log2(q/a)*q)
    return (Dpa + Dqa)/2