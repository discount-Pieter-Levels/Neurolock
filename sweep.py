import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from neurolock import fuzzy
d=np.load("windows_physionet.npz"); We,Wq,se,sq=d["We"],d["Wq"],d["se"],d["sq"]; subs=np.arange(1,110)
Re=np.array([We[se==s].mean(0) for s in subs]); Rq=np.array([Wq[sq==s].mean(0) for s in subs])
folds=[]
for tr,te in KFold(5,shuffle=True,random_state=0).split(subs):
    m1,m2=np.isin(se,subs[tr]),np.isin(sq,subs[tr])
    lda=LinearDiscriminantAnalysis(solver="eigen",shrinkage="auto").fit(np.vstack([We[m1],Wq[m2]]),np.concatenate([se[m1],sq[m2]]))
    P=lda.transform(np.vstack([Re[tr],Rq[tr]])); mu,sd=P.mean(0),P.std(0)
    folds.append(((lda.transform(Re[te])-mu)/sd,(lda.transform(Rq[te])-mu)/sd))
def run(k,keep,r):
    g=i=ng=ni=0
    for Pe,Pq in folds:
        Pe,Pq=Pe[:,:k],Pq[:,:k]; n=len(Pe); code=fuzzy.RepetitionCode(r)
        for a in range(n):
            m=np.sort(np.argsort(-np.abs(Pe[a]))[:keep]) if keep<k else np.arange(k)
            L=(len(m)//r)*r; m=m[:L]
            _,rec=fuzzy.enroll((Pe[a,m]>0).astype(np.uint8),code)
            for b in range(n):
                ok=fuzzy.recover((Pq[b,m]>0).astype(np.uint8),rec)[1]
                if a==b: g+=ok; ng+=1
                else: i+=ok; ni+=1
    return g/ng,i/ni,L//r
print("dims keep r  key  GAR    FAR")
for k in [16,24,32,48,86]:
    for keep in sorted({k, k//2}):
        for r in [3,5,7]:
            G,F,K=run(k,keep,r)
            if K>=6: print(f"{k:4d} {keep:4d} {r:2d} {K:4d}  {G:.3f}  {F:.4f}")
