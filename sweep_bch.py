"""PhysioNet (109 people, single session): repetition vs BCH on the LDA-projection lock."""
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
def gray2(z):
    q=np.digitize(z,[-0.6745,0,0.6745]); return np.array([[0,0],[0,1],[1,1],[1,0]],np.uint8)[q].reshape(*z.shape[:-1],-1)
def run(code,k,keep,enc):
    g=i=ng=ni=0
    for Pe,Pq in folds:
        Pe,Pq=Pe[:,:k],Pq[:,:k]; n=len(Pe)
        for a in range(n):
            m=np.sort(np.argsort(-np.abs(Pe[a]))[:keep])
            A=enc(Pe[a,m]); kk,rec=fuzzy.enroll(A,code)
            for b in range(n):
                ok=fuzzy.recover(enc(Pq[b,m]),rec)[1]
                if a==b: g+=ok; ng+=1
                else: i+=ok; ni+=1
    return g/ng,i/ni,len(kk)
b1=lambda z:(z>0).astype(np.uint8)
print("enc   dims keep code        key  GAR    FAR")
for name,enc in [("1bit",b1),("2bit",gray2)]:
  for k,keep in [(86,43),(86,86),(48,48),(32,32)]:
    for code in [fuzzy.RepetitionCode(5),fuzzy.BCHCode(31,3),fuzzy.BCHCode(31,4),fuzzy.BCHCode(31,5),fuzzy.BCHCode(63,7),fuzzy.BCHCode(63,10),fuzzy.BCHCode(63,11)]:
      if code.usable(len(enc(np.zeros(keep))))==0: continue
      G,F,K=run(code,k,keep,enc)
      if K>=8: print(f"{name} {k:4d} {keep:4d} {str(code.spec()):26s} {K:3d}  {G:.3f}  {F:.4f}",flush=True)
