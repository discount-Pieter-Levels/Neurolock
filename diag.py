import numpy as np, mne
from neurolock import data, features, fuzzy
mne.set_log_level("ERROR")
E,Q=[],[]
for s in range(1,110):
    e,q,sf=data.physionet(s); E.append(features.band_features(e,sf)); Q.append(features.band_features(q,sf))
E,Q=np.array(E),np.array(Q); med,mad=features.fit_population(E); code=fuzzy.RepetitionCode(7)
n=len(E); acc=np.zeros((n,n),bool); ham=np.zeros((n,n))
for i in range(n):
    w,z=features.to_bits(E[i],med,mad); m=features.reliable_mask(z,105); _,rec=fuzzy.enroll(w[m],code)
    for j in range(n):
        wq,_=features.to_bits(Q[j],med,mad); ham[i,j]=np.mean(w[m]!=wq[m]); acc[i,j]=fuzzy.recover(wq[m],rec)[1]
off=~np.eye(n,dtype=bool)
print("impostor ham pctiles 1/5/50:",np.percentile(ham[off],[1,5,50]).round(3))
fa_row=(acc&off).sum(1); fa_col=(acc&off).sum(0)
print("enrollees with >=1 FA:",(fa_row>0).sum(),"max FA per enrollee:",fa_row.max())
print("queries causing FA, top 5 counts:",np.sort(fa_col)[-5:], " n queries with any FA:",(fa_col>0).sum())
# expected FA if bits independent with each pair's ham
from scipy.stats import binom
p=binom.sf(3,7,ham[off]); print("independence-predicted FAR:",np.mean((1-p)**15))
# where are the 105 bits? feature correlation
print("mean |corr| between features:",np.abs(np.corrcoef(E.T))[np.triu_indices(E.shape[1],1)].mean().round(3))
ones=[]
for i in range(n):
    w,z=features.to_bits(E[i],med,mad); m=features.reliable_mask(z,105); ones.append(w[m].mean())
ones=np.array(ones); print("frac of 1s in enrolled bits, pctiles 5/25/50/75/95:",np.percentile(ones,[5,25,50,75,95]).round(2))
u,s,_=np.linalg.svd(E-E.mean(0),full_matrices=False); print("variance share PC1..5:",(s[:5]**2/(s**2).sum()).round(3))
