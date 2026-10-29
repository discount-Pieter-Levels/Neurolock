"""Projection-based lock vs median-split lock, vector matching, and PCA clustering.

All tests split by PERSON (5 folds): the projection and thresholds are learned on
training people only, then the lock is enrolled/tested on held-out people.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
from neurolock import data, features, fuzzy

N = 109
CACHE = "windows_physionet.npz"


def windows(x, sf, epoch_s=4.0):
    step = int(epoch_s * sf)
    return np.array([features.band_features(x[:, i:i + step], sf, epoch_s)
                     for i in range(0, x.shape[1] - step + 1, step)])


if not os.path.exists(CACHE):
    We, Wq, se, sq = [], [], [], []
    for s in range(1, N + 1):
        e, q, sf = data.physionet(s)
        a, b = windows(e, sf), windows(q, sf)
        We.append(a); Wq.append(b); se += [s] * len(a); sq += [s] * len(b)
        print(f"subject {s}", end="\r")
    np.savez(CACHE, We=np.vstack(We), Wq=np.vstack(Wq), se=se, sq=sq)
d = np.load(CACHE)
We, Wq, se, sq = d["We"], d["Wq"], d["se"], d["sq"]
subs = np.arange(1, N + 1)
Re = np.array([We[se == s].mean(0) for s in subs])   # recording-level features
Rq = np.array([Wq[sq == s].mean(0) for s in subs])


def eer(gen, imp):
    ths = np.unique(np.concatenate([gen, imp]))
    frr = np.array([(gen < t).mean() for t in ths]); far = np.array([(imp >= t).mean() for t in ths])
    k = np.argmin(np.abs(frr - far)); return float((frr[k] + far[k]) / 2)


def gray2(z):
    """2-bit Gray quantisation by quartile of a standardised value: 00,01,11,10."""
    q = np.digitize(z, [-0.6745, 0, 0.6745])
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.uint8)[q].reshape(*z.shape[:-1], -1)


def lock_rates(Be, Bq, r):
    """Fuzzy commitment with repetition code r over all bits. Returns GAR, FAR, key bits."""
    n, L = Be.shape; L = (L // r) * r; code = fuzzy.RepetitionCode(r)
    g, i_ = [], []
    for a in range(n):
        _, rec = fuzzy.enroll(Be[a, :L], code)
        for b in range(n):
            (g if a == b else i_).append(fuzzy.recover(Bq[b, :L], rec)[1])
    return np.mean(g), np.mean(i_), L // r, len(g), len(i_)


results = {"median_split": {}, "projection": {}}
agg = {k: {"g": [], "i": []} for k in ["vec_cos", "vec_euc", "bits1", "bits2", "median_bits"]}
lockacc = {}
kf = KFold(5, shuffle=True, random_state=0)
for fold, (tr, te) in enumerate(kf.split(subs)):
    trs, tes = subs[tr], subs[te]
    mtr = np.isin(se, trs); mtrq = np.isin(sq, trs)
    # projection learned on training people, both recordings, so it favours session-stable directions
    X = np.vstack([We[mtr], Wq[mtrq]]); y = np.concatenate([se[mtr], sq[mtrq]])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, y)
    Ptr = lda.transform(np.vstack([Re[tr], Rq[tr]]))
    mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-9          # public thresholds from training people
    Pe = (lda.transform(Re[te]) - mu) / sd; Pq = (lda.transform(Rq[te]) - mu) / sd
    D = Pe.shape[1]
    # vector matching (upper bound; template stored in the clear)
    cos = (Pe / np.linalg.norm(Pe, axis=1, keepdims=True)) @ (Pq / np.linalg.norm(Pq, axis=1, keepdims=True)).T
    euc = -np.linalg.norm(Pe[:, None] - Pq[None], axis=2)
    off = ~np.eye(len(te), dtype=bool)
    for k, S in [("vec_cos", cos), ("vec_euc", euc)]:
        agg[k]["g"] += list(np.diag(S)); agg[k]["i"] += list(S[off])
    B1e, B1q = (Pe > 0).astype(np.uint8), (Pq > 0).astype(np.uint8)
    B2e, B2q = gray2(Pe), gray2(Pq)
    # old scheme on the same held-out people, thresholds from training people
    med = np.median(Re[tr], 0)
    Me, Mq = (Re[te] > med).astype(np.uint8), (Rq[te] > med).astype(np.uint8)
    for k, (A, B) in [("bits1", (B1e, B1q)), ("bits2", (B2e, B2q)), ("median_bits", (Me, Mq))]:
        sim = -(A[:, None] != B[None]).mean(2)
        agg[k]["g"] += list(np.diag(sim)); agg[k]["i"] += list(sim[off])
        for r in [3, 5, 7, 9, 11]:
            lockacc.setdefault((k, r), []).append(lock_rates(A, B, r))
    print(f"fold {fold}: {len(te)} test people, {D} projected dims")

out = {"eer": {k: eer(np.array(v["g"]), np.array(v["i"])) for k, v in agg.items()}, "lock": {}}
for (k, r), rows in sorted(lockacc.items()):
    rows = np.array(rows)
    ng, ni = rows[:, 3].sum(), rows[:, 4].sum()
    out["lock"][f"{k}_r{r}"] = {"GAR": float((rows[:, 0] * rows[:, 3]).sum() / ng),
                                "FAR": float((rows[:, 1] * rows[:, 4]).sum() / ni),
                                "key_bits": int(rows[:, 2].min()), "n_impostor": int(ni)}
# feature redundancy: how many directions carry the variance
pc = PCA().fit(Re); out["pca_var_top5"] = pc.explained_variance_ratio_[:5].round(3).tolist()
out["pca_dims_for_90pct"] = int(np.searchsorted(np.cumsum(pc.explained_variance_ratio_), 0.9) + 1)

# ---- PCA clustering: does unsupervised PCA separate people? ----
clus = {}
Z = (We - We.mean(0)) / We.std(0)
for k in [2, 5, 10, 20, 50, 100]:
    p = PCA(k).fit(Z); Ze = p.transform(Z)
    Zq = p.transform((Wq - We.mean(0)) / We.std(0))
    Ce = np.array([Ze[se == s].mean(0) for s in subs]); Cq = np.array([Zq[sq == s].mean(0) for s in subs])
    nn = np.argmin(np.linalg.norm(Cq[:, None] - Ce[None], axis=2), axis=1)
    km = KMeans(N, n_init=3, random_state=0).fit_predict(Ze)
    clus[k] = {"silhouette_by_person": float(silhouette_score(Ze, se, sample_size=5000, random_state=0)),
               "kmeans_ARI": float(adjusted_rand_score(se, km)),
               "nearest_centroid_id_acc": float((nn == np.arange(N)).mean())}
out["pca_clustering"] = clus
print(json.dumps(out, indent=1))
json.dump(out, open("results_projection.json", "w"), indent=1)

# figures
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
p = PCA(2).fit(Z); Z2 = p.transform(Z)
lda_all = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(We, se); L2 = lda_all.transform(We)[:, :2]
pick = subs[:12]
for s in pick:
    m = se == s
    ax[0].scatter(*Z2[m].T, s=10, label=f"S{s}"); ax[1].scatter(*L2[m].T, s=10)
ax[0].set_title("PCA (unsupervised), 12 people, 4-s windows"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[1].set_title("LDA (supervised), same 12 people"); ax[1].set_xlabel("LD1"); ax[1].set_ylabel("LD2")
ax[0].legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig("pca_vs_lda.png", dpi=140)
