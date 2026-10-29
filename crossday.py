"""Cross-session test on OpenNeuro ds004148 (60 people, eyes-open rest, 3 sessions).

Session 2 was ~90 min after session 1; session 3 was ~1 month later.
Enrol on the first half of session 1. Query with the second half of:
  S1 (same recording), S2 (same day), S3 (a month later).
All queries are 2.5 min, so only the time gap differs.

The LDA projection is learned on training people only (5 folds split by person),
using all their sessions. Test people are never seen during training.
"""
import glob, json, os, sys
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from neurolock import features, fuzzy

ROOT = os.path.expanduser("~/neurolock_run/ds004148")
CACHE = "windows_ds004148.npz"
mne.set_log_level("ERROR")


def windows(x, sf, epoch_s=4.0):
    step = int(epoch_s * sf)
    return np.array([features.band_features(x[:, i:i + step], sf, epoch_s)
                     for i in range(0, x.shape[1] - step + 1, step)])


if not os.path.exists(CACHE):
    rows = {}
    for s in range(1, 61):
        W = []
        for ses in (1, 2, 3):
            f = f"{ROOT}/sub-{s:02d}/ses-session{ses}/eeg/sub-{s:02d}_ses-session{ses}_task-eyesopen_eeg.vhdr"
            if not os.path.exists(f):
                break
            raw = mne.io.read_raw_brainvision(f, preload=True).pick("eeg")
            raw.filter(1, 40)
            W.append(windows(raw.get_data(), raw.info["sfreq"]))
        if len(W) == 3:
            rows[s] = W
        print(f"subject {s}: {'ok' if len(W) == 3 else 'missing session'}", flush=True)
    subs = np.array(sorted(rows))
    n = min(len(w) for W in rows.values() for w in W)       # equalise length
    arr = np.array([[w[:n] for w in rows[s]] for s in subs])  # (people, 3 sessions, windows, feats)
    np.savez(CACHE, X=arr, subs=subs)
d = np.load(CACHE)
X, subs = d["X"], d["subs"]
P, _, nw, F = X.shape
h = nw // 2
print(f"{P} people, {nw} windows of 4 s per session, {F} features")

ENR = X[:, 0, :h].mean(1)                                   # enrolment: S1 first half
QRY = {"S1_same_recording": X[:, 0, h:].mean(1),
       "S2_90min": X[:, 1, h:].mean(1),
       "S3_1month": X[:, 2, h:].mean(1)}


def eer(gen, imp):
    ths = np.unique(np.concatenate([gen, imp]))
    frr = np.array([(gen < t).mean() for t in ths]); far = np.array([(imp >= t).mean() for t in ths])
    k = np.argmin(np.abs(frr - far)); return float((frr[k] + far[k]) / 2)


def gray2(z):
    q = np.digitize(z, [-0.6745, 0, 0.6745])
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.uint8)[q].reshape(*z.shape[:-1], -1)


CODES = [("rep", 3), ("rep", 5), ("rep", 7),
         ("bch31", 2), ("bch31", 3), ("bch31", 4), ("bch31", 5),
         ("bch63", 5), ("bch63", 7), ("bch63", 10), ("bch63", 11)]


def make(c):
    kind, p = c
    return fuzzy.RepetitionCode(p) if kind == "rep" else fuzzy.BCHCode(31 if kind == "bch31" else 63, p)


acc = {}
for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(subs)):
    Xtr = X[tr].reshape(-1, F)
    ytr = np.repeat(np.arange(len(tr)), 3 * nw)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xtr, ytr)
    Rtr = lda.transform(X[tr].mean(2).reshape(-1, F))
    mu, sd = Rtr.mean(0), Rtr.std(0) + 1e-9
    Pe = (lda.transform(ENR[te]) - mu) / sd
    off = ~np.eye(len(te), dtype=bool)
    for cond, Q in QRY.items():
        Pq = (lda.transform(Q[te]) - mu) / sd
        cos = (Pe / np.linalg.norm(Pe, axis=1, keepdims=True)) @ (Pq / np.linalg.norm(Pq, axis=1, keepdims=True)).T
        bits = {"bits1": ((Pe > 0).astype(np.uint8), (Pq > 0).astype(np.uint8)),
                "bits2": (gray2(Pe), gray2(Pq))}
        a = acc.setdefault(cond, {"sim": {}, "ber": [], "lock": {}})
        a["sim"].setdefault("vec_cos", []).append((np.diag(cos), cos[off]))
        for bk, (A, B) in bits.items():
            sim = -(A[:, None] != B[None]).mean(2)
            a["sim"].setdefault(bk, []).append((np.diag(sim), sim[off]))
            if bk == "bits1":
                a["ber"] += list(-np.diag(sim))
            for c in CODES:
                code = make(c)
                if code.usable(A.shape[1]) == 0:
                    continue
                g, i_ = [], []
                for x in range(len(te)):
                    k, rec = fuzzy.enroll(A[x], code)
                    for y in range(len(te)):
                        (g if x == y else i_).append(fuzzy.recover(B[y], rec)[1])
                a["lock"].setdefault(f"{bk}|{c[0]}{'_r' if c[0]=='rep' else '_t'}{c[1]}", []).append(
                    (np.mean(g) * len(g), len(g), np.sum(i_), len(i_), len(k)))
    print(f"fold {fold} done ({len(te)} test people, {Pe.shape[1]} dims)", flush=True)

out = {"n_people": int(P), "conditions": {}}
for cond, a in acc.items():
    r = {"bit_error_rate_genuine_mean": float(np.mean(a["ber"])),
         "eer": {k: eer(np.concatenate([g for g, _ in v]), np.concatenate([i for _, i in v])) for k, v in a["sim"].items()},
         "lock": {}}
    for k, rows in a["lock"].items():
        rows = np.array(rows)
        r["lock"][k] = {"GAR": float(rows[:, 0].sum() / rows[:, 1].sum()),
                        "FAR": float(rows[:, 2].sum() / rows[:, 3].sum()),
                        "key_bits": int(rows[:, 4].min()), "n_imp": int(rows[:, 3].sum())}
    out["conditions"][cond] = r
json.dump(out, open("results_crossday.json", "w"), indent=1)

print("\nEER (lower is better)")
print(f"{'condition':20s} {'vec_cos':>8s} {'bits2':>8s} {'bits1':>8s}  genuine bit-error")
for cond, r in out["conditions"].items():
    e = r["eer"]; print(f"{cond:20s} {e['vec_cos']:8.3f} {e['bits2']:8.3f} {e['bits1']:8.3f}  {r['bit_error_rate_genuine_mean']:.3f}")
print("\nLock: GAR / FAR  (key bits)")
keys = list(next(iter(out["conditions"].values()))["lock"])
print(f"{'config':18s}" + "".join(f"{c:>24s}" for c in out["conditions"]))
for k in keys:
    row = "".join(f"{out['conditions'][c]['lock'][k]['GAR']:>11.2f} / {out['conditions'][c]['lock'][k]['FAR']:.3f} ({out['conditions'][c]['lock'][k]['key_bits']:2d})" for c in out["conditions"])
    print(f"{k:18s}{row}")
