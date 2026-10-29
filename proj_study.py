"""Which projection makes the EEG signature survive across sessions? (ds004148)

Protocol (same as crossday.py, so numbers are comparable):
  enrol  = mean of the first half of session 1
  query  = mean of the second half of S1 (same recording), S2 (+90 min), S3 (+1 month)
  5 outer folds split by PERSON; test people are never seen in training.

Projections (all fitted on training people only):
  raw        z-scored features, cosine
  pca_w      PCA + whitening (unsupervised), k dims
  lda_win    shrinkage LDA on 4 s windows (= crossday.py baseline)
  lda_rec    shrinkage LDA on quarter-recording means (12 per person, 3 sessions),
             so within-person scatter is dominated by SESSION drift, not window noise
  nap+lda    nuisance attribute projection (speaker-verification trick): remove the
             top-k directions of within-person between-session change, then lda_rec
  pair_lr    logistic regression on (a-b)^2 in a PCA-whitened space, trained on
             cross-session same-person pairs vs different-person pairs = a learned
             diagonal metric, optimised directly for cross-session verification
Hyper-parameters AND the overall winner (feature set x method) are chosen by inner
4-fold CV on the training people (criterion: mean EER of S2 and S3). Outer test
numbers of the chosen config are the honest result; the full table is diagnostic.
-> results_proj.json
"""
import itertools, json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from neurolock import fuzzy

d = np.load("feats_ds004148.npz")
FEATS = {"relpow19": d["relpow"], "psd": d["psd"], "cov": d["cov"]}
w61 = np.load("windows_ds004148.npz")
assert (w61["subs"] == d["subs"]).all()
n = min(FEATS["psd"].shape[2], w61["X"].shape[2])
FEATS["relpow61"] = w61["X"]
FEATS = {k: v[:, :, :n].astype(np.float64) for k, v in FEATS.items()}
FEATS["relpow19+cov"] = np.concatenate([FEATS["relpow19"], FEATS["cov"]], -1)
FEATS["psd+cov"] = np.concatenate([FEATS["psd"], FEATS["cov"]], -1)
P = len(d["subs"])
h = n // 2
QUERIES = ("S1_same_recording", "S2_90min", "S3_1month")
GRID = {"raw": [None], "pca_w": [10, 20, 40], "lda_win": [None], "lda_rec": [None],
        "nap+lda": [1, 2, 4, 8, 16], "pair_lr": [20, 40]}


def eer(g, i):
    t = np.unique(np.concatenate([g, i]))
    frr = np.array([(g < x).mean() for x in t]); far = np.array([(i >= x).mean() for x in t])
    k = np.argmin(np.abs(frr - far)); return float((frr[k] + far[k]) / 2)


def quarters(X):
    """(people, 3, n, D) -> samples (people*12, D), person ids, session ids."""
    q = n // 4
    S = X[:, :, :4 * q].reshape(X.shape[0], 3, 4, q, -1).mean(3)
    return S.reshape(-1, S.shape[-1]), np.repeat(np.arange(X.shape[0]), 12), np.tile(np.repeat([0, 1, 2], 4), X.shape[0])


def lda(Xs, y):
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xs, y)


def fit(method, hp, X):
    """Fit on training people X (p, 3, n, D). Returns embed(V) and score(E, Q)."""
    Xs, y, s = quarters(X)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    z = lambda V: (V - mu) / sd
    Zs = z(Xs)
    cos = lambda E, Q: (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (Q / np.linalg.norm(Q, axis=1, keepdims=True)).T
    if method == "raw":
        f = z
    elif method == "pca_w":
        p = PCA(hp, whiten=True).fit(Zs); f = lambda V: p.transform(z(V))
    elif method == "lda_win":
        W = z(X.reshape(-1, X.shape[-1])); m = lda(W, np.repeat(np.arange(X.shape[0]), 3 * n))
        f = lambda V: m.transform(z(V))
    elif method == "lda_rec":
        m = lda(Zs, y); f = lambda V: m.transform(z(V))
    elif method == "nap+lda":
        # session means minus person mean = between-session change of each person
        sm = np.array([Zs[(y == a) & (s == b)].mean(0) for a in range(X.shape[0]) for b in range(3)])
        dev = sm - np.repeat(sm.reshape(-1, 3, sm.shape[1]).mean(1), 3, 0)
        U = np.linalg.svd(dev, full_matrices=False)[2][:hp].T
        nap = lambda V: V - (V @ U) @ U.T
        m = lda(nap(Zs), y); f = lambda V: m.transform(nap(z(V)))
    elif method == "pair_lr":
        p = PCA(hp, whiten=True).fit(Zs); Ps = p.transform(Zs)
        rng = np.random.default_rng(0)
        gi = [(i, j) for i in range(len(y)) for j in range(i + 1, len(y)) if y[i] == y[j] and s[i] != s[j]]
        ii = [(i, j) for i, j in rng.integers(0, len(y), (8 * len(gi), 2)) if y[i] != y[j]]
        A = np.array([(Ps[i] - Ps[j]) ** 2 for i, j in gi + ii])
        lab = np.r_[np.ones(len(gi)), np.zeros(len(ii))]
        lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(A, lab)
        emb = lambda V: p.transform(z(V))
        return emb, lambda E, Q: lr.decision_function(((E[:, None] - Q[None]) ** 2).reshape(-1, E.shape[1])).reshape(len(E), len(Q))
    return f, lambda E, Q: cos(E, Q)


def evaluate(method, hp, X, tr, te):
    """Train on people tr, return per-query (genuine, impostor) scores on people te."""
    emb, score = fit(method, hp, X[tr])
    E = emb(X[te, 0, :h].mean(1))
    qs = {"S1_same_recording": X[te, 0, h:].mean(1), "S2_90min": X[te, 1, h:].mean(1), "S3_1month": X[te, 2, h:].mean(1)}
    off = ~np.eye(len(te), dtype=bool)
    out = {}
    for k, Q in qs.items():
        S = score(E, emb(Q)); out[k] = (np.diag(S), S[off])
    return out, emb, E, {k: emb(v) for k, v in qs.items()}


def inner_select(X, tr):
    """Best hp per (feature, method) and best overall, by inner CV on training people."""
    crit = {}
    for (fname, Xf), method in itertools.product(X.items(), GRID):
        for hp in GRID[method]:
            g = {q: [] for q in QUERIES[1:]}; i_ = {q: [] for q in QUERIES[1:]}
            for itr, ite in KFold(4, shuffle=True, random_state=1).split(tr):
                r = evaluate(method, hp, Xf, tr[itr], tr[ite])[0]
                for q in g:
                    g[q].append(r[q][0]); i_[q].append(r[q][1])
            crit[(fname, method, hp)] = np.mean([eer(np.concatenate(g[q]), np.concatenate(i_[q])) for q in g])
    best_hp = {}
    for (fname, method, hp), c in crit.items():
        if (fname, method) not in best_hp or c < crit[(fname, method, best_hp[(fname, method)])]:
            best_hp[(fname, method)] = hp
    overall = min(crit, key=crit.get)
    return best_hp, overall, crit


CODES = [("bch31", 3), ("bch31", 5), ("bch63", 7), ("bch63", 10), ("rep", 5), ("rep", 7)]


def lock_rates(Eb, Qb):
    out = {}
    for kind, p in CODES:
        code = fuzzy.RepetitionCode(p) if kind == "rep" else fuzzy.BCHCode(31 if kind == "bch31" else 63, p)
        if code.usable(Eb.shape[1]) == 0:
            continue
        g, i_, kb = [], [], None
        for x in range(len(Eb)):
            k, rec = fuzzy.enroll(Eb[x], code); kb = len(k)
            for y in range(len(Qb)):
                (g if x == y else i_).append(fuzzy.recover(Qb[y], rec)[1])
        out[f"{kind}_{p}"] = (int(np.sum(g)), len(g), int(np.sum(i_)), len(i_), kb)
    return out


if __name__ == "__main__":
    pool = {}                      # (feature, method) -> query -> [(g, i)]
    chosen, chosen_scores, lock = [], {q: [] for q in QUERIES}, {q: {} for q in QUERIES}
    ber = {q: [] for q in QUERIES}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(P))):
        best_hp, overall, crit = inner_select(FEATS, tr)
        chosen.append({"fold": fold, "feature": overall[0], "method": overall[1], "hp": overall[2],
                       "inner_crit": float(crit[overall])})
        print(f"fold {fold}: chose {overall} (inner S2/S3 EER {crit[overall]:.3f})", flush=True)
        for (fname, method), hp in best_hp.items():
            r = evaluate(method, hp, FEATS[fname], tr, te)[0]
            for q, gi in r.items():
                pool.setdefault((fname, method), {}).setdefault(q, []).append(gi)
        r, emb, E, Qs = evaluate(overall[1], overall[2], FEATS[overall[0]], tr, te)
        for q in QUERIES:
            chosen_scores[q].append(r[q])
        if overall[1] != "pair_lr":
            # key-lock on the chosen embedding: 1 bit per dim (sign after training z-score)
            emb_tr = emb(quarters(FEATS[overall[0]][tr])[0])
            m, s_ = emb_tr.mean(0), emb_tr.std(0) + 1e-9
            Eb = ((E - m) / s_ > 0).astype(np.uint8)
            for q in QUERIES:
                Qb = ((Qs[q] - m) / s_ > 0).astype(np.uint8)
                ber[q] += list((Eb != Qb).mean(1))
                for k, v in lock_rates(Eb, Qb).items():
                    lock[q].setdefault(k, []).append(v)

    table = {f"{f}|{m}": {q: eer(np.concatenate([g for g, _ in v]), np.concatenate([i for _, i in v]))
                          for q, v in qs.items()} for (f, m), qs in pool.items()}
    res = {"n_people": int(P), "windows_per_session": int(n),
           "chosen_per_fold": chosen,
           "chosen_EER": {q: eer(np.concatenate([g for g, _ in v]), np.concatenate([i for _, i in v]))
                          for q, v in chosen_scores.items()},
           "chosen_bit_error_genuine": {q: float(np.mean(v)) if v else None for q, v in ber.items()},
           "chosen_lock": {q: {k: {"GAR": sum(a[0] for a in v) / sum(a[1] for a in v),
                                   "FAR": sum(a[2] for a in v) / sum(a[3] for a in v),
                                   "key_bits_min": min(a[4] for a in v)} for k, v in L.items()} for q, L in lock.items()},
           "table_EER_outer": table}
    json.dump(res, open("results_proj.json", "w"), indent=1)
    print("\nOuter-test EER per feature|method (hp tuned in inner CV)")
    print(f"{'':24s}" + "".join(f"{q:>20s}" for q in QUERIES))
    for k, v in sorted(table.items(), key=lambda kv: kv[1]["S3_1month"]):
        print(f"{k:24s}" + "".join(f"{v[q]:20.3f}" for q in QUERIES))
    print("\nCHOSEN by inner CV:", [(c["feature"], c["method"], c["hp"]) for c in chosen])
    print("chosen EER:", {q: round(v, 3) for q, v in res["chosen_EER"].items()})
    print("chosen genuine bit error:", res["chosen_bit_error_genuine"])
    for q, L in res["chosen_lock"].items():
        print(q, {k: f"{v['GAR']:.2f}/{v['FAR']:.3f} ({v['key_bits_min']}b)" for k, v in L.items()})
