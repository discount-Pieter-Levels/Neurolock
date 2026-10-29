"""End-to-end proof-of-personhood evaluation on static data (Phase 2).

  python evaluate_pop.py            (about 15-25 min on a 16-thread laptop)

Outputs results_pop/results_pop.json, plots, and results_pop/pop_params.pkl
(public parameters + thresholds for run_pop_demo.py; no templates).

PROTOCOL
  * Vein people: FV-USM, seed-0 tune/test split by person (as Phase 1).
    Enrol = session 1, probe = session 2 (cross-session).
  * EEG subjects: PhysioNet MMI (109), seed-7 tune/test split by subject.
    Enrol = run 1 rest, verify = rest periods of runs 3/7/11 (SAME DAY: optimistic).
    Cross-month check on OpenNeuro ds004148 (S1 vs +90 min vs +1 month).
  * VIRTUAL PEOPLE: one FV-USM person + one EEG subject, paired at random
    within the same half. ASSUMES vein and EEG are independent.
  * Every threshold/setting is chosen on TUNE; every reported rate is TEST.
  * SIMULATED pieces (labelled everywhere): SSVEP responses (synthetic
    oscillation added to real resting EEG) and finger-pulse videos.
"""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurolock import fuzzy
from neurolock.vein import code as vcode
from neurolock.liveness import pulse as P_, ssvep as SS
from neurolock.pop import crypto
from neurolock.pop.biometrics import (ATTEMPTS, METHODS, PERM_SEED, BrainKey, EEGData, VeinData,
                                      gray2, split_people)

OUT = "results_pop"
os.makedirs(OUT, exist_ok=True)
VCACHE = os.path.expanduser("~/.cache/neurolock_vein")
PCACHE = os.path.expanduser("~/.cache/neurolock_pop")
SIM = "SIMULATED"
R = {"labels": {"ssvep": "protocol test on simulated responses - not physiological evidence",
                "pulse": "synthetic finger videos (simulated); no real finger video on disk",
                "identities": "virtual (chimeric) people: FV-USM person + EEG subject, assumed independent",
                "eeg_same_day": "PhysioNet enrol/verify recordings are minutes apart (optimistic)"}}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def eer(g, i):
    t = np.unique(np.concatenate([g, i])); t = t[:: max(1, len(t) // 4000)]
    frr = np.array([(g < x).mean() for x in t]); far = np.array([(i >= x).mean() for x in t])
    k = np.argmin(abs(frr - far)); return float((frr[k] + far[k]) / 2), float(t[k])


def ci95_upper_zero(n):
    """'Rule of three': 0 events in n trials -> 95 % upper bound ~ 3/n."""
    return 3.0 / n if n else 1.0


# ===================================================================== STEP 0 inventory
log("STEP 0  inventory")
VD = VeinData()
ED = EEGData()
occ = np.load(f"{PCACHE}/occipital_physionet.npz")
SF = float(occ["sf"])
ds = np.load("windows_ds004148.npz")
inv = {"FV-USM": {"people": len(VD.people), "fingers_per_person": 4, "sessions": 2,
                  "images_per_finger_per_session": 6, "location": "~/data/FV-USM (raw_data only)"},
       "PhysioNet_EEGMMI": {"subjects": len(ED.subjects), "channels": 64, "sfreq_hz": 160,
                            "recordings": "run 1 (rest, enrol) + rest periods of runs 3/7/11 (verify)",
                            "location": "~/mne_data/MNE-eegbci-data"},
       "OpenNeuro_ds004148": {"subjects": int(ds["X"].shape[0]), "sessions": 3,
                              "gaps": "S2 +90 min, S3 +1 month", "location": "~/neurolock_run/ds004148"},
       "eegmat (Kaggle mental arithmetic)": "NOT on disk (used 2026-10-15 elsewhere); skipped",
       "SSVEP dataset": "NONE on disk -> SSVEP is SIMULATED on real resting EEG",
       "pulse videos (~/data/pulse_videos)": "absent -> pulse tested on synthetic videos only"}
R["inventory"] = inv
print(json.dumps(inv, indent=1))

# ===================================================================== STEP 1 virtual identities
v_tune, v_test = split_people(VD.people, seed=0)
rng = np.random.default_rng(7)
eeg_subs = np.array(ED.subjects); rng.shuffle(eeg_subs)
e_tune, e_test = sorted(eeg_subs[: len(eeg_subs) // 2].tolist()), sorted(eeg_subs[len(eeg_subs) // 2:].tolist())


def pair(vp, es, seed):
    r = np.random.default_rng(seed)
    n = min(len(vp), len(es))
    return list(zip(r.permutation(vp)[:n].tolist(), r.permutation(es)[:n].tolist()))


VIRT = {"tune": pair(v_tune, e_tune, 11), "test": pair(v_test, e_test, 12)}
R["virtual_people"] = {"tune": len(VIRT["tune"]), "test": len(VIRT["test"]),
                       "vein_people_tune_test": [len(v_tune), len(v_test)],
                       "eeg_subjects_tune_test": [len(e_tune), len(e_test)]}
log("STEP 1  virtual people:", R["virtual_people"])

# ===================================================================== STEP 2 vein layer
log("STEP 2  vein layer: calibrate on tune")
SM = {m: np.load(f"{VCACHE}/scoremat_{m}.npz") for m in METHODS}


def X5(m, sp):
    X = SM[m][f"{sp}_X"]; n = X.shape[0] // 6
    return X.reshape(n, 6, n, 6)


def finger_rows(sp):
    meta = SM["gabor"]["meta"]; r = SM["gabor"][f"{sp}_rows"][::6]
    return meta[r, 0], meta[r, 1]


# fusion normalisation: tune impostor stats of 'max over 6 enrol images'
norm = {}
for m in METHODS:
    B = X5(m, "tune").max(axis=1); n = B.shape[0]
    imp = B[~np.eye(n, dtype=bool)].ravel()
    norm[m] = (float(imp.mean()), float(imp.std()))


def fused_B(sp):
    return sum((X5(m, sp).max(axis=1) - norm[m][0]) / norm[m][1] for m in METHODS)    # (n, n, 6)


def consistency(sp):
    W = sum((SM[m][f"{sp}_W2"] - norm[m][0]) / norm[m][1] for m in METHODS)          # (n, 6, 6)
    return np.stack([np.mean([W[:, x, y] for x in a for y in a if x < y], axis=0) for a in ATTEMPTS], 1)


FB = {sp: fused_B(sp) for sp in ("tune", "test")}
Q = {sp: consistency(sp) for sp in ("tune", "test")}
q_thr = float(np.quantile(Q["tune"][:, 0], 0.10))      # Phase 1b: retake the worst 10 % of attempts


def finger_matrix(sp):
    """(claimed finger, probe finger) score with retake, max over the 3 images."""
    q = Q[sp]; use2 = (q[:, 0] < q_thr) & (q[:, 1] >= q[:, 0])
    A = np.stack([FB[sp][:, :, a].max(axis=2) for a in ATTEMPTS], 2)
    return A[:, np.arange(len(use2)), use2.astype(int)], float(use2.mean())


FM = {}; retake_rate = {}
for sp in ("tune", "test"):
    FM[sp], retake_rate[sp] = finger_matrix(sp)


def person_matrix(sp, people, fingers):
    """(enrolled person, probe person) mean score over the chosen fingers."""
    p, s = finger_rows(sp)
    idx = {(pp, ss): k for k, (pp, ss) in enumerate(zip(p, s))}
    M = np.zeros((len(people), len(people)))
    for f in fingers:
        ix = [idx[(q, f)] for q in people]
        M += FM[sp][np.ix_(ix, ix)]
    return M / len(fingers)


# choose the 2 fingers on tune (all 6 pairs of the 4 slots)
pairs_ = [(a, b) for a in range(1, 5) for b in range(a + 1, 5)]
tune_eer = {}
for fp in pairs_:
    M = person_matrix("tune", v_tune, fp); off = ~np.eye(len(M), dtype=bool)
    tune_eer[fp] = eer(np.diag(M), M[off])[0]
FINGERS = min(tune_eer, key=tune_eer.get)
R["vein_finger_pair_tune_EER"] = {f"{a}+{b}": v for (a, b), v in tune_eer.items()}
R["vein_fingers_chosen"] = list(FINGERS)
log("  fingers chosen on tune:", FINGERS, f"(tune EER {100 * tune_eer[FINGERS]:.2f}%)",
    "FV-USM slots: 1 L-index, 2 L-middle, 3 R-index, 4 R-middle")


def virtual_impostors(sp, people, fingers, n=4_000_000, seed=3):
    """Person-level impostor scores from fingers of DIFFERENT people (independence),
    to reach FMR ~1e-6. Phase 1b found real pairs give 3-9x higher FMR than this."""
    r = np.random.default_rng(seed)
    pools = []
    for f in fingers:
        M = person_matrix(sp, people, (f,)); pools.append(M[~np.eye(len(M), dtype=bool)])
    return np.mean([pl[r.integers(0, len(pl), n)] for pl in pools], axis=0)


VI_tune = virtual_impostors("tune", v_tune, FINGERS)
thr_verify = float(np.quantile(VI_tune, 1 - 1e-4))    # 1:1 verification, FMR 1e-4
thr_dedup = float(np.quantile(VI_tune, 1 - 1e-5))     # 1:N dedup, FMR 1e-5 per comparison
VM_PARAMS = {"norm": norm, "q_thr": q_thr, "fingers": FINGERS,
             "thr_verify": thr_verify, "thr_dedup": thr_dedup}

# ---- commitment mode: codes for every image, then exact unlock rule per BCH block
log("  vein commitment: codes for all images")


def _codes(m):
    return np.array([vcode.to_code(x, "com") for x in VD.maps[m]], np.uint8)


with Pool(2) as p:
    CODES = dict(zip(METHODS, p.map(_codes, METHODS)))


def commit_errors(m, people, f, K):
    """errors[a, b, j, :] = bits where probe image j of person b's finger f
    differs from person a's enrolled (masked, shuffled) code."""
    perm = np.random.default_rng(PERM_SEED).permutation(K)
    C = CODES[m]
    W, MASK = [], []
    for a in people:
        w, rel = vcode.enroll_bits(C[[VD.ix[(a, f, 1, i)] for i in range(1, 7)]])
        mk = vcode.reliable_mask(rel, K); W.append(w[mk][perm]); MASK.append(mk)
    PR = np.array([C[[VD.ix[(b, f, 2, i)] for i in range(1, 7)]] for b in people])    # (n, 6, 3072)
    err = np.zeros((len(people), len(people), 6, K), np.uint8)
    for ai in range(len(people)):
        err[ai] = PR[:, :, MASK[ai]][:, :, perm] ^ W[ai][None, None]
    return err


def block_ok(err, t, n=255):
    """Exact BCH(n, t) unlock rule (fuzzy.py after the fix): every block <= t errors."""
    c = fuzzy.BCHCode(n, t)
    ok = np.ones(err.shape[:-1], bool); i = 0
    for s in c._blocks(err.shape[-1]):
        ok &= err[..., i:i + s].sum(-1) <= t; i += s
    return ok, c.key_bits(err.shape[-1])


def commit_rates(people, m, K, t, errs=None):
    """unlock[f][a, b, j] for each finger; person-level 'any finger, first n tries'."""
    errs = errs or [commit_errors(m, people, f, K) for f in FINGERS]
    U = []
    for e in errs:
        ok, kb = block_ok(e, t)
        U.append(ok)
    U = np.stack(U)                                       # (2, n, n, 6)
    out = {"key_bits": int(kb)}
    n = len(people); off = ~np.eye(n, dtype=bool)
    for tries in (1, 3, 6):
        anyf = U[:, :, :, :tries].any(axis=3)            # finger opens within `tries` images
        for pol, P in (("any", anyf.any(0)), ("both", anyf.all(0))):
            out[f"GAR_{pol}_{tries}"] = float(np.diag(P).mean())
            out[f"FAR_{pol}_{tries}"] = float(P[off].mean())
    return out


# Phase 2b: codes are aligned to a coarse vein-density helper (neurolock/vein/align.py)
# instead of centre of mass. Centre-of-mass rows are kept for comparison.
import align_commit as AC
assert tuple(AC.FINGERS) == tuple(FINGERS), "align_commit.py assumes the same 2 fingers"


def align_data(people):
    with Pool(16) as p:
        out = p.map(AC.codes_for, [(a, f, people) for a in people for f in FINGERS], chunksize=2)
    return dict(zip([(a, f) for a in people for f in FINGERS], out))


AD_tune = align_data(v_tune)
rows = [{"align": w, "K": K, "t": t, **AC.unlock_matrix(v_tune, AD_tune, K, t, w)}
        for w in ("helper", "com") for K in (3072, 1536, 768) for t in (15, 23, 31, 39, 47)]
cands = [r for r in rows if r["align"] == "helper" and r["FAR_6tries"] == 0 and r["key_bits"] >= 128]
best_c = max(cands, key=lambda r: (r["GAR_6tries"], r["key_bits"]))
VC_PARAMS = {"K": best_c["K"], "code": {"code": "bch", "n": 255, "t": best_c["t"]}, "method": AC.METHOD, "align": "helper"}
R["vein_commit_tune_choice"] = best_c
R["vein_commit_tune_sweep"] = rows
log("  commitment chosen on tune:", VC_PARAMS, f"tune GAR(any finger, 6 tries) {best_c['GAR_6tries']:.2f}")

# ===================================================================== STEP 3a brain key
log("STEP 3a EEG brain key: calibrate on tune")
Re = {s: ED.recording(s, "enrol") for s in ED.subjects}
Rq = {s: ED.recording(s, "query") for s in ED.subjects}


def fit_proj(subs):
    m1, m2 = np.isin(ED.se, subs), np.isin(ED.sq, subs)
    Xw = np.vstack([ED.We[m1], ED.Wq[m2]]); y = np.concatenate([ED.se[m1], ED.sq[m2]])
    Rtr = np.array([Re[s] for s in subs] + [Rq[s] for s in subs])
    return BrainKey.fit_projection(Xw, y, Rtr)


def eeg_bits(proj, feats, dims, enc):
    z = ((proj["lda"].transform(feats) - proj["mu"]) / proj["sd"])[:, :dims]
    return z, ((z > 0).astype(np.uint8) if enc == "1bit" else gray2(z))


def eeg_lock(proj, subs, dims, keep, enc, n, t):
    """Exact unlock matrix (enrolled subject a, query subject b) under BCH(n, t)."""
    ze, _ = eeg_bits(proj, np.array([Re[s] for s in subs]), dims, enc)
    zq, _ = eeg_bits(proj, np.array([Rq[s] for s in subs]), dims, enc)
    c = fuzzy.BCHCode(n, t)
    U = np.zeros((len(subs), len(subs)), bool)
    for a in range(len(subs)):
        mk = np.sort(np.argsort(-np.abs(ze[a]))[:keep])
        enc_f = (lambda z: (z > 0).astype(np.uint8)) if enc == "1bit" else (lambda z: gray2(z))
        wa = enc_f(ze[a][mk]); wq = enc_f(zq[:, mk])
        e = wq ^ wa[None]
        ok = np.ones(len(subs), bool); i = 0
        if c.usable(e.shape[1]) == 0:
            return None, 0
        for s in c._blocks(e.shape[1]):
            ok &= e[:, i:i + s].sum(1) <= t; i += s
        U[a] = ok
    return U, c.key_bits(wa.size)


EEG_CFGS = [(d, k, enc, n, t) for d in (16, 24, 32) for frac in (1.0, 0.75) for enc in ("1bit", "2bit")
            for k in [int(d * frac)] for n, ts in ((31, (2, 3, 4, 5, 6, 7)), (63, (5, 7, 9, 11, 13)))
            for t in ts]
# inner 3-fold CV inside the tune subjects (projection never sees the subjects it is scored on)
folds = np.array_split(np.random.default_rng(0).permutation(e_tune), 3)
cv = {c: {"g": [], "i": [], "kb": 0} for c in EEG_CFGS}
for k in range(3):
    te_ = sorted(folds[k].tolist()); tr_ = sorted(set(e_tune) - set(te_))
    proj = fit_proj(tr_)
    for cfg in EEG_CFGS:
        U, kb = eeg_lock(proj, te_, *cfg)
        if U is None:
            continue
        off = ~np.eye(len(te_), dtype=bool)
        cv[cfg]["g"] += np.diag(U).tolist(); cv[cfg]["i"] += U[off].tolist(); cv[cfg]["kb"] = kb
eeg_rows = [{"dims": c[0], "keep": c[1], "enc": c[2], "n": c[3], "t": c[4], "key_bits": v["kb"],
             "GAR_tune": float(np.mean(v["g"])), "FAR_tune": float(np.mean(v["i"])), "n_imp_tune": len(v["i"])}
            for c, v in cv.items() if v["g"]]
ec = [r for r in eeg_rows if r["FAR_tune"] == 0 and r["key_bits"] >= 16]
best_e = max(ec, key=lambda r: (r["GAR_tune"], r["key_bits"]))
PROJ = fit_proj(e_tune)                                   # final projection: all tune subjects
BK_PARAMS = {"proj": PROJ, "dims": best_e["dims"], "keep": best_e["keep"], "enc": best_e["enc"],
             "code": {"code": "bch", "n": best_e["n"], "t": best_e["t"]}}
R["brain_key_tune_choice"] = best_e
log("  brain key chosen on tune:", {k: best_e[k] for k in ("dims", "keep", "enc", "n", "t", "key_bits", "GAR_tune")})

# test-half brain key + EEG degrees of freedom
U, kb = eeg_lock(PROJ, e_test, best_e["dims"], best_e["keep"], best_e["enc"], best_e["n"], best_e["t"])
off = ~np.eye(len(e_test), dtype=bool)
_, Be = eeg_bits(PROJ, np.array([Re[s] for s in e_test]), best_e["dims"], best_e["enc"])
_, Bq = eeg_bits(PROJ, np.array([Rq[s] for s in e_test]), best_e["dims"], best_e["enc"])
hd = (Be[:, None] != Bq[None]).mean(2)
p_hat, sd_hat = float(hd[off].mean()), float(hd[off].std())
R["brain_key_test"] = {"GAR": float(np.diag(U).mean()), "FAR": float(U[off].mean()),
                       "FAR_95pct_upper_if_zero": ci95_upper_zero(int(off.sum())),
                       "impostor_pairs": int(off.sum()), "key_bits": int(kb),
                       "dof": {"bits": int(Be.shape[1]), "impostor_HD_p": p_hat, "impostor_HD_sd": sd_hat,
                               "genuine_HD_mean": float(np.diag(hd).mean()),
                               "N_daugman": p_hat * (1 - p_hat) / sd_hat ** 2},
                       "note": "PhysioNet enrol/verify same day (minutes apart): optimistic"}
log("  brain key TEST:", {k: R["brain_key_test"][k] for k in ("GAR", "FAR", "key_bits")},
    f"EEG DoF N = {R['brain_key_test']['dof']['N_daugman']:.1f}")

# cross-session check on ds004148 (same recipe, projection fit on its tune half)
Xd = ds["X"]; nd = Xd.shape[0]; h = Xd.shape[2] // 2
perm_d = np.random.default_rng(7).permutation(nd); dt, dv = perm_d[: nd // 2], perm_d[nd // 2:]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Xw = Xd[dt].reshape(-1, Xd.shape[-1]); yw = np.repeat(dt, 3 * Xd.shape[2])
lda_d = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xw, yw)
Ptr = lda_d.transform(Xd[dt].mean(2).reshape(-1, Xd.shape[-1])); mu_d, sd_d = Ptr.mean(0), Ptr.std(0) + 1e-9
dims_d = min(best_e["dims"], Ptr.shape[1]); keep_d = min(best_e["keep"], dims_d)


def zd(F):
    return ((lda_d.transform(F) - mu_d) / sd_d)[:, :dims_d]


enc_f = (lambda z: (z > 0).astype(np.uint8)) if best_e["enc"] == "1bit" else gray2
ze = zd(Xd[dv, 0, :h].mean(1))
cross = {}
for name, sess, sl in (("same_recording", 0, slice(h, None)), ("+90min", 1, slice(h, None)), ("+1month", 2, slice(h, None))):
    zq = zd(Xd[dv, sess, sl].mean(1)); c = fuzzy.BCHCode(best_e["n"], best_e["t"])
    U2 = np.zeros((len(dv), len(dv)), bool)
    for a in range(len(dv)):
        mk = np.sort(np.argsort(-np.abs(ze[a]))[:keep_d]); e = enc_f(zq[:, mk]) ^ enc_f(ze[a][mk])[None]
        ok = np.ones(len(dv), bool); i = 0
        for s in c._blocks(e.shape[1]):
            ok &= e[:, i:i + s].sum(1) <= best_e["t"]; i += s
        U2[a] = ok
    o2 = ~np.eye(len(dv), dtype=bool)
    cross[name] = {"GAR": float(np.diag(U2).mean()), "FAR": float(U2[o2].mean())}
R["brain_key_ds004148_cross_session"] = {"test_subjects": int(len(dv)), "dims": dims_d, "keep": keep_d, **cross}
log("  brain key ds004148 (test 30):", cross)

# ===================================================================== STEP 3b SSVEP (SIMULATED)
log("STEP 3b SSVEP challenge-response (SIMULATED responses on real resting EEG)")
WEIGHTS = [0.7] * 5 + [1.0] * 3                      # PO channels weaker than O channels
eeg_index = {s: k for k, s in enumerate(occ["subjects"].tolist())}


def trial(subj, which, K, seg_s, amp, r, response=True, for_challenge=None):
    """One trial: returns (genuine pass, replay pass, rest pass, injected pass, per-segment correct list)."""
    rest = occ[which][eeg_index[subj]]
    need = int(K * seg_s * SF)
    o = r.integers(0, rest.shape[1] - need)
    seg = rest[:, o:o + need]
    ch = SS.make_challenge(K, seg_s, rng=r)
    x = SS.simulate_response(seg, SF, ch, amp, r, WEIGHTS)
    return ch, x, seg


def ssvep_rhos(subjs, which, K, seg_s, amp, n_trials, seed):
    """Per segment: (correct-class?, rho) for genuine responses, and best rho for rest."""
    r = np.random.default_rng(seed)
    G, Rr = [], []
    for s in subjs:
        for _ in range(n_trials):
            ch, x, seg = trial(s, which, K, seg_s, amp, r)
            for xs, f in zip(SS.segments(x, SF, K, seg_s), ch["freqs"]):
                b, rho, _ = SS.classify_segment(xs, SF); G.append((np.isclose(b, f), rho))
            for xs in SS.segments(seg, SF, K, seg_s):
                Rr.append(SS.classify_segment(xs, SF)[1])
    return np.array(G), np.array(Rr)


SEGS = (0.5, 1.0, 1.5, 2.0, 3.0)
AMPS = (0.2, 0.3, 0.5, 0.7)
MAIN_AMP, MAIN_SEG, MAIN_K = 0.5, SS.SEG_S, 4
# rho threshold per segment length, chosen on TUNE resting EEG: reject the 80 % least
# confident rest segments (a random segment then has to be confident AND right)
rho_thr = {}
for sg in SEGS:
    _, rr = ssvep_rhos(e_tune[:30], "later", 4, sg, MAIN_AMP, 3, seed=int(sg * 10))
    rho_thr[sg] = float(np.quantile(rr, 0.80))
R["ssvep_rho_thr_tune"] = rho_thr


def ssvep_eval(subjs, K, seg_s, amp, n_trials, seed, tries=1):
    """tries = how many fresh challenges a person may attempt; pass if any passes.
    Attackers get the same number of tries (each with a new challenge)."""
    r = np.random.default_rng(seed)
    gp, rp, rep, inj, seg_acc = [], [], [], [], []
    for s in subjs:
        for _ in range(n_trials):
            g_ok = rp_ok = rep_ok = inj_ok = False
            for _t in range(tries):
                ch, x, seg = trial(s, "later", K, seg_s, amp, r)
                ok, per = SS.verify_response(x, SF, ch, rho_thr[seg_s]); g_ok |= ok; seg_acc += per
                if _t == 0:
                    x0 = x                       # the attacker's recording (made for the first challenge)
                chB = SS.make_challenge(K, seg_s, rng=r)
                # replay: a recording made for ANOTHER challenge, presented for this one
                rep_ok |= SS.verify_response(x0, SF, chB, rho_thr[seg_s])[0]
                # resting EEG, no stimulus
                rp_ok |= SS.verify_response(seg, SF, chB, rho_thr[seg_s])[0]
                # attacker who KNOWS challenge B and synthesises the response in software
                xs = SS.simulate_response(seg, SF, chB, amp, r, WEIGHTS)
                inj_ok |= SS.verify_response(xs, SF, chB, rho_thr[seg_s])[0]
            gp.append(g_ok); rp.append(rp_ok); rep.append(rep_ok); inj.append(inj_ok)
    return {"true_pass": float(np.mean(gp)), "segment_accept": float(np.mean(seg_acc)),
            "replay_pass": float(np.mean(rep)), "rest_pass": float(np.mean(rp)),
            "injected_synthetic_pass": float(np.mean(inj)), "trials": len(gp)}


ss_res = {"label": R["labels"]["ssvep"], "by_seg_and_amp": {}, "by_K": {}}
for amp in AMPS:
    for sg in SEGS:
        r_ = ssvep_eval(e_test, 4, sg, amp, 4, seed=100 + int(amp * 10) + int(sg * 100))
        ss_res["by_seg_and_amp"][f"amp{amp}_seg{sg}"] = r_
MAX_TRIES = 3
for tries in (1, MAX_TRIES):
    for K in (1, 2, 4, 6):
        r_ = ssvep_eval(e_test, K, MAIN_SEG, MAIN_AMP, 8, seed=500 + K + 10 * tries, tries=tries)
        for k_ in ("rest_pass", "replay_pass"):
            r_[k_ + "_95pct_upper"] = ci95_upper_zero(r_["trials"]) if r_[k_] == 0 else r_[k_]
        ss_res["by_K"][f"K{K}_tries{tries}"] = r_
R["ssvep"] = ss_res
MAIN_SS = ss_res["by_K"][f"K{MAIN_K}_tries{MAX_TRIES}"]
log("  SSVEP main (amp 0.5, 1.5 s, K=4):", "1 try", ss_res["by_K"][f"K{MAIN_K}_tries1"], "| 3 tries", MAIN_SS)

# ===================================================================== STEP 3c SSVEP consent (SIMULATED)
log("STEP 3c SSVEP confirm-target consent (SIMULATED responses) - replaces the EEG brain key")


def consent_eval(subjs, n_trials, seed, tries=MAX_TRIES):
    """Outcomes ('confirm' counts as consent). A genuine user retries while the
    answer is 'none'; an explicit 'deny' ends the attempt."""
    r = np.random.default_rng(seed)
    need = int(MAIN_K * MAIN_SEG * SF)
    res = {k: [] for k in ("confirm_1try", "confirm", "deny_gives_consent", "deny_detected",
                           "rest_gives_consent", "replay_gives_consent", "injected_gives_consent")}

    def rec(s):
        rest = occ["later"][eeg_index[s]]; o = r.integers(0, rest.shape[1] - need); return rest[:, o:o + need]

    for s in subjs:
        for _ in range(n_trials):
            got, first = False, None
            for t_ in range(tries):
                ch = SS.make_consent_challenge("vote yes on proposal 7", MAIN_K, MAIN_SEG, rng=r)
                x = SS.simulate_response(rec(s), SF, SS.attended(ch, "confirm"), MAIN_AMP, r, WEIGHTS)
                dcs = SS.decide_consent(x, SF, ch, rho_thr[MAIN_SEG])[0]
                if t_ == 0:
                    first, x_old = dcs, x
                if dcs == "confirm":
                    got = True; break
                if dcs == "deny":
                    break
            res["confirm_1try"].append(first == "confirm"); res["confirm"].append(got)
            # the person looks at DENY (up to `tries` challenges): must never give consent
            dg, dd = False, False
            for _t in range(tries):
                ch = SS.make_consent_challenge("vote yes on proposal 7", MAIN_K, MAIN_SEG, rng=r)
                dcs = SS.decide_consent(SS.simulate_response(rec(s), SF, SS.attended(ch, "deny"), MAIN_AMP, r, WEIGHTS),
                                        SF, ch, rho_thr[MAIN_SEG])[0]
                dg |= dcs == "confirm"; dd |= dcs == "deny"
                if dcs != "none":
                    break
            res["deny_gives_consent"].append(dg); res["deny_detected"].append(dd)
            # attackers get `tries` fresh challenges each
            rg = rp = inj = False
            for _t in range(tries):
                ch = SS.make_consent_challenge("transfer all funds", MAIN_K, MAIN_SEG, rng=r)
                rg |= SS.decide_consent(rec(s), SF, ch, rho_thr[MAIN_SEG])[0] == "confirm"
                rp |= SS.decide_consent(x_old, SF, ch, rho_thr[MAIN_SEG])[0] == "confirm"
                xs = SS.simulate_response(rec(s), SF, SS.attended(ch, "confirm"), MAIN_AMP, r, WEIGHTS)
                inj |= SS.decide_consent(xs, SF, ch, rho_thr[MAIN_SEG])[0] == "confirm"
            res["rest_gives_consent"].append(rg); res["replay_gives_consent"].append(rp)
            res["injected_gives_consent"].append(inj)
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["trials"] = len(res["confirm"])
    for k in ("deny_gives_consent", "rest_gives_consent", "replay_gives_consent"):
        out[k + "_95pct_upper"] = ci95_upper_zero(out["trials"]) if out[k] == 0 else out[k]
    return out


CONSENT = consent_eval(e_test, 8, seed=900)
R["ssvep_consent"] = {"label": R["labels"]["ssvep"], "K": MAIN_K, "seg_s": MAIN_SEG, "amp": MAIN_AMP,
                      "max_tries": MAX_TRIES, **CONSENT}
log("  consent:", CONSENT)

# ===================================================================== STEP 4 pulse (SIMULATED videos)
log("STEP 4  pulse liveness (SIMULATED videos)")
import cv2


def pulse_snrs(people, seed, n=200):
    """SNR (dB) of n SIMULATED live videos and n videos of a still (printed) REAL vein image."""
    r = np.random.default_rng(seed)
    imgs = [VD.samples[VD.ix[(p_, 1, 1, 1)]].path for p_ in people[:40]]
    live, still = [], []
    for k in range(n):
        v = P_.synthetic_finger_video(seconds=10, bpm=r.uniform(50, 110), depth=r.uniform(0.002, 0.006),
                                      noise=r.uniform(0.4, 1.5), seed=seed * 10000 + k)
        live.append(P_.check_pulse(v, 30, -np.inf)[2])
        img = cv2.resize(cv2.imread(imgs[k % len(imgs)], 0), (80, 60))
        v = P_.synthetic_finger_video(seconds=10, still_image=img, noise=r.uniform(0.4, 1.5), seed=seed * 10000 + 5000 + k)
        still.append(P_.check_pulse(v, 30, -np.inf)[2])
    return np.array(live), np.array(still)


lt, stl = pulse_snrs(v_tune, 1)
# Phase 1 threshold (3 dB) was never tuned; on tune stills it passes too many.
# New threshold: just above the highest tune still-image SNR.
PULSE_THR = float(max(P_.MIN_SNR_DB, np.max(stl) + 0.5))
ls, sts = pulse_snrs(v_test, 2)
R["pulse"] = {"label": R["labels"]["pulse"], "threshold_snr_db": PULSE_THR,
              "threshold_note": "chosen on TUNE simulated stills (max tune still SNR + 0.5 dB); Phase 1 used 3 dB",
              "old_3dB_threshold_test": {"live_pass": float((ls >= 3).mean()), "still_image_pass": float((sts >= 3).mean())},
              "live_pass": float((ls >= PULSE_THR).mean()), "still_image_pass": float((sts >= PULSE_THR).mean()),
              "live_snr_median_db": float(np.median(ls)), "still_snr_max_db": float(sts.max()),
              "n_each": 200, "real_video": "none on disk (~/data/pulse_videos absent): skipped",
              "heartbeat_cross_check": "finger pulse <-> EEG heartbeat artifact: NOT testable on static data (Phase 3)"}
log("  pulse:", {k: R["pulse"][k] for k in ("live_pass", "still_image_pass")})

# ===================================================================== STEP 6a layer table (fast, exact rules)
log("STEP 6  end-to-end on TEST virtual people")
VP = VIRT["test"]; nV = len(VP)
vt_people = [v for v, _ in VP]; et_subs = [e for _, e in VP]
PMt = person_matrix("test", vt_people, FINGERS)
vein_match = PMt >= thr_verify                                    # (claimed, presenter)
vein_commit = AC.unlock_matrix(vt_people, align_data(vt_people), VC_PARAMS["K"], best_c["t"], "helper",
                               return_matrix=True)                 # any finger, 6 tries

Ue, _ = eeg_lock(PROJ, et_subs, best_e["dims"], best_e["keep"], best_e["enc"], best_e["n"], best_e["t"])
# each presenter's own liveness outcomes (one simulated pulse video + one SSVEP trial each)
r = np.random.default_rng(77)
pulse_ok = np.array([P_.check_pulse(P_.synthetic_finger_video(bpm=r.uniform(50, 110),
                     depth=r.uniform(0.002, 0.006), noise=r.uniform(0.4, 1.5), seed=5000 + k), 30, PULSE_THR)[0]
                     for k in range(nV)])
ssvep_ok = []
for k, s in enumerate(et_subs):
    ok = False
    for _ in range(MAX_TRIES):
        ch, x, _ = trial(s, "later", MAIN_K, MAIN_SEG, MAIN_AMP, r)
        ok |= SS.verify_response(x, SF, ch, rho_thr[MAIN_SEG])[0]
    ssvep_ok.append(ok)
consent_ok = []
for k, s in enumerate(et_subs):
    ok = False
    for _ in range(MAX_TRIES):
        ch = SS.make_consent_challenge("vote yes on proposal 7", MAIN_K, MAIN_SEG, rng=r)
        rest = occ["later"][eeg_index[s]]; need = int(MAIN_K * MAIN_SEG * SF); o = r.integers(0, rest.shape[1] - need)
        dcs = SS.decide_consent(SS.simulate_response(rest[:, o:o + need], SF, SS.attended(ch, "confirm"), MAIN_AMP, r, WEIGHTS),
                                SF, ch, rho_thr[MAIN_SEG])[0]
        if dcs != "none":
            ok = dcs == "confirm"; break
    consent_ok.append(ok)
consent_ok = np.array(consent_ok)
ssvep_ok = np.array(ssvep_ok)
offV = ~np.eye(nV, dtype=bool)
layers = {}
for mode, V in (("matcher", vein_match), ("commitment", vein_commit)):
    live_b = (pulse_ok & ssvep_ok)[None, :].repeat(nV, 0)          # presenter b's liveness
    steps = [("vein", V), ("+ pulse", V & pulse_ok[None, :]), ("+ SSVEP (3 tries)", V & live_b),
             ("+ SSVEP consent (high-stakes)", V & live_b & consent_ok[None, :]),
             ("[dropped] + EEG brain key instead", V & live_b & Ue)]
    layers[mode] = [{"layers": n_, "GAR": float(np.diag(M).mean()), "zero_effort_FAR": float(M[offV].mean()),
                     "impostor_pairs": int(offV.sum())} for n_, M in steps]
R["layer_table"] = layers
for mode in layers:
    for row in layers[mode]:
        log(f"  {mode:10s} {row['layers']:18s} GAR {100 * row['GAR']:5.1f}%   FAR {row['zero_effort_FAR']:.4f}")
R["vein_test_detail"] = {
    "person_EER_matcher": eer(np.diag(PMt), PMt[offV])[0],
    "retake_rate_test": retake_rate["test"],
    "VI_test_FMR_at_thr_verify": float((virtual_impostors("test", v_test, FINGERS) >= thr_verify).mean()),
    "commit_test_all_62": {w: AC.unlock_matrix(v_test, align_data(v_test), VC_PARAMS["K"], best_c["t"], w)
                           for w in ("helper", "com")}}

# ===================================================================== STEP 6b real pipeline run
log("STEP 6b real pipeline (Argon2id, AES-GCM, Ed25519, LocalLedger) on every test virtual person")
from neurolock.pop.enrol import Capture, Rejected, Station
from neurolock.pop.ledger import LocalLedger, LedgerError
from neurolock.pop.verify import verify
from neurolock.pop import delegation as DL

PARAMS = {"vein_matcher": VM_PARAMS, "vein_commit": VC_PARAMS, "brain_key": BK_PARAMS,
          "ssvep": {"rho_thr": rho_thr[MAIN_SEG], "K": MAIN_K, "seg_s": MAIN_SEG, "amp_sim": MAIN_AMP,
                    "max_tries": MAX_TRIES},
          "pulse": {"min_snr_db": PULSE_THR},
          "fingers": FINGERS, "virtual_test": VP, "occ_weights": WEIGHTS}
pickle.dump(PARAMS, open(f"{OUT}/pop_params.pkl", "wb"))


def capture(v, e, vein_session, r, which_eeg, ssvep_challenge=None, live=True, eeg_feat=None):
    cap = Capture(label=f"vein {v} / eeg {e}")
    cap.vein_enrol = {f: VD.images(v, f, vein_session, range(6)) for f in FINGERS}
    cap.vein_probe = {f: [VD.images(v, f, vein_session, a) for a in ATTEMPTS] for f in FINGERS}
    cap.pulse = (P_.synthetic_finger_video(bpm=r.uniform(55, 100), depth=r.uniform(0.002, 0.006),
                                           noise=r.uniform(0.4, 1.5), seed=int(r.integers(1e9))) if live else
                 P_.synthetic_finger_video(still_image=cv2.resize(cv2.imread(VD.samples[VD.ix[(v, FINGERS[0], 1, 1)]].path, 0), (80, 60)),
                                           noise=r.uniform(0.4, 1.5), seed=int(r.integers(1e9))), 30)
    rest = occ[which_eeg][eeg_index[e]]; need = int(MAIN_K * MAIN_SEG * SF)

    def respond(ch, fixed=ssvep_challenge):
        """SIMULATED: the person's real resting EEG + a response to the flicker
        they look at. Liveness challenge: the shown sequence. Consent challenge:
        the sequence of the button they look at (cap.intent). If `fixed` is
        set, the 'person' is a replay of a response recorded for an old challenge."""
        o = r.integers(0, rest.shape[1] - need)
        if fixed is not None:
            target = fixed
        else:
            target = SS.attended(ch, cap.intent) if "confirm" in ch else ch
        return SS.simulate_response(rest[:, o:o + need], SF, target, MAIN_AMP, r, WEIGHTS), SF
    cap.ssvep = respond
    return cap


quiet = lambda *a, **k: None
st = Station(PARAMS, LocalLedger)
r = np.random.default_rng(2026)
order = r.permutation(nV)
creds, enrol_times, false_dup, other_reject = {}, [], [], []
for k in order:
    v, e = VP[k]; ch = SS.make_challenge(MAIN_K, MAIN_SEG)
    t = time.perf_counter()
    try:
        creds[k], _ = st.enrol(capture(v, e, 1, r, "enrol"), ch, log=quiet)
    except Rejected as ex:
        (false_dup if ex.layer == "dedup" else other_reject).append((int(k), ex.layer))
    enrol_times.append(time.perf_counter() - t)
log(f"  enrolled {len(creds)}/{nV}; false duplicates {len(false_dup)}; other rejections {other_reject}")

ver = {f"{m}_{h}": [] for m in ("commitment", "matcher") for h in ("normal", "high_stakes")}; ver_times = []
for k, cred in creds.items():
    v, e = VP[k]
    for key in ver:
        mode, hs = key.split("_", 1)
        ch = SS.make_challenge(MAIN_K, MAIN_SEG)
        t = time.perf_counter()
        try:
            verify(st, cred, capture(v, e, 2, r, "later"), ch, mode=mode, high_stakes=(hs == "high_stakes"),
                   action="vote yes on proposal 7", log=quiet)
            ver[key].append((int(k), "pass"))
        except Rejected as ex:
            ver[key].append((int(k), ex.layer))
        ver_times.append(time.perf_counter() - t)
imp_runs = []
keys = list(creds)
for j in range(80):
    a, b = r.choice(keys, 2, replace=False)
    v, e = VP[b]; ch = SS.make_challenge(MAIN_K, MAIN_SEG)
    try:
        verify(st, creds[a], capture(v, e, 2, r, "later"), ch, mode="commitment", high_stakes=True,
               action="vote yes on proposal 7", log=quiet)
        imp_runs.append("ACCEPTED")
    except Rejected as ex:
        imp_runs.append(ex.layer)
# re-enrolment with session-2 veins
reenrol = []
for k in creds:
    v, e = VP[k]; ch = SS.make_challenge(MAIN_K, MAIN_SEG)
    try:
        st.enrol(capture(v, e, 2, r, "later"), ch, log=quiet)
        reenrol.append("not caught")
    except Rejected as ex:
        reenrol.append(ex.layer)


def summ(lst):
    out = {}
    for x in lst:
        x = x[1] if isinstance(x, tuple) else x
        out[x] = out.get(x, 0) + 1
    return out


R["pipeline_run"] = {
    "enrolled": len(creds), "of": nV, "false_duplicates_at_enrolment": len(false_dup),
    "other_enrol_rejections": summ(other_reject),
    "verify_outcomes": {m: summ(v) for m, v in ver.items()},
    "verify_GAR": {m: float(np.mean([x[1] == "pass" for x in v])) for m, v in ver.items()},
    "impostor_attempts_commitment": summ(imp_runs),
    "re_enrolment": summ(reenrol), "re_enrolment_catch_rate": float(np.mean([x == "dedup" for x in reenrol])),
    "timing_s": {"argon2id_one_call": crypto.time_kdf(3), "argon2_params": crypto.ARGON2,
                 "enrol_mean": float(np.mean(enrol_times)), "enrol_max (largest registry)": float(np.max(enrol_times)),
                 "verify_mean": float(np.mean(ver_times))},
    "ledger_chain_valid": st.ledger.verify_chain()}
log("  pipeline:", json.dumps(R["pipeline_run"], indent=1))

# ===================================================================== attack table
log("  attack table")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from neurolock.pop.enrol import device_keys, pub_hex
passing = [k for k, o in ver["matcher_normal"] if o == "pass"]
k0 = passing[0]; v0, e0 = VP[k0]
ch = SS.make_challenge(MAIN_K, MAIN_SEG)
proof, S0, id_sk = verify(st, creds[k0], capture(v0, e0, 2, r, "later"), ch, mode="matcher", log=quiet)
agent1, agent2 = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
d1 = DL.make_delegation(id_sk, creds[k0]["id_pub"], S0, pub_hex(agent1), "fieldstation-vote", ["vote"])
did = st.ledger.post_delegation(d1)
try:
    st.ledger.post_delegation(DL.make_delegation(id_sk, creds[k0]["id_pub"], S0, pub_hex(agent2), "fieldstation-vote", ["vote"]))
    second = "accepted (BAD)"
except LedgerError as ex:
    second = f"rejected: {ex}"
other_app = st.ledger.post_delegation(DL.make_delegation(id_sk, creds[k0]["id_pub"], S0, pub_hex(agent2), "fieldstation-forum", ["post"]))
st.ledger.revoke(DL.make_revocation(id_sk, creds[k0]["id_pub"], did))
ok_after, why = DL.agent_action_allowed(st.ledger, did, agent1, {"vote": "yes"})
# printed vein image (no pulse), through the real pipeline
ch = SS.make_challenge(MAIN_K, MAIN_SEG)
try:
    verify(st, creds[k0], capture(v0, e0, 2, r, "later", live=False), ch, mode="matcher", log=quiet); printed = "accepted (BAD)"
except Rejected as ex:
    printed = f"rejected by {ex.layer}"
# replayed EEG: response recorded for challenge A presented for challenge B
chA, chB = SS.make_challenge(MAIN_K, MAIN_SEG), SS.make_challenge(MAIN_K, MAIN_SEG)
capA = capture(v0, e0, 2, r, "later", ssvep_challenge=chA)     # replays the chA recording
try:
    verify(st, creds[k0], capA, chB, mode="matcher", log=quiet); replay = "accepted"
except Rejected as ex:
    replay = f"rejected by {ex.layer}"
# person looks at DENY during a high-stakes action
capD = capture(v0, e0, 2, r, "later"); capD.intent = "deny"
try:
    verify(st, creds[k0], capD, SS.make_challenge(MAIN_K, MAIN_SEG), mode="matcher", high_stakes=True,
           action="transfer all funds", log=quiet); deny_res = "consent given (BAD)"
except Rejected as ex:
    deny_res = f"rejected by {ex.layer}"
# replayed CONFIRM recording from an earlier consent, against a new action
old = SS.make_consent_challenge("vote yes on proposal 7", MAIN_K, MAIN_SEG)
capR = capture(v0, e0, 2, r, "later")
live_respond = capR.ssvep
capR.ssvep = lambda ch: live_respond(ch) if "confirm" not in ch else live_respond(ch, fixed=SS.attended(old, "confirm"))
try:
    verify(st, creds[k0], capR, SS.make_challenge(MAIN_K, MAIN_SEG), mode="matcher", high_stakes=True,
           action="transfer all funds", log=quiet); rep_res = "consent given (BAD)"
except Rejected as ex:
    rep_res = f"rejected by {ex.layer}"
lay = R["layer_table"]
att = [
    {"attack": "Impostor's own veins (zero effort)", "stopped_by": "vein layer",
     "measured": f"matcher FAR {lay['matcher'][0]['zero_effort_FAR']:.4f}, commitment FAR "
                 f"{lay['commitment'][0]['zero_effort_FAR']:.4f} over {offV.sum()} pairs; real pipeline: "
                 f"{summ(imp_runs)} of 80"},
    {"attack": "Same person enrolling twice (session-2 veins)", "stopped_by": "dedup",
     "measured": f"caught {R['pipeline_run']['re_enrolment_catch_rate'] * 100:.1f}% ({summ(reenrol)})"},
    {"attack": "Static / printed vein image (no pulse) [SIMULATED video]", "stopped_by": "pulse",
     "measured": f"still-image pass rate {R['pulse']['still_image_pass']:.3f} (n=200); pipeline: {printed}"},
    {"attack": "Replayed EEG for a new challenge [SIMULATED SSVEP]", "stopped_by": "SSVEP",
     "measured": f"replay pass {MAIN_SS['replay_pass']:.4f} (n={MAIN_SS['trials']}); pipeline: {replay}"},
    {"attack": "Resting EEG, no response [SIMULATED SSVEP]", "stopped_by": "SSVEP",
     "measured": f"rest pass {MAIN_SS['rest_pass']:.4f} (n={MAIN_SS['trials']})"},
    {"attack": "Person looks at DENY for a high-stakes action [SIMULATED SSVEP]", "stopped_by": "SSVEP consent",
     "measured": f"consent given {CONSENT['deny_gives_consent']:.4f} (n={CONSENT['trials']}); pipeline: {deny_res}"},
    {"attack": "Replayed CONFIRM recording for a new action [SIMULATED SSVEP]", "stopped_by": "SSVEP consent",
     "measured": f"consent given {CONSENT['replay_gives_consent']:.4f} (n={CONSENT['trials']}); pipeline: {rep_res}"},
    {"attack": "Second agent delegation, same app", "stopped_by": "nullifier", "measured": second},
    {"attack": "Using a revoked delegation", "stopped_by": "ledger", "measured": f"allowed={ok_after}: {why}"},
    {"attack": "Software-injected synthetic signals (attacker knows challenge)", "stopped_by": "NOT STOPPED (needs trusted hardware)",
     "measured": f"synthetic SSVEP pass {MAIN_SS['injected_synthetic_pass']:.3f}"},
]
R["attack_table"] = att
R["delegation"] = {"first": "accepted", "second_same_app": second, "different_app": "accepted",
                   "after_revocation": why}
for a in att:
    log(f"  {a['attack']:60s} -> {a['measured']}")

# ===================================================================== plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, mode in enumerate(("matcher", "commitment")):
    rows_ = layers[mode]
    ax[j].bar(range(len(rows_)), [100 * x["GAR"] for x in rows_], color="#4C78A8")
    ax[j].set_xticks(range(len(rows_))); ax[j].set_xticklabels([x["layers"] for x in rows_], fontsize=8)
    for i, x in enumerate(rows_):
        ax[j].text(i, 100 * x["GAR"] + 1, f"{100 * x['GAR']:.0f}%\nFAR {x['zero_effort_FAR']:.3f}", ha="center", fontsize=7)
    ax[j].set_ylim(0, 115); ax[j].set_ylabel("genuine accept rate (%)")
    ax[j].set_title(f"vein {mode} mode: adding layers (test, n={nV})", fontsize=9)
plt.suptitle("Pulse + SSVEP are SIMULATED; virtual people (vein + EEG independent)", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/layer_contribution.png", dpi=140); plt.close()

plt.figure(figsize=(6, 4))
for amp in AMPS:
    plt.plot(SEGS, [100 * ss_res["by_seg_and_amp"][f"amp{amp}_seg{sg}"]["true_pass"] for sg in SEGS], "o-",
             label=f"amplitude {amp} x RMS")
plt.xlabel("segment length (s)"); plt.ylabel("true pass rate, K=4 (%)")
plt.title("SSVEP challenge: SIMULATED responses on real resting EEG\n(protocol test, not physiological evidence)", fontsize=9)
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/ssvep_vs_segment.png", dpi=140); plt.close()

pr = R["pipeline_run"]
plt.figure(figsize=(5, 3.5))
vals = [pr["enrolled"], pr["false_duplicates_at_enrolment"], int(pr["re_enrolment_catch_rate"] * pr["enrolled"])]
plt.bar(["enrolled", "false duplicates", "re-enrolments caught"], vals, color=["#4C78A8", "#E45756", "#54A24B"])
for i, x in enumerate(vals):
    plt.text(i, x + 0.5, str(x), ha="center")
plt.title(f"Dedup on {nV} test virtual people (real pipeline)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/dedup_summary.png", dpi=140); plt.close()

ok_all = {k for k, o in ver["commitment_high_stakes"] if o == "pass"}
ok_norm = [k for k, o in ver["commitment_normal"] if o == "pass"]
asha = sorted(ok_all)[0] if ok_all else ok_norm[0]
bob = next(k for k in range(nV) if k != asha)
PARAMS["demo"] = {"asha": int(asha), "bob": int(bob),
                  "note": f"Asha = a test person who passed every layer in the evaluation run "
                          f"({len(ok_all)}/{len(creds)} did including SSVEP consent, {len(ok_norm)}/{len(creds)} without)"}
pickle.dump(PARAMS, open(f"{OUT}/pop_params.pkl", "wb"))
R["demo_cast"] = PARAMS["demo"]
R["params_public"] = {"vein_matcher": {k: (list(v) if isinstance(v, tuple) else v) for k, v in VM_PARAMS.items()},
                      "vein_commit": VC_PARAMS, "ssvep_rho_thr": rho_thr[MAIN_SEG], "pulse_min_snr_db": PULSE_THR,
                      "brain_key": {k: v for k, v in BK_PARAMS.items() if k != "proj"}}
R["runtime_s"] = time.time() - T0
json.dump(R, open(f"{OUT}/results_pop.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done ->", f"{OUT}/results_pop.json")
