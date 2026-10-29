"""Does the finger-vein layer get good enough for dedup / proof of personhood?

Uses score matrices from scoremat_vein.py (cross-session: enrol = session 1,
probe = session 2). Each idea is stacked on the previous one:

  A  baseline: one enrol image vs one probe image (as in evaluate_vein.py)
  B  multi-enrol: max over the 6 stored enrol images
  C  fusion: gabor + maxcurv scores (z-normalised on tune impostors)
  D  multi-probe: a capture attempt = 3 images; score = mean over them
  E  quality retake: if the 3 images of an attempt disagree with each other,
     reject the attempt and use the second attempt
  F  multi-finger: person score = mean of k fingers (k = 1, 2, 4), same slot

Every threshold / choice is fixed on the TUNE people; numbers are TEST people.
Dedup is extrapolated to large populations from the per-comparison false match
rate (FMR): P(new person falsely flagged) = 1 - (1 - FMR)^N.
"""
import json
import os
import numpy as np

cache = os.path.expanduser("~/.cache/neurolock_vein")
M = {m: np.load(f"{cache}/scoremat_{m}.npz") for m in ("gabor", "maxcurv")}
rng = np.random.default_rng(1)
R = {}


def eer(g, i):
    t = np.unique(np.concatenate([g, i]))
    t = t[:: max(1, len(t) // 4000)]
    frr = np.array([(g < x).mean() for x in t]); far = np.array([(i >= x).mean() for x in t])
    k = np.argmin(abs(frr - far))
    return float((frr[k] + far[k]) / 2), float(t[k])


def X5(m, sp):
    """(fingers, 6 enrol, fingers, 6 probe) score tensor + same-session blocks."""
    z = M[m]; X = z[f"{sp}_X"]; n = X.shape[0] // 6
    return X.reshape(n, 6, n, 6), z[f"{sp}_W1"], z[f"{sp}_W2"]


def finger_ids(sp):
    meta = M["gabor"]["meta"]; r = M["gabor"][f"{sp}_rows"][::6]
    return meta[r, 0], meta[r, 1]       # person, finger slot


def pairs(T):
    """genuine diag / impostor off-diag of an (n, n, ...) array."""
    n = T.shape[0]; off = ~np.eye(n, dtype=bool)
    return T[np.arange(n), np.arange(n)].ravel(), T[off].ravel()


def report(name, gt, it, g, i):
    e_tune, thr = eer(gt, it)
    e, _ = eer(g, i)
    R[name] = {"test_EER": e, "tune_EER": e_tune,
               "test_FAR_at_tune_thr": float((i >= thr).mean()),
               "test_FRR_at_tune_thr": float((g < thr).mean())}
    print(f"{name:34s} test EER {100*e:5.2f}%   (tune {100*e_tune:5.2f}%)")


# ---------------------------------------------------------------- A, B per method
T = {}
for m in M:
    for sp in ("tune", "test"):
        X, _, _ = X5(m, sp)
        T[m, sp, "A"] = X[:, 0, :, :]                        # enrol image 0 only
        T[m, sp, "B"] = X.max(axis=1)                         # (n, n, 6 probes)
    for st in "AB":
        gt, it = pairs(T[m, "tune", st]); g, i = pairs(T[m, "test", st])
        report(f"{st} {m}", gt, it, g, i)

# ---------------------------------------------------------------- C fusion
norm = {}
for m in M:
    _, it = pairs(T[m, "tune", "B"]); norm[m] = (it.mean(), it.std())
F = {sp: sum((T[m, sp, "B"] - norm[m][0]) / norm[m][1] for m in M) for sp in ("tune", "test")}
report("C fusion (B)", *pairs(F["tune"]), *pairs(F["test"]))


# ---------------------------------------------------------------- D multi-probe
def W_fused(sp):
    return sum((X5(m, sp)[2] - norm[m][0]) / norm[m][1] for m in M)  # (n, 6, 6) session 2


ATT = [(0, 1, 2), (3, 4, 5)]
D = {}
for agg in ("mean", "median", "max"):
    f = getattr(np, agg)
    D[agg] = {sp: np.stack([f(F[sp][:, :, a], axis=2) for a in ATT], axis=2) for sp in F}
best = min(D, key=lambda a: eer(*pairs(D[a]["tune"]))[0])
R["D_aggregate_chosen_on_tune"] = best
report(f"D multi-probe 3 imgs ({best})", *pairs(D[best]["tune"]), *pairs(D[best]["test"]))
DD = D[best]


# ---------------------------------------------------------------- E quality retake
def consistency(sp):
    W = W_fused(sp)
    return np.stack([np.mean([W[:, x, y] for x in a for y in a if x < y], axis=0) for a in ATT], 1)


Q = {sp: consistency(sp) for sp in F}
best_e = None
for frac in (0.0, 0.05, 0.1, 0.2, 0.3):
    qthr = np.quantile(Q["tune"][:, 0], frac) if frac else -np.inf

    def pick(sp):
        q = Q[sp]
        use2 = (q[:, 0] < qthr) & (q[:, 1] >= q[:, 0])        # retake only if it helps
        a = np.where(use2, 1, 0)
        return DD[sp][:, np.arange(len(a)), a], use2.mean()   # rows: claimed, cols: probe finger

    Et, _ = pick("tune"); Es, rate = pick("test")
    e = eer(*pairs(Et[..., None]))[0]
    if best_e is None or e < best_e[0]:
        best_e = (e, frac, Et, Es, rate)
_, frac, Et, Es, rate = best_e
R["E_reject_fraction_chosen_on_tune"] = frac; R["E_test_retake_rate"] = float(rate)
report(f"E quality retake (rej {int(frac*100)}%)", *pairs(Et[..., None]), *pairs(Es[..., None]))
EF = {"tune": Et, "test": Es}       # (n claimed, n probe) one attempt per probe finger


# ---------------------------------------------------------------- same-person correlation
p, s = finger_ids("test")
off_slot = (p[:, None] == p[None, :]) & (s[:, None] != s[None, :])
diff = p[:, None] != p[None, :]
R["impostor_same_person_other_finger_mean_z"] = float(EF["test"][off_slot].mean())
R["impostor_stranger_mean_z"] = float(EF["test"][diff].mean())


# ---------------------------------------------------------------- F multi-finger, person level
def person_mats(sp):
    """per slot: (people, people) attempt scores, enrolled person x probe person."""
    p, s = finger_ids(sp); P = np.unique(p)
    out = []
    for k in np.unique(s):
        ix = [np.where((p == q) & (s == k))[0][0] for q in P]
        out.append(EF[sp][np.ix_(ix, ix)])
    return np.stack(out)       # (4 slots, people, people)


PM = {sp: person_mats(sp) for sp in F}
# correlation of impostor scores across slots for the same (enrolled, probe) person pair
off = ~np.eye(PM["test"].shape[1], dtype=bool)
C = np.corrcoef(PM["test"][:, off])
R["impostor_cross_slot_corr_mean"] = float(C[~np.eye(4, dtype=bool)].mean())
print("impostor score corr across finger slots:", round(R["impostor_cross_slot_corr_mean"], 3))

SUBSETS = {1: [(0,)], 2: [(0, 1), (2, 3), (0, 2), (1, 3)], 4: [(0, 1, 2, 3)]}
NV = 4_000_000


def virtual_impostors(sp, sub):
    """Impostor person scores from fingers of *different* people (independence),
    so FMR can be measured down to ~1e-6. Validated by the cross-slot corr above."""
    pm = PM[sp]; n = pm.shape[1]; off = ~np.eye(n, dtype=bool)
    pools = [pm[k][off] for k in sub]
    return np.mean([pl[rng.integers(0, len(pl), NV)] for pl in pools], axis=0)


ded = {}
for k, subs in SUBSETS.items():
    gt = np.concatenate([PM["tune"][list(sb)].mean(0).diagonal() for sb in subs])
    gs = np.concatenate([PM["test"][list(sb)].mean(0).diagonal() for sb in subs])
    it_real = np.concatenate([PM["tune"][list(sb)].mean(0)[~np.eye(PM["tune"].shape[1], dtype=bool)] for sb in subs])
    is_real = np.concatenate([PM["test"][list(sb)].mean(0)[~np.eye(PM["test"].shape[1], dtype=bool)] for sb in subs])
    report(f"F {k} finger(s), person level", gt, it_real, gs, is_real)
    vt = np.concatenate([virtual_impostors("tune", sb) for sb in subs])
    vs = np.concatenate([virtual_impostors("test", sb) for sb in subs])
    rows = []
    for fmr_target in (1e-3, 1e-4, 1e-5, 1e-6):
        thr = np.quantile(vt, 1 - fmr_target)
        fmr = max((vs >= thr).mean(), 1 / len(vs))
        rows.append({"tune_FMR_target": fmr_target, "test_FMR_virtual": float((vs >= thr).mean()),
                     "test_FMR_real_pairs": float((is_real >= thr).mean()),
                     "real_impostor_pairs": int(len(is_real)),
                     "catch_rate_test": float((gs >= thr).mean()),
                     "genuine_test": int(len(gs)),
                     "N_people_for_1pct_false_dup": float(0.01 / fmr)})
        r = rows[-1]
        print(f"   FMR target {fmr_target:.0e}: catch {100*r['catch_rate_test']:5.1f}%  "
              f"test FMR {r['test_FMR_virtual']:.1e} (real {r['test_FMR_real_pairs']:.1e})  "
              f"-> ~{r['N_people_for_1pct_false_dup']:.0e} people")
    ded[k] = rows
R["dedup_by_fingers"] = ded
json.dump(R, open("results_vein/results_improve.json", "w"), indent=1)
