"""Does capture-quality selection fix the vein failures? (Phase 2b diagnosis)

Failures concentrate in captures whose brightness differs from enrolment
(exposure / illumination drift on the FV-USM rig) and in broken ROI crops.
A live system films ~30 frames/s, so it can keep capturing until it has good
frames. We emulate that with the 6 session-2 images: drop broken crops and
use the 3 whose brightness is closest to the enrolment median.

Quality per image (no identity information):
  brightness  = mean of the low-pass ROI strip (enrolment median is stored as
                a helper value: one public number per finger)
  broken crop = across-finger / along-finger gradient ratio < 1.0
                (streaky strip = edge detection failed)
Thresholds/settings fixed by inspection on TUNE; rates on TEST.
"""
import json
import numpy as np
SCR = "/tmp/claude-1000/-home-dhruv/ec0478ac-f9a6-4630-986a-ab707be5f591/scratchpad"
c = np.load(f"{SCR}/contrast.npy"); d = np.load(f"{SCR}/roi_check.npz")
bright, ratio, meta = c[:, 1], d["ratio"], d["meta"]
key = {tuple(m): k for k, m in enumerate(meta)}
M = [np.load(f"/home/dhruv/.cache/neurolock_vein/scoremat_{m}.npz") for m in ("gabor", "maxcurv")]
R = {}


def eer(g, i):
    t = np.unique(np.concatenate([g, i])); t = t[:: max(1, len(t) // 3000)]
    frr = np.array([(g < x).mean() for x in t]); far = np.array([(i >= x).mean() for x in t])
    k = np.argmin(abs(frr - far)); return (frr[k] + far[k]) / 2


norm = []
for Z in M:
    n = Z["tune_X"].shape[0] // 6; B = Z["tune_X"].reshape(n, 6, n, 6).max(1)
    imp = B[~np.eye(n, dtype=bool)]; norm.append((imp.mean(), imp.std()))

for sp in ("tune", "test"):
    n = M[0][f"{sp}_X"].shape[0] // 6; rows = M[0][f"{sp}_rows"][::6]; mm = M[0]["meta"]
    pf = [(mm[r, 0], mm[r, 1]) for r in rows]
    F = sum((Z[f"{sp}_X"].reshape(n, 6, n, 6).max(1) - mu) / sd for Z, (mu, sd) in zip(M, norm))  # (n,n,6)
    # enrol-side: drop broken enrol images (recompute max over kept enrol images)
    ok_e = np.array([[ratio[key[(p, f, 1, i)]] >= 1.0 for i in range(1, 7)] for p, f in pf])
    Xs = [Z[f"{sp}_X"].reshape(n, 6, n, 6) for Z in M]
    Fq = sum((np.where(ok_e[:, :, None, None], X, -np.inf).max(1) - mu) / sd for X, (mu, sd) in zip(Xs, norm))
    b1 = np.array([np.median([bright[key[(p, f, 1, i)]] for i in range(1, 7)]) for p, f in pf])
    b2 = np.array([[bright[key[(p, f, 2, i)]] for i in range(1, 7)] for p, f in pf])
    br = np.array([[ratio[key[(p, f, 2, i)]] < 1.0 for i in range(1, 7)] for p, f in pf])
    out = {}
    for name, FF, pick in (
            ("baseline: images 1-3", F, lambda a: np.tile([0, 1, 2], (n, 1))),
            ("any 3 images, no quality (upper bound of luck)", F, None),
            ("quality: drop broken, 3 closest in brightness", Fq, "q")):
        # claimed a, probe finger b: probe images are chosen from b's OWN capture (no identity info)
        if pick is None:
            S = FF.max(2)                                            # best of all 6
        else:
            if pick == "q":
                # brightness distance to the CLAIMED finger's enrolment brightness (public helper)
                S = np.zeros((n, n))
                for a in range(n):
                    dist = np.abs(np.log(b2 / b1[a])) + 10 * br               # (n probe fingers, 6)
                    idx = np.argsort(dist, axis=1)[:, :3]
                    S[a] = np.take_along_axis(FF[a], idx, axis=1).max(1)
            else:
                S = FF[:, :, :3].max(2)
        g = np.diag(S); i_ = S[~np.eye(n, dtype=bool)]
        out[name] = {"finger_EER": float(eer(g, i_))}
        # person level (2 fingers: R-index + R-middle, as in Phase 2)
        persons = sorted({p for p, _ in pf}); fi = {k: j for j, k in enumerate(pf)}
        P = np.zeros((len(persons), len(persons)))
        for f in (3, 4):
            ix = [fi[(p, f)] for p in persons]; P += S[np.ix_(ix, ix)] / 2
        gP = np.diag(P); iP = P[~np.eye(len(persons), dtype=bool)]
        out[name]["person_EER_2fingers"] = float(eer(gP, iP))
        if sp == "tune":
            R.setdefault("thr", {})[name] = float(np.sort(i_)[-max(1, int(1e-3 * len(i_)))])
        thr = R["thr"][name]
        out[name]["finger_FRR_at_tune_FMR1e-3"] = float((g < thr).mean())
        out[name]["finger_FMR_test"] = float((i_ >= thr).mean())
    R[sp] = out
    print(sp); [print(f"  {k:48s}", {a: round(b, 4) for a, b in v.items()}) for k, v in out.items()]
json.dump(R, open("results_vein/results_quality.json", "w"), indent=1)
