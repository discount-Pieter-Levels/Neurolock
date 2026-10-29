"""Export REAL intermediate results of the NeuroLock pipeline for the interactive
web demo (demo_web/index.html). Run from the repo root with ../venv:
    ../venv/bin/python demo_web/export_demo_data.py
Writes demo_web/data.js. Vein data = FV-USM test people; SSVEP = simulated on real EEG
(same simulate_response as the evaluation), labelled as such in the page."""
import base64, json, pickle, os, time, hashlib
import cv2, numpy as np
from neurolock.pop.biometrics import VeinData, VeinMatcher, VeinCommitment, ATTEMPTS, METHODS
from neurolock.vein import preprocess as pp, extract as ex, match, align as valign, code as vcode
from neurolock.liveness import ssvep as SS, pulse as PU
from neurolock import fuzzy

T0 = time.time()
P = pickle.load(open("results_pop/pop_params.pkl", "rb"))
VD = VeinData(); M = VeinMatcher(P["vein_matcher"]); C = VeinCommitment(P["vein_commit"])
F = P["fingers"]; VP = P["virtual_test"]
ASHA, BOB = VP[P["demo"]["asha"]], VP[P["demo"]["bob"]]
OTHERS = [p for k, p in enumerate(VP) if k > 1][:11]

def png(a, q=None):
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    ok, b = cv2.imencode(".png" if q is None else ".jpg", a, [] if q is None else [cv2.IMWRITE_JPEG_QUALITY, q])
    return ("data:image/png;base64," if q is None else "data:image/jpeg;base64,") + base64.b64encode(b).decode()

def raw(v, f, s, i):
    return cv2.imread(VD.samples[VD.ix[(v, f, s, i)]].path, 0)

def stages(v, f, s, i):
    g = raw(v, f, s, i)
    L, R, tip = pp.finger_edges(g)
    strip = pp.roi(g); en = pp.enhance(strip)
    gb, mc = ex.extract(en, "gabor"), ex.extract(en, "maxcurv")
    k = VD.ix[(v, f, s, i)]
    assert (VD.maps["maxcurv"][k] == mc).mean() > 0.97, "cache mismatch"
    small = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
    ys = np.arange(0, g.shape[0], 8)
    return {"raw": png(small, 82), "rawW": small.shape[1], "rawH": small.shape[0],
            "edges": [[int(y // 2), float(L[y] / 2), float(R[y] / 2)] for y in ys],
            "tip": int(tip // 2), "top": int(max(tip - pp.TIP_MARGIN - pp.LENGTH, 0) // 2),
            "bottom": int((tip - pp.TIP_MARGIN) // 2),
            "roi": png(strip), "enh": png(en),
            "gabor": png(255 - gb * 255), "maxcurv": png(255 - mc * 255)}

def capture(person, session):
    v = person[0]
    enrol = {f: VD.images(v, f, session, range(6)) for f in F}
    probe = {f: [VD.images(v, f, session, a) for a in ATTEMPTS] for f in F}
    return enrol, probe

def overlay(a, b):
    dy, dx = match.best_shift(a, b)
    bs = np.roll(np.roll(b, dy, 0), dx, 1)
    img = np.full(a.shape + (3,), 250, np.uint8)
    img[(a > 0) & (bs == 0)] = (230, 170, 40)   # BGR: template only = blue-ish
    img[(bs > 0) & (a == 0)] = (60, 60, 230)    # probe only = red
    img[(a > 0) & (bs > 0)] = (30, 30, 30)      # both = black
    return png(img), [dy, dx]

out = {"params": {"thr_verify": P["vein_matcher"]["thr_verify"], "thr_dedup": P["vein_matcher"]["thr_dedup"],
                  "rho_thr": P["ssvep"]["rho_thr"], "freqs": list(SS.FREQS), "seg_s": SS.SEG_S,
                  "K": P["ssvep"]["K"], "bch": P["vein_commit"]["code"], "Kbits": P["vein_commit"]["K"],
                  "fingers": list(F), "pulse_snr": P["pulse"]["min_snr_db"]}}
fA = F[0]
out["asha"] = {"id": [int(x) for x in ASHA], "s1": stages(ASHA[0], fA, 1, 1), "s2": stages(ASHA[0], fA, 2, 1)}
out["bob"] = {"id": [int(x) for x in BOB], "s2": stages(BOB[0], fA, 2, 1)}

# matcher scores ------------------------------------------------------------
eA1, _ = capture(ASHA, 1); _, pA2 = capture(ASHA, 2); _, pB2 = capture(BOB, 2)
sAA, _ = M.person_score(eA1, pA2); sBA, _ = M.person_score(eA1, pB2)
mA1 = VD.maps["maxcurv"][VD.ix[(ASHA[0], fA, 1, 1)]]
ovA, shA = overlay(mA1, VD.maps["maxcurv"][VD.ix[(ASHA[0], fA, 2, 1)]])
ovB, shB = overlay(mA1, VD.maps["maxcurv"][VD.ix[(BOB[0], fA, 2, 1)]])
rawA = match.score(mA1, VD.maps["maxcurv"][VD.ix[(ASHA[0], fA, 2, 1)]])
rawB = match.score(mA1, VD.maps["maxcurv"][VD.ix[(BOB[0], fA, 2, 1)]])
out["match"] = {"asha": sAA, "bob": sBA, "ovA": ovA, "ovB": ovB, "shiftA": shA, "shiftB": shB,
                "diceA": rawA, "diceB": rawB}
print("scores", sAA, sBA, time.time() - T0)

# score distributions over test people (session-1 enrol vs session-2 probe) ----
pool = VP[:30]
caps = {p: capture(p, 1)[0] for p in pool}; probes = {p: capture(p, 2)[1] for p in pool}
gen, imp = [], []
for a in pool:
    for b in pool:
        s, _ = M.person_score(caps[a], probes[b])
        (gen if a == b else imp).append(round(s, 2))
out["dist"] = {"gen": gen, "imp": imp, "n": len(pool)}
print("dist", len(gen), len(imp), time.time() - T0)

# dedup: registry of OTHERS (+ Asha's session-1); newcomer = Asha's session-2 enrolment images
reg = [("Asha", eA1)] + [(f"#{k+1:03d}", capture(p, 1)[0]) for k, p in enumerate(OTHERS)]
def as_probe(enr):
    return {f: [{m: [enr[f][m][i] for i in a] for m in METHODS} for a in ATTEMPTS] for f in F}
eA2 = capture(ASHA, 2)[0]
out["dedup"] = {"names": [n for n, _ in reg],
                "asha2": [round(M.person_score(e, as_probe(eA2))[0], 2) for _, e in reg],
                "bob": [round(M.person_score(e, as_probe(capture(BOB, 2)[0]))[0], 2) for _, e in reg]}
print("dedup", out["dedup"], time.time() - T0)

# fuzzy commitment (maxcurv), both fingers, as recover_S tries them ------------
def commit_for(f):
    enrol_maps = eA1[f]["maxcurv"]
    helper = valign.make_helper(enrol_maps)
    w, rel = vcode.enroll_bits(C.codes(enrol_maps, helper))
    mask = vcode.reliable_mask(rel, C.K)
    wk = w[mask][C.perm]
    k, rec = fuzzy.enroll(wk, fuzzy.code_from_spec(C.spec))
    n = C.spec["n"]; L = len(rec["helper"])
    def probe_info(maps):
        best = None
        for t, c in enumerate(C.codes(maps, helper), 1):
            cw = c[mask][C.perm]
            kk, ok = fuzzy.recover(cw, rec)
            err = (cw[:L] != wk[:L]).astype(np.uint8)
            info = {"try": t, "ok": bool(ok), "blocks": [int(err[i:i + n].sum()) for i in range(0, L, n)],
                    "errbits": np.packbits(err).tolist(), "ber": float(err.mean())}
            if best is None or ok or info["ber"] < best["ber"]:
                best = info
            if ok:
                break
        return best
    return {"finger": int(f), "n": n, "t": C.spec["t"], "L": L, "keybits": int(len(k)), "keyhash": rec["hash"],
            "wbits": np.packbits(wk[:L]).tolist(),
            "asha": probe_info([m for att in pA2[f] for m in att["maxcurv"]]),
            "bob": probe_info([m for att in pB2[f] for m in att["maxcurv"]])}
out["commit"] = [commit_for(f) for f in F]
for c in out["commit"]:
    print("commit f", c["finger"], c["asha"]["ok"], c["asha"]["try"], c["asha"]["blocks"], c["bob"]["ok"], c["bob"]["blocks"])

# SSVEP: real resting occipital EEG + simulated response -----------------------
occ = np.load(os.path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz"))
sf = float(occ["sf"]); eix = {s: i for i, s in enumerate(occ["subjects"].tolist())}
rest = occ["later"][eix[ASHA[1]]]
segn = int(SS.SEG_S * sf); rng = np.random.default_rng(7)
NSEG = 12
grid = []   # grid[seg][target] -> {rhos, oz}; target index len(FREQS) = looking away
oz = 5 if rest.shape[0] > 5 else 0
for s in range(NSEG):
    o = int(rng.integers(0, rest.shape[1] - segn))
    base = rest[:, o:o + segn]
    row = []
    for ti in range(len(SS.FREQS) + 1):
        if ti < len(SS.FREQS):
            x = SS.simulate_response(base, sf, {"freqs": [SS.FREQS[ti]], "seg_s": SS.SEG_S}, P["ssvep"]["amp_sim"], rng, P["occ_weights"])
        else:
            x = base.copy()
        seg = x[:, int(SS.LATENCY_S * sf):]
        _, _, rhos = SS.classify_segment(seg, sf)
        z = x[oz]; z = (z - z.mean()) / (z.std() + 1e-9)
        row.append({"rhos": [round(float(r), 3) for r in rhos], "oz": [round(float(v), 2) for v in z[::2]]})
    grid.append(row)
out["ssvep"] = {"sf": sf / 2, "grid": grid, "channels": int(rest.shape[0])}

# pulse: synthetic finger video brightness traces (live + still photo), for fallback
for name, kw in (("live", dict(bpm=72, seed=3)),
                 ("photo", dict(still_image=cv2.resize(raw(ASHA[0], fA, 1, 1), (80, 60)), seed=4))):
    fr = PU.synthetic_finger_video(**kw)
    sig = PU.brightness_trace(fr)
    ok, bpm, snr, _ = PU.check_pulse(fr, 30, P["pulse"]["min_snr_db"])
    out.setdefault("pulse", {})[name] = {"trace": [round(float(v), 3) for v in sig], "ok": bool(ok),
                                         "bpm": float(bpm), "snr": float(snr)}
print("pulse", {k: (v["ok"], v["bpm"], v["snr"]) for k, v in out["pulse"].items()})

open("demo_web/data.js", "w").write("window.NL = " + json.dumps(out) + ";\n")
print("wrote demo_web/data.js", os.path.getsize("demo_web/data.js") // 1024, "KB", round(time.time() - T0), "s")
