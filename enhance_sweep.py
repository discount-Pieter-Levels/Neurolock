"""Illumination-robust enhancement: does it fix the dark / exposure-drift failures?

Chosen on TUNE people only (finger-level fused EER and FRR at FMR 1e-3,
cross-session). Variants of preprocess.enhance (everything else unchanged):
  V0 current : CLAHE
  V1 flat    : divide by a heavy blur (removes uneven/changed illumination) -> CLAHE
  V2 blur+flat: light Gaussian denoise (sigma 1.5) -> flat-field -> CLAHE
  V3 nlm     : non-local-means denoise -> CLAHE
Writes vein-map caches fvusm_<method>_<variant>.npz next to the Phase 1 ones.
"""
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from neurolock.vein import data, extract, match, preprocess

CACHE = os.path.expanduser("~/.cache/neurolock_vein")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 12))


def flat(s):
    f = s.astype(np.float32) / (cv2.GaussianBlur(s.astype(np.float32), (0, 0), 15) + 1)
    return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


VARIANTS = {
    "V0": lambda s: s,
    "V1": flat,
    "V2": lambda s: flat(cv2.GaussianBlur(s, (0, 0), 1.5)),
    "V3": lambda s: cv2.fastNlMeansDenoising(s, None, h=7, templateWindowSize=7, searchWindowSize=15),
}
samples = data.load("fvusm", os.path.expanduser("~/data/FV-USM"))


def make(args):
    k, v, m = args
    strip = preprocess.roi(data.load_image(samples[k]))
    return extract.extract(clahe.apply(VARIANTS[v](strip)).astype(np.float32) / 255.0, m)


def maps_for(v, m):
    f = f"{CACHE}/fvusm_{m}_{v}.npz" if v != "V0" else f"{CACHE}/fvusm_{m}.npz"
    if not os.path.exists(f):
        with Pool(16) as p:
            mp = np.array(p.map(make, [(k, v, m) for k in range(len(samples))], chunksize=16), np.uint8)
        np.savez_compressed(f, bits=np.packbits(mp.reshape(len(mp), -1), axis=1), h=mp.shape[1], w=mp.shape[2])
    z = np.load(f)
    return np.unpackbits(z["bits"], axis=1)[:, : z["h"] * z["w"]].reshape(-1, z["h"], z["w"])


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "tune"
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
    rng = np.random.default_rng(0)
    people = np.array(sorted({s.person for s in samples})); rng.shuffle(people)
    sel = set(people[: len(people) // 2].tolist() if split == "tune" else people[len(people) // 2:].tolist())
    ids = sorted({(s.person, s.finger) for s in samples if s.person in sel})
    ix = {(s.person, s.finger, s.session, s.index): k for k, s in enumerate(samples)}
    E = {i: [ix[(*i, 1, j)] for j in range(1, 7)] for i in ids}
    P = {i: [ix[(*i, 2, j)] for j in range(1, 7)] for i in ids}
    MAPS = {}

    def job(a):
        """genuine: probe image j vs max over 6 enrol; impostor: same vs 12 random other fingers."""
        r = np.random.default_rng(hash(a) % 2**32)
        others = [ids[k] for k in r.choice(len(ids), 13, replace=False) if ids[k] != a][:12]
        out = []
        for m in ("gabor", "maxcurv"):
            mp = MAPS[m]
            g = [max(match.score(mp[e], mp[p]) for e in E[a]) for p in P[a]]
            im = [max(match.score(mp[e], mp[P[b][j]]) for e in E[a]) for b in others for j in (0, 3)]
            out.append((g, im))
        return out

    for v in variants:
        t = time.time()
        MAPS = {m: maps_for(v, m) for m in ("gabor", "maxcurv")}
        with Pool(16) as p:
            res = p.map(job, ids, chunksize=4)
        fused_g, fused_i = 0, 0
        for mi in range(2):
            g = np.array([r_[mi][0] for r_ in res]); i_ = np.array([r_[mi][1] for r_ in res])
            mu, sd = i_.mean(), i_.std()
            fused_g = fused_g + (g - mu) / sd; fused_i = fused_i + (i_ - mu) / sd
        g, i_ = fused_g.ravel(), fused_i.ravel()
        thr = np.quantile(i_, 1 - 1e-3)
        ts = np.quantile(np.concatenate([g, i_]), np.linspace(0, 1, 3000))
        frr = np.array([(g < x).mean() for x in ts]); far = np.array([(i_ >= x).mean() for x in ts])
        k = np.argmin(abs(frr - far))
        # failures among the darkest probes (bottom quartile of strip brightness proxy: fused not needed)
        print(f"{split} {v}: finger EER {100 * (frr[k] + far[k]) / 2:.2f}%  FRR@FMR1e-3 {100 * (g < thr).mean():.2f}%  "
              f"({len(g)} gen / {len(i_)} imp, {time.time() - t:.0f}s)", flush=True)
