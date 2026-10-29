"""Full cross-session score matrices for FV-USM, used by improve_vein.py.

For each split (same seed-0 people split as evaluate_vein.py) and method:
  X[i, j] = match.score(session-1 image i, session-2 image j)   all fingers
  W1/W2   = same-finger, same-session 6x6 scores (for capture-quality checks)
Saved to ~/.cache/neurolock_vein/scoremat_<method>.npz (scores only, no templates).
"""
import os, sys
from multiprocessing import Pool
import numpy as np
from neurolock.vein import data, match

method = sys.argv[1]
cache = os.path.expanduser("~/.cache/neurolock_vein")
samples = data.load("fvusm", os.path.expanduser("~/data/FV-USM"))
z = np.load(f"{cache}/fvusm_{method}.npz")
maps = np.unpackbits(z["bits"], axis=1)[:, : z["h"] * z["w"]].reshape(-1, z["h"], z["w"])

rng = np.random.default_rng(0)
people = np.array(sorted({s.person for s in samples}))
rng.shuffle(people)                       # identical to evaluate_vein.py
half = len(people) // 2
split_of = {p: ("tune" if k < half else "test") for k, p in enumerate(people)}


def idx(split, sess):
    k = [i for i, s in enumerate(samples) if split_of[s.person] == split and s.session == sess]
    return sorted(k, key=lambda i: (samples[i].person, samples[i].finger, samples[i].index))


def row(args):
    i, cols = args
    return [match.score(maps[i], maps[j]) for j in cols]


out = {}
with Pool(16) as p:
    for sp in ("tune", "test"):
        r, c = idx(sp, 1), idx(sp, 2)
        out[f"{sp}_rows"], out[f"{sp}_cols"] = np.array(r), np.array(c)
        out[f"{sp}_X"] = np.array(p.map(row, [(i, c) for i in r], chunksize=8), np.float32)
        for sess, k in ((1, r), (2, c)):   # within-finger, within-session 6x6 blocks
            blocks = [k[f:f + 6] for f in range(0, len(k), 6)]
            out[f"{sp}_W{sess}"] = np.array(
                p.map(row, [(i, b) for b in blocks for i in b], chunksize=8), np.float32
            ).reshape(len(blocks), 6, 6)
        print(sp, out[f"{sp}_X"].shape, flush=True)
meta = np.array([(s.person, s.finger, s.session, s.index) for s in samples])
np.savez_compressed(f"{cache}/scoremat_{method}.npz", meta=meta, **out)
