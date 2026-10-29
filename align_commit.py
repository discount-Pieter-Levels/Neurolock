"""Commitment mode: can a coarse alignment helper close the gap to 'oracle' alignment?

Phase 1 found centre-of-mass alignment loses ~30 points of GAR vs aligning the
probe to the stored vein map ('oracle', not private). Here the helper data
gets a COARSE vein-density map instead: the enrolment vein maps averaged,
blurred (sigma 6 px) and shrunk 4x to 25 x 75 numbers, quantised to 4 levels.
At unlock, the probe's own coarse map is slid over it (+-10 coarse px = +-40 px
along, +-4 coarse px = +-16 px across) and the best shift is applied before
computing the bit code. Fine vein detail (what the key is made from) is not in
the helper; the coarse map does leak where the big veins run (privacy cost,
not quantified here).

Settings (K, t) chosen on TUNE, rates on TEST. People/fingers as Phase 2:
2 fingers (R-index, R-middle), unlock if either finger opens within 6 probe images.
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from neurolock import fuzzy
from neurolock.pop.biometrics import PERM_SEED, VeinData, split_people
from neurolock.vein import code as vcode

VD = VeinData()
FINGERS = (3, 4)
METHOD = "maxcurv"
MAPS = VD.maps[METHOD]
from neurolock.vein.align import best_shift, coarse, make_helper


def codes_for(args):
    """For finger (person, f): helper, aligned enrol codes, and per-claim aligned probe codes."""
    person, f, claimants = args
    enrol = [MAPS[VD.ix[(person, f, 1, i)]] for i in range(1, 7)]
    helper = make_helper(enrol)
    enc = np.array([vcode.to_code(vcode.shift(m, *best_shift(helper, m))) for m in enrol])
    # probes of every claimant person b (genuine when b == person), aligned to THIS helper
    pr = np.array([[vcode.to_code(vcode.shift(MAPS[VD.ix[(b, f, 2, i)]], *best_shift(helper, MAPS[VD.ix[(b, f, 2, i)]])))
                    for i in range(1, 7)] for b in claimants])
    com = np.array([[vcode.to_code(MAPS[VD.ix[(b, f, 2, i)]], "com") for i in range(1, 7)] for b in claimants])
    enc_com = np.array([vcode.to_code(m, "com") for m in enrol])
    return enc, pr, enc_com, com


def unlock_matrix(people, data, K, t, which, return_matrix=False):
    perm = np.random.default_rng(PERM_SEED).permutation(K)
    U = []
    for f in FINGERS:
        Uf = np.zeros((len(people), len(people), 6), bool)
        for a in range(len(people)):
            enc, pr, enc_com, com = data[(people[a], f)]
            E_, P_ = (enc, pr) if which == "helper" else (enc_com, com)
            w, rel = vcode.enroll_bits(E_)
            mk = vcode.reliable_mask(rel, K)
            err = P_[:, :, mk][:, :, perm] ^ w[mk][perm][None, None]
            c = fuzzy.BCHCode(255, t); ok = np.ones(err.shape[:2], bool); i = 0
            for s in c._blocks(K):
                ok &= err[:, :, i:i + s].sum(-1) <= t; i += s
            Uf[a] = ok
        U.append(Uf)
    U = np.stack(U)
    anyf = U.any(axis=3).any(axis=0); kb = fuzzy.BCHCode(255, t).key_bits(K)
    if return_matrix:
        return anyf
    off = ~np.eye(len(people), dtype=bool)
    one = U[:, :, :, 0].any(axis=0)
    return {"GAR_6tries": float(np.diag(anyf).mean()), "FAR_6tries": float(anyf[off].mean()),
            "GAR_1img": float(np.diag(one).mean()), "key_bits": int(kb), "impostor_pairs": int(off.sum())}


if __name__ == "__main__":
    tune, test = split_people(VD.people, 0)
    res = {}
    for sp, people in (("tune", tune), ("test", test)):
        with Pool(16) as p:
            out = p.map(codes_for, [(a, f, people) for a in people for f in FINGERS], chunksize=2)
        data = {(a, f): o for (a, f), o in zip([(a, f) for a in people for f in FINGERS], out)}
        res[sp] = {}
        for which in ("com", "helper"):
            for K in (3072, 1536, 768):
                for t in (15, 23, 31, 39, 47):
                    res[sp][f"{which}_K{K}_t{t}"] = unlock_matrix(people, data, K, t, which)
        print(sp, "done", flush=True)
    chosen = {}
    for which in ("com", "helper"):
        c = [k for k, v in res["tune"].items() if k.startswith(which) and v["FAR_6tries"] == 0 and v["key_bits"] >= 128]
        best = max(c, key=lambda k: (res["tune"][k]["GAR_6tries"], res["tune"][k]["key_bits"]))
        chosen[which] = {"setting": best, "tune": res["tune"][best], "test": res["test"][best]}
        print(which, best, "TUNE", res["tune"][best], "\n   TEST", res["test"][best])
    json.dump({"chosen": chosen, "all": res}, open("results_vein/results_align_helper.json", "w"), indent=1)
