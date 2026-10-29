"""Finger-vein layer, Phase 1: does it identify fingers, and how much entropy?

  python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method maxcurv --cross-session
  python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method gabor --cross-session
  python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method gabor          # same-session

Protocol
  identity  = (person, finger)
  enroll    = cross-session: all session-1 images; same-session: session-1 images 1-3
  probe     = cross-session: all session-2 images; same-session: session-1 images 4-6
  genuine   = every (enroll, probe) pair of the same identity
  impostor  = first enroll image of A vs first probe image of B, for every A != B
  split     = people shuffled (seed) into TUNE and TEST halves. Thresholds and
              settings are chosen on TUNE only; headline numbers come from TEST.

Outputs (in --out): results_vein_<method>_<cross|same>.json and PNGs.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

from neurolock import fuzzy
from neurolock.vein import data, preprocess, extract, match, code

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=list(data.LOADERS), default="fvusm")
ap.add_argument("--path", required=True)
ap.add_argument("--method", choices=list(extract.METHODS), default="maxcurv")
ap.add_argument("--cross-session", action="store_true",
                help="genuine = session 1 vs session 2 (headline). Default: same session")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--jobs", type=int, default=os.cpu_count())
ap.add_argument("--out", default=".")
ap.add_argument("--cache", default=os.path.expanduser("~/.cache/neurolock_vein"),
                help="vein maps are cached here (biometric templates: keep private)")
args = ap.parse_args()
mode = "cross" if args.cross_session else "same"
tag = f"{args.method}_{mode}"
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.cache, exist_ok=True)
t_start = time.time()
rng = np.random.default_rng(args.seed)

# ------------------------------------------------------------------ 1. load
samples = data.load(args.data, os.path.expanduser(args.path))
summ = data.summary(samples)
print("dataset:", json.dumps(summ))


# ------------------------------------------------------------------ 2+3. vein maps
def make_map(s):
    strip = preprocess.roi(data.load_image(s))
    return extract.extract(preprocess.enhance(strip), args.method)


cache_file = os.path.join(args.cache, f"{args.data}_{args.method}.npz")
if os.path.exists(cache_file):
    z = np.load(cache_file)
    maps = np.unpackbits(z["bits"], axis=1)[:, : z["h"] * z["w"]].reshape(-1, z["h"], z["w"])
    print(f"vein maps loaded from cache ({len(maps)})")
else:
    t = time.time()
    with Pool(args.jobs) as p:
        maps = np.array(p.map(make_map, samples, chunksize=16), np.uint8)
    h, w = maps.shape[1:]
    np.savez_compressed(cache_file, bits=np.packbits(maps.reshape(len(maps), -1), axis=1), h=h, w=w)
    print(f"vein maps computed in {time.time() - t:.0f}s")
assert len(maps) == len(samples)

# example panels: raw | enhanced strip | vein map, for 2 fingers x 2 sessions
ex = [i for i, s in enumerate(samples) if s.index == 1 and s.person in (1, 2) and s.finger == 1]
fig, axes = plt.subplots(len(ex), 3, figsize=(11, 1.9 * len(ex)))
for row, i in zip(axes, ex):
    s = samples[i]
    raw = data.load_image(s)
    row[0].imshow(raw, cmap="gray"); row[0].set_title(f"raw p{s.person} f{s.finger} s{s.session}", fontsize=8)
    row[1].imshow(preprocess.enhance(preprocess.roi(raw)), cmap="gray"); row[1].set_title("ROI + CLAHE", fontsize=8)
    row[2].imshow(maps[i], cmap="gray"); row[2].set_title(f"vein map ({args.method})", fontsize=8)
    for a in row:
        a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(args.out, f"vein_examples_{args.method}.png"), dpi=130)
plt.close()

# ------------------------------------------------------------------ split + sets
people = np.array(sorted({s.person for s in samples}))
rng.shuffle(people)
half = len(people) // 2
split_of = {p: ("tune" if k < half else "test") for k, p in enumerate(people)}


def is_enroll(s):
    return s.session == 1 and (mode == "cross" or s.index <= 3)


def is_probe(s):
    return s.session == 2 if mode == "cross" else (s.session == 1 and s.index >= 4)


ids = sorted({data.identity(s) for s in samples})
E = {i: [] for i in ids}   # identity -> sample indices
P = {i: [] for i in ids}
for k, s in enumerate(samples):
    if is_enroll(s):
        E[data.identity(s)].append(k)
    elif is_probe(s):
        P[data.identity(s)].append(k)
for i in ids:
    E[i].sort(key=lambda k: samples[k].index)
    P[i].sort(key=lambda k: samples[k].index)
split_ids = {sp: [i for i in ids if split_of[i[0]] == sp] for sp in ("tune", "test")}


# ------------------------------------------------------------------ 4. match scores
def _scores_for(a):
    """Genuine scores of identity a, and impostor scores a(enroll[0]) vs every b."""
    ids_here = split_ids[split_of[a[0]]]
    gen = [match.score(maps[e], maps[p]) for e in E[a] for p in P[a]]
    imp, same_person = [], []
    for b in ids_here:
        if b == a:
            continue
        imp.append(match.score(maps[E[a][0]], maps[P[b][0]]))
        same_person.append(b[0] == a[0])
    return gen, imp, same_person


def scores(split):
    with Pool(args.jobs) as p:
        out = p.map(_scores_for, split_ids[split], chunksize=4)
    g = np.concatenate([o[0] for o in out])
    i = np.concatenate([o[1] for o in out])
    sp = np.concatenate([o[2] for o in out]).astype(bool)
    return g, i, sp


t = time.time()
S = {sp: scores(sp) for sp in ("tune", "test")}
print(f"match scores in {time.time() - t:.0f}s")


# ------------------------------------------------------------------ 5. metrics
def roc(gen, imp):
    """FAR and FRR at every candidate threshold (accept if score >= thr)."""
    thr = np.unique(np.concatenate([gen, imp]))
    gs, is_ = np.sort(gen), np.sort(imp)
    frr = np.searchsorted(gs, thr, side="left") / len(gs)          # genuine below thr
    far = 1 - np.searchsorted(is_, thr, side="left") / len(is_)    # impostor >= thr
    return thr, far, frr


def eer(gen, imp):
    thr, far, frr = roc(gen, imp)
    k = np.argmin(np.abs(far - frr))
    return float((far[k] + frr[k]) / 2), float(thr[k])


def far_at_frr(gen, imp, target):
    """Lowest FAR achievable while rejecting at most `target` of genuine pairs."""
    thr, far, frr = roc(gen, imp)
    ok = frr <= target
    return float(far[ok].min())


def rates_at(gen, imp, thr):
    return {"FAR": float(np.mean(imp >= thr)), "FRR": float(np.mean(gen < thr))}


def block(gen, imp, sp):
    e, _ = eer(gen, imp)
    return {
        "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
        "impostor_pairs_same_person_other_finger": int(sp.sum()),
        "EER": e,
        "FAR_at_FRR_1pct": far_at_frr(gen, imp, 0.01),
        "FAR_at_FRR_5pct": far_at_frr(gen, imp, 0.05),
        "EER_different_people_only": eer(gen, imp[~sp])[0],
        "EER_same_person_other_finger_only": eer(gen, imp[sp])[0] if sp.any() else None,
        "mean_score_genuine": float(gen.mean()),
        "mean_score_impostor_diff_person": float(imp[~sp].mean()),
        "mean_score_impostor_same_person": float(imp[sp].mean()) if sp.any() else None,
    }


g_tu, i_tu, _ = S["tune"]
g_te, i_te, sp_te = S["test"]
_, thr_eer_tune = eer(g_tu, i_tu)
thr_zero_far_tune = float(i_tu.max()) + 1e-9      # strictest: no tune impostor accepted
verification = {
    "test": block(*S["test"]),
    "tune": block(*S["tune"]),
    "threshold_from_tune_at_EER": thr_eer_tune,
    "test_rates_at_tune_EER_threshold": rates_at(g_te, i_te, thr_eer_tune),
    "threshold_from_tune_zero_FAR": thr_zero_far_tune,
    "test_rates_at_tune_zero_FAR_threshold": rates_at(g_te, i_te, thr_zero_far_tune),
}
print("verification (test):", json.dumps(verification["test"], indent=1))

label = "session 1 vs session 2" if mode == "cross" else "same session (optimistic)"
plt.figure(figsize=(7, 4))
bins = np.linspace(0, max(1e-3, max(g_te.max(), i_te.max())), 60)
plt.hist(i_te[~sp_te], bins, alpha=0.55, density=True, label="different person")
plt.hist(i_te[sp_te], bins, alpha=0.55, density=True, label="same person, other finger")
plt.hist(g_te, bins, alpha=0.55, density=True, label=f"same finger ({label})")
plt.axvline(thr_eer_tune, ls="--", c="k", label="threshold (EER, tune set)")
plt.xlabel("match score (vein overlap)"); plt.ylabel("density")
plt.title(f"Finger-vein scores, {args.method}, test half (n={len(split_ids['test'])} fingers)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(args.out, f"scores_vein_{tag}.png"), dpi=150); plt.close()

from scipy.stats import norm
thr, far, frr = roc(g_te, i_te)
plt.figure(figsize=(5, 5))
eps = 1e-5
plt.plot(norm.ppf(np.clip(far, eps, 1 - eps)), norm.ppf(np.clip(frr, eps, 1 - eps)))
ticks = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
plt.xticks(norm.ppf(ticks), [f"{t:g}" for t in ticks]); plt.yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
plt.xlim(norm.ppf(eps), norm.ppf(0.6)); plt.ylim(norm.ppf(eps), norm.ppf(0.6))
plt.plot([-5, 5], [-5, 5], ls=":", c="gray")
plt.xlabel("false accept rate"); plt.ylabel("false reject rate")
plt.title(f"DET, {args.method}, {label}\nEER = {verification['test']['EER']:.2%} (test half)", fontsize=9)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(args.out, f"det_vein_{tag}.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ 6. bit codes, entropy
t = time.time()


def _code(k):
    return code.to_code(maps[k], "com")


with Pool(args.jobs) as p:
    C = np.array(p.map(_code, range(len(samples)), chunksize=64), np.uint8)   # 'com' codes
NB = C.shape[1]

test_ids = split_ids["test"]
imp_hd, gen_hd = [], []
for a in test_ids:
    for e in E[a]:
        for p_ in P[a]:
            gen_hd.append(np.mean(C[e] != C[p_]))
    for b in test_ids:
        if b != a:
            imp_hd.append(np.mean(C[E[a][0]] != C[P[b][0]]))
imp_hd, gen_hd = np.array(imp_hd), np.array(gen_hd)
p_hat, sd = float(imp_hd.mean()), float(imp_hd.std())
dof = p_hat * (1 - p_hat) / sd ** 2
log2_n_safe = 0.5 * (np.log2(0.02) + dof)          # n ~ sqrt(0.02 * 2^N)
entropy = {
    "code_bits": int(NB), "grid": list(code.GRID), "alignment": "com (centre of vein mass)",
    "impostor_pairs": int(len(imp_hd)),
    "impostor_HD_mean_p": p_hat, "impostor_HD_std": sd,
    "genuine_HD_mean": float(gen_hd.mean()),
    "degrees_of_freedom_N": float(dof),
    "safe_community_size_1pct_collision": float(2 ** log2_n_safe),
    "safe_community_log2": float(log2_n_safe),
    "eeg_comparison": "not available: EEG DoF has not been computed in this repo",
}
print("entropy:", json.dumps(entropy, indent=1))

plt.figure(figsize=(7, 4))
bins = np.linspace(0, 1, 101)
plt.hist(imp_hd, bins, density=True, alpha=0.6, label="different fingers")
plt.hist(gen_hd, bins, density=True, alpha=0.6, label=f"same finger ({label})")
Nr = max(int(round(dof)), 1)
xs = np.arange(Nr + 1)
plt.plot(xs / Nr, binom.pmf(xs, Nr, p_hat) * Nr, "k-", lw=1.2,
         label=f"binomial fit: N={dof:.0f}, p={p_hat:.3f}")
plt.xlabel("fractional Hamming distance between bit codes"); plt.ylabel("density")
plt.title(f"Bit-code entropy ({NB} bits, {args.method}, test half)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(args.out, f"hd_vein_{tag}.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ 6b. fuzzy commitment
# For each identity: enrollment bits = majority vote over its enrollment codes,
# reliability = how consistently each bit agreed. A probe is a single image.
# The fuzzy commitment unlocks iff the error-correcting code can fix the bit
# differences. For a repetition code that means: every group of r bits has
# <= (r-1)//2 errors. For BCH(n, t): every block has <= t errors. We compute
# this exactly from the error pattern for every pair, then confirm it against
# the real fuzzy.enroll/recover on a sample of pairs.
PERM = np.random.default_rng(12345).permutation(NB)   # fixed public bit shuffle:
# neighbouring cells are correlated; shuffling spreads a misaligned patch over
# many code groups instead of wiping out one group.


def _aligned(ref, k):
    """Code of image k after shifting it onto map ref (oracle alignment)."""
    return code.to_code(code.shift(maps[k], *match.best_shift(ref, maps[k])))


def _codes_for(job):
    """Enrollment codes, genuine probe codes and impostor probe codes of identity a."""
    a, align = job
    sid = split_ids[split_of[a[0]]]
    if align == "oracle":
        ref = maps[E[a][0]]
        enc = [_aligned(ref, e) for e in E[a]]
        gen = [_aligned(ref, k) for k in P[a]]
        imp = [_aligned(ref, P[b][0]) for b in sid if b != a]
    else:
        enc = [C[e] for e in E[a]]
        gen = [C[k] for k in P[a]]
        imp = [C[P[b][0]] for b in sid if b != a]
    return np.array(enc), np.array(gen), np.array(imp)


CODES_CACHE = {}


def errors_for(split, align, K):
    """Bit-error vectors (masked to K reliable bits, then shuffled)."""
    if (split, align) not in CODES_CACHE:
        with Pool(args.jobs) as p:
            CODES_CACHE[(split, align)] = p.map(_codes_for, [(a, align) for a in split_ids[split]])
    gen_err, imp_err = [], []
    for enc, gen, imp in CODES_CACHE[(split, align)]:
        w, rel = code.enroll_bits(enc)
        m = code.reliable_mask(rel, K)
        gen_err.append((w[m] ^ gen[:, m])[:, PERM_K[K]])
        imp_err.append((w[m] ^ imp[:, m])[:, PERM_K[K]])
    return np.concatenate(gen_err).astype(np.uint8), np.concatenate(imp_err).astype(np.uint8)


def unlocks(err, spec):
    """Exact unlock decision for each error vector under an ECC spec."""
    if spec["code"] == "rep":
        r = spec["r"]
        L = (err.shape[1] // r) * r
        groups = err[:, :L].reshape(len(err), -1, r).sum(axis=2)
        return np.all(groups <= (r - 1) // 2, axis=1)
    # BCH: a block decodes iff it has <= t errors. (fuzzy.py now treats a
    # failed block as a failure; the old galois pass-through quirk is fixed.
    # More than t errors can also mis-decode to a wrong codeword: wrong key,
    # rejected by the hash, so it is a failure either way.)
    c = fuzzy.BCHCode(spec["n"], spec["t"])
    ok = np.ones(len(err), bool)
    i = 0
    for s in c._blocks(err.shape[1]):
        ok &= err[:, i:i + s].sum(axis=1) <= c.t
        i += s
    return ok


K_LIST = [NB, NB // 2, NB // 4]
PERM_K = {K: np.random.default_rng(K).permutation(K) for K in K_LIST}
CODES = [{"code": "rep", "r": r} for r in (3, 5, 7, 9, 15, 25)] + \
        [{"code": "bch", "n": 255, "t": t} for t in (15, 23, 31, 39, 47, 55, 63)]
ALIGNS = ["com", "oracle"]

fc_rows = []
for align in ALIGNS:
    for K in K_LIST:
        err = {sp: errors_for(sp, align, K) for sp in ("tune", "test")}
        for spec in CODES:
            cobj = fuzzy.code_from_spec(spec)
            kb = cobj.key_bits(K)
            if kb <= 0:
                continue
            row = {"align": align, "reliable_bits": K, **spec, "key_bits": int(kb)}
            for sp in ("tune", "test"):
                ge, ie = err[sp]
                row[f"GAR_{sp}"] = float(unlocks(ge, spec).mean())
                row[f"FAR_{sp}"] = float(unlocks(ie, spec).mean())
                row[f"bit_error_genuine_{sp}"] = float(ge.mean())
            fc_rows.append(row)
print(f"fuzzy commitment sweep in {time.time() - t:.0f}s")

# choose on TUNE: highest GAR with zero tune FAR and >= 64 key bits, per alignment
chosen = {}
for align in ALIGNS:
    cands = [r for r in fc_rows if r["align"] == align and r["FAR_tune"] == 0 and r["key_bits"] >= 64]
    if cands:
        best = max(cands, key=lambda r: (r["GAR_tune"], r["key_bits"]))
        chosen[align] = best

# Confirm the exact rule against the real fuzzy.enroll / fuzzy.recover, for
# each chosen setting, on random genuine + impostor pairs of the test half.
# recover() only depends on the bit errors (helper XOR probe = codeword XOR
# errors), so enrolling an all-zero template and presenting the error vector
# runs exactly the same arithmetic as the real pair.
def _real_unlock(job):
    err, spec = job
    _, rec = fuzzy.enroll(np.zeros(len(err), np.uint8), fuzzy.code_from_spec(spec))
    return fuzzy.recover(err, rec)[1]


check = {}
for align, row in chosen.items():
    spec = {k: row[k] for k in ("code", "r", "n", "t") if k in row}
    ge, ie = errors_for("test", align, row["reliable_bits"])
    pick_g = rng.choice(len(ge), min(150, len(ge)), replace=False)
    pick_i = rng.choice(len(ie), min(150, len(ie)), replace=False)
    errs = np.concatenate([ge[pick_g], ie[pick_i]])
    # single process on purpose: the galois decoder hangs in forked workers
    real = np.array([_real_unlock((e, spec)) for e in errs])
    rule = unlocks(errs, spec)
    check[align] = {"pairs": int(len(errs)), "agree": int((real == rule).sum()),
                    "real_unlocks_genuine": int(real[:len(pick_g)].sum()),
                    "real_unlocks_impostor": int(real[len(pick_g):].sum())}
print("fuzzy check (real recover vs exact rule):", check)

fuzzy_res = {"enroll": "majority vote of enrollment images, k most reliable bits",
             "probe": "single image", "settings": fc_rows,
             "chosen_on_tune": chosen, "real_fuzzy_check": check}

# ------------------------------------------------------------------ 6c. dedup simulation
# Enroll test-half fingers one at a time (enrollment images). Each new finger is
# compared with every finger enrolled before it (max score over their enrollment
# images, using its first enrollment image). Flag "duplicate" if >= threshold.
# Then present a probe image of every enrolled finger: is it caught?
t = time.time()
order = [split_ids["test"][j] for j in rng.permutation(len(split_ids["test"]))]


def _vs_all(k):
    return [max(match.score(maps[e], maps[k]) for e in E[b]) for b in order]


with Pool(args.jobs) as p:
    new_vs = np.array(p.map(_vs_all, [E[a][0] for a in order], chunksize=4))   # row: newcomer
    probe_vs = np.array(p.map(_vs_all, [P[a][0] for a in order], chunksize=4))  # row: returning
dedup = {"fingers_enrolled": len(order),
         "comparisons_at_enrollment": int(len(order) * (len(order) - 1) / 2)}
for name, thr_ in (("tune_EER_threshold", thr_eer_tune), ("tune_zero_FAR_threshold", thr_zero_far_tune)):
    false_dup = sum(int(new_vs[j, :j].max() >= thr_) for j in range(1, len(order)))
    caught = probe_vs.max(axis=1) >= thr_
    right = np.argmax(probe_vs, axis=1) == np.arange(len(order))
    dedup[name] = {"threshold": thr_, "false_duplicates": false_dup,
                   "false_duplicate_rate": false_dup / (len(order) - 1),
                   "re_enrollment_catch_rate": float(caught.mean()),
                   "catch_and_correct_identity_rate": float((caught & right).mean())}
print("dedup:", json.dumps(dedup, indent=1), f"({time.time() - t:.0f}s)")

# ------------------------------------------------------------------ write
res = {
    "dataset": args.data, "path_layout": "raw_data only (extractedvein ignored)",
    "summary": summ, "method": args.method, "mode": mode, "seed": args.seed,
    "identity": "(person, finger); 4 fingers of a person = 4 identities",
    "split": {"people_tune": int(half), "people_test": int(len(people) - half),
              "finger_identities_tune": len(split_ids["tune"]),
              "finger_identities_test": len(split_ids["test"])},
    "verification": verification, "entropy": entropy,
    "fuzzy_commitment": fuzzy_res, "dedup": dedup,
    "runtime_s": round(time.time() - t_start, 1), "cpu_jobs": args.jobs,
}
out = os.path.join(args.out, f"results_vein_{tag}.json")
json.dump(res, open(out, "w"), indent=2)
print("wrote", out, f"total {res['runtime_s']}s")
