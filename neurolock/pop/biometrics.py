"""Biometric layers used by the PoP pipeline (built on Phase 1/1b code).

VEIN, matcher mode   (accurate, less private: the station keeps vein maps)
  finger score  = for each of the 3 probe images, max fused score over the 6
                  enrol images; then max over the 3 probe images (Phase 1b D).
  fused score   = z_gabor + z_maxcurv, each z-scored with tune impostor stats.
  quality retake= if the 3 probe images of an attempt disagree with each other
                  (mean fused pairwise score below a tune threshold), take a
                  second attempt (Phase 1b E).
  person score  = mean over the 2 enrolled fingers.

VEIN, commitment mode (private: the station keeps only helper data)
  vein map -> 3072-bit code (code.py), aligned to a coarse vein-density helper
  stored with the commitment (vein/align.py, Phase 2b) -> majority vote
  over enrol images -> K most reliable bits -> public shuffle -> BCH fuzzy
  commitment per finger. Unlock: try each probe image in turn (multi-probe
  retry); a finger unlocks if any image decodes.

EEG brain key
  76/256 band-power features per 4 s window -> recording mean -> LDA projection
  learned on TUNE subjects only -> standardise with tune stats -> keep the
  `keep` dims farthest from 0 for this person (public mask) -> 1 bit (sign) or
  2 bits (Gray quartile) per dim -> BCH fuzzy commitment.

Nothing in this file writes templates to disk; callers must keep caches private.
"""
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from neurolock import fuzzy
from neurolock.vein import align as valign, code as vcode, data as vdata, match

CACHE = os.path.expanduser("~/.cache/neurolock_vein")
METHODS = ("gabor", "maxcurv")
ATTEMPTS = ((0, 1, 2), (3, 4, 5))


# ================================================================ vein data
class VeinData:
    """Vein maps for every FV-USM image, both extraction methods (from the Phase 1 cache)."""

    def __init__(self, path="~/data/FV-USM"):
        self.samples = vdata.load("fvusm", os.path.expanduser(path))
        self.maps = {}
        for m in METHODS:
            z = np.load(f"{CACHE}/fvusm_{m}.npz")
            self.maps[m] = np.unpackbits(z["bits"], axis=1)[:, : z["h"] * z["w"]].reshape(-1, z["h"], z["w"])
        self.ix = {(s.person, s.finger, s.session, s.index): k for k, s in enumerate(self.samples)}
        self.people = sorted({s.person for s in self.samples})

    def images(self, person, finger, session, idxs):
        """dict method -> list of vein maps."""
        ks = [self.ix[(person, finger, session, i + 1)] for i in idxs]
        return {m: [self.maps[m][k] for k in ks] for m in METHODS}


def split_people(people, seed=0):
    """Same seed-0 tune/test split by person as evaluate_vein.py."""
    rng = np.random.default_rng(seed)
    p = np.array(sorted(people)); rng.shuffle(p)
    h = len(p) // 2
    return sorted(p[:h].tolist()), sorted(p[h:].tolist())


# ================================================================ vein matcher mode
class VeinMatcher:
    def __init__(self, params):
        """params: norm {method: (mu, sd)}, q_thr (quality), fingers (slots)."""
        self.norm = params["norm"]; self.q_thr = params["q_thr"]; self.fingers = params["fingers"]

    def fused(self, enrol, probe):
        """Fused score of one enrol image set vs one probe image (max over enrol)."""
        return sum((max(match.score(e, probe[m]) for e in enrol[m]) - self.norm[m][0]) / self.norm[m][1]
                   for m in METHODS)

    def finger_score(self, enrol, probe_imgs):
        """enrol: {method: [6 maps]}; probe_imgs: {method: [3 maps]} -> max over probes."""
        n = len(probe_imgs[METHODS[0]])
        return max(self.fused(enrol, {m: probe_imgs[m][j] for m in METHODS}) for j in range(n))

    def consistency(self, imgs):
        """Mean fused pairwise score among the images of one attempt."""
        n = len(imgs[METHODS[0]]); s = []
        for a in range(n):
            for b in range(a + 1, n):
                s.append(sum((match.score(imgs[m][a], imgs[m][b]) - self.norm[m][0]) / self.norm[m][1]
                             for m in METHODS))
        return float(np.mean(s))

    def pick_attempt(self, attempts):
        """Quality retake: use attempt 1 unless its images disagree and attempt 2 is better."""
        q = [self.consistency(a) for a in attempts]
        if q[0] < self.q_thr and len(attempts) > 1 and q[1] >= q[0]:
            return attempts[1], True
        return attempts[0], False

    def person_score(self, templates, capture):
        """templates/capture: {finger: images}. capture[f] = list of attempts."""
        scores, retakes = [], 0
        for f in self.fingers:
            att, r = self.pick_attempt(capture[f])
            retakes += r
            scores.append(self.finger_score(templates[f], att))
        return float(np.mean(scores)), retakes


# ================================================================ vein commitment mode
PERM_SEED = 12345


class VeinCommitment:
    def __init__(self, params):
        """params: K (reliable bits), code spec, method."""
        self.K = params["K"]; self.spec = params["code"]; self.method = params["method"]
        self.perm = np.random.default_rng(PERM_SEED).permutation(self.K)

    def codes(self, maps, helper):
        return np.array([valign.aligned_code(helper, m) for m in maps])

    def enrol(self, enrol_maps):
        """-> (key bits, public record incl. reliable-bit mask and coarse alignment helper)."""
        helper = valign.make_helper(enrol_maps)
        w, rel = vcode.enroll_bits(self.codes(enrol_maps, helper))
        mask = vcode.reliable_mask(rel, self.K)
        k, rec = fuzzy.enroll(w[mask][self.perm], fuzzy.code_from_spec(self.spec))
        rec["mask"] = mask.tolist()
        rec["align_helper"] = helper.astype(np.uint8).tolist()
        return k, rec

    def unlock(self, rec, probe_maps):
        """Multi-probe retry: try each image; return (key or None, tries used)."""
        mask = np.array(rec["mask"])
        helper = np.array(rec["align_helper"], np.float32)
        for t, c in enumerate(self.codes(probe_maps, helper), 1):
            k, ok = fuzzy.recover(c[mask][self.perm], rec)
            if ok:
                return k, t
        return None, len(probe_maps)


# ================================================================ EEG brain key
def gray2(z):
    q = np.digitize(z, [-0.6745, 0, 0.6745])
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.uint8)[q].reshape(*z.shape[:-1], -1)


class EEGData:
    """PhysioNet features from the Phase 1 cache: run-1 rest (enrol) and
    rest periods of runs 3/7/11 (query). SAME-DAY recordings -> optimistic."""

    def __init__(self, path="windows_physionet.npz"):
        d = np.load(path)
        self.We, self.Wq, self.se, self.sq = d["We"], d["Wq"], d["se"], d["sq"]
        self.subjects = sorted(set(self.se.tolist()))

    def recording(self, s, which):
        W, S = (self.We, self.se) if which == "enrol" else (self.Wq, self.sq)
        return W[S == s].mean(0)


class BrainKey:
    def __init__(self, params):
        self.p = params

    @staticmethod
    def fit_projection(Wtrain, ytrain, R_train):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Wtrain, ytrain)
        P = lda.transform(R_train)
        return {"lda": lda, "mu": P.mean(0), "sd": P.std(0) + 1e-9}

    def project(self, feat):
        pr = self.p["proj"]
        return ((pr["lda"].transform(feat[None]) - pr["mu"]) / pr["sd"])[0, : self.p["dims"]]

    def _bits(self, z):
        return (z > 0).astype(np.uint8) if self.p["enc"] == "1bit" else gray2(z[None])[0]

    def enrol(self, feat):
        z = self.project(feat)
        mask = np.sort(np.argsort(-np.abs(z))[: self.p["keep"]])
        k, rec = fuzzy.enroll(self._bits(z[mask]), fuzzy.code_from_spec(self.p["code"]))
        rec["mask"] = mask.tolist()
        return k, rec

    def unlock(self, rec, feat):
        z = self.project(feat)[np.array(rec["mask"])]
        k, ok = fuzzy.recover(self._bits(z), rec)
        return k if ok else None

    def code_bits(self, feat):
        """Full (unmasked) bit string, used only for the entropy (DoF) estimate."""
        return self._bits(self.project(feat))
