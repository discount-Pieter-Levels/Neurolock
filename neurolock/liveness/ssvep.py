"""SSVEP challenge-response liveness.

Idea: when you look at a light flickering at f Hz, your visual cortex
(back of the head, channels O1/Oz/O2) oscillates at f, 2f, 3f. A server
sends a random sequence of flicker frequencies; a live person watching the
screen produces EEG that follows that sequence. A recording made earlier
for a different sequence does not.

  challenge  = K random targets from FREQS (60 Hz screen renders these
               cleanly: 60/f is a whole number of frames), SEG_S each,
               plus a random nonce and a timestamp (stops whole-message replay).
  detector   = CCA: for each segment, how well can a mix of the occipital
               channels be matched by sin/cos at f, 2f, 3f? The largest
               canonical correlation rho(f) is computed for every candidate
               f; the segment's answer is argmax_f rho(f), accepted only if
               rho is above a threshold (chosen on tune people).
  pass       = all K segments answered with the challenged frequency, in order.

!! There is no SSVEP dataset on disk. All SSVEP numbers in this project
!! come from simulate_response(): a weak synthetic oscillation added to REAL
!! resting EEG. They are a PROTOCOL TEST ON SIMULATED RESPONSES, NOT
!! PHYSIOLOGICAL EVIDENCE that a real person's SSVEP would pass.
"""
import secrets
import time
import numpy as np

FREQS = (6.0, 60 / 9, 7.5, 60 / 7, 10.0, 12.0, 15.0)   # 6, 6.67, 7.5, 8.57, 10, 12, 15 Hz
SEG_S = 1.5
LATENCY_S = 0.14       # visual pathway delay; analysis skips it
HARMONICS = 3


def make_challenge(K=4, seg_s=SEG_S, rng=None):
    pick = (rng.integers(0, len(FREQS), K) if rng is not None
            else [secrets.randbelow(len(FREQS)) for _ in range(K)])
    return {"freqs": [FREQS[i] for i in pick], "seg_s": seg_s,
            "nonce": secrets.token_hex(16), "ts": time.time()}


def references(f, n, sf, harmonics=HARMONICS):
    t = np.arange(n) / sf
    return np.vstack([fn(2 * np.pi * h * f * t) for h in range(1, harmonics + 1)
                      for fn in (np.sin, np.cos)])


def cca_rho(X, Y):
    """Largest canonical correlation between rows of X (channels) and Y (refs).
    QR of each centred block, then the top singular value of Qx^T Qy."""
    X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
    qx, _ = np.linalg.qr(X.T); qy, _ = np.linalg.qr(Y.T)
    return float(np.linalg.svd(qx.T @ qy, compute_uv=False)[0])


def classify_segment(x, sf, freqs=FREQS):
    """x: (occipital channels, samples). Returns (best freq, best rho, all rhos)."""
    rhos = np.array([cca_rho(x, references(f, x.shape[1], sf)) for f in freqs])
    k = int(np.argmax(rhos))
    return freqs[k], float(rhos[k]), rhos


def segments(eeg, sf, K, seg_s, latency_s=LATENCY_S):
    """Cut the response into K analysis windows, each skipping the latency."""
    n, d = int(seg_s * sf), int(latency_s * sf)
    return [eeg[:, k * n + d:(k + 1) * n] for k in range(K)]


def verify_response(eeg, sf, challenge, rho_thr):
    """True iff every segment's CCA answer equals the challenged frequency
    and is confident (rho >= rho_thr)."""
    K = len(challenge["freqs"])
    out = []
    for seg, f in zip(segments(eeg, sf, K, challenge["seg_s"]), challenge["freqs"]):
        best, rho, _ = classify_segment(seg, sf)
        out.append(bool(np.isclose(best, f) and rho >= rho_thr))
    return all(out), out


# ---------------------------------------------------------------- SIMULATION
def simulate_response(rest, sf, challenge, snr, rng, weights=None, latency_s=LATENCY_S):
    """SIMULATED SSVEP: add a weak f/2f/3f oscillation to real resting EEG.

    rest   : (occipital channels, >= K*seg_s*sf samples) real resting EEG
    snr    : amplitude of the fundamental relative to each channel's RMS
             (0.1-0.3 is a weak-to-moderate SSVEP; real SSVEPs at the scalp
             are ~1-5 uV against ~10-30 uV background)
    Each segment: random phase, harmonic amplitudes 1, 0.5, 0.25, onset
    after `latency_s`, slightly different strength per channel (Oz strongest)."""
    n = int(challenge["seg_s"] * sf)
    K = len(challenge["freqs"])
    x = rest[:, : K * n].copy()
    rms = x.std(axis=1, keepdims=True)
    w = np.ones((x.shape[0], 1)) if weights is None else np.asarray(weights)[:, None]
    d = int(latency_s * sf)
    t = np.arange(n - d) / sf
    for k, f in enumerate(challenge["freqs"]):
        ph = rng.uniform(0, 2 * np.pi, 3)
        s = sum(a * np.sin(2 * np.pi * h * f * t + p)
                for h, a, p in zip((1, 2, 3), (1, 0.5, 0.25), ph))
        x[:, k * n + d:(k + 1) * n] += snr * rms * w * s[None] * np.sqrt(2)
    return x


# ================================================================ consent (confirm target)
def make_consent_challenge(action: str, K=4, seg_s=SEG_S, rng=None):
    """Consent by attention. The screen shows the action text and two buttons,
    CONFIRM and DENY, each flickering. Every segment both buttons get NEW random
    frequencies (always different from each other), so the frequency sequence
    of CONFIRM is unique to this action + nonce. The user keeps looking at the
    button they mean; their EEG then follows that button's sequence.

    No brain *identity* is needed: identity comes from the veins in the same
    session; the EEG only shows a live person choosing CONFIRM for THIS action."""
    import hashlib
    r = rng if rng is not None else np.random.default_rng(secrets.randbits(64))
    confirm, deny = [], []
    for _ in range(K):
        a, b = r.choice(len(FREQS), 2, replace=False)
        confirm.append(FREQS[a]); deny.append(FREQS[b])
    return {"action": action, "action_hash": hashlib.sha256(action.encode()).hexdigest(),
            "confirm": confirm, "deny": deny, "seg_s": seg_s,
            "nonce": secrets.token_hex(16), "ts": time.time()}


def decide_consent(eeg, sf, ch, rho_thr):
    """'confirm' if every segment follows the CONFIRM button (confident CCA
    answer), 'deny' if every segment follows DENY, otherwise 'none' (looked
    away, no response, or a recording made for another challenge)."""
    K = len(ch["confirm"])
    ans = []
    for seg, fc, fd in zip(segments(eeg, sf, K, ch["seg_s"]), ch["confirm"], ch["deny"]):
        best, rho, _ = classify_segment(seg, sf)
        ans.append("c" if (np.isclose(best, fc) and rho >= rho_thr) else
                   "d" if (np.isclose(best, fd) and rho >= rho_thr) else "-")
    if all(a == "c" for a in ans):
        return "confirm", ans
    if all(a == "d" for a in ans):
        return "deny", ans
    return "none", ans


def attended(ch, intent):
    """The frequency sequence a person looking at `intent` ('confirm'/'deny') sees."""
    return {"freqs": ch[intent], "seg_s": ch["seg_s"]}
