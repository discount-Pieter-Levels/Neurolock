"""Extra EEG feature sets for the projection study (ds004148, 60 people x 3 sessions).

Per 4 s window, three feature families (all on the 19 standard 10-20 channels so
they also exist on PhysioNet / a consumer headset):
  relpow : log relative band power, 4 bands (the current NeuroLock feature)    76
  psd    : log power spectrum in 2 Hz bins 2-40 Hz, each channel's spectrum
           normalised by its total power (removes gain / impedance changes)   361
  cov    : log-Euclidean channel covariance (Riemannian tangent space), 1-40 Hz,
           trace-normalised (again removes overall gain)                      190
Saved per window so recordings can be averaged in any way later.
-> feats_ds004148.npz : {name: (people, 3 sessions, windows, dims)}, subs
"""
import os
import numpy as np
import mne
from scipy.linalg import logm
from scipy.signal import welch
from multiprocessing import Pool

ROOT = os.path.expanduser("~/neurolock_run/ds004148")
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
BANDS = [(4, 8), (8, 13), (13, 20), (20, 30)]
EPOCH = 4.0
mne.set_log_level("ERROR")


def window_feats(x, sf):
    f, p = welch(x, fs=sf, nperseg=int(EPOCH * sf), axis=-1)
    tot = p[:, (f >= 4) & (f <= 30)].sum(1, keepdims=True)
    rel = np.log(np.hstack([p[:, (f >= lo) & (f < hi)].sum(1, keepdims=True) for lo, hi in BANDS]) / tot)
    bins = np.hstack([p[:, (f >= lo) & (f < lo + 2)].sum(1, keepdims=True) for lo in range(2, 40, 2)])
    psd = np.log(bins / bins.sum(1, keepdims=True))
    c = np.cov(x)
    c = c / np.trace(c) + 1e-6 * np.eye(len(c))
    L = logm(c).real
    iu = np.triu_indices(len(c))
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))          # tangent-space weighting
    return rel.ravel(), psd.ravel(), L[iu] * w


def subject(s):
    out = []
    for ses in (1, 2, 3):
        f = f"{ROOT}/sub-{s:02d}/ses-session{ses}/eeg/sub-{s:02d}_ses-session{ses}_task-eyesopen_eeg.vhdr"
        if not os.path.exists(f):
            return s, None
        raw = mne.io.read_raw_brainvision(f, preload=True)
        names = {c.lower(): c for c in raw.ch_names}
        raw.pick([names[c.lower()] for c in CH19])
        raw.filter(1, 40)
        x, sf = raw.get_data(), raw.info["sfreq"]
        step = int(EPOCH * sf)
        out.append([window_feats(x[:, i:i + step], sf) for i in range(0, x.shape[1] - step + 1, step)])
    return s, out


if __name__ == "__main__":
    with Pool(8) as p:
        res = [r for r in p.map(subject, range(1, 61)) if r[1] is not None]
    subs = np.array([s for s, _ in res])
    n = min(len(w) for _, W in res for w in W)
    out = {}
    for k, name in enumerate(("relpow", "psd", "cov")):
        out[name] = np.array([[[win[k] for win in W[ses][:n]] for ses in range(3)] for _, W in res], np.float32)
        print(name, out[name].shape)
    np.savez("feats_ds004148.npz", subs=subs, **out)
