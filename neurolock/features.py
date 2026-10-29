"""EEG -> feature vector -> bit string.

First principles: a person's resting EEG has a fairly stable *spectral shape*
(e.g. how strong alpha is at the back of the head vs theta at the front).
Raw voltages never repeat, but relative band power per channel roughly does.
"""
import numpy as np
from scipy.signal import welch

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta_lo": (13, 20), "beta_hi": (20, 30)}


def band_features(data, sfreq, epoch_s=4.0):
    """data: (n_channels, n_samples) in volts. Returns log relative band power,
    averaged over epochs -> vector of length n_channels * n_bands."""
    n_ch, n_s = data.shape
    step = int(epoch_s * sfreq)
    epochs = [data[:, i:i + step] for i in range(0, n_s - step + 1, step)]
    feats = []
    for ep in epochs:
        f, p = welch(ep, fs=sfreq, nperseg=min(step, 256), axis=-1)
        total = p[:, (f >= 4) & (f <= 30)].sum(axis=1, keepdims=True)
        bp = [p[:, (f >= lo) & (f < hi)].sum(axis=1, keepdims=True) for lo, hi in BANDS.values()]
        feats.append(np.log(np.hstack(bp) / total).ravel())
    return np.mean(feats, axis=0)


def fit_population(feature_matrix):
    """Public parameters: per-feature median and spread across many people."""
    med = np.median(feature_matrix, axis=0)
    mad = np.median(np.abs(feature_matrix - med), axis=0) + 1e-9
    return med, mad


def to_bits(feat, med, mad):
    """Each bit = is this person above or below the population median here?"""
    return (feat > med).astype(np.uint8), np.abs(feat - med) / mad


def reliable_mask(z_enroll, n_bits):
    """Keep the n_bits features farthest from the median for THIS person:
    those bits are least likely to flip between sessions. The mask is
    stored as public helper data."""
    return np.sort(np.argsort(-z_enroll)[:n_bits])
