"""Binary vein map -> fixed-length bit code.

Why: the fuzzy commitment (neurolock/fuzzy.py) and the entropy measurement
both need a fixed-length bit string, where bit i always means "the same spot
on the finger".

  1. Align   - 'none':   trust the ROI normalisation (edges + fingertip)
               'com':    shift the map so its vein centre of mass sits in the
                         middle. Uses the image alone, so it is allowed at
                         unlock time (no stored template needed).
               'oracle': shift the probe to its best match against the
                         enrolled vein map. Needs the raw template stored, so
                         it is NOT privacy-preserving; reported only as an
                         upper bound on what better alignment could buy.
  2. Smooth  - blur the thin vein lines so a 1-2 px misalignment does not
               flip every bit.
  3. Shrink  - average down to a GRID (32 x 96 = 3072 cells).
  4. Threshold at this image's median cell -> about half the bits are 1,
               which is what makes the bits informative.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

GRID = (32, 96)   # rows x cols -> 3072 bits
SMOOTH = 2.0      # Gaussian blur (pixels) before shrinking


def shift(m, dy, dx):
    """Shift a map by (dy, dx), filling with zeros (no wrap-around)."""
    return ndi.shift(m.astype(np.float32), (dy, dx), order=0, cval=0)


def center_of_mass_shift(m):
    h, w = m.shape
    if m.sum() == 0:
        return 0, 0
    cy, cx = ndi.center_of_mass(m)
    return int(round(h / 2 - cy)), int(round(w / 2 - cx))


def to_code(m, align="none"):
    if align == "com":
        m = shift(m, *center_of_mass_shift(m))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), SMOOTH)
    g = cv2.resize(m, (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA)
    return (g > np.median(g)).astype(np.uint8).ravel()


def enroll_bits(codes):
    """Combine several enrollment codes: majority vote per bit, plus how
    reliable each bit is (1 = all images agreed, 0 = a coin flip)."""
    frac = np.mean(codes, axis=0)
    return (frac > 0.5).astype(np.uint8), np.abs(frac - 0.5) * 2


def reliable_mask(reliability, k):
    """Indices of the k most reliable bits (ties broken by position)."""
    order = np.argsort(-reliability, kind="stable")
    return np.sort(order[:k])
