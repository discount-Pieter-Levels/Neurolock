"""Enhanced finger strip -> binary vein map (same size for every image).

Veins absorb near-infrared light, so in the image they are dark valleys running
through a brighter finger. Both methods below look for "valley-shaped" lines.

maxcurv  Miura et al. 2007, "Extraction of finger-vein patterns using maximum
         curvature points in image profiles".
         Look at 1-D brightness profiles in 4 directions. A vein crossing the
         profile makes it curve upward (a dip). Score each dip by how sharply
         it curves and how wide it is, put that score at the centre of the dip,
         then keep scores that are supported by their neighbours (veins are
         lines, noise dots are not).
gabor    Even-symmetric Gabor filters: small striped templates that respond
         strongly to a dark line of a given width and orientation. Take the
         strongest response over 8 orientations, then keep the strongest
         VEIN_FRACTION of pixels.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

BORDER = 10          # rows at the finger edge are dominated by the edge itself
VEIN_FRACTION = 0.25  # gabor: fraction of pixels called "vein" (fixed, not tuned)


def _clear_border(v):
    v = v.copy()
    v[:BORDER] = 0
    v[-BORDER:] = 0
    v[:, :BORDER] = 0
    v[:, -BORDER:] = 0
    return v


# ---------------------------------------------------------------- max curvature
# 4 profile directions as unit steps (dy, dx): horizontal, vertical, 2 diagonals
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def maxcurv_score(img, sigma=5.0):
    """Miura step 1+2: per-pixel vein evidence (float map, >= 0)."""
    # Gaussian-smoothed derivatives of the image
    fy = ndi.gaussian_filter(img, sigma, order=(1, 0))
    fx = ndi.gaussian_filter(img, sigma, order=(0, 1))
    fyy = ndi.gaussian_filter(img, sigma, order=(2, 0))
    fxx = ndi.gaussian_filter(img, sigma, order=(0, 2))
    fxy = ndi.gaussian_filter(img, sigma, order=(1, 1))

    V = np.zeros_like(img)
    for dy, dx in _DIRS:
        n = np.hypot(dy, dx)
        uy, ux = dy / n, dx / n
        d1 = uy * fy + ux * fx                                   # slope along the profile
        d2 = uy * uy * fyy + 2 * uy * ux * fxy + ux * ux * fxx  # bend along the profile
        k = d2 / (1 + d1 ** 2) ** 1.5                            # curvature
        pos = k > 0                                              # dip (valley) regions

        # group positive-curvature pixels into runs along this direction only
        struct = np.zeros((3, 3), bool)
        struct[1, 1] = struct[1 + dy, 1 + dx] = struct[1 - dy, 1 - dx] = True
        lab, n_runs = ndi.label(pos, structure=struct)
        if n_runs == 0:
            continue
        idx = np.arange(1, n_runs + 1)
        peak = ndi.maximum(k, lab, idx)               # sharpest bend in each dip
        width = ndi.sum(pos, lab, idx)                # how wide the dip is
        where = ndi.maximum_position(k, lab, idx)     # dip centre
        ys, xs = np.array(where).T
        np.add.at(V, (ys, xs), peak * width)          # score = curvature x width
    return V


def maxcurv(img):
    V = maxcurv_score(img)
    # Miura step 3 ("connect vein centres"): along each direction, a pixel's
    # support is the WEAKER of its two sides, where each side is the best of
    # the next 2 pixels. Pixels on a line get support from both sides (and
    # small gaps get filled); isolated noise dots get none. Take the best
    # direction.
    def sh(a, dy, dx):  # value of the neighbour at offset (dy, dx)
        return np.roll(a, (-dy, -dx), axis=(0, 1))
    G = np.zeros_like(V)
    for dy, dx in _DIRS:
        fwd = np.maximum(sh(V, dy, dx), sh(V, 2 * dy, 2 * dx))
        bwd = np.maximum(sh(V, -dy, -dx), sh(V, -2 * dy, -2 * dx))
        G = np.maximum(G, np.minimum(fwd, bwd))
    G = _clear_border(G)
    nz = G[G > 0]
    thr = np.median(nz) if len(nz) else 0.0         # Miura's binarisation rule
    return (G > thr).astype(np.uint8)


# ---------------------------------------------------------------- gabor
def _gabor_bank(n_orient=8, lam=10.0, sigma=3.5, ksize=21):
    bank = []
    for i in range(n_orient):
        theta = np.pi * i / n_orient
        k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma=0.6, psi=0)
        k -= k.mean()                  # zero mean: flat regions give zero response
        bank.append(k.astype(np.float32))
    return bank


_BANK = _gabor_bank()


def gabor(img):
    inv = (1.0 - img).astype(np.float32)               # make veins bright ridges
    resp = np.max([cv2.filter2D(inv, -1, k) for k in _BANK], axis=0)
    resp = _clear_border(np.maximum(resp, 0))
    inner = resp[BORDER:-BORDER, BORDER:-BORDER]
    thr = np.quantile(inner, 1 - VEIN_FRACTION)
    return (resp > thr).astype(np.uint8)


METHODS = {"maxcurv": maxcurv, "gabor": gabor}


def extract(img, method):
    return METHODS[method](img)
