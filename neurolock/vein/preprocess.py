"""Raw finger image -> fixed-size, straightened, contrast-enhanced finger strip.

Why: two photos of the same finger differ in where the finger sits, how wide it
looks and how bright the light is. Matching works far better if we first map
every finger into the same frame.

Steps (finger is vertical in FV-USM, tip at the bottom):
  1. Blur, then split finger from background with Otsu's threshold
     (the finger glows under IR, the background is dark).
  2. For every row, find the left and right finger edge.
     Starting from the Otsu edge, snap to the strongest horizontal gradient
     nearby (the column-wise gradient the task asks for).
  3. Find the fingertip: the lowest row where the finger is still at least
     half its typical width. Keep a fixed length LENGTH above it.
     Anchoring at the tip removes up/down shifts between photos.
  4. Straighten: resample each row so that [left edge, right edge] becomes
     exactly WIDTH pixels. This removes finger width, sideways shift and small
     tilts in one go.
  5. Rotate to a horizontal strip (H=WIDTH rows, W=OUT_LEN columns) and apply
     CLAHE (local histogram equalisation) so veins stand out everywhere,
     not only in the brightest part of the finger.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter

WIDTH = 100      # output height in pixels (across the finger)
OUT_LEN = 300    # output width in pixels (along the finger)
LENGTH = 360     # raw-image rows kept above the fingertip
TIP_MARGIN = 30  # rows dropped just above the tip (rounded, poorly lit)


MIN_WIDTH = 80   # px; real fingers are ~110-190 px wide in FV-USM


def _finger_mask(g, mode):
    """Binary finger mask and the column of the finger axis.

    'otsu':     Otsu on the whole image, axis = brightest column. Works on ~99 %
                of FV-USM captures.
    'fallback': used when 'otsu' finds an implausibly narrow finger. That happens
                when a neighbouring finger / the hand pokes in from the top: it
                is much brighter, so it drags Otsu's threshold up (only a sliver
                of the imaged finger survives) and steals the brightest column.
                Threshold relative to the background instead, then keep the
                bright region with most area in the lower half (the tip sits on
                the rest) and take the axis there.
    """
    h, w = g.shape
    if mode == "otsu":
        _, mask = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask, int(np.argmax(g.astype(float).mean(axis=0)))
    low = g[h // 2: h - 20]
    bg, fg = np.percentile(low, 20), np.percentile(low, 85)
    mask = (g > bg + 0.5 * (fg - bg)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(mask)
    if n < 2:
        return mask, int(np.argmax(g.astype(float).mean(axis=0)))
    ll = lab[h // 2: h - 20]
    k = 1 + int(np.argmax(np.bincount(ll.ravel(), minlength=n)[1:]))
    cols = np.nonzero(ll == k)[1]
    return (lab == k).astype(np.uint8), int(np.median(cols))


def finger_edges(gray):
    """Return per-row left/right edge columns and a fingertip row."""
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    res = _edges(g, *_finger_mask(g, "otsu"))
    if res[3] < MIN_WIDTH:
        alt = _edges(g, *_finger_mask(g, "fallback"))
        if alt[3] > res[3]:
            res = alt
    return res[:3]


def _edges(g, mask, centre):
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    h, w = g.shape
    left = np.zeros(h, int)
    right = np.zeros(h, int)
    for y in range(h):
        row = mask[y]
        # walk out from the centre until we leave the finger
        l = centre
        while l > 0 and row[l - 1]:
            l -= 1
        r = centre
        while r < w - 1 and row[r + 1]:
            r += 1
        # snap to the strongest edge within +-15 px (dark->bright on the left,
        # bright->dark on the right)
        lo, hi = max(l - 15, 0), min(l + 16, centre)
        if hi > lo:
            l = lo + int(np.argmax(gx[y, lo:hi]))
        lo, hi = max(r - 15, centre + 1), min(r + 16, w)
        if hi > lo:
            r = lo + int(np.argmin(gx[y, lo:hi]))
        left[y], right[y] = l, r

    # remove single-row glitches
    left = median_filter(left, 15)
    right = median_filter(right, 15)

    width = right - left
    typical = np.median(width[h // 4: 3 * h // 4])
    wide = np.nonzero(width > 0.5 * typical)[0]
    tip = int(wide.max()) if len(wide) else h - 1

    # The finger is close to a straight tube, but per-row edges are noisy where
    # the image is dim (top of the frame). Fit a straight line to each edge
    # using only rows whose width looks normal, ignoring the rounded tip, and
    # use the fitted lines. Two refit passes drop outlier rows.
    ys = np.arange(h)
    ok = (ys < tip - 40) & (np.abs(width - typical) < 0.25 * typical)
    if ok.sum() < 50:            # fall back to raw edges if detection is poor
        return left, right, tip, float(np.median(right - left))
    for edge in (left, right):
        sel = ok.copy()
        for _ in range(3):
            a, b = np.polyfit(ys[sel], edge[sel], 1)
            resid = np.abs(edge - (a * ys + b))
            sel = ok & (resid < max(3.0, 2.5 * np.median(resid[sel])))
        edge[:] = np.round(a * ys + b).astype(int)
    return left, right, tip, float(np.median(right - left))


def roi(gray):
    """Raw grayscale image -> straightened finger strip (WIDTH x OUT_LEN, uint8)."""
    left, right, tip = finger_edges(gray)
    bottom = tip - TIP_MARGIN
    top = max(bottom - LENGTH, 0)
    rows = np.linspace(top, bottom, OUT_LEN)          # sample positions along the finger
    out = np.zeros((OUT_LEN, WIDTH), np.float32)
    ys = np.arange(gray.shape[0])
    lf = np.interp(rows, ys, left)
    rf = np.interp(rows, ys, right)
    for i, y in enumerate(rows):
        xs = np.linspace(lf[i], rf[i], WIDTH)         # WIDTH samples between the edges
        out[i] = cv2.remap(gray.astype(np.float32), xs[None].astype(np.float32),
                           np.full((1, WIDTH), y, np.float32), cv2.INTER_LINEAR)[0]
    strip = out.T[::-1]                               # horizontal; rows = across finger
    strip = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return strip


def enhance(strip):
    """CLAHE contrast enhancement (float image in [0, 1])."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 12))
    return clahe.apply(strip).astype(np.float32) / 255.0
