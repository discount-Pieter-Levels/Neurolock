"""Compare two binary vein maps (Miura-style matching).

The two images are never perfectly aligned, so we slide one over the other
by up to +-SHIFT_Y rows and +-SHIFT_X columns and keep the best overlap.

Score at one shift = 2 * |A and B| / (|A| + |B|)   (Dice overlap, 0..1)
where B is the probe with a margin cut off on every side (so it can move
around inside A), and A is the window of the template under B.
Score = max over all shifts. 1 = identical, ~0.1-0.2 = unrelated.
"""
import cv2
import numpy as np

SHIFT_Y = 15  # chosen on the tune half (EER), see VEIN_README
SHIFT_X = 40  # chosen on the tune half (EER), see VEIN_README


def score(a, b):
    a = a.astype(np.float32)
    core = b[SHIFT_Y:-SHIFT_Y, SHIFT_X:-SHIFT_X].astype(np.float32)
    # overlap |A_window and B_core| for every shift, in one call
    both = cv2.matchTemplate(a, core, cv2.TM_CCORR)
    # |A_window| for every shift (correlate with a block of ones)
    a_cnt = cv2.matchTemplate(a, np.ones_like(core), cv2.TM_CCORR)
    denom = a_cnt + core.sum()
    s = np.where(denom > 0, 2 * both / np.maximum(denom, 1), 0)
    return float(s.max())


def best_shift(a, b):
    """(dy, dx) that best aligns probe b to template a."""
    a = a.astype(np.float32)
    core = b[SHIFT_Y:-SHIFT_Y, SHIFT_X:-SHIFT_X].astype(np.float32)
    both = cv2.matchTemplate(a, core, cv2.TM_CCORR)
    a_cnt = cv2.matchTemplate(a, np.ones_like(core), cv2.TM_CCORR)
    s = 2 * both / np.maximum(a_cnt + core.sum(), 1)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(y - SHIFT_Y), int(x - SHIFT_X)
