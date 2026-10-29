"""Template-free alignment for the vein fuzzy commitment (Phase 2b).

The helper data stores a COARSE vein-density map of the enrolled finger:
enrolment vein maps, centre-of-mass aligned, averaged, blurred (sigma 6 px),
shrunk 4x to 25 x 75 and quantised to 4 levels. At unlock the probe's own
coarse map is slid over it (+-40 px along the finger, +-16 px across) and the
best shift is applied before the bit code is computed.

Why it helps: centre-of-mass alignment is thrown off by a few extra or missing
veins; matching the coarse layout is far more stable (Phase 2b: commitment GAR
69 % -> 82 % on test, FAR still 0, key unchanged at 594 bits).
Privacy cost: the coarse map reveals roughly where the large veins run
(1,875 numbers with 2 bits each). The fine detail that forms the key is not in it.
"""
import cv2
import numpy as np
from . import code as vcode

CO = (25, 75)    # coarse grid (rows, cols): 4x smaller than the 100 x 300 map
SY, SX = 4, 10   # search range in coarse pixels (x4 = full-res pixels)


def coarse(m):
    g = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6)
    return cv2.resize(g, (CO[1], CO[0]), interpolation=cv2.INTER_AREA)


def make_helper(enrol_maps):
    c = np.mean([coarse(vcode.shift(m, *vcode.center_of_mass_shift(m))) for m in enrol_maps], 0)
    return np.digitize(c, np.quantile(c, [0.25, 0.5, 0.75])).astype(np.float32)


def best_shift(helper, probe_map):
    p = coarse(probe_map)
    pad = cv2.copyMakeBorder(helper, SY, SY, SX, SX, cv2.BORDER_CONSTANT, value=float(helper.mean()))
    r = cv2.matchTemplate(pad, p.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(np.argmax(r), r.shape)
    return 4 * (y - SY), 4 * (x - SX)


def aligned_code(helper, m):
    return vcode.to_code(vcode.shift(m, *best_shift(helper, m)))
