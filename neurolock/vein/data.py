"""Finger-vein dataset loaders.

Every loader returns a list of Sample(person, finger, session, index, path).
Images are read later with load_image(sample), so listing a dataset is cheap.

A *finger identity* is (person, finger): the 4 fingers of one person count as
4 identities, because their vein patterns are unrelated.

To add another dataset (UTFVP, SDUMLA-HMT, HKPU, or the live IR camera in
Phase 2), write one function that walks its folder and returns Samples, and
register it in LOADERS. Nothing else in the pipeline needs to change.
"""
import os
import re
from collections import Counter, namedtuple

import cv2

Sample = namedtuple("Sample", "person finger session index path")


def identity(s):
    return (s.person, s.finger)


def fvusm(root):
    """FV-USM, as published (Dec 2013) and mirrored on Kaggle.

    Layout (verified 2026-10-26 on the Kaggle zip):
      <root>/[Published_database_FV-USM_Dec2013/]{1st,2nd}_session/raw_data/PPP_F/NN.jpg
        PPP = person 001..123, F = finger 1..4, NN = image 01..06
    Raw images: 640x480 8-bit grayscale JPEG, finger vertical, tip at bottom.
    The same zip also contains `extractedvein/` (the authors' own ROI/vein
    output). We deliberately ignore it and use raw_data only.
    """
    sub = os.path.join(root, "Published_database_FV-USM_Dec2013")
    if os.path.isdir(sub):
        root = sub
    out = []
    for sess_dir, sess in (("1st_session", 1), ("2nd_session", 2)):
        base = os.path.join(root, sess_dir, "raw_data")
        if not os.path.isdir(base):
            raise FileNotFoundError(base)
        for d in sorted(os.listdir(base)):
            m = re.fullmatch(r"(\d{3})_(\d)", d)
            if not m:
                continue
            for f in sorted(os.listdir(os.path.join(base, d))):
                n = re.fullmatch(r"(\d{2})\.jpg", f, re.I)
                if n:
                    out.append(Sample(int(m[1]), int(m[2]), sess, int(n[1]),
                                      os.path.join(base, d, f)))
    return out


LOADERS = {"fvusm": fvusm}


def load(name, path):
    return LOADERS[name](path)


def load_image(sample):
    img = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {sample.path}")
    return img


def summary(samples):
    """Counts as a dict (no image data)."""
    per = Counter((identity(s), s.session) for s in samples)
    img = load_image(samples[0])
    return {
        "images": len(samples),
        "people": len({s.person for s in samples}),
        "finger_identities": len({identity(s) for s in samples}),
        "fingers_per_person": sorted(Counter(s.person for s in samples if s.session == 1
                                             and s.index == 1).values())[0],
        "sessions": sorted({s.session for s in samples}),
        "images_per_finger_per_session": sorted(set(per.values())),
        "image_size_hw": list(img.shape),
    }
