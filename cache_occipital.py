"""Cache parieto-occipital resting EEG (PO7, PO3, POz, PO4, PO8, O1, Oz, O2) for the SSVEP protocol test.

PhysioNet EEG Motor Movement/Imagery, 109 subjects, 160 Hz:
  'enrol' = run 1 (1 min eyes-open rest)
  'later' = rest (T0) periods of runs 3/7/11, joined (minutes later, same day)
Band-pass 2-48 Hz (keeps the 3rd harmonic of 15 Hz = 45 Hz).
Raw EEG is personal data: the cache lives in ~/.cache/neurolock_pop (private).
"""
import os
import numpy as np
import mne
from mne.datasets import eegbci

mne.set_log_level("ERROR")
OUT = os.path.expanduser("~/.cache/neurolock_pop/occipital_physionet.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CH = ["Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2.."]   # parieto-occipital: where SSVEPs are strongest
SF = 160.0
enrol, later = [], []
for s in range(1, 110):
    raws = [mne.io.read_raw_edf(f, preload=True) for f in eegbci.load_data(s, [1, 3, 7, 11], update_path=False)]
    segs = []
    for k, r in enumerate(raws):
        r.pick(CH).filter(2.0, 48.0)
        d = r.get_data()
        if k == 0:
            e = d
        else:
            for ann in r.annotations:
                if ann["description"] == "T0":
                    a = int(ann["onset"] * SF)
                    segs.append(d[:, a:a + int(ann["duration"] * SF)])
    enrol.append(e[:, : int(55 * SF)])           # equal length for every subject
    later.append(np.hstack(segs)[:, : int(55 * SF)])
    print(s, e.shape, np.hstack(segs).shape, flush=True)
np.savez(OUT, enrol=np.array(enrol, np.float32), later=np.array(later, np.float32),
         subjects=np.arange(1, 110), sf=SF, channels=CH)
print("saved", OUT)
