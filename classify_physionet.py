"""LDA / logistic-regression baseline on the same band-power features as the lock.

Train on per-window features from the enrolment recording (run 1, eyes-open rest),
test on windows from the query recording (T0 rest periods of runs 3/7/11).
Reports window accuracy, recording accuracy (mean of window scores) and EER.
"""
import argparse, json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neurolock import data, features

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", type=int, default=109)
args = ap.parse_args()


def windows(x, sf, epoch_s=4.0):
    step = int(epoch_s * sf)
    return np.array([features.band_features(x[:, i:i + step], sf, epoch_s)
                     for i in range(0, x.shape[1] - step + 1, step)])


Xe, ye, Xq, yq = [], [], [], []
for s in range(1, args.subjects + 1):
    e, q, sf = data.physionet(s)
    we, wq = windows(e, sf), windows(q, sf)
    Xe.append(we); ye += [s] * len(we)
    Xq.append(wq); yq += [s] * len(wq)
    print(f"subject {s:3d}: {len(we)} enrol / {len(wq)} query windows", end="\r")
Xe, Xq, ye, yq = np.vstack(Xe), np.vstack(Xq), np.array(ye), np.array(yq)


def eer(scores, labels):
    """scores: (n_trials, n_classes) claimed-identity scores; genuine = own class."""
    gen = scores[np.arange(len(labels)), labels]
    mask = np.ones_like(scores, bool); mask[np.arange(len(labels)), labels] = False
    imp = scores[mask]
    ths = np.unique(np.concatenate([gen, imp]))
    frr = np.array([(gen < t).mean() for t in ths])
    far = np.array([(imp >= t).mean() for t in ths])
    k = np.argmin(np.abs(frr - far))
    return float((frr[k] + far[k]) / 2)


res = {}
subs = np.unique(ye)
for name, clf in [("LDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                  ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)))]:
    clf.fit(Xe, ye)
    win_acc = float((clf.predict(Xq) == yq).mean())
    lp = clf.predict_log_proba(Xq)
    rec = np.array([lp[yq == s].mean(axis=0) for s in subs])
    idx = np.arange(len(subs))
    rec_acc = float((rec.argmax(axis=1) == idx).mean())
    res[name] = {"window_acc": win_acc, "recording_acc": rec_acc, "EER_recording": eer(rec, idx)}
print()
res["n_subjects"] = int(len(subs)); res["chance"] = 1 / len(subs)
print(json.dumps(res, indent=2))
json.dump(res, open("results_physionet_classifier.json", "w"), indent=2)
