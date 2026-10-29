"""Measure the neural signature: does the same person unlock, and do others fail?

  python evaluate.py --data synthetic --subjects 40
  python evaluate.py --data physionet --subjects 109      # on your laptop
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neurolock import data, features, fuzzy

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["synthetic", "physionet", "kaggle"], default="synthetic")
ap.add_argument("--subjects", type=int, default=40)
ap.add_argument("--bits", type=int, default=105, help="reliable bits kept per person")
ap.add_argument("--r", type=int, default=7, help="repetition factor (odd)")
ap.add_argument("--path", help="folder with SubjectXX_1/_2 files (kaggle)")
ap.add_argument("--mode", choices=["task", "rest"], default="task")
args = ap.parse_args()
data.KAGGLE_DIR, data.KAGGLE_MODE = args.path, args.mode

loader = getattr(data, args.data)
subs = list(range(1, args.subjects + 1))
E, Q = [], []
for s in subs:
    e, q, sf = loader(s)
    E.append(features.band_features(e, sf))
    Q.append(features.band_features(q, sf))
    print(f"subject {s:3d} loaded", end="\r")
E, Q = np.array(E), np.array(Q)
args.bits = min(args.bits, (E.shape[1] // args.r) * args.r)  # 19-ch data has only 76 features

# Population stats from enrollment sessions only (in deployment: a public reference cohort)
med, mad = features.fit_population(E)
code = fuzzy.RepetitionCode(args.r)

genuine, impostor, g_ham, i_ham = [], [], [], []
for i in range(len(subs)):
    w, z = features.to_bits(E[i], med, mad)
    mask = features.reliable_mask(z, args.bits)
    _, rec = fuzzy.enroll(w[mask], code)
    for j in range(len(subs)):
        wq, _ = features.to_bits(Q[j], med, mad)
        ham = float(np.mean(w[mask] != wq[mask]))
        _, ok = fuzzy.recover(wq[mask], rec)
        (genuine if i == j else impostor).append(ok)
        (g_ham if i == j else i_ham).append(ham)

res = {
    "dataset": args.data, "subjects": len(subs), "bits": args.bits, "r": args.r,
    "key_bits": args.bits // args.r,
    "genuine_accept_rate": float(np.mean(genuine)),
    "false_accept_rate": float(np.mean(impostor)),
    "mean_bit_error_genuine": float(np.mean(g_ham)),
    "mean_bit_error_impostor": float(np.mean(i_ham)),
}
print(json.dumps(res, indent=2))
json.dump(res, open(f"results_{args.data}_{args.mode}.json", "w"), indent=2)

plt.figure(figsize=(7, 4))
bins = np.linspace(0, 0.8, 41)
plt.hist(i_ham, bins, alpha=0.6, density=True, label="different person")
plt.hist(g_ham, bins, alpha=0.6, density=True, label="same person, later session")
plt.axvline(((args.r - 1) // 2) / args.r, ls="--", c="k", label="ECC tolerance (approx.)")
plt.xlabel("fraction of signature bits that flipped")
plt.ylabel("density")
plt.title(f"Neural signature stability ({args.data}/{args.mode}, n={len(subs)})")
plt.legend()
plt.tight_layout()
plt.savefig(f"hamming_{args.data}_{args.mode}.png", dpi=150)
