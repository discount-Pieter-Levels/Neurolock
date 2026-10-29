"""Heart rate + simple liveness from a fingertip video (PPG).

  python pulse_from_video.py --video finger.mp4

Each heartbeat pushes a little more blood into the fingertip, which absorbs
a little more light, so the video gets slightly darker once per beat.

  1. Per frame: mean brightness of the finger region (the central half of
     the frame; red channel for a phone + flashlight, grayscale for IR).
  2. Detrend: remove slow drifts (auto-exposure, finger pressure).
  3. Band-pass 0.7-3.0 Hz (42-180 bpm): keep only heart-rate frequencies.
  4. Heart rate = strongest frequency in that band; beats = signal peaks.
  5. Liveness score = spectral SNR: power near the heart-rate peak (and its
     2nd harmonic) divided by the rest of the band, in dB. A photo, a
     silicone finger or a still frame has no clear peak -> rejected.

Outputs pulse_<video>.json and pulse_<video>.png.
"""
import argparse
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, detrend, filtfilt, find_peaks, welch

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--channel", choices=["red", "gray"], default="red")
ap.add_argument("--min-snr-db", type=float, default=3.0, help="liveness threshold")
ap.add_argument("--out", default=".")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
sig = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    roi = f[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if roi.ndim == 3:
        roi = roi[..., 2] if args.channel == "red" else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sig.append(roi.mean())
cap.release()
sig = np.array(sig, float)
if len(sig) < fps * 5:
    raise SystemExit(f"need at least 5 s of video, got {len(sig) / fps:.1f} s")

x = detrend(sig)
b, a = butter(3, [0.7, 3.0], btype="band", fs=fps)
x = filtfilt(b, a, x)
x = -x                                  # more blood = darker: flip so beats are peaks

# 8 s windows, zero-padded so the peak frequency is read to ~0.5 bpm
f, pxx = welch(x, fs=fps, nperseg=min(len(x), int(fps * 8)), nfft=8192)
band = (f >= 0.7) & (f <= 3.0)
f0 = f[band][np.argmax(pxx[band])]
# 'signal' = within 0.25 Hz of the peak or its 2nd harmonic (an 8 s window
# smears a pure tone over about +-0.25 Hz)
near = band & ((np.abs(f - f0) <= 0.25) | (np.abs(f - 2 * f0) <= 0.25))
snr_db = 10 * np.log10(pxx[near].sum() / max(pxx[band & ~near].sum(), 1e-12))

peaks, _ = find_peaks(x, distance=int(fps / 3.0))   # at most 3 beats per second
beats = peaks / fps
live = bool(snr_db >= args.min_snr_db)
res = {"video": args.video, "fps": fps, "seconds": round(len(sig) / fps, 2),
       "heart_rate_bpm": round(float(f0 * 60), 1),
       "beats_detected": int(len(beats)),
       "beat_times_s": [round(float(t), 3) for t in beats],
       "liveness_snr_db": round(float(snr_db), 2), "live": live}
print(json.dumps({k: v for k, v in res.items() if k != "beat_times_s"}, indent=2))

name = os.path.splitext(os.path.basename(args.video))[0]
json.dump(res, open(os.path.join(args.out, f"pulse_{name}.json"), "w"), indent=2)
fig, ax = plt.subplots(2, 1, figsize=(8, 5))
t = np.arange(len(x)) / fps
ax[0].plot(t, x, lw=0.8)
ax[0].plot(beats, x[peaks], "r.", ms=5)
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("PPG (filtered)")
ax[0].set_title(f"{res['heart_rate_bpm']} bpm, SNR {snr_db:.1f} dB -> {'LIVE' if live else 'REJECT'}")
ax[1].semilogy(f * 60, pxx)
ax[1].axvspan(42, 180, alpha=0.1)
ax[1].set_xlim(0, 240); ax[1].set_xlabel("bpm"); ax[1].set_ylabel("power")
plt.tight_layout()
plt.savefig(os.path.join(args.out, f"pulse_{name}.png"), dpi=130)
