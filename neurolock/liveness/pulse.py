"""Pulse (PPG) liveness from a finger video.

Each heartbeat pushes a little more blood into the finger, which absorbs a
little more light, so the image gets slightly darker once per beat.

  1. Per frame: mean brightness of the central half of the frame.
  2. Detrend: remove slow drifts (auto-exposure, finger pressure).
  3. Band-pass 0.7-3.0 Hz (42-180 bpm): keep only heart-rate frequencies.
  4. Heart rate = strongest frequency in that band; beats = signal peaks.
  5. Liveness score = spectral SNR (dB): power near the heart-rate peak and
     its 2nd harmonic, divided by the rest of the band. A printed vein image
     or a still frame has no clear peak, so it is rejected.

check_pulse(frames, fps) -> (ok, bpm, snr_db, beat_times)
"""
import numpy as np
from scipy.signal import butter, detrend, filtfilt, find_peaks, welch

MIN_SNR_DB = 3.0      # liveness threshold (same as pulse_from_video.py)
MIN_SECONDS = 5.0


def brightness_trace(frames, channel="gray"):
    """frames: iterable of HxW (gray) or HxWx3 (BGR) arrays -> 1-D signal."""
    sig = []
    for f in frames:
        h, w = f.shape[:2]
        roi = f[h // 4: 3 * h // 4, w // 4: 3 * w // 4].astype(np.float32)
        if roi.ndim == 3:
            roi = roi[..., 2] if channel == "red" else roi.mean(axis=2)
        sig.append(roi.mean())
    return np.array(sig, float)


def analyse(sig, fps, min_snr_db=MIN_SNR_DB):
    """Brightness trace -> dict with heart rate, SNR, beats, live flag."""
    if len(sig) < fps * MIN_SECONDS:
        return {"live": False, "heart_rate_bpm": 0.0, "snr_db": -np.inf, "beat_times": [],
                "reason": f"need >= {MIN_SECONDS} s of video"}
    x = detrend(sig)
    if np.allclose(x, 0):                       # perfectly still image: no signal at all
        return {"live": False, "heart_rate_bpm": 0.0, "snr_db": -np.inf, "beat_times": [],
                "reason": "no brightness variation"}
    b, a = butter(3, [0.7, 3.0], btype="band", fs=fps)
    x = -filtfilt(b, a, x)                      # more blood = darker: flip so beats are peaks
    f, pxx = welch(x, fs=fps, nperseg=min(len(x), int(fps * 8)), nfft=8192)
    band = (f >= 0.7) & (f <= 3.0)
    f0 = f[band][np.argmax(pxx[band])]
    near = band & ((np.abs(f - f0) <= 0.25) | (np.abs(f - 2 * f0) <= 0.25))
    snr_db = 10 * np.log10(pxx[near].sum() / max(pxx[band & ~near].sum(), 1e-30))
    peaks, _ = find_peaks(x, distance=int(fps / 3.0))
    return {"live": bool(snr_db >= min_snr_db), "heart_rate_bpm": float(f0 * 60),
            "snr_db": float(snr_db), "beat_times": (peaks / fps).tolist(), "trace": x}


def check_pulse(frames, fps, min_snr_db=MIN_SNR_DB, channel="gray"):
    r = analyse(brightness_trace(frames, channel), fps, min_snr_db)
    return r["live"], r["heart_rate_bpm"], r["snr_db"], r["beat_times"]


# ---------------------------------------------------------------- test videos (SIMULATED)
def synthetic_finger_video(seconds=10, fps=30, bpm=72, depth=0.004, noise=0.6,
                           drift=2.0, seed=0, still_image=None, size=(60, 80)):
    """SIMULATED finger video, for testing only.

    still_image=None : a 'live' finger: brightness dips by `depth` (0.4 %,
                       a typical PPG modulation) once per heartbeat, plus
                       sensor noise and a slow exposure drift.
    still_image=img  : a printed/static image filmed by the same camera:
                       sensor noise and drift, but no heartbeat."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(90, 160, size) if still_image is None else still_image.astype(float)
    t = np.arange(int(seconds * fps)) / fps
    hb = np.zeros(len(t)) if still_image is not None else \
        depth * (np.sin(2 * np.pi * bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * bpm / 60 * t))
    slow = drift * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
    return [np.clip(base * (1 - hb[i]) + slow[i] + rng.normal(0, noise, size), 0, 255)
            for i in range(len(t))]
