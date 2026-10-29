"""Two sessions per subject: (enroll_signal, query_signal, sfreq).

physionet: EEG Motor Movement/Imagery (109 subjects, 64 ch, 160 Hz).
  enroll = run 1 (eyes-open rest, 1 min)
  query  = rest periods (T0) cut out of runs 3/7/11, recorded minutes later.
  NOTE: same-day recordings, so this is an optimistic test of stability.
synthetic: fake EEG with per-person spectral fingerprints, for plumbing tests only.
"""
import numpy as np

SFREQ = 160.0


def physionet(subject):
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")

    def load(runs):
        raws = [mne.io.read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs, update_path=True)]
        for r in raws:
            r.filter(1.0, 40.0)
        return raws

    enroll = load([1])[0].get_data()
    rest = []
    for raw in load([3, 7, 11]):
        d = raw.get_data()
        for ann in raw.annotations:
            if ann["description"] == "T0":
                a = int(ann["onset"] * SFREQ)
                rest.append(d[:, a:a + int(ann["duration"] * SFREQ)])
    return enroll, np.hstack(rest), SFREQ


KAGGLE_DIR = None    # set by evaluate.py / demo.py --path
KAGGLE_MODE = "task"  # "task": enroll=rest(_1), query=arithmetic(_2); "rest": two halves of _1


def _read_any(stem):
    """Read Subject##_k as .edf or .csv; returns (n_channels, n_samples), sfreq."""
    import glob, os
    hits = glob.glob(os.path.join(KAGGLE_DIR, "**", stem + ".*"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{stem}.edf/.csv not found under {KAGGLE_DIR}")
    f = sorted(hits)[0]
    if f.lower().endswith(".edf"):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(f, preload=True)
        raw.pick([c for c in raw.ch_names if "ECG" not in c.upper() and "EKG" not in c.upper()])
        raw.filter(1.0, 40.0)
        return raw.get_data(), raw.info["sfreq"]
    import pandas as pd
    from scipy.signal import butter, sosfiltfilt
    df = pd.read_csv(f).select_dtypes("number")
    df = df[[c for c in df.columns if not any(k in c.upper() for k in ("TIME", "ECG", "EKG", "INDEX", "UNNAMED"))]]
    sf = 500.0  # eegmat sampling rate
    x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=sf, output="sos"), df.to_numpy().T, axis=1)
    return x, sf


def kaggle(subject):
    """Kaggle 'Complete EEG dataset' (= PhysioNet eegmat: Subject00..Subject35,
    _1 = resting baseline, _2 = mental arithmetic, same sitting)."""
    s = subject - 1
    rest, sf = _read_any(f"Subject{s:02d}_1")
    if KAGGLE_MODE == "rest":
        h = rest.shape[1] // 2
        return rest[:, :h], rest[:, h:], sf
    task, _ = _read_any(f"Subject{s:02d}_2")
    return rest, task, sf


def synthetic(subject, n_ch=64, seconds=60, session_noise=0.35):
    rng_subj = np.random.default_rng(subject)
    fingerprint = rng_subj.normal(0, 1, (n_ch, 4))           # stable per person
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    centers = [6, 10, 16, 25]

    def session(seed):
        rng = np.random.default_rng(seed)
        amp = np.exp(0.6 * (fingerprint + rng.normal(0, session_noise, fingerprint.shape)))
        x = rng.normal(0, 1, (n_ch, t.size)).cumsum(axis=1) * 0.05  # brown-ish noise
        x -= x.mean(axis=1, keepdims=True)
        for b, fc in enumerate(centers):
            phase = rng.uniform(0, 2 * np.pi, (n_ch, 1))
            fm = fc + rng.normal(0, 0.5, (n_ch, 1))
            x += amp[:, b:b + 1] * np.sin(2 * np.pi * fm * t + phase)
        return x * 1e-5

    return session(subject * 1000 + 1), session(subject * 1000 + 2), SFREQ
