import time, mne
from mne.datasets import eegbci
mne.set_log_level("ERROR")
for s in range(1, 110):
    for tries in range(8):
        try:
            eegbci.load_data(s, [1, 3, 7, 11], update_path=True); break
        except Exception as e:
            print(f"S{s:03d} try {tries}: {type(e).__name__}", flush=True); time.sleep(10 * (tries + 1))
    else:
        raise SystemExit(f"gave up on S{s:03d}")
    print(f"S{s:03d} ok", flush=True)
