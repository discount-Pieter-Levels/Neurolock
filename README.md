# NeuroLock

Proof of personhood using finger veins and EEG.

A person proves they are one live, unique human without any biometric data being stored:

- **Identity:** the finger-vein pattern (near-infrared) unlocks a secret through a fuzzy commitment with a BCH error-correcting code. The vein image is not kept.
- **Uniqueness:** new enrolments are checked against everyone already enrolled.
- **Liveness:** pulse in the finger video, plus the EEG response to flickering targets (SSVEP).
- **Consent:** the person looks at CONFIRM or DENY; each flickers at a fresh random rate.
- **Ledger:** hash-chained records on Logos (LEZ) holding commitments only.

See `NEUROLOCK_OVERVIEW.md` for how it works and the results.

## Layout

    neurolock/            pipeline code (vein/, liveness/, pop/, fuzzy.py, shamir.py)
    evaluate_vein.py      vein matching evaluation
    evaluate_pop.py       full pipeline evaluation (writes results_pop/)
    run_pop_demo.py       narrated command-line demo
    logos/                Logos registry program (Rust) and ledger client
    logos/basecamp/       Logos Basecamp apps (neurolock-demo is the interactive demo)
    demo_web/             interactive demo as a web page
    *_README.md           detailed notes per part

## Setup

    python -m venv venv && . venv/bin/activate
    pip install numpy scipy scikit-learn opencv-python mne cryptography argon2-cffi galois matplotlib

## Data

Not included. Download separately:

- FV-USM finger-vein database (raw_data), placed at `~/data/FV-USM`
- PhysioNet EEG Motor Movement/Imagery (downloaded by MNE to `~/mne_data`)
- OpenNeuro ds004148 (for the cross-session EEG test)

FV-USM images are licensed for research use, so no vein images are in this repository.

## Run

    python evaluate_vein.py
    python evaluate_pop.py
    python run_pop_demo.py

Interactive demo (needs the data above):

    PYTHONPATH=. python demo_web/export_demo_data.py    # builds demo_web/data.js
    python3 -m http.server -d demo_web 8000             # open http://localhost:8000

Basecamp app:

    cd logos/basecamp/neurolock-demo
    python3 build_assets.py && ./install.sh             # then restart Logos Basecamp

## Status

- Vein matching: 0.4% equal error rate with two fingers, visits two weeks apart (FV-USM test half).
- SSVEP responses and finger-pulse videos are simulated; no recordings exist yet.
- Scanner hardware is not built.
- Logos registry program passes its tests; deployment is waiting for the LEZ testnet to resume.
