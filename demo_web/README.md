# NeuroLock interactive station demo

    cd ~/neurolock_run/neurolock/demo_web && python3 -m http.server 8000
    # open http://localhost:8000  (or just open index.html)

Offline, single page. Simplified 5-step story (scan → pulse → unlock → consent → attacks). Full detailed version: index_v2_full.html.
`data.js` comes from the real pipeline: `cd .. && PYTHONPATH=. ../venv/bin/python demo_web/export_demo_data.py` (~40 s).
REAL: FV-USM images and every intermediate stage, matcher scores, dedup scores, the 30-person score distribution,
fuzzy-commitment bit errors per BCH block (Asha finger 3 fails, finger 4 opens: real), CCA rho values, check_pulse results.
Inputs are loaded by drag-and-drop (or click) from sample trays: 3 recorded FV-USM scans, Asha's PhysioNet resting EEG, 2 synthetic finger videos.
SIMULATED: SSVEP response (simulate_response on real PhysioNet resting EEG) and the finger-pulse videos.
No camera use. The earlier webcam version is kept as index_v1_backup.html for later camera integration.
Mouse hover = gaze. Ledger hashes = live SHA-256 in the browser (not the Logos testnet).
