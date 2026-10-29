#!/bin/bash
# Rebuild vein maps + score matrices after the ROI fallback fix, then rerun Phase 1b.
set -e
cd "$(dirname "$0")"
PY=../venv/bin/python
for m in gabor maxcurv; do
  $PY evaluate_vein.py --path ~/data/FV-USM --method $m --cross-session --out results_vein
done
for m in gabor maxcurv; do $PY scoremat_vein.py $m; done
$PY improve_vein.py
echo DONE $(date)
