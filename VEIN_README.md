# Neurolock finger-vein layer — Phase 1 (public data)

Goal: show that finger veins can tell fingers apart **across sessions**, and
measure how much entropy they carry, using the public FV-USM database. There's
no live hardware yet.

## Setup
    pip install numpy scipy opencv-python scikit-image matplotlib galois

(`galois` is only needed for the BCH codes in `neurolock/fuzzy.py`.)

## Dataset: FV-USM
- Source: Kaggle mirror of FV-USM (`fvusmnew.zip`, 220 MB). The original
  database is from Universiti Sains Malaysia (Asaari, Suandi & Rosdi, *Expert
  Systems with Applications* 41(7), 2014). Cite that paper when using it.
- **License: not verified.** FV-USM is distributed for non-commercial
  research use. Check the Kaggle page's license field before publishing results
  or demoing them outside a research context.
- Layout, verified by inspection on 2026-10-26:

  `Published_database_FV-USM_Dec2013/{1st,2nd}_session/{raw_data,extractedvein}/PPP_F/NN.jpg`

  | | |
  |---|---|
  | people | 123 |
  | fingers per person | 4 (finger identities: 492) |
  | sessions | 2 |
  | images per finger per session | 6 |
  | total raw images | 5,904 |
  | raw image | 640×480 8-bit grayscale JPEG, finger vertical, tip at bottom |

- The zip also contains `extractedvein/`, the authors' own processed images
  (20,665 files in session 1, which looks like several crops per image). **We use
  `raw_data` only.** Only `raw_data` was extracted, to `~/data/FV-USM`, kept
  apart from the code.

## How to run
    python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method maxcurv --cross-session
    python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method gabor   --cross-session
    python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method maxcurv   # same-session
    python evaluate_vein.py --data fvusm --path ~/data/FV-USM --method gabor     # same-session
    python pulse_from_video.py --video finger.mp4

Add `--out results_vein` to choose where outputs go. Runtime on this laptop (16
CPU threads): about 4.5 min per cross-session run, about 3 min per same-session
run, plus about 1.5 min the first time each method's vein maps are computed.

Vein maps are cached in `~/.cache/neurolock_vein/`. **They are biometric
templates: keep that folder private and delete it when done.** The JSON results
contain only scores and rates, never templates.

## Pipeline (files in `neurolock/vein/`)
| file | what it does |
|---|---|
| `data.py` | dataset → list of `(person, finger, session, index, path)`. Identity = (person, finger). |
| `preprocess.py` | Otsu finger mask → per-row edges snapped to the strongest column-wise gradient → straight-line fit of each edge → find the fingertip → straighten into a 100×300 strip anchored at the tip → CLAHE |
| `extract.py` | `maxcurv`: Miura maximum curvature (σ=5). `gabor`: 8-orientation even Gabor bank, top 25 % of responses = vein. |
| `match.py` | Miura-style matching: Dice overlap of vein pixels, maximised over shifts of ±40 px along and ±15 px across the finger |
| `code.py` | vein map → 3,072-bit code (32×96 grid): align by vein centre of mass, blur, shrink, threshold at the image's median |
| `evaluate_vein.py` | scores, EER/DET, entropy, fuzzy commitment, dedup; writes JSON and plots |

### Protocol
- **Cross-session (headline):** enrol with all 6 session-1 images, probe with
  session-2 images.
- **Same-session (optimistic):** enrol with session-1 images 1–3, probe with
  images 4–6.
- **Genuine pairs:** every enrol × probe pair of the same finger.
- **Impostor pairs:** first enrol image of A vs first probe image of B, for
  every A ≠ B in the same half. No subsampling beyond "one image each". Pairs of
  the same person's different fingers are reported separately.
- **Tune/test split:** the 123 people are shuffled (seed 0) into 61 tune / 62
  test people (244 / 248 finger identities). Every threshold and setting is
  chosen on the tune half. All numbers below are from the **test half** unless
  marked.
- **Chosen on tune:**
  - the shift range: ±20/±10 gave tune EER 11.0 %, ±40/±15 gave 7.5 %, with Gabor;
  - the decision thresholds;
  - the fuzzy-commitment setting.

  Fixed by design and not tuned on scores: preprocessing sizes, σ=5 (picked
  from looking at 3 images), the Gabor bank and its 25 % fraction, and the code
  grid.

## Results (test half, 248 finger identities)

### Verification
| method | protocol | genuine / impostor pairs | **EER** | FAR @ FRR=1 % | FAR @ FRR=5 % | test FAR / FRR at tune-EER threshold |
|---|---|---|---|---|---|---|
| maxcurv | **cross-session** | 8,928 / 61,256 | **8.2 %** | 85 % | 27 % | 8.4 % / 8.2 % |
| gabor | **cross-session** | 8,928 / 61,256 | **7.5 %** | 82 % | 18 % | 6.3 % / 8.0 % |
| maxcurv | same-session (optimistic) | 2,232 / 61,256 | 4.9 % | 71 % | 4.5 % | 4.7 % / 4.9 % |
| gabor | same-session (optimistic) | 2,232 / 61,256 | 5.1 % | 68 % | 5.7 % | 3.6 % / 5.4 % |

- Same person's other finger as impostor (744 pairs): EER 8.8 % for maxcurv
  and 8.1 % for gabor, cross-session. That is about the same as for strangers,
  so a person's 4 fingers act as roughly independent identities.
- At the strictest tune threshold (no tune impostor accepted), the test FAR is
  0.003 %, but FRR is 27 % (maxcurv) / 31 % (gabor) cross-session.

### Entropy (Daugman degrees of freedom, cross-session codes, 61,256 impostor pairs)
| method | impostor HD mean p | std σ | genuine HD mean | **DoF N = p(1−p)/σ²** | community size at 1 % collision, n ≈ √(0.02·2^N) |
|---|---|---|---|---|---|
| maxcurv | 0.415 | 0.040 | 0.276 | **155** | 2^74.8 |
| gabor | 0.418 | 0.051 | 0.287 | **94** | 2^44.3 |

EEG comparison: not available, because the EEG pipeline in this repo never
computed a DoF.

### Fuzzy commitment (enrol = majority vote of enrolment codes; probe = one image)
Settings chosen on tune: highest tune GAR with zero tune FAR and ≥ 64 key bits.

| method | protocol | alignment | bits kept | code | key bits | GAR (test) | FAR (test) |
|---|---|---|---|---|---|---|---|
| maxcurv | cross | centre of mass | 1,536 | BCH(255, t=31) | 330 | **41 %** | 0 / 61,256 |
| gabor | cross | centre of mass | 3,072 | BCH(255, t=63) | 108 | **35 %** | 2 / 61,256 |
| maxcurv | cross | oracle (upper bound) | 1,536 | BCH(255, t=39) | 282 | 70 % | 4 / 61,256 |
| gabor | cross | oracle (upper bound) | 1,536 | BCH(255, t=31) | 330 | 57 % | 0 / 61,256 |
| maxcurv | same | centre of mass | 3,072 | BCH(255, t=63) | 108 | 57 % | 0 / 61,256 |
| gabor | same | centre of mass | 3,072 | BCH(255, t=55) | 252 | 40 % | 0 / 61,256 |

- The full sweep is in the JSON files:
  - repetition codes r = 3…25 and BCH t = 15…63;
  - 3,072, 1,536 or 768 most reliable bits kept;
  - both alignments.
- Repetition codes do worse than BCH at the same key length. For example, with
  maxcurv cross-session and all bits, r=25 gives a 122-bit key at 56 % GAR,
  but 0.1 % of impostors unlock (65 of 61,256).
- "Oracle" alignment shifts the probe onto the stored vein map. It is
  **not privacy-preserving**: it needs the raw template. It is listed only to
  show how much better alignment could gain (about +20–30 points GAR).
- Unlock decisions are computed from the exact bit-error pattern for every
  pair. The rule was checked against the real `fuzzy.enroll` / `fuzzy.recover`
  on 300 random pairs per chosen setting: **2,400 / 2,400 agreed.**

### Dedup simulation (test half, 248 fingers enrolled in random order)
| method | protocol | threshold (from tune) | false duplicates at enrolment | re-enrolment caught | caught as the right finger |
|---|---|---|---|---|---|
| maxcurv | cross | zero tune FAR | 5 / 247 | 88 % | 88 % |
| gabor | cross | zero tune FAR | 1 / 247 | 85 % | 84 % |
| maxcurv | cross | tune EER | 244 / 247 | 100 % | 93 % |
| gabor | cross | tune EER | 218 / 247 | 100 % | 94 % |

The EER threshold is useless for dedup: an 8 % false match rate compounds over
hundreds of earlier enrolments. Only the strict threshold works, and even then
1 in 7 or 8 returning fingers gets through as "new".

### Plots (`results_vein/`)
- `vein_examples_<method>.png`: raw / ROI+CLAHE / vein map
- `scores_vein_<method>_<cross|same>.png`: genuine vs impostor histogram
- `det_vein_*.png`: DET curves
- `hd_vein_*.png`: impostor Hamming distances with the binomial fit

## Limitations
1. **The EER is mediocre.** Published FV-USM results often report 1–5 %, but
   many use the provided ROIs, same-session protocols or learned features. Our
   7.5–8.2 % cross-session figure is from raw images with simple, explainable
   methods, and it is the honest number for this pipeline.
2. **About 8 % of genuine cross-session pairs score like impostors** (FRR at
   the tune EER threshold; the left tail in the score histograms). These are most likely bad crops or a rotated or
   pressed finger. The fingertip anchor is imprecise: 16 % of genuine pairs
   needed the maximum ±20 px shift before we widened it. A per-image quality
   check that rejects bad captures and asks for a retake would probably help
   more than any matcher change. This has not been tested yet.
3. **The DoF / safe-community figure is not a security claim.**
   - N = 155 says the 3,072-bit codes behave like about 155 independent coin
     flips *between different fingers*.
   - It does not account for how noisy *the same finger* is: genuine HD is about
     0.28 against an impostor mean of about 0.42. In practice, how big a group
     can enrol before two fingers get confused is set by the matching
     threshold, and the dedup test shows that already at 248 fingers.
   - Impostor p is 0.415 rather than 0.5: all fingers share common structure,
     such as veins running along the finger.
4. **The fuzzy commitment works only about 35–41 % of the time cross-session.**
   That is not usable alone. Retries with several probe images, better
   alignment that needs no template, or a template-protected matcher are the
   next steps.
5. **The reliable-bit mask is stored in public.** It reveals which grid cells
   are stable for that finger, which leaks a little information. That leak has
   not been quantified.
6. **FIXED 2026-10-27 (see POP_README "Decoder fix"). Vein numbers barely changed.** Original note: **`fuzzy.py` decoder quirk (affects EEG too).** When a BCH block has more
   than t errors, the `galois` decoder gives up and returns the first k bits of
   the noisy word as they are. If those key bits happen to be error-free, the
   block "decodes" correctly anyway. This is safe for impostors here, because
   it needs k clean bits in all 12 blocks. It is modelled exactly in
   `evaluate_vein.py`. Separately, `galois` decoding hangs inside
   `multiprocessing` fork workers, so the real-decoder check runs in one
   process.
7. **Small, single population.** 123 people, one device, two sessions some
   weeks apart. Ageing, temperature and different sensors are untested.
8. **`pulse_from_video.py` is untested on real video.** On synthetic video it
   read 72.3 bpm for a 72 bpm signal (SNR 10.8 dB, accepted) and rejected a
   still video (−4 dB).

## Phase 2: live IR camera
Only the loader changes. In `neurolock/vein/data.py`:
1. Write `def ircam(root):` returning `Sample(person, finger, session, index,
   path)` for your captured images, and add it to `LOADERS`.
2. If the camera shows the finger horizontally, rotate in the loader so the
   finger is vertical with the tip at the bottom; that is what `preprocess.roi`
   expects.
3. Re-run `evaluate_vein.py --data ircam --path ...`.

The ROI, extraction, matching, codes and fuzzy commitment stay the same.
Retuning the shift range and thresholds on the new data's tune half is
expected.

## Phase 1b: stacking improvements (2026-10-27, `scoremat_vein.py` + `improve_vein.py`)
Cross-session, thresholds from tune, numbers from test. Output: `results_vein/results_improve.json`.

| step | test EER |
|---|---|
| A baseline, 1 enrol vs 1 probe image (gabor / maxcurv) | 8.4 % / 9.8 % |
| B max over 6 enrol images | 4.6 % / 4.5 % |
| C fuse gabor + maxcurv | 3.4 % |
| D probe = 3 images, max | 1.8 % |
| E quality retake (10 % of attempts redone) | 1.6 % |
| F 2 fingers per person | 0.8 % |
| F 4 fingers per person | 0 % (62 people, so a 95 % upper bound of about 5 %) |

Dedup at large scale, using person-level false match rate (FMR) per comparison:
- Low FMRs were measured on "virtual" people built from fingers of different people. This treats fingers as independent; their impostor scores correlate at r = 0.08 across finger slots.
- **Real person pairs give 3–9× higher FMR than the virtual ones**, so the virtual figures are optimistic.

| fingers | catch rate | virtual FMR | real-pair FMR | people at a 1 % false-dup risk per enrolee |
|---|---|---|---|---|
| 2 | 97 % | 7e-7 | 0 / 15k | ~1–10 k |
| 4 | 100 % (62/62) | 2e-6 | 0 / 3.8k | ~1–5 k |

Conclusion: good for a single site of thousands of people. Not good enough for a global proof-of-personhood registry, which needs FMR around 1e-9 or lower.
