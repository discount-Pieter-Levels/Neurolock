# Neurolock: proof-of-personhood pipeline, Phase 2 (static data)

Phase 2 connects finger veins, pulse, EEG and a signed agent delegation into
one pipeline, and evaluates it end to end on data already on disk. There is
no live hardware and no blockchain: the ledger is a local stub behind a
small interface.

> **Bottom line (Phase 2b, 27 Oct)**
> - **Roles:** veins prove *who* (uniqueness); EEG proves *live, present and
>   consenting* (SSVEP). EEG is never used as a key.
> - **The resting-EEG "brain key" is dropped.** It opened for 0 % of people
>   after a month. It is replaced by an **SSVEP CONFIRM / DENY consent
>   challenge**.
>
> | real pipeline, cross-session | normal verification | high-stakes (with consent) |
> |---|---|---|
> | matcher mode | **96 %** | **86 %** (18 % with the brain key) |
> | private commitment mode | **78 %** | **66 %** (10 % with the brain key) |
>
> - **Private-mode veins improved from 69 % to 82 %** (either finger, 6
>   images), still with 0 false accepts in 3,782 pairs. The gain comes from a
>   coarse alignment helper.
> - **Dedup:** 0 false duplicates, 98 % of re-enrolments caught.
> - **Every attack on the list is stopped, except signals injected in
>   software.** That includes looking at DENY (0 / 440 gave consent) and
>   replaying a CONFIRM recording for a new action (0 / 440).
> - **Is vein accuracy a data problem? Mostly, yes.** Remaining failures
>   track lighting and exposure changes between FV-USM sessions (see "Vein
>   diagnosis").
> - **SSVEP and pulse results come from simulated signals.** They test the
>   protocol; they are not physiological evidence.

## Phase 2b changes (27 Oct)

### 1. SSVEP consent replaces the brain key (`ssvep.make_consent_challenge`, `decide_consent`, `verify.consent`)
**How it works:**
- The screen shows the action text (for example "vote yes on proposal 7")
  and two buttons, CONFIRM and DENY.
- Every 1.5 s both buttons get new random frequencies, always different from
  each other, chosen from 6–15 Hz.
- The person keeps looking at the button they mean.
- Decision rule:
  - all 4 segments follow CONFIRM → **confirm**;
  - all 4 follow DENY → **deny**, which stops immediately;
  - anything else → **none**, and the person can retry (up to 3 fresh
    challenges).
- On confirm, the device signs `{action_hash, nonce, decision}` with the
  identity key.
- No brain identity is needed. Identity comes from the veins in the same
  session; EEG only shows a live person choosing CONFIRM for this action.

**Results** (SIMULATED responses, 440 trials on 55 test subjects, 3 tries):

| case | outcome |
|---|---|
| genuine confirm | 91.6 % (70 % on the first try) |
| looks at DENY | 0 / 440 gave consent (DENY detected in 93 %) |
| resting EEG | 0 / 440 gave consent |
| replayed CONFIRM recording for a new action | 0 / 440 gave consent |
| synthetic injection by an attacker who knows the challenge | 91 % (**not stopped**, same limit as liveness) |

### 2. Vein diagnosis: data or method?
`results_vein/diag_worst_fingers.png`, `quality_vein.py`, `enhance_sweep.py`.
Cross-session, single probe image, fused matcher, FMR 1e-3.

| finding | numbers | data or method? |
|---|---|---|
| Same-session vs cross-session failure | 1.5–2.2 % vs 8.3–9.0 % | session drift |
| Failures concentrate in a few fingers | 10 % of people hold 58 % of failures; 7–11 fingers fail every image | |
| Bad fingers are inconsistent even within one session | self-consistency 10–15 vs 20 for good fingers | capture quality (data) |
| Brightness change from enrolment to probe | within ±10 %: 4–6 % fail; more than ±20 %: 22–29 % fail | **exposure / illumination drift on the FV-USM rig (data)** |
| Dark captures (bottom brightness quartile) | 15–18 % fail vs 2–3 % for the brightest | data (sensor noise at low light) |
| Broken ROI crops (streaky strips) | 69 / 5,904 images; 3 of the 20 worst test fingers | **method bug**: FIXED 28 Oct (see below) |
| Large placement shift (knuckle more than 40 px off) | 0 of the 20 worst fingers | not the cause |

**Fixes tried, each chosen on tune and checked on test:**
- **Pick the 3 best-quality frames** (brightness closest to enrolment, skip
  broken crops): no gain. Bad fingers fail on all 6 images, because the whole
  session was lit differently.
- **Flat-field correction before CLAHE** (V1): tune FRR 9.0 % → 7.2 %, but
  test 8.1 % → 8.3 %. Not real. Blur and non-local-means denoising were no
  better.
- **Coarse alignment helper for the commitment** (`neurolock/vein/align.py`):
  **test GAR 69 % → 82 %, FAR 0 / 3,782, same 594-bit key.** This was a method
  problem, and it is fixed. The cost: the helper reveals roughly where the
  large veins run, as 25 × 75 cells at 2 bits each.

**Verdict:**
- The matcher at person level (2 fingers) is already at about 0.2 % EER. Its
  remaining failures are mostly a data / capture problem: exposure changes
  between sessions, plus faint veins in some people.
- In live hardware this is fixable on the capture side:
  - fixed IR illumination power with exposure locked to the enrolment value;
  - a live quality gate (reject dark or noisy frames, re-capture);
  - a finger guide.
- On FV-USM, nothing more can be recovered from images where the veins
  aren't visible.
- The private mode's gap was mostly method (alignment), and is now much
  smaller.

## Setup
    cd ~/neurolock_run && source venv/bin/activate      # numpy scipy sklearn mne galois cryptography
    uv pip install opencv-python-headless scikit-image argon2-cffi matplotlib   # added in Phase 2
    cd neurolock
    python cache_occipital.py      # once, ~2 min: parieto-occipital rest EEG -> ~/.cache/neurolock_pop
    python scoremat_vein.py gabor && python scoremat_vein.py maxcurv   # once, ~12 min (Phase 1b)
    python evaluate_pop.py         # ~11 min -> results_pop/
    python run_pop_demo.py         # ~17 s, narrated stage demo

Private caches hold biometric templates and raw EEG. **Delete them after use:**
`~/.cache/neurolock_vein`, `~/.cache/neurolock_pop`.
`results_pop/pop_params.pkl` holds public parameters only: thresholds, the LDA
projection and fusion statistics. No templates.

## Data (Step 0 inventory)
| dataset | used for | people | what it contains | location |
|---|---|---|---|---|
| FV-USM | veins | 123 (61 tune / 62 test) | 4 fingers, 2 sessions, 6 images each | `~/data/FV-USM` (raw_data only) |
| PhysioNet EEG Motor Movement/Imagery | brain key, SSVEP background | 109 (54 / 55) | 64 ch, 160 Hz; run 1 rest = enrol, rest of runs 3/7/11 = verify (**same day**) | `~/mne_data/MNE-eegbci-data` |
| OpenNeuro ds004148 | brain key across sessions | 60 (30 / 30) | eyes-open rest; S2 = +90 min, S3 = +1 month | `~/neurolock_run/ds004148` |
| Kaggle eegmat (mental arithmetic) | not used | | **not on this machine** | |
| SSVEP dataset | | | **none on disk** → SSVEP is simulated | |
| finger pulse videos | | | **none** (`~/data/pulse_videos` absent) → simulated | |

## Assumptions (read these before quoting any number)
1. **Virtual (chimeric) people.**
   - Each test "person" is one FV-USM person paired with one PhysioNet
     subject, paired at random within the same half: 55 test people and 54
     tune people.
   - This assumes vein and EEG traits are independent. That is standard in
     multimodal biometrics research, but not proven.
2. **PhysioNet EEG enrol and verify recordings are minutes apart.** That is
   optimistic, so the cross-session ds004148 numbers are the realistic ones.
3. **SSVEP responses are simulated.**
   - A weak 3-harmonic oscillation (random phase, 140 ms latency) is added to
     the subject's real resting parieto-occipital EEG.
   - The main operating point uses amplitude 0.5 × channel RMS. Results are
     reported across amplitudes 0.2–0.7.
   - **These are protocol tests on simulated responses, not physiological evidence.**
4. **Pulse videos are synthetic.**
   - "Live" videos dim by 0.2–0.6 % per heartbeat, plus sensor noise and drift.
   - The printed-vein attack is a still video of a **real** FV-USM image, with
     the same noise and drift.
   - Real videos will have motion and pressure artefacts that these do not.
5. **Tune/test discipline.**
   - Every threshold and setting is chosen on tune people.
   - Every rate reported here is from test people.
   - Exception: the SSVEP amplitude is an assumption, not tuned.

## Pipeline
| layer | signal | job | code |
|---|---|---|---|
| Who | finger veins, 2 fingers (R-index + R-middle, chosen on tune) | uniqueness, dedup, unlocking the identity secret | `neurolock/pop/biometrics.py` |
| Alive | finger pulse (PPG) | cheap liveness | `neurolock/liveness/pulse.py` |
| Present | SSVEP flicker challenge-response (EEG) | strong liveness, defeats replay | `neurolock/liveness/ssvep.py` |
| Consent | EEG brain key → consent signing key | high-stakes actions | `biometrics.BrainKey` |
| Act | Ed25519 delegation + nullifier | one agent per human per app | `neurolock/pop/delegation.py` |
| Ledger | `LedgerInterface` / `LocalLedger` | commitments, delegations, revocations | `neurolock/pop/ledger.py` |

**Enrolment** (`pop/enrol.py`, `Station.enrol`):
1. Vein dedup against every registered template. Templates are encrypted at
   rest with AES-GCM under the station key.
2. Pulse check, then SSVEP (up to 3 fresh challenges).
3. The device generates a random identity secret `S`.
4. `S` is wrapped under Argon2id(each finger's fuzzy-commitment key); either
   finger can unwrap it. EEG brain key → Argon2id → Ed25519 consent key.
5. The ledger receives `C = H(S)`, `id_pub` (derived from `S`) and
   `consent_pub`, signed by the station. No templates and no raw data go to
   the ledger.

**Verification** (`pop/verify.py`):
1. Vein unlock → `S`. Two modes:
   - commitment: private, the station stores helper data only;
   - matcher: more accurate, the station keeps the encrypted template plus an
     escrowed `S`.
2. Pulse + SSVEP on a fresh challenge.
3. For high-stakes actions only: brain key → consent key, which must match
   the ledger.
4. Sign the challenge nonce.

**Delegation:**
- The human signs `{agent_pub, app_id, scope, expiry, nullifier = H(S‖app_id)}`.
- The ledger rejects a second active delegation with the same nullifier.
- Revocation is a signed record.

**Argon2id:** t = 3, m = 64 MiB, p = 1, with a 16-byte random salt per key.
One call takes 0.115 s on this laptop.

## Results (test half)

> The sections below are the **Phase 2** run (centre-of-mass alignment,
> brain key as consent). Phase 2b numbers replace them where they differ.
>
> **Phase 2b layer table** (`results_pop/results_pop.json`, 55 people,
> 2,970 zero-effort impostor pairs):
>
> | layers | matcher GAR | matcher FAR | commitment GAR | commitment FAR |
> |---|---|---|---|---|
> | vein | 98.2 % | 1 / 2,970 | **80.0 %** | 0 |
> | + pulse [sim] | 98.2 % | 1 / 2,970 | 80.0 % | 0 |
> | + SSVEP liveness, 3 tries [sim] | 87.3 % | 1 / 2,970 | 70.9 % | 0 |
> | + SSVEP consent (high-stakes) [sim] | **83.6 %** | 1 / 2,970 | **67.3 %** | 0 |
> | (dropped) + EEG brain key instead | 18.2 % | 0 | 18.2 % | 0 |
>
> **Phase 2b real pipeline:**
> - enrolled 50 / 55 (5 failed SSVEP on all 3 tries);
> - verification:
>
>   | mode | normal | high-stakes |
>   |---|---|---|
>   | matcher | 96 % | 86 % |
>   | commitment | 78 % | 66 % |
>
> - 80 / 80 impostors rejected;
> - re-enrolment caught 49 / 50;
> - enrolment 1.2 s, verification 0.15 s.

### Layer contribution (Phase 2) (`results_pop/layer_contribution.png`)
55 test virtual people, 2,970 zero-effort impostor pairs (the impostor
presents their own veins and brain, and is live).

| layers | matcher GAR | matcher FAR | commitment GAR | commitment FAR |
|---|---|---|---|---|
| vein | 98.2 % | 0.03 % (1 pair) | 65.5 % | 0 |
| + pulse [sim] | 98.2 % | 0.03 % | 65.5 % | 0 |
| + SSVEP, 3 tries [sim] | 87.3 % | 0.03 % | 60.0 % | 0 |
| + EEG brain key | **18.2 %** | 0 | **12.7 %** | 0 |

- Pulse and SSVEP do not lower zero-effort FAR, and they shouldn't: the
  impostor is a live person. Their job is the attacks listed below.
- The EEG key removes the matcher's single false accept, but at a cost of
  80 points of GAR.

**Real pipeline run** (Argon2id, AES-GCM, Ed25519, ledger; every test person,
in random order):

| | result |
|---|---|
| enrolled | 51 / 55 (4 failed SSVEP on all 3 tries) |
| verification, matcher mode, normal / high-stakes | **82 %** / 18 % |
| verification, commitment mode, normal / high-stakes | **59 %** / 10 % |
| 80 random impostor attempts (commitment mode) | 80 / 80 rejected, all by the vein layer |
| time to enrol | 1.3 s mean, 2.3 s with 50 people registered |
| time to verify | 0.17 s |
| ledger hash chain | valid |

### Veins
- **Matcher:**
  - person-level EER 0.24 %;
  - verification threshold set at FMR 1e-4 on tune, measured test FMR
    5.6e-5 (virtual impostors);
  - dedup threshold set at FMR 1e-5 on tune.
- **Commitment:**
  - setup chosen on tune: maxcurv, 1,536 reliable bits, BCH(255, t = 23),
    giving a **594-bit key**;
  - test, all 62 people, either finger opens:

    | tries | 1 image | 3 images | 6 images |
    |---|---|---|---|
    | GAR | 58 % | 65 % | **69 %** |
    | FAR | 0 | 0 | 0 |

  - Requiring both fingers gives 37 %.
- Multi-probe retries raised GAR from 41 % (Phase 1, one finger, one image)
  to 69 %.

### Dedup (real pipeline, `results_pop/dedup_summary.png`)
- 51 people enrolled, **0 false duplicates**.
- Re-enrolment with session-2 veins: **50 / 51 caught (98 %)**.

### EEG brain key
| test | GAR | FAR | key bits |
|---|---|---|---|
| PhysioNet, same day (55 test subjects) | 21.8 % | 0 / 2,970 | 16 |
| ds004148, same recording | 86.7 % | 0 | 16 |
| ds004148, +90 min | 13.3 % | 0 | 16 |
| ds004148, +1 month | **0 %** | 0 | 16 |

- Setting chosen on tune with 3-fold CV inside tune: LDA with 32 dimensions,
  2-bit Gray coding, BCH(63, t = 11).
- **EEG degrees of freedom** (Daugman, 64-bit codes, test impostors):
  p = 0.472, σ = 0.071, so **N ≈ 49**. Finger-vein codes give N = 94–155
  (Phase 1).
- EEG carries real identity information within a session, but it drifts
  within hours. Even with a perfect code, a 16-bit key can be brute-forced:
  2¹⁶ Argon2id guesses take about 2 hours on one core.

### SSVEP challenge-response (SIMULATED, `results_pop/ssvep_vs_segment.png`)
8 parieto-occipital channels, CCA against f/2f/3f, 7 targets (6–15 Hz). The
confidence threshold per segment length was set on tune resting EEG.

| K targets | live pass, 1 try | live pass, 3 tries | replay pass, 3 tries | rest pass, 3 tries | synthetic injection |
|---|---|---|---|---|---|
| 1 | 88 % | 99.8 % | 37 % | 10 % | 99.5 % |
| 2 | 83 % | 98.9 % | 6.1 % | 0.9 % | 98.6 % |
| **4** (used) | 74 % | **89.5 %** | **0.2 %** (1 / 440) | **0 / 440** (< 0.7 %) | 92.5 % |
| 6 | 60 % | 82 % | 0 / 440 | 0 / 440 | 86 % |

- At 1.5 s segments, true pass depends strongly on the assumed amplitude:
  5 % at 0.2× RMS, 10 % at 0.3×, 67 % at 0.5×, 96 % at 0.7× (K = 4, one
  try).
- Longer segments help. At 0.5× RMS, a 3 s segment gives 98 %.
- **Only a real SSVEP recording can say which amplitude is realistic.**

### Pulse (SIMULATED videos)
- The Phase 1 threshold of 3 dB let through **5.5 % of still-image videos**.
- New threshold, chosen on tune stills: 6.85 dB.
- On test: live 200 / 200 pass, still printed-vein image 0 / 200 pass.
- Live synthetic videos have about 30 dB SNR, far cleaner than a real finger
  video will be.

### Attack table
| attack | should be stopped by | measured |
|---|---|---|
| Impostor's own veins | vein layer | commitment FAR 0 / 2,970; matcher 1 / 2,970; real pipeline 80 / 80 rejected |
| Same person enrolling twice | dedup | 50 / 51 caught |
| Printed / still vein image, no pulse [sim] | pulse | 0 / 200 pass; real pipeline rejected by pulse |
| Replayed EEG for a new challenge [sim] | SSVEP | 1 / 440 pass (3 tries); real pipeline rejected by SSVEP |
| Resting EEG, no response [sim] | SSVEP | 0 / 440 |
| Second agent delegation, same app | nullifier | rejected |
| Using a revoked delegation | ledger | rejected |
| Software-injected synthetic signals | **not stopped** | 92.5 % pass. Needs trusted sensor hardware |

## Decoder fix (`neurolock/fuzzy.py`): Phase 1 numbers, re-reported
- **The bug:** when a BCH block had more than t errors, `galois` returned the
  noisy bits unchanged. If the key bits happened to be clean, the block
  counted as decoded.
- **The fix:** `decode_checked` reads galois's error count, which is −1 on
  failure, and any failed block now fails the unlock. The exact rule in
  `evaluate_vein.py` was updated to match.
- **Check:** the real decoder agreed with the rule on 1,200 / 1,200 sampled
  pairs.

**Veins (Phase 1, chosen settings, test).** Essentially unchanged:

| method | GAR before → after | FAR before → after |
|---|---|---|
| gabor | 34.9 % → 34.3 % | 2 → 1 of 61,256 |
| maxcurv | 41.3 % | 0 (unchanged) |
| oracle alignment (both methods) | unchanged | unchanged |

**EEG, PhysioNet** (`results_vein_postfix/sweep_bch_postfix.log`). The quirk
was inflating both GAR and FAR. Examples:

| setting | GAR before → after | FAR before → after |
|---|---|---|
| 1-bit, 86 dims / keep 43, BCH(31, 4) | 63.3 % → 56.0 % | 0.22 % → 0 |
| same, BCH(31, 5) | 67.9 % → 66.1 % | 0.22 % → 0.04 % |

**EEG, cross-session ds004148** (`results_vein_postfix/results_crossday_postfix.json`):
- The large changes are all in configurations with very short keys. For
  example, 1-bit BCH(63, 10) has a 2-bit key; at +1 month its GAR went
  48 % → 7 % and its FAR 24 % → 0.
- Repetition-code results are unaffected.

## Limitations
1. **Static data cannot prove real-time liveness.** Pulse and SSVEP were
   tested on simulated signals.
2. **Software-injected signals are not stopped.** An attacker who controls
   the capture software and knows the challenge can synthesise both signals.
   Defending against that needs attested sensors or trusted hardware.
3. **The EEG brain key fails across sessions** (0 % at one month) and gives
   only a 16-bit key. **It was dropped in Phase 2b** in favour of SSVEP
   consent. It is still measured in `evaluate_pop.py` as a research result.
   - SSVEP consent proves a live person attended CONFIRM. It does not prove
     *which* brain did it: the vein capture in the same session does that.
   - Binding the two needs one trusted capture device (Phase 3).
4. **Virtual identities:** independence of vein and EEG traits is assumed.
5. **Commitment mode** opens for 82 % of people within 6 images (Phase 2b,
   up from 69 %). Its helper data leaks the reliable-bit mask and a coarse
   vein-density map. The matcher mode is more
   accurate but the station holds templates and escrowed secrets; these are
   encrypted, and the station key can be destroyed after the event.
6. **Nullifier scheme:** the delegation is signed by `id_pub`, so apps can
   link one person across apps. A real deployment should use a
   zero-knowledge membership proof (Semaphore-style) instead.
7. **Scale:** dedup was tested on 51 people. Phase 1b extrapolates the
   2-finger dedup to about 1–10 k people at a 1 % false-duplicate risk per
   enrolee. That is not global scale.
8. **The finger-pulse ↔ EEG-heartbeat cross-check is not testable** without
   simultaneous recordings.

## Phase 3: live hardware (what to swap)
| piece | replace | with |
|---|---|---|
| IR vein camera | `neurolock/vein/data.py`: add `ircam(root)` to `LOADERS` | captured images, rotated so the finger is vertical with the tip at the bottom |
| live EEG | `Capture.ssvep` responder (`run_pop_demo.capture`) and `EEGData` | LSL stream (`pylsl`): show the flicker, record K × 1.5 s from PO/O channels |
| SSVEP stimulus | none yet | a 60 Hz screen flicker renderer for `make_challenge()` frequencies |
| live pulse | `synthetic_finger_video` | frames from the IR camera during the vein capture → `pulse.check_pulse(frames, fps)` |
| heartbeat cross-check | not built | match beat times from `check_pulse` to the ECG artefact in the EEG |
| ledger | `LocalLedger` | a Logos implementation of `LedgerInterface` (4 methods) |
| SSVEP amplitude | assumed 0.5× RMS | measure it; re-tune `rho_thr` on real responses |

## Files
- `neurolock/liveness/{ssvep.py,pulse.py}`
- `neurolock/pop/{biometrics.py,crypto.py,enrol.py,verify.py,delegation.py,ledger.py}`
- Phase 2b:
  - `neurolock/vein/align.py` and `align_commit.py` (alignment helper);
  - `quality_vein.py` and `enhance_sweep.py` (diagnosis);
  - `results_vein/{diag_worst_fingers.png,results_quality.json,results_align_helper.json}`.
- `neurolock/fuzzy.py` (fix)
- `evaluate_pop.py`, `run_pop_demo.py`, `cache_occipital.py`
- `results_pop/`: `results_pop.json`, `layer_contribution.png`,
  `ssvep_vs_segment.png`, `dedup_summary.png`, `pop_params.pkl`
- Re-runs after the decoder fix: `results_vein_postfix/`. Pre-fix copies:
  `prefix_backup/`.

### ROI fix (28 Oct 2026)
**Cause:** a much brighter neighbouring finger or hand enters from the top of the frame. It pushes Otsu's
threshold up (only a sliver of the imaged finger survives) and steals the "brightest column" used as the finger axis.
**Fix** (`neurolock/vein/preprocess.py`): the original path runs first. If the fitted finger is narrower than
80 px (real ones are about 147 px), a fallback runs: a background-relative threshold, keeping the bright region with
the most area in the lower half.
- Crops narrower than 80 px: 78 -> 0. Streaky crops: 69 -> 9. Only 78 of 5,904 crops change.
- The 9 left are tilted fingers that give wedge-shaped strips (a different, unfixed failure).

Cross-session test EER, before -> after (old results in `results_vein_pre_roifix/`):

| | before | after |
|---|---|---|
| single image gabor / maxcurv | 8.4 / 9.8 % | 7.9 / 9.3 % |
| fused, max over enrol | 3.4 % | 2.8 % |
| 3-image probe | 1.8 % | 1.8 % |
| person level, 1 / 2 / 4 fingers | 3.2 / 0.8 / 0 % | 1.6 / 0.4 / 0 % |

Tuning now picks no quality retake (it was rejecting 10 %), so the retake was mostly catching broken crops.
