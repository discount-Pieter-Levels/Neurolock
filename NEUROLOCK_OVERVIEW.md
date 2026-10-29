# NeuroLock: Proof of Personhood

Slide-ready notes. Each `##` section is roughly one slide.

---

## 1. The problem

- Online systems (voting, airdrops, DAOs, aid) need to know: **is this one real, live, unique human?**
- Today's answers either fail (bots and fake accounts are cheap) or cost privacy (iris and face databases are honeypots).
- **Goal:** prove "one live human, agreeing right now" **without storing anyone's biometrics.**

---

## 2. The idea in one line

**Your finger veins become a cryptographic key, your heartbeat and brain prove you're alive, and a public ledger stores only hashes.**

| Question | Answered by | How |
|---|---|---|
| Who are you? | Finger veins (near-infrared) | Vein pattern unlocks a secret (fuzzy commitment) |
| Are you unique? | Duplicate check at enrolment | New veins compared against everyone enrolled |
| Are you alive? | Pulse in the finger | A heartbeat changes how much light gets through |
| Do you mean it? | Brain response to flicker (SSVEP) | You look at CONFIRM or DENY; your brain echoes its flicker rate |
| Can it be audited? | Logos ledger | Hash-chained records: commitments only, no biometrics |

---

## 3. How it works

1. **Scan.** Near-infrared light passes through the finger. Veins absorb it and show up as dark lines.
2. **Extract.** Find the finger edges → straighten → enhance contrast → vein map (maximum curvature + Gabor filters).
3. **Lock (enrolment).** The vein map becomes a code of about 1,530 stable bits. A random secret is locked with it using a **fuzzy commitment** with a **BCH error-correcting code**. The vein image is then deleted. Only "helper data", which looks like noise, is kept.
4. **Unlock (return visit).** A fresh scan is never identical, so some bits flip. The code fixes up to **23 flips per 255-bit block**. The same person stays under that limit; a different person lands about 30% away and can't unlock.
5. **Alive and consenting.** CONFIRM and DENY flicker at random rates that change every 1.5 s. Your visual cortex follows the one you look at. A recording made earlier can't follow a sequence that didn't exist yet.
6. **Record.** The ledger stores `commitment = H("commit", secret)`, the identity public key, a station signature and the signed consent. **No images, templates or EEG anywhere.**

---

## 4. Results (held-out test people)

| Metric | Result |
|---|---|
| Vein match error rate (EER), 2 fingers, visits ~2 weeks apart | **0.4%** |
| 1 finger per person | 1.6% |
| Full pipeline, wrong person accepted | **1 in 2,970** |
| Duplicate sign-ups caught | **98%** (0 false duplicates) |
| Fuzzy-commitment unlock (genuine), with alignment helper | 82% (0 false unlocks) |

- Data: FV-USM finger-vein database (123 people, 4 fingers, 2 sessions); PhysioNet EEG (109 subjects); OpenNeuro ds004148 (60 subjects, recorded again after 90 min and after 1 month).
- A bug fix in finger-edge detection (28 Oct) halved the vein error rate (fused EER 3.4% → 2.8%).

---

## 5. What we learned (pivots)

- **The brain as a key: tried, then dropped.** EEG identity works within a session (EER about 2%), but it drifts: 16% after a month (8.7% with better features), and about 34% of bits flip, so no stable key is possible. **So EEG is used for liveness and consent, not identity.**
- **The error-correcting code isn't the limit; bit stability is.** Moving from a repetition code to BCH lowered false accepts but didn't raise genuine unlocks.
- **Two fingers are enough for a community** (about 1–10k people). A global registry would need about a 1-in-a-billion false-match rate, so it would need more fingers or other signals.
- **One brain step, not two.** Following CONFIRM's random flicker sequence already proves liveness, so consent and liveness can share one check.

---

## 6. Logos integration

- **Registry program** on the Logos LEZ (Rust, risc0): checks station signatures, enforces a party window, computes a nullifier so one person can't register twice. **14/14 tests pass.**
- Python ledger client verified against the live testnet. Faucet and first transactions included.
- **Status:** deployment is paused because the LEZ testnet halted (block 68,453); it resumes when blocks advance.
- **The interactive demo runs inside Logos Basecamp** as a native app (QML), within Basecamp's plugin sandbox.

---

## 7. The interactive demo

Five steps, one action each. The audience drags in recorded samples:

1. **Scan.** Drop Asha's (or Bob's) finger scan. The extraction stages play in the scanner.
2. **Pulse.** Drop "live finger" or "printed photo". A heartbeat is found, or it isn't.
3. **Unlock.** Watch the bit flips. Asha: finger 3 stays locked, finger 4 opens (real result). Bob: rejected.
4. **Consent.** Drop Asha's EEG and hold the mouse on CONFIRM or DENY. A signed record joins the hash chain.
5. **Attacks.** Six attacks (impostor, photo, double sign-up, replayed consent, coercion, ledger edit), each stopped by a named check.

**Real:** the FV-USM scans and every processing stage, all scores and bit-error counts, the CCA brain-response detector, the pulse detector, SHA-256 hashing.
**Simulated:** the flicker response (added to real resting EEG) and the finger-pulse videos. No such recordings exist yet.
**Not built:** the scanner hardware. The demo deliberately uses recorded samples instead of a webcam.

---

## 8. Limits and next steps

- **Trusted hardware:** anyone who controls the capture software could fake a pulse and a brain signal. That needs a secure sensor.
- **Real recordings:** collect actual SSVEP EEG and NIR finger videos to replace the simulated parts.
- **Scale:** more fingers or fused signals for registries larger than about 10k people.
- **Deploy:** put the registry on the LEZ testnet once it resumes.
- **Hardware:** an NIR finger capture rig (prototype capture script in progress).

---

### Where things are

- Pipeline code: `~/neurolock_run/neurolock/neurolock/`
- Browser demo: `demo_web/index.html` (detailed version: `index_v2_full.html`)
- Basecamp app: `logos/basecamp/neurolock-demo/` (`install.sh` installs it)
- Detailed write-ups: `VEIN_README.md`, `POP_README.md`, `PROJ_README.md`, `LOGOS_README.md`
