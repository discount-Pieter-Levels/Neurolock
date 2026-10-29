# EEG projection study: does any projection make the brain signature survive across sessions?

**28 Oct 2026.** Scripts: `proj_features.py` (features) and `proj_study.py` (study). Results: `results_proj.json`, `results_proj.log`.

**Data:** OpenNeuro ds004148, 60 people, eyes-open rest. Session 2 was +90 min, session 3 was +1 month.
**Protocol:** the same as `crossday.py`. Enrol on the first half of S1; query with the second half of S1, S2 and S3.
**Validation:**
- 5 outer folds, split by person.
- Hyper-parameters, *and the winning feature set x method*, are picked by inner 4-fold CV on the training people
  only. The criterion is the mean EER at S2 and S3.

## Result (winner chosen by inner CV, outer test)
| | same recording | +90 min | +1 month |
|---|---|---|---|
| before: relpow + LDA on windows (`crossday.py`) | 1.7 % | 6.7 % | 16.3 % |
| **chosen: psd + LDA on recording means / NAP** | **0.0 %** | **7.0 %** | **8.7 %** |
| genuine bit error (before 14 / 28 / 37 %) | 18 % | 29 % | 34 % |
| best key-lock at +1 month | | | rep r7: GAR 37 % / FAR 3.2 %, 5-bit key; BCH at FAR 0: GAR <= 8 % |

## What mattered
- **Features beat projections.**
  - `psd` is each 10-20 channel's spectrum in 2 Hz bins (2-40 Hz), normalised by that channel's own total power.
    This removes gain and impedance drift between sessions.
  - Riemannian covariance (`cov`) separates people perfectly within a recording, but drifts badly over a month
    (28-36 % EER).
- **Train the projection on recording means, not windows.**
  - LDA fitted on 4 s windows learns to ignore window noise. Fitted on quarter-recording means from 3 sessions, it
    learns to ignore *session drift*.
  - With `psd`: 12.0 % -> 6.7 % at one month.
- **NAP** (projecting out the top within-person between-session directions) is about equal to LDA on recording
  means.
- **Pairwise logistic regression on (a-b)^2 was worst**, probably overfit with 48 training people.
- **PCA** (unsupervised) helps with `psd` only (11.7 %).

## Caveats
- **Small samples:** 60 genuine comparisons per condition, so 8.7 % = 5 misses. The 95 % interval is roughly 3-18 %.
- **The inner-CV choice varies by fold** (PCA / LDA on recording means / NAP), although it is always the `psd`
  features.
- **One dataset, one task** (eyes-open rest). Not yet checked on PhysioNet or on a consumer headset.

## Verdict
- **Score:** EER at one month roughly halves. This is useful for EEG as a liveness / identity *score*.
- **Keys:** still impractical. A third of the bits flip after a month, and at FAR ~ 0 the lock opens for under 10 %
  of people. The Phase 2b decision (no brain key) stands.
