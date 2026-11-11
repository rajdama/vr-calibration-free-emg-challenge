# Aura: Calibration-Free EMG Challenge (PiEEG, Kaggle)

Baseline for cross-subject hand-gesture recognition from 8-channel EMG + IMU.
Competition: https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg

## Run

Unzip the competition data into `data/` (so that `data/train/*.xlsx`, `data/test/*.xlsx`
and `data/sample_submission.csv` exist), then:

```
pip install pandas numpy scipy scikit-learn openpyxl joblib
python load.py    # xlsx -> train.pkl / test.pkl
python feats.py   # windowed features -> Ftr.pkl / Fte.pkl
python final.py   # validation + submission.csv
```

## Approach

- Test segments are 1.4 s, mean-removed clips, so matching 1.4 s windows are cut from the
  guided-label runs of the training recordings (gesture windows 0.4-1.4 s after the cue).
- ~270 features per window: per-channel EMG amplitude, spectrum, cross-channel ratios and
  correlations, plus gyro/accelerometer motion statistics.
- Features are standardised within each recording (test files use their own unlabeled
  segments), concatenated with the raw features.
- Extra-Trees classifier with class balancing, averaged over 3 seeds.

## Validation (macro-F1)

| Split | Score |
|---|---|
| Held-out recordings (28 files) | 0.61 |
| Held-out subject-like groups (14) | 0.39 |

Several training files appear to be the same person, so the second number is the realistic one.

## Known caveats

- Where in a gesture the test clips were cut is unknown; the offset used here is inferred
  from their motion profile.
- Every training recording runs fist, open, pinch, point, thumb-up in the same order, so
  gesture is confounded with time within the session.
