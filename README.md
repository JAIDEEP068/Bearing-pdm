# Predictive Maintenance of Rotating Machinery Bearing Fault Diagnosis via Vibration Analysis

Classifies the health state of a rolling-element bearing from raw accelerometer
vibration, using the **CWRU (Case Western Reserve University) bearing dataset**.
Two complementary modelling tracks are built and compared:

1. **Classical ML** (SVM / Random Forest / XGBoost) on hand-engineered
   time-domain + FFT + envelope-spectrum features.
2. **1D-CNN** that learns directly from the raw 2048-sample waveform.

## Results (held-out 20% test set, 10 classes)

| Model | Features | Test accuracy |
|---|---|---|
| SVM / RF / XGBoost | time-domain only (9 feats) | 93.5 - 95.0 % |
| SVM | time + FFT + envelope (28 feats) | 98.3 % |
| Random Forest | time + FFT + envelope (28 feats) | 98.7 % |
| **XGBoost** | time + FFT + envelope (28 feats) | **98.9 %** |
| **1D-CNN** | raw waveform (2048 samples) | **99.6 %** |

**Headline finding:** adding frequency-domain (FFT) and envelope-spectrum
features lifts the classical models by 4–5 accuracy points over the
time-domain-only baseline. The CNN matches this *without* manual feature
engineering by learning filters directly from the signal.

## The problem

A rolling bearing has four parts that can fail: the **outer race**, **inner
race**, **rolling balls**, and **cage**. A local defect (a pit/spall) causes a
sharp mechanical impact every time a ball rolls over it. These impacts repeat at
a characteristic frequency that depends *only on bearing geometry and shaft
speed* so the repetition rate tells you **which** part is damaged.

For the CWRU drive-end bearing (SKF 6205, shaft ≈ 29.5 Hz at 1772 rpm):

| Defect | Frequency multiplier (× shaft) | Frequency |
|---|---|---|
| Outer race (BPFO) | 3.585 | ≈ 106 Hz |
| Inner race (BPFI) | 5.415 | ≈ 160 Hz |
| Ball (BSF) | 4.714 | ≈ 139 Hz |
| Cage (FTF) | 0.398 | ≈ 12 Hz |

The 10 classes are: **Normal**, plus **Ball / Inner-Race / Outer-Race** faults
at three severities each (0.007", 0.014", 0.021" defect diameter).

## Pipeline

```
.mat raw signals (48 kHz, drive-end accelerometer)
        │
        ├─ window into non-overlapping 2048-sample frames  (~230 / class, 2317 total)
        │
        ├─► ENGINEERED FEATURES (28 per window) ─► SVM / RF / XGBoost
        │       • time-domain (9): max,min,mean,sd,rms,skewness,kurtosis,crest,form
        │       • FFT spectral (13): centroid, spread, entropy, peak, 8 band energies
        │       • envelope (6): amplitude + energy-share at BPFO/BPFI/BSF
        │
        └─► RAW WINDOWS (2048 samples) ─► 1D-CNN
```

### Feature families :what each one "sees"

- **Time-domain stats** capture *how impulsive / energetic* the vibration is.
  `rms` rises with fault energy; `kurtosis` and `crest` spike when sharp impacts
  appear. Good at "is something wrong + how bad", weaker at "which part".
- **FFT spectral features** describe *where* energy sits in frequency. Different
  faults excite different resonance bands, so band energies and spectral
  shape separate fault types.
- **Envelope-spectrum features** are the physics-aware core. A defect
  *amplitude-modulates* a high-frequency structural resonance. Band-passing
  (2-6 kHz), taking the **Hilbert envelope** (demodulation), then FFT-ing the
  envelope reveals a peak at the defect frequency. Reading the amplitude at
  BPFO / BPFI / BSF directly encodes *which element* is damaged. This is the
  standard technique in industrial condition monitoring.

See `outputs/09_envelope_physics.png`: the Outer-Race envelope spectrum peaks
on the 106 Hz BPFO line, Inner-Race on 160 Hz BPFI, Normal shows no defect peak.

## Files

| File | Purpose |
|---|---|
| `features.py` | feature-extraction functions (time / FFT / envelope) |
| `build_dataset.py` | windows all `.mat` files, builds `engineered_features.csv` + `raw_windows.npz`, validates against the original CSV |
| `models_classical.py` | trains SVM/RF/XGBoost, time-only vs full comparison, confusion matrix, importances |
| `cnn.py` | trains + evaluates the 1D-CNN |
| `physics_plot.py` | envelope-analysis visualization |
| `main.py` | runs the whole pipeline |
| `outputs/` | all plots, metrics JSON, saved models (`xgb_model.joblib`, `cnn_model.keras`) |

## Reproduce

```bash
pip install -r requirements.txt
python main.py          # full pipeline
```

## Honest notes / limitations (read before quoting the numbers)

- **Single operating load (1 HP).** Models are trained and tested at one speed/
  load. Real deployment needs multiple loads; accuracy typically drops when you
  train on one load and test on another (domain shift). This is a known CWRU
  caveat and a natural next step.
- **Window-level split, not recording-level.** Train/test windows come from the
  same continuous recordings (standard in CWRU papers). Because adjacent windows
  share operating conditions, the 99% numbers are *optimistic* relative to a
  truly held-out machine. A stricter evaluation splits by time segment or by
  recording recommended before claiming production readiness.
- **Seeded (artificial) faults.** CWRU faults are EDM-machined, cleaner than
  natural wear. Field signals are noisier.
- **Validation done:** our recomputed time-domain `rms` matches the original
  provided CSV to within 0.006, confirming correct signal reading/windowing.

## What FFT and the CNN added (the point of the project)

The provided CSV had only the 9 time-domain features and capped the classical
models at 94–95%. Going back to the **raw signal** unlocked:
- real **FFT/spectral** features and **envelope analysis** → +4-5% (→ 98.9%),
- a genuine **1D-CNN** on the waveform → 99.6%, with the network implicitly
  learning frequency-selective filters that the engineered features encode by hand.
