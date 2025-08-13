# Smartphone Decimeter 2022 — TCN Residual Corrector

A compact, end-to-end **deep learning** solution for the **Google/Kaggle Smartphone Decimeter 2022** challenge.  
We train a **Temporal Convolutional Network (TCN; Bai et al., 2018)** to learn **per-timestamp ENU corrections (ΔE, ΔN)** over a baseline smartphone trajectory, and evaluate with the official **GSDC score** (mean of p50 & p95 horizontal errors per phone).

---

## TL;DR

- **Objective:** For each phone trace, predict accurate per-timestamp latitude/longitude that minimize the **GSDC score** (average of the 50th & 95th percentile horizontal errors).  
- **Approach:** Convert baseline lat/lon to a local **ENU** frame, engineer temporal features (velocities, speed, heading, Δt), and train a **TCN** to predict **ΔE, ΔN** residuals. Add the residuals back to the baseline, then convert to lat/lon for submission/evaluation.  
- **Why TCN:** Causal, **dilated convolutions** capture long context with low latency and stable gradients—well-suited to smoothing/correction of noisy position time series.  
- **Data:** Not included (point the scripts to your local GSDC’22 copy). A **synthetic demo** is provided to sanity-check the pipeline.

---

## Repository structure
```python
├── README.md
├── requirements.txt
├── configs.yaml
└── src/
├── init.py
├── data.py # feature engineering (ENU, vel, speed, heading), windowing
├── model_tcn.py # Temporal Convolutional Network (Bai et al., 2018)
├── train_tcn.py # training loop (Huber loss, AdamW, early stopping)
├── predict_tcn.py # inference + path reconstruction to lat/lon
└── evaluate.py # GSDC scoring: mean of p50/p95 per phone

```

## Data & expected CSVs

Use competition data to create baseline CSVs (you can start from Kaggle’s baselines or your own). The scripts accept common column aliases.

#### Required columns per row

phone — e.g., 2020-05-15-US-MTV-1_Pixel4

millisSinceGpsEpoch (or UnixTimeMillis)

latDeg (or LatitudeDegrees)

lonDeg (or LongitudeDegrees)

#### Training needs two CSVs

train_input.csv — baseline lat/lon per timestamp per phone

train_gt.csv — ground-truth lat/lon (same keys)

#### Validation/Test prediction

val_input.csv / test_input.csv — baseline only

Output pred.csv will contain phone, millisSinceGpsEpoch, latDeg, lonDeg.

## Quick start

###### 0) Create synthetic train/val CSVs
```bash
python -m src.make_synth --out_root demo --n_traces 6 --len 600
```

###### 1) Train TCN on synthetic training set
```bash
python -m src.train_tcn \
  --train_csv demo/train_input.csv \
  --gt_csv    demo/train_gt.csv \
  --out       runs/tcn_demo \
  --epochs    5
```
###### 2) Predict on synthetic validation set
```bash
python -m src.predict_tcn \
  --input_csv demo/val_input.csv \
  --ckpt      runs/tcn_demo/best.pt \
  --out_csv   runs/tcn_demo/val_pred.csv
```
###### 3) Evaluate (GSDC score) on synthetic validation set
```bash
python -m src.evaluate \
  --pred_csv  runs/tcn_demo/val_pred.csv \
  --gt_csv    demo/val_gt.csv \
  --report    runs/tcn_demo/val_metrics.json
```




Features (per timestamp): ENU baseline (E, N), velocities (vE, vN), speed, heading (sin/cos), and Δt (clamped).

Targets: residuals ΔE, ΔN (ENU) between ground truth and baseline.

Loss: Huber on ΔE,ΔN (robust to outliers).

Optimizer: AdamW; early stopping on validation loss.

Inference: Sliding windows with overlap; predictions overlap-averaged. Residuals added to baseline ENU, then converted back to lat/lon.


## Metric (GSDC score)

For each phone/trace:

Compute horizontal (2D) errors per timestamp (great-circle distance in meters).

Take p50 and p95, average them.

Average the result across phones.
Lower is better. The scripts implement this in evaluate.py.

