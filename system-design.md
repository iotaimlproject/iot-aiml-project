# IoT-AIML Project — System Design

## Overview

OEE (Overall Equipment Effectiveness) forecasting and conveyor speed optimization using an ensemble of Keras ANNs, with an experimental TimeFM2.5 transformer endpoint. Real-time PLC data flows through Node-RED → FastAPI → trained models → dashboard display + PLC control signal.

---

## Project Structure

```
E:\Projects\iot-aiml-project\
├── api/                              # FastAPI application (sibling to ml_model/)
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # Settings: model_dir, scaler paths, speed map
│   ├── database.py                   # MySQL connection
│   ├── requirements.txt              # API deps (fastapi, uvicorn, tensorflow, joblib, etc.)
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── oee.py                    # Pydantic models for request/response
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py                # /predict-oee + /predict-oee-v2 routes
│   │
│   └── services/
│       ├── __init__.py
│       ├── predictor.py              # M1 ensemble + M2 classifier inference
│       └── timefm_predictor.py       # TimeFM2.5 inference (experimental)
│
├── ml_model/                         # Training pipeline (Python modules, not notebooks)
│   ├── __init__.py
│   ├── requirements-ml.txt           # keras, tensorflow, timesfm, sklearn, etc.
│   │
│   ├── data/
│   │   ├── clean_production.py       # Parse raw CSV → production_clean.csv
│   │   └── generate_synthetic.py     # 5-state Markov + causal model → 10k syn rows
│   │
│   ├── features/
│   │   └── pipeline.py               # 12-feature contract definition
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── config.py                 # Paths, hyperparams, seeds, speed map
│   │   ├── models.py                 # build_m1(), build_m2() factories
│   │   ├── train_m1.py               # 5-seed ensemble training loop
│   │   ├── train_m2.py               # 10-class classifier with class weights
│   │   ├── evaluate.py               # Per-state metrics, confusion matrix, ablation
│   │   ├── export.py                 # Save .keras, .pkl, model_card.json
│   │   └── main.py                   # CLI: python -m ml_model.train.main --m1 --m2
│   │
│   ├── timefm/
│   │   ├── __init__.py
│   │   ├── config.py                 # Window, horizon, d_model
│   │   ├── encoder.py                # TimeFM2.5 frozen backbone → embeddings
│   │   ├── head_m1.py                # Regression head (embeddings → OEE)
│   │   ├── head_m2.py                # Classification head (embeddings → speed 1-10)
│   │   ├── train.py                  # Train both heads on synthetic embeddings
│   │   ├── predict.py                # Full pipeline: encode → M1 head → M2 head
│   │   └── main.py                   # CLI: python -m ml_model.timefm.main
│   │
│   ├── models/                       # Trained artifacts (loaded by api/ at runtime)
│   │   ├── scaler_m1.pkl
│   │   ├── scaler_m2.pkl
│   │   ├── m1_oee_seed{0-4}.keras    # 5 M1 ensemble models
│   │   ├── m2_speed_optimizer.keras  # 10-class classifier
│   │   ├── timefm_m1_head.keras      # Trained TimeFM OEE head
│   │   ├── timefm_m2_head.keras      # Trained TimeFM speed head
│   │   └── model_card.json           # Train metadata per run
│   │
│   ├── data/                         # Generated datasets
│   │   ├── production_final.csv      # Raw 55-row seed (from factory)
│   │   ├── production_clean.csv      # Parsed + engineered (step 1 output)
│   │   └── syn_oee_10k.csv           # 10,000 synthetic rows (step 2 output)
│   │
│   └── notebooks/                    # Legacy notebooks (reference only, not for training)
│       ├── 01_process_production_data.ipynb
│       ├── 02_generate_synthetic_data.ipynb
│       ├── 03_train_m1.ipynb
│       └── 04_train_m2.ipynb
│
├── flows.json                        # Node-RED flows
├── system-design.md                  # This file
└── README.md
```

---

## Feature Engineering (The Contract)

### Input Features (12 dimensions, no state input)

The model receives **12 numerical features** per inference request. The API has NO state field — the model infers operating conditions from the raw metrics alone.

| # | Feature | Type | Range | Source |
|---|---------|------|-------|--------|
| 1 | Availability | int | 0–100 | PLC / Node-RED |
| 2 | Performance | int | 0–100 | PLC / Node-RED |
| 3 | Quality | int | 0–100 | PLC / Node-RED |
| 4 | Current_OEE | int | 0–100 | Computed: A×P×Q / 10000 |
| 5 | Current_Speed | int | 1–10 | PLC (conveyor speed level) |
| 6 | DownTime_sec | int | 0+ | PLC accumulated downtime |
| 7 | OEE_Delta | int | -100–100 | Current_OEE - Previous_OEE |
| 8 | Part_SLNo | int | 1+ | Part serial within batch |
| 9 | pos_ratio | float | 0.0–1.0 | Part_SLNo / total_batch_size |
| 10 | OEE_lag1 | int | 0–100 | Last reading's OEE (from RollingBuffer) |
| 11 | OEE_roll5_mean | float | 0–100 | Rolling mean of last 5 OEEs |
| 12 | OEE_trend3 | float | -50–50 | OEE change over last 3 readings |

### RollingBuffer (api/services/predictor.py)

Maintains last **20 readings** in memory (covers ~40s at 2s polling). Provides:

| Feature | Computation |
|---------|-------------|
| OEE_lag1 | buffer[-2].oee (second-to-last) |
| OEE_roll5_mean | mean of last 5 buffer entries |
| OEE_trend3 | buffer[-1].oee - buffer[-3].oee |
| OEE_min5 | min of last 5 entries |
| OEE_max5 | max of last 5 entries |
| OEE_slope5 | linear regression slope over last 5 entries |

Buffer **resets** when `batch_part_no` changes (new batch detected).

---

## M1: OEE Forecaster (Regression)

**Purpose**: Predict OEE value ~10 minutes ahead (0–100).

**Architecture**: 5-model ensemble, each a Keras Sequential ANN:
```
Input(12) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.1) → Dense(32, ReLU) → Dense(1)
```

**Training details**:
- Loss: MAE (mean absolute error)
- Optimizer: Adam, lr=0.001
- Batch size: 32
- Early stopping: patience=15, monitor=val_loss
- Ensemble seeds: 0, 1, 2, 3, 4 (different random init)
- Split: 80/20 temporal (time-ordered, not random)
- Scaler: StandardScaler (per-feature, fitted on train set)

**Inference**:
```
pred_oee = mean(model_0(x), model_1(x), ..., model_4(x))
confidence = 100 * (1 - std(preds) / 60), clipped to [0, 100]
```

**Target**: `Predicted_OEE` — actual OEE value 12 steps ahead in the dataset (~10 min at ~50s/row cadence). Computed by rolling forward: for each row, look up the OEE of the row ~12 positions later (within the same batch). This ensures the target reflects real temporal trajectory, not stationary state averages.

---

## M2: Speed Optimizer (10-Class Classification)

**Purpose**: Recommend conveyor speed level (1–10) that maximizes future OEE.

**Architecture**: Single Keras Sequential ANN:
```
Input(12) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.1) → Dense(10, softmax)
```

**Training details**:
- Loss: sparse categorical crossentropy
- Class weights: `n_samples / (n_classes × bin_count)` to handle imbalance
- Optimizer: Adam, lr=0.001
- Batch size: 32
- Early stopping: patience=15, monitor=val_accuracy
- Scaler: StandardScaler (separate from M1 scaler)

**Inference**:
```
probs = model(x)                         # 10 probabilities
speed_idx = argmax(probs)                # 0-9
recommended_speed = speed_idx + 1         # 1-10
speed_confidence = probs[speed_idx] * 100
```

### Graduated Post-Processing

After raw M2 output, a control policy smooths speed changes:

```python
delta = recommended_speed - current_speed
max_delta = 2 if (oee_trend > 0 and confidence > 85) else 1
delta = np.clip(delta, -max_delta, max_delta)
final_speed = int(current_speed + delta)
```

This ensures:
- **Normal operation**: speed changes by max ±1 per inference
- **Improving trend + high confidence**: allows ±2 (faster recovery)
- **Downtrend/emergency**: drops speed by 1 per step (gradual deceleration)
- **No abrupt jumps**: prevents mechanical stress on conveyor

**Target**: `Recommended_Speed` — precomputed by brute-force simulation: for each training row, try all 10 speed levels (1–10), compute resulting OEE via the causal model, pick the speed that gives the highest OEE.

---

## Data Generation (`generate_synthetic.py`)

### 5-State Markov Machine

| State | Description | Self-loop | Avg OEE |
|-------|-------------|-----------|---------|
| NORMAL | Baseline stable operation | 0.94 | ~60 |
| HIGH_LOAD | Performance up, quality down | 0.80 | ~55 |
| MINOR_STOPPAGE | Moderate degradation | 0.65 | ~35 |
| MAJOR_STOPPAGE | Heavy degradation | 0.82 | ~20 |
| RECOVERY | Gradual improvement | 0.15 | ~50 |

### Causal Model (Speed → OEE)

Speed causally drives Performance, Quality, and Availability with calibrated tradeoffs:

```
Performance = base + speed * 5 + state_mod - pos_ratio * 8 + noise
Quality = base - penalty * (speed - 1) + state_mod + noise
Availability = base + state_mod - pos_ratio * (speed / 2) + noise
OEE = (Availability * Performance * Quality) / 10000
```

The penalty term ensures that **speed 10 is not always optimal** — there's a real tradeoff:

| Speed | Perf boost | Quality penalty | Avail penalty | Net OEE effect |
|-------|-----------|----------------|---------------|----------------|
| 1–3 | Low | None | None | Safe, low output |
| 4–6 | Medium | Mild | Mild | Balanced sweet spot |
| 7–9 | High | Moderate | Moderate | Good when stable |
| 10 | Max | Heavy | Heavy | Risky, defects ↑ |

Optimal speed depends on state + position:

| State | Optimal Speed Range |
|-------|-------------------|
| NORMAL | 6–9 |
| HIGH_LOAD | 4–7 |
| MINOR_STOPPAGE | 2–5 |
| MAJOR_STOPPAGE | 1–3 |
| RECOVERY | 2→3→5→7 (increasing) |

### Batch Types

6 batch types with varying OEE baselines (mirroring real factory batches):

| Type | OEE Base | Degrade Rate | Size Range |
|------|---------|-------------|-----------|
| WM-23-A-1 | 50 | 0.15 | 30–60 |
| WM-23-A-2 | 55 | 0.12 | 25–50 |
| Ra_21_A_1 | 45 | 0.18 | 20–40 |
| Ra_21_A_2 | 60 | 0.10 | 30–55 |
| Ra_21_A_3 | 40 | 0.20 | 15–35 |
| Ra_21_A_4 | 65 | 0.08 | 35–70 |

### Lag Feature Computation

Computed per-batch (reset at batch boundary):
- `OEE_lag1` = shift(1), filled with current value for row 0
- `OEE_roll5_mean` = rolling(5, min_periods=1).mean()
- `OEE_trend3` = diff(2), clipped to [-50, 50]
- `pos_ratio` = (Part_SLNo - 1) / batch_size

---

## API Endpoints

### `POST /predict-oee` (Production)

**Request** (Pydantic schema — no state field):
```json
{
  "availability": 85,
  "performance": 72,
  "quality": 95,
  "current_oee": 58,
  "current_speed": 6,
  "downtime_sec": 12,
  "oee_delta": 3,
  "part_slno": 15,
  "total_batch_size": 50,
  "batch_part_no": "WM-23-A-1_b42"
}
```

**Response**:
```json
{
  "pred_oee_10m": 61.3,
  "confidence_pct": 94.2,
  "recommended_speed": 7,
  "confidence_speed_pct": 87.5,
  "change_needed": true,
  "batch_position": "15/50"
}
```

**Inference Flow**:
1. Parse request → validate with Pydantic
2. Detect batch change → reset RollingBuffer if new batch
3. Push current_oee to RollingBuffer
4. Build 12-feature vector (inline, mirrors `features/pipeline.py` contract)
5. Scale with scaler_m1 → run 5 M1 models → mean + std → pred_oee + confidence
6. Scale with scaler_m2 → run M2 model → argmax → raw recommendation
7. Apply graduated post-processing (clip to ±1 / ±2)
8. Push to OEE history buffer → return response

### `POST /predict-oee-v2` (Experimental — TimeFM2.5)

**Request**:
```json
{
  "series": [
    {
      "availability": 85,
      "performance": 72,
      "quality": 95,
      "current_oee": 58,
      "current_speed": 6,
      "downtime_sec": 12,
      "oee_delta": 3,
      "part_slno": 15,
      "pos_ratio": 0.3
    },
    { "...": "..." }
  ]
}
```

**Response**:
```json
{
  "predicted_trajectory": [58.1, 59.2, 60.0, 61.5, 62.3],
  "pred_oee_10m": 62.3,
  "confidence_lower": 57.8,
  "confidence_upper": 66.9,
  "recommended_speed": 7,
  "speed_confidence_pct": 85.1
}
```

**TimeFM2.5 Pipeline**:
1. Format series as `[batch=1, time_steps=N, features=9]` numpy array
2. Run TimeFM2.5 encoder (frozen, 200M params) → embeddings
3. **M1 Head**: Dense(128→64→1) regression → future OEE trajectory
4. **M2 Head**: Dense(64→10) softmax → speed recommendation
5. Both heads are lightweight (few thousand params), trained on synthetic embeddings
6. Full TimeFM2.5 model is NOT fine-tuned — only the heads are trained

---

## Data Flow (Node-RED → API → Dashboard)

### Trigger Cadence

```
PLC polling (every 2s)
  └→ Node-RED Supervisor DB tab (stores raw values in flow context)
      └→ Performance DB tab (computes Availability, Performance, Quality, OEE)
          └→ AI Optimize tab (triggered PER PRODUCED PART, not every 2s)
              ├─ 7 Store X functions (track value changes)
              ├─ Build Payload function (assemble JSON)
              ├─ POST /predict-oee
              ├─ Parse Response function
              └─ Update dashboard (3 charts + 3 text widgets)
```

**Key rule**: The 2s PLC data is stored in flow context but does NOT trigger ML inference. Only the **part completion event** triggers the AI Optimize flow, ensuring:
- Stable feature vectors (part-level granularity)
- RollingBuffer has sufficient data between calls
- No API flooding from 2s polling
- Dashboard updates at meaningful cadence (~30–60s)

### Dashboard Widgets (AI Dashboard group)

| Widget | Data Source | Type |
|--------|------------|------|
| OEE Prediction | pred_oee_10m | Chart (ui_chart) |
| Speed Prediction | recommended_speed | Chart (ui_chart) |
| Profit Prediction | derived metric | Chart (ui_chart) |
| Predicted OEE | pred_oee_10m | Text (ui_text) |
| Predicted Speed | recommended_speed | Text (ui_text) |
| Confidence | confidence_pct | Text (ui_text) |

### PLC Writeback (Phase 2)

When `change_needed == true`:
```
Node-RED → PLC write to speed register
  ├─ If recommended_speed > current_speed: increase gradually (1 level per cycle)
  ├─ If recommended_speed < current_speed: decrease immediately (safety)
  └─ Cooldown: minimum 3 cycles between writes (prevent oscillation)
```

---

## Training CLI

### `python -m ml_model.train.main`

| Flag | Description |
|------|-------------|
| `--m1` | Train M1 ensemble |
| `--m2` | Train M2 classifier |
| `--data path` | Path to training CSV (default: data/syn_oee_10k.csv) |
| `--split float` | Temporal split ratio (default: 0.8) |
| `--epochs int` | Max epochs (default: 200) |
| `--seed int` | Random seed (default: 42) |

### `python -m ml_model.timefm.main`

| Flag | Description |
|------|-------------|
| `--train` | Train M1 + M2 heads |
| `--predict` | Run inference on test set |
| `--window int` | Lookback window (default: 30 steps) |
| `--horizon int` | Forecast horizon (default: 12 steps) |

### Model Card (`models/model_card.json`)

Exported after every training run:
```json
{
  "train_date": "2026-07-10T15:30:00",
  "data_hash": "sha256:abc123...",
  "data_rows": 10000,
  "m1_val_mae": 1.77,
  "m1_val_r2": 0.968,
  "m2_val_accuracy": 0.9915,
  "m2_class_weights": {"1": 8.5, "2": 7.2, ..., "10": 0.85},
  "features": ["Availability", "Performance", ...],
  "speed_range": "1-10",
  "state_input": false,
  "training_script": "train/main.py",
  "seed": 42
}
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No state input to API | Model learns operating conditions from raw metrics; removes external dependency on state computation |
| Speed 1-10 discrete | Matches PLC conveyor control; 10 classes give sufficient granularity |
| ±1 speed clipping | Prevents mechanical shock; gradual changes match operator behavior |
| 5-ensemble for M1 | Reduces prediction variance; confidence derived from ensemble spread |
| Separate scalers per model | M1 regression vs M2 classification have different scale sensitivities |
| RollingBuffer(20) in memory | Avoids MySQL query on critical path; sufficient for 40s of 2s data |
| TimeFM2.5 as separate endpoint | Experimental comparison; no risk to production flow |
| Frozen TimeFM backbone | 200M params too large to train; heads are lightweight and fast to train |
| Temporal (not random) split | Prevents data leakage; respects production time ordering |
| Class weights for M2 | Handles speed imbalance (speed 10 is common, speed 1 is rare) |

---

## Future Considerations

- **Online learning**: Periodically retrain on real production data collected from Node-RED
- **Anomaly detection flag**: Add `unexpected_state: true` to response when feature distribution falls outside training range
- **Multi-step forecast**: Extend M1 to predict OEE at t+10, t+20, t+30 minutes (3-head output)
- **RL-based speed controller**: Replace M2 with a DQN agent that learns optimal speed through cumulative reward
- **Model versioning**: Keep last 5 model generations in `models/archive/` with git-lfs tracking
- **A/B testing**: Route % of traffic to TimeFM2.5 endpoint and compare MAE / recommendation acceptance rate
