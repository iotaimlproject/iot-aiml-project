# Physical AI Closed Loop — System Design

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Node-RED v4.1.7 (dashboard v3.6.6)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ AI Optimize Flow (per-part trigger)              │   │
│  │                                                  │   │
│  │  Part produced → gather metrics →                │   │
│  │  POST /predict-oee →                             │   │
│  │    ├─ Update OEE chart (current vs predicted)    │   │
│  │    ├─ Update Confidence gauge                     │   │
│  │    ├─ Update Speed indicator                     │   │
│  │    └─ If change_needed:                          │   │
│  │         AI Switch ON  → write speed to PLC (S7)  │   │
│  │         AI Switch OFF → flash "suggest speed X"  │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Supervisor DB / Performance DB (sensor data)     │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP POST (live metrics)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (api/)                                         │
│                                                         │
│  POST /predict-oee                                       │
│    ├─ Compute pos_ratio from part_slno / total_batch_size│
│    ├─ Build feature vector (12 features)                 │
│    ├─ Maintain rolling buffer (last 5 readings)          │
│    │    └─ Compute OEE_lag1, OEE_roll5_mean, OEE_trend3 │
│    ├─ M1: 5 ANN ensemble → pred_oee_10m + confidence %  │
│    ├─ M2: ANN classifier  → recommended_speed + conf %  │
│    ├─ Log to predictions_log (MySQL)                     │
│    └─ Return combined response                           │
│                                                         │
│  Files: main.py, config.py, database.py                  │
│         schemas/oee.py                                   │
│         services/predictor.py (M1+M2 combined)           │
│         routers/predict.py (single endpoint)              │
└──────────────────────┬──────────────────────────────────┘
                       │ load at startup
                       ▼
┌─────────────────────────────────────────────────────────┐
│  ml_model/                                               │
│                                                         │
│  models/                                                 │
│    ├─ m1_oee_seed0.keras  ─┐                             │
│    ├─ m1_oee_seed1.keras   ├─ M1 ensemble (5 models)    │
│    ├─ m1_oee_seed2.keras   │                             │
│    ├─ m1_oee_seed3.keras   │                             │
│    ├─ m1_oee_seed4.keras  ─┘                             │
│    ├─ m2_speed_optimizer.keras  ← M2 classifier          │
│    └─ scaler.pkl              ← StandardScaler           │
│                                                         │
│  data/                                                   │
│    ├─ syn_oee_10k.csv   ← training data (10k rows)      │
│    └─ production_clean.csv  ← real seed (55 rows)       │
│                                                         │
│  notebooks/                                              │
│    ├─ 01_process_production_data.ipynb  ← done          │
│    ├─ 02_generate_synthetic_data.ipynb  ← simulator     │
│    ├─ 03_train_m1.ipynb                  ← M1 training  │
│    └─ 04_train_m2.ipynb                  ← M2 training  │
└──────────────────────────────────────────────────────────┘
```

## Data Flow (per part produced)

### 1. Node-RED Sends

```json
POST /predict-oee
{
  "batch_part_no": "WM-23-A-1_b7",
  "part_slno": 5,
  "total_batch_size": 12,
  "current_speed": 80,
  "availability": 88,
  "performance": 82,
  "quality": 86,
  "current_oee": 62,
  "downtime_sec": 0,
  "oee_delta": -2,
  "state": "NORMAL"
}
```

### 2. FastAPI Processes

```python
pos_ratio = request.part_slno / request.total_batch_size  # 5/12 = 0.417

features = [
    request.availability,       # 88
    request.performance,        # 82
    request.quality,            # 86
    request.current_oee,        # 62
    request.current_speed,      # 80
    request.downtime_sec,       # 0
    request.oee_delta,          # -2
    request.part_slno,          # 5
    pos_ratio,                  # 0.417
    rolling.OEE_lag1,           # from buffer
    rolling.OEE_roll5_mean,     # from buffer
    rolling.OEE_trend3,         # from buffer
    onehot_state_NORMAL,        # 1
    onehot_state_HIGH_LOAD,     # 0
    onehot_state_MINOR_STOP,    # 0
    onehot_state_MAJOR_STOP,    # 0
    onehot_state_RECOVERY       # 0
] → scaled by scaler.transform()
```

### 3. FastAPI Returns

```json
{
  "pred_oee_10m": 64.2,
  "confidence_pct": 91.5,
  "recommended_speed": 60,
  "confidence_speed_pct": 88.0,
  "change_needed": true,
  "batch_position": "5/12",
  "reasoning": "Part 5/12, NORMAL state, OEE 62 trending down -2. Speed 80→60 recovers expected OEE to 64.2 (+2.2 pts)."
}
```

### 4. Node-RED Consumes

| Field | Dashboard Element | Action |
|-------|------------------|--------|
| `pred_oee_10m` | OEE chart line | Show predicted OEE |
| `confidence_pct` | Gauge | 0-100% needle |
| `recommended_speed` | Speed chart bar | Show recommended speed |
| `change_needed` | → if True + AI Switch ON | Write speed to PLC (S7) |
| `change_needed` | → if True + AI Switch OFF | Flash suggestion on dashboard |
| `batch_position` | Status text | "Part 5 of 12" |

## Model Architecture

### M1: OEE Forecaster (Regression)

| Aspect | Detail |
|--------|--------|
| Type | Keras Sequential ANN |
| Architecture | Dense(128)→Dropout(0.2)→Dense(64)→Dropout(0.1)→Dense(32)→Dense(1) |
| Output | `pred_oee_10m` (scalar, int 0-100) |
| Loss | MSE |
| Optimizer | Adam (lr grid: 0.001, 0.0005, 0.0001) |
| Input | 17 features (scaled) |
| Ensemble | 5 models, seeds 0-4, average prediction |
| Confidence | `max(0, 100 * (1 - ensemble_std / 60))` |
| Training target | `Predicted_OEE_t10` — conditional expected OEE (E[OEE_{t+12} \| current_state]) via 12-step transition matrix |
| Batch size | Grid search: 16, 32, 64 |
| Patience | EarlyStopping(patience=15), best weights restored |
| Validation | 20% holdout |

### M2: Speed Optimizer (Classification)

| Aspect | Detail |
|--------|--------|
| Type | Keras Sequential ANN |
| Architecture | Dense(64)→Dropout(0.2)→Dense(32)→Dropout(0.1)→Dense(5, softmax) |
| Output | 5-class probability [20, 40, 60, 80, 100] |
| Loss | SparseCategoricalCrossentropy |
| Optimizer | Adam (lr grid: 0.001, 0.0005) |
| Input | 17 features (scaled) |
| Prediction | `argmax(probs) → speed_level`, confidence = `max(probs) * 100` |
| Training target | `Recommended_Speed` — precomputed optimal speed from simulator |
| Grid search | Layer sizes [64→32, 128→64→32], dropout [0.1, 0.2, 0.3] |
| Validation | 20% holdout |

### Feature Vector (17 dimensions)

| # | Feature | Range | Type | Source |
|---|---------|-------|------|--------|
| 1 | Availability | 0-100 | int | Live sensor |
| 2 | Performance | 0-100 | int | Live sensor |
| 3 | Quality | 0-100 | int | Live sensor |
| 4 | Current_OEE | 0-100 | int | Computed |
| 5 | Current_Speed_pct | 20/40/60/80/100 | int | Live sensor |
| 6 | DownTime_sec | ≥0 | int | Live sensor |
| 7 | OEE_Delta | -100 to 100 | int | current - previous |
| 8 | Part_SLNo | 1-N | int | From request |
| 9 | pos_ratio | 0-1 | float | part_slno / total_batch_size |
| 10 | OEE_lag1 | 0-100 | int | Rolling buffer |
| 11 | OEE_roll5_mean | 0-100 | float | Rolling buffer |
| 12 | OEE_trend3 | -100 to 100 | float | Rolling buffer |
| 13-17 | _state_onehot | 0/1 | binary | One-hot encoded |

### Why No Rule Layer

M2's training target `Recommended_Speed` was computed by the simulator's `recommended_speed()` function, which brute-forces all feasible speed levels and picks the one maximizing projected OEE. M2 learns this optimal mapping from features alone.

At inference:
- If recommended_speed == current_speed → system is optimized, no action
- If recommended_speed != current_speed → AI determined a change is optimal
- **No hardcoded thresholds** (OEE < 85, OEE_Delta < 0, etc.)
- **No state-based feasibility rules** — M2 learns from data which speeds are valid per state
- **No trigger_reason** — change_needed = (recommended_speed != current_speed) is the trigger

The "reasoning" field in the response is a template string generated by the API for dashboard display, not a model output.

## API Implementation Details

### `api/` Structure

```
api/
├── main.py                 # FastAPI app, CORS, startup (load models)
├── config.py               # Model paths, speed levels, MySQL config
├── database.py             # MySQL connection, fetch_last_n_rows(), log_prediction()
├── schemas/
│   └── oee.py              # Pydantic models: PredictRequest, PredictResponse
├── services/
│   └── predictor.py        # RollingBuffer, M1EnsemblePredictor, M2Optimizer, predict()
└── routers/
    └── predict.py          # POST /predict-oee endpoint
```

### Rolling Buffer (in predictor.py)

```python
class RollingBuffer:
    maxlen: int = 5
    store: list[dict]  # last N readings
    
    def push(self, reading: dict)  # add latest, trim to maxlen
    def oee_lag(self, n: int = 1) -> int  # OEE from n steps ago
    def oee_rolling_mean(self, n: int = 5) -> float  # mean of last n OEE values
    def oee_trend(self, n: int = 3) -> float  # OEE[n-1] - OEE[0], positive = improving
    def reset(self)  # clear buffer (e.g., batch change)
```

Buffering is per-batch. When `batch_part_no` changes, reset buffer.

### MySQL Tables

```sql
-- Live metrics (written by Node-RED)
CREATE TABLE manufacture_ai_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    op_data_slno INT,
    batch_part_no VARCHAR(50),
    part_slno INT,
    availability INT,
    performance INT,
    quality INT,
    current_oee INT,
    current_speed INT,
    downtime_sec INT,
    oee_delta INT,
    state VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction audit log (written by FastAPI)
CREATE TABLE predictions_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_part_no VARCHAR(50),
    part_slno INT,
    pos_ratio FLOAT,
    current_oee INT,
    current_speed INT,
    pred_oee_10m FLOAT,
    confidence_pct FLOAT,
    recommended_speed INT,
    speed_confidence_pct FLOAT,
    change_needed BOOLEAN,
    features JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single endpoint (`/predict-oee`) | Node-RED makes one call per part → gets prediction + recommendation + confidence + trigger |
| `pos_ratio` computed in API | Node-RED sends `part_slno` and `total_batch_size`; API computes ratio. Keeps training & inference consistent. |
| Rolling buffer over MySQL queries | Lag features needed at inference time. MySQL query adds latency. Buffer is O(1) per push. |
| MySQL still used | Audit trail, retraining dataset, dashboard history queries. Not in critical path. |
| Keras Sequential ANN (not RF/XGB) | Must match old architecture pattern. ANN learns smooth decision boundaries from the deterministic `recommended_speed` target. |
| Ensemble for confidence | 5 models with different seeds → variance measures uncertainty. Robust to single-model overconfidence. |
| `change_needed = recommended != current` | The ONLY trigger. No rule thresholds. M2's training target already encodes the optimal speed. |

## Implementation Order

### Phase 1: Data Pipeline (done ± fixes)
1. ✅ `01_process_production_data.ipynb` — executed
2. [ ] `02_generate_synthetic_data.ipynb` — remove `Part_SLNo` and `_pos_ratio` from `drop_cols`, add lag feature columns, re-execute

### Phase 2: Training
3. [ ] `03_train_m1.ipynb` — Keras ANN ensemble, 5 seeds, grid search, save models + scaler
4. [ ] `04_train_m2.ipynb` — Keras ANN classifier, grid search, save model

### Phase 3: API
5. [ ] `api/config.py` — conveyor speed levels, single model paths, MySQL config
6. [ ] `api/schemas/oee.py` — PredictRequest, PredictResponse
7. [ ] `api/services/predictor.py` — RollingBuffer, M1+M2, combined predict()
8. [ ] `api/routers/predict.py` — single endpoint
9. [ ] `api/database.py` — log_prediction(), no inference queries
10. [ ] `api/main.py` — load models at startup

### Phase 4: Node-RED
11. [ ] Update "AI Optimize" flow — single HTTP POST, AI Switch routing to PLC

### Phase 5: Integration
12. [ ] End-to-end test: inject data → API → Node-RED → dashboard → simulated PLC

## `Recommended_Speed` Ground Truth (from 02 notebook)

```python
def recommended_speed(state, pos_ratio):
    feasible = {
        'NORMAL': [20,40,60,80,100],
        'HIGH_LOAD': [60,80,100],
        'MINOR_STOPPAGE': [20,40,60],
        'MAJOR_STOPPAGE': [20,40],
        'RECOVERY': [20,40,60,80]
    }[state]
    best_spd, best_oee = None, -1
    for spd in feasible:
        perf = int(np.clip(round(40 + (spd/20)*8 + state_bonus[state] - pos_ratio*8), 10, 100))
        qual = int(np.clip(round(95 - (spd/20-1)*3 + state_qual_penalty[state]), 20, 100))
        avail = int(np.clip(round(90 + state_avail_offset[state] - pos_ratio*(spd/20)*4), 20, 100))
        oee = int((avail * perf * qual) / 10000)
        if oee > best_oee:
            best_oee, best_spd = oee, spd
    return best_spd
```

All state-speed relationships are encoded in the training target. M2 learns this function. No runtime rules needed.
