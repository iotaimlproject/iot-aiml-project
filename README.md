# IoT-AIML Project

Manufacturing OEE (Overall Equipment Effectiveness) prediction and optimization platform combining IoT data acquisition (Node-RED), machine learning forecasting (ANN), and a REST API for real-time inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Node-RED (Data Acquisition)                  │
│  Siemens S7-1200 ───► S7 Protocol ──┐                               │
│  Modbus TCP Devices ───► Modbus ─────┤                              │
│  OPC-UA Servers ───────► OPC-UA ─────┼──► MySQL/MariaDB Database    │
│  Dashboard UI (OEE, Controls) ───────┘                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI (REST API on :8000)                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │ /predict-  │  │ /optimize-   │  │ /health                      │ │
│  │ oee        │  │ oee          │  │                              │ │
│  └──────┬─────┘  └──────┬───────┘  └──────────────────────────────┘ │
│         │               │                                           │
│         ▼               ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │           ANN Models (5 horizons: 30m, 1h, 2h, 6h, 8h)  │        │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                    Jupyter Notebooks (ML Pipeline)                  │
│  synthetic_data_generation.ipynb  ──►  ann_future_oee_forecasting   │
│  (generates 19,520-row synthetic    │  .ipynb                       │
│   OEE timeseries)                   │  (trains 5 ANN models)        │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Node-RED (`flows.json`, `settings.js`)

IoT data collection layer running on Node-RED with:

| Feature | Details |
|---------|---------|
| **Siemens S7-1200** | Connects to `10.11.15.193` (rack 0, slot 1/2) for real-time PLC data |
| **Modbus TCP** | Client connecting to `192.168.0.123` |
| **MySQL** | Writes to databases: `akash`, `course`, `lathe`, `miet` |
| **Dashboard** | OEE monitoring, machine control panels, CAD viewer, home automation |
| **Flow Tabs** | Supervisor DB, Performance DB, Idea Lab Dashboard, Actuator DB |

Dependencies: `node-red-contrib-modbus`, `node-red-contrib-opcua`, `node-red-node-s7`, `node-red-node-mysql`, `node-red-dashboard`

### 2. ML Models (`ml_model/`)

#### 2a. Synthetic Data Generation

`synthetic_data_generation.ipynb` — generates 19,520 rows of realistic OEE timeseries data.

- **Source**: Seed data from `data/lathe_job_status_data.csv` (~20 rows)
- **Generation method**: AR(1) process for Availability/Performance/Quality with φ=0.998 + RPM random walk (300-500)
- **Downtime events**: 1,200 events placed at lowest OEE windows, exponential durations (~14 min mean)
- **Targets**: 5 forecast horizons (30m, 1h, 2h, 6h, 8h) created via shift(-N) from current OEE
- **Output**: `data/syn_timeseries_oee.csv` (1.6 MB, 19,520 rows × 12 columns, no nulls)

| Column | Description |
|--------|-------------|
| `timestamp` | 1-min resolution, 13.6 day span |
| `rpm` | Spindle speed (300-500) |
| `availability` | AR(1) process, clipped 40-100 |
| `performance` | AR(1) + RPM coupling, clipped 30-100 |
| `quality` | AR(1) process, clipped 40-100 |
| `current_oee` | A×P×Q / 10000, clipped 10-100 |
| `downtime_minutes` | 0 or exponential(14) duration |
| `target_oee_30m/1h/2h/6h/8h` | Future OEE at each horizon |

#### 2b. ANN Forecasting

`ann_future_oee_forecasting.ipynb` — trains 5 independent ANN models, one per horizon.

**Architecture**: `Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.1) → Dense(16) → Dense(1)`
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Early stopping: patience=10, min_delta=1e-5
- Train/val/test split: 80/10/10 (chronological)

**Performance (on test set)**:

| Horizon |   R²   |  MAE  | RMSE  | MAPE% | Bias  | Quality |
|---------|--------|-------|-------|-------|-------|---------|
| 30m     | 0.8024 | 1.50  | 1.85  | 2.45  | -0.16 | Good    |
| 1h      | 0.6677 | 1.92  | 2.39  | 3.12  | -0.01 | Fair    |
| 2h      | 0.3742 | 2.62  | 3.20  | 4.30  | -0.70 | Weak    |
| 6h      | -1.3745| 4.17  | 4.87  | 7.07  | -2.94 | Fails   |
| 8h      | -0.4287| 2.95  | 3.58  | 4.95  | -0.24 | Fails   |

**Key finding**: The feedforward ANN works for short horizons (30m-1h) but lacks the temporal context needed for 6h/8h. Long horizons would benefit from LSTM/GRU with lookback windows or engineered lag features.

**Saved models** (in `ml_model/models/`):
- `ann_oee_{30m,1h,2h,6h,8h}.keras` — 5 Keras models (~72 KB each)
- `scaler.pkl` — StandardScaler for feature normalization

### 3. FastAPI (`api/`)

REST API serving predictions and RPM optimization via 5 ANN models.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model load status |
| POST | `/predict-oee` | Predict OEE for all 5 horizons |
| POST | `/optimize-oee` | Find optimal RPM to reach target OEE |

#### `/predict-oee`

**Request**:
```json
{
  "current_rpm": 400,
  "current_oee": 62.0,
  "availability": 88.0,
  "performance": 82.0,
  "quality": 86.0,
  "downtime_minutes": 0.0
}
```
All fields optional except `current_rpm` and `current_oee` — missing A/P/Q/downtime auto-fetched from MySQL.

**Response**:
```json
{
  "current_rpm": 400,
  "current_oee": 62.0,
  "predictions": {
    "pred_30m": 61.62,
    "pred_1h": 61.99,
    "pred_2h": 61.44,
    "pred_6h": 61.38,
    "pred_8h": 61.64
  }
}
```

#### `/optimize-oee`

**Request**:
```json
{
  "target_oee": 70.0,
  "horizon": "1h",
  "current_rpm": 400,
  "current_oee": 62.0,
  "availability": 88.0,
  "performance": 82.0,
  "quality": 86.0,
  "downtime_minutes": 0.0
}
```

**Response**:
```json
{
  "current_oee": 62.0,
  "target_oee": 70.0,
  "optimal_rpm": 400,
  "optimal_hz": 40.0,
  "predicted_oee": 62.0,
  "feasible": false,
  "search_range": {"min_rpm": 300, "max_rpm": 500}
}
```

> Note: Optimizer performs a brute-force RPM sweep (300-500, step 5). If no RPM achieves the target, `feasible` is `false` and `optimal_rpm` stays at `current_rpm`.

#### Database Fallback

When A/P/Q/downtime are omitted from requests, the API fetches the latest row from MySQL table `manufacture_ai_data` (configurable via settings).

### 4. Database

| Setting | Default |
|---------|---------|
| Host | localhost |
| Port | 3306 |
| User | root |
| Password | iotaimlproject |
| Database | lathe |
| Table | manufacture_ai_data |

Connection: `mysql+asyncmy://` via the `databases` library.

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Node-RED)
- MySQL/MariaDB
- TensorFlow-compatible environment

### Installation

```bash
# Clone and enter project
cd iot-aiml-project

# Python dependencies (API + ML)
pip install -r api/requirements.txt

# Node-RED dependencies
npm install
```

### Configuration

Environment variables (`.env` file in project root, all optional):

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=iotaimlproject
DB_DATABASE=lathe
DB_TABLE=manufacture_ai_data
MODEL_DIR=ml_model/models
```

### Running

```bash
# Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Node-RED (separate terminal)
node-red
```

## Project Structure

```
iot-aiml-project/
├── api/                          # FastAPI REST service
│   ├── main.py                   # App entry, lifespan, health endpoint
│   ├── config.py                 # Pydantic Settings (DB, model paths)
│   ├── database.py               # Async MySQL connection
│   ├── requirements.txt          # Python dependencies
│   ├── routers/
│   │   ├── predict.py            # POST /predict-oee
│   │   └── optimize.py           # POST /optimize-oee
│   ├── schemas/
│   │   └── oee.py                # Pydantic request/response models
│   └── services/
│       ├── predictor.py          # ANN model loading + prediction
│       └── optimizer.py          # RPM sweep optimization
├── ml_model/                     # ML pipeline
│   ├── synthetic_data_generation.ipynb   # Data generation notebook
│   ├── ann_future_oee_forecasting.ipynb  # ANN training notebook
│   ├── regress_model.ipynb               # (legacy)
│   ├── syn_data.ipynb                    # (legacy)
│   ├── data/
│   │   ├── syn_timeseries_oee.csv        # Generated: 19,520 rows
│   │   ├── syn_lathe_jsdata.csv          # (legacy)
│   │   └── lathe_job_status_data.csv     # Seed data
│   └── models/
│       ├── ann_oee_{30m,1h,2h,6h,8h}.keras  # Trained ANN models
│       └── scaler.pkl                       # Feature scaler
├── lib/flows/                    # Node-RED subflow storage
├── flows.json                    # Node-RED flow definitions (39,810 lines)
├── settings.js                   # Node-RED runtime config (623 lines)
├── package.json                  # Node-RED dependencies
└── report.tex / report.pdf       # (project report artifacts)
```

## Development Notes

### Import Path Convention

The API uses relative imports with lowercase `api.` prefix (e.g., `from api.config import settings`). Keep this consistent — Python module resolution is case-sensitive regardless of filesystem.

### Notebook Execution Order

`ann_future_oee_forecasting.ipynb` has 16 cells that must run sequentially:
1. Imports
2. Load CSV (cell 2 — verified executed)
3. Feature/target split + scaling
4. `build_oee_model` (redefined in cell 5 — the cell 4 version is unused dead code)
5. Initialization + `build_oee_model` (actual version with Dropout)
6-10. Train each horizon (30m, 1h, 2h, 6h, 8h) — individual cells to avoid MCP timeout
11. Save models to `.keras` + scaler
12. Print results table
13. Scatter/bar/time-series plots
14. Feature importance (first-layer weight magnitudes)
15. Comprehensive evaluation (residuals, training history, metrics table)

### Known Limitations

- **6h/8h horizons**: Feedforward ANN with 6 static features lacks temporal context. R² < 0 for long horizons. Upgrade path: LSTM/GRU with lookback windows.
- **Optimizer retracing**: TensorFlow retraces the prediction graph on each RPM sweep iteration (minor warmup penalty, ~6 calls logged).
- **Security**: DB credentials are hardcoded in `config.py`. Use `.env` file or env vars in production.

## License

MIT
