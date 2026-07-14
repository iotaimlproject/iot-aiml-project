from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODEL_DIR = PROJECT_ROOT / "ml_model"
DATA_DIR = ML_MODEL_DIR / "data"
MODELS_DIR = ML_MODEL_DIR / "models"
TIMEFM_DIR = ML_MODEL_DIR / "timefm"

TRAIN_CSV = str(DATA_DIR / "syn_oee_10k.csv")
HEAD_PATH = str(MODELS_DIR / "timefm_m1_head.pt")

CONTEXT_WINDOW = 30
FORECAST_HORIZON = 10

OEE_CHANNEL = "Current_OEE"
COVARIATE_CHANNELS = ["Availability", "Performance", "Quality", "Current_Speed_pct", "_pos_ratio"]
BATCH_ID = "Batch_PartNo"
TIME_ORDER = "_pos_ratio"

SPEED_MAP = {20: 0, 40: 1, 60: 2, 80: 3, 100: 4}
INV_SPEED_MAP = {v: k for k, v in SPEED_MAP.items()}
N_SPEED_CLASSES = len(SPEED_MAP)

TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
MAX_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

HEAD_HIDDEN = 64
HEAD_DROPOUT = 0.2
