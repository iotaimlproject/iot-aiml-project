from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODEL_DIR = PROJECT_ROOT / "ml_model"
DATA_DIR = ML_MODEL_DIR / "data"
MODELS_DIR = ML_MODEL_DIR / "models"
PLOTS_DIR = ML_MODEL_DIR / "plots"

TRAIN_CSV = str(DATA_DIR / "syn_oee_10k.csv")

BASE = ["Availability","Performance","Quality","Current_OEE",
        "Current_Speed_pct","DownTime_sec","OEE_Delta",
        "Part_SLNo","_pos_ratio",
        "Planned_Prod_Duration","Production_Duration","Production_Delay_sec",
        "Prev_Speed_pct","Speed_Delta"]
LAGS = ["OEE_lag1","OEE_roll5_mean","OEE_trend3",
        "OEE_min5","OEE_max5","OEE_range5"]

FEATURE_NAMES = BASE + LAGS
N_FEATURES = len(FEATURE_NAMES)

TARGET_M1 = "Predicted_OEE_t1"

M1_SEEDS = list(range(20))

M1_CONFIG = {
    "layers": [256, 128, 64],
    "dropout": [0.2, 0.1, 0.0],
    "activation": "relu",
    "use_batch_norm": True,
    "output_activation": "linear",
    "loss": "huber",
    "learning_rate": 0.0008,
    "batch_size": 256,
    "max_epochs": 400,
    "early_stopping_patience": 40,
}


