import json
import logging
from datetime import datetime, timezone
from hashlib import sha256

import joblib
import numpy as np

from ml_model.train.config import MODELS_DIR, FEATURE_NAMES, TRAIN_CSV

logger = logging.getLogger(__name__)


def save_model_m1(model, scaler):
    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "m1_regressor.keras"
    model.save(str(path))
    logger.info(f"Saved {path}")
    scaler_path = out_dir / "scaler_m1.pkl"
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"Saved {scaler_path}")


def save_model_m2(model, scaler):
    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "m2_classifier.keras"
    model.save(str(path))
    logger.info(f"Saved {path}")
    scaler_path = out_dir / "scaler_m2.pkl"
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"Saved {scaler_path}")


def _data_hash(csv_path):
    try:
        with open(csv_path, "rb") as f:
            return sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def write_model_card(m1_val_metrics, m2_val_metrics, csv_path=None, data_rows=None):
    csv_path = csv_path or TRAIN_CSV
    if data_rows is None:
        data_rows = 0

    def _val(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    card = {
        "train_date": datetime.now(timezone.utc).isoformat(),
        "data_hash": _data_hash(csv_path),
        "data_rows": data_rows,
        "model_type": "m1_regressor + m2_classifier",
        "features": FEATURE_NAMES,
        "state_input": False,
        "model_version": 4,
        "m1": {
            "target": "Predicted_OEE_t1 (regression)",
            "val_mae": _val(m1_val_metrics.get("val_mae", 0)),
            "val_r2": _val(m1_val_metrics.get("val_r2", 0)),
        },
        "m2": {
            "target": "Recommended_Speed (5-class, 20-100)",
            "n_classes": 5,
            "val_accuracy": _val(m2_val_metrics.get("val_accuracy", 0)),
        },
    }
    out_path = MODELS_DIR / "model_card.json"
    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info(f"Saved {out_path}")
    return card
