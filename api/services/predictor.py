import logging
from pathlib import Path

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from api.config import settings

logger = logging.getLogger(__name__)


class RollingBuffer:
    def __init__(self, maxlen: int = 5):
        self.maxlen = maxlen
        self.store: list[dict] = []

    def push(self, oee: int) -> None:
        self.store.append({"oee": oee})
        if len(self.store) > self.maxlen:
            self.store.pop(0)

    def oee_lag(self, n: int = 1) -> int:
        if len(self.store) > n:
            return self.store[-(n + 1)]["oee"]
        return self.store[-1]["oee"] if self.store else 50

    def oee_roll5_mean(self) -> float:
        vals = [r["oee"] for r in self.store]
        return round(sum(vals) / len(vals), 1) if vals else 50.0

    def oee_trend3(self) -> float:
        if len(self.store) >= 3:
            return float(self.store[-1]["oee"] - self.store[-3]["oee"])
        return 0.0

    def reset(self) -> None:
        self.store.clear()


class PredictorService:
    def __init__(self):
        self.m1_models: list = []
        self.m2_model = None
        self.scaler_m1 = None
        self.scaler_m2 = None
        self._loaded = False
        self.buffer = RollingBuffer()
        self.current_batch: str | None = None

    def _build_features(self, req) -> np.ndarray:
        pos_ratio = req.part_slno / req.total_batch_size
        state_onehot = [0] * 5
        state_map = {"NORMAL": 3, "HIGH_LOAD": 0, "MINOR_STOPPAGE": 2,
                     "MAJOR_STOPPAGE": 1, "RECOVERY": 4}
        idx = state_map.get(req.state, 3)
        state_onehot[idx] = 1

        features = [
            req.availability, req.performance, req.quality, req.current_oee,
            req.current_speed, req.downtime_sec, req.oee_delta,
            req.part_slno, pos_ratio,
            self.buffer.oee_lag(1), self.buffer.oee_roll5_mean(), self.buffer.oee_trend3(),
        ] + state_onehot

        return np.array([features])

    def load(self):
        if self._loaded:
            return
        logger.info("Loading M1 ensemble (5 models)...")
        base = Path(settings.model_dir)
        for seed in range(5):
            path = base / f"m1_oee_seed{seed}.keras"
            self.m1_models.append(load_model(str(path)))
        logger.info(f"  Loaded {len(self.m1_models)} M1 models")

        logger.info("Loading M2 classifier...")
        self.m2_model = load_model(str(base / "m2_speed_optimizer.keras"))
        logger.info("  M2 model loaded")

        self.scaler_m1 = joblib.load(settings.scaler_m1_path)
        self.scaler_m2 = joblib.load(settings.scaler_m2_path)
        self._loaded = True

    def predict(self, req) -> dict:
        if req.batch_part_no != self.current_batch:
            self.buffer.reset()
            self.current_batch = req.batch_part_no

        features = self._build_features(req)
        features_m1 = self.scaler_m1.transform(features)

        # M1: ensemble prediction
        preds = []
        for model in self.m1_models:
            p = model.predict(features_m1, verbose=0)[0][0]
            preds.append(p)
        pred_oee = float(np.clip(np.mean(preds), 0, 100))
        pred_std = float(np.std(preds))
        oee_range = 60.0
        confidence = max(0.0, min(100.0, 100.0 * (1.0 - pred_std / oee_range)))

        # M2: speed classification
        features_m2 = self.scaler_m2.transform(features)
        probs = self.m2_model.predict(features_m2, verbose=0)[0]
        speed_idx = int(np.argmax(probs))
        recommended_speed = settings.inv_speed_map[speed_idx]
        speed_confidence = float(probs[speed_idx]) * 100.0

        self.buffer.push(req.current_oee)

        return {
            "pred_oee_10m": round(pred_oee, 1),
            "confidence_pct": round(confidence, 1),
            "recommended_speed": recommended_speed,
            "confidence_speed_pct": round(speed_confidence, 1),
            "change_needed": recommended_speed != req.current_speed,
            "batch_position": f"{req.part_slno}/{req.total_batch_size}",
        }


predictor = PredictorService()
