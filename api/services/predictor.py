import logging
from pathlib import Path

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from api.config import settings
from ml_model.features.pipeline import (
    compute_lag_features,
    build_feature_vector,
)

logger = logging.getLogger(__name__)

from ml_model.train.config import M1_SEEDS

SPEED_PERF_BOOST = {1: 0, 2: 4, 3: 9, 4: 15, 5: 22, 6: 30, 7: 39, 8: 49, 9: 60, 10: 72}
SPEED_PENALTY_QUALITY = {1: 0, 2: 0, 3: 1, 4: 4, 5: 8, 6: 14, 7: 21, 8: 30, 9: 40, 10: 52}
SPEED_PENALTY_AVAIL = {1: 0, 2: 0, 3: 0, 4: 1, 5: 3, 6: 6, 7: 10, 8: 15, 9: 22, 10: 30}


class RollingBuffer:
    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self.store: list[dict] = []

    def push(self, oee: int, speed_pct: int) -> None:
        self.store.append({"oee": oee, "speed_pct": speed_pct})
        if len(self.store) > self.maxlen:
            self.store.pop(0)

    def prev_speed_pct(self) -> int | None:
        if len(self.store) >= 2:
            return self.store[-2]["speed_pct"]
        return None

    def reset(self) -> None:
        self.store.clear()

    def __len__(self):
        return len(self.store)


def _ensemble_predict(models, X):
    preds = [m.predict(X, verbose=0)[0][0] for m in models]
    raw_mean = float(np.mean(preds))
    raw_std = float(np.std(preds))
    return raw_mean, raw_std


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def estimate_oee_at_speed(availability, performance, quality, current_speed_level, target_speed_level):
    delta_perf = SPEED_PERF_BOOST[target_speed_level] - SPEED_PERF_BOOST[current_speed_level]
    delta_qual = -(SPEED_PENALTY_QUALITY[target_speed_level] - SPEED_PENALTY_QUALITY[current_speed_level])
    delta_avail = -(SPEED_PENALTY_AVAIL[target_speed_level] - SPEED_PENALTY_AVAIL[current_speed_level])

    est_perf = clamp(performance + delta_perf, 0, 100)
    est_qual = clamp(quality + delta_qual, 0, 100)
    est_avail = clamp(availability + delta_avail, 0, 100)

    return (est_avail * est_perf * est_qual) / 10000.0


def _find_best_speed(availability, performance, quality, current_speed_level):
    current_oee = estimate_oee_at_speed(availability, performance, quality,
                                         current_speed_level, current_speed_level)
    best_speed = current_speed_level
    best_oee = current_oee

    for s in range(1, 11):
        oee = estimate_oee_at_speed(availability, performance, quality,
                                     current_speed_level, s)
        if oee > best_oee:
            best_oee = oee
            best_speed = s

    return best_speed, best_oee


class PredictorService:
    def __init__(self):
        self.m1_models = []
        self.scaler_m1 = None
        self._loaded = False
        self.buffer = RollingBuffer()
        self.current_batch: str | None = None

    def load(self):
        if self._loaded:
            return
        base = Path(settings.model_dir)
        logger.info("Loading M1 ensemble...")
        self.m1_models = [
            load_model(str(base / f"m1_oee_seed{s}.keras")) for s in M1_SEEDS
        ]
        self.scaler_m1 = joblib.load(str(base / "scaler_m1.pkl"))
        self._loaded = True

    def _build_req_fields(self, req):
        return {
            "availability": req.availability,
            "performance": req.performance,
            "quality": req.quality,
            "current_oee": req.current_oee,
            "current_speed": req.current_speed,
            "downtime_sec": req.downtime_sec,
            "oee_delta": req.oee_delta,
            "part_slno": req.part_slno,
            "pos_ratio": (req.part_slno - 1) / max(req.total_batch_size - 1, 1),
            "planned_prod_duration": req.planned_prod_duration,
            "production_duration": req.production_duration,
            "production_delay_sec": req.production_delay_sec,
        }

    def _run_m1(self, fields, lag):
        features = build_feature_vector(fields, lag)
        X = self.scaler_m1.transform(np.array([features], dtype=np.float32))
        raw_mean, raw_std = _ensemble_predict(self.m1_models, X)
        conf = max(0, 100 - raw_std * 10)
        if conf < 20:
            pred = int(round(fields["current_oee"]))
        else:
            pred = int(round(raw_mean))
        return int(np.clip(pred, 0, 100)), round(conf, 1), raw_mean

    def predict(self, req) -> dict:
        if req.batch_part_no != self.current_batch:
            self.buffer.reset()
            self.current_batch = req.batch_part_no

        current_speed_pct = req.current_speed * 10
        prev_speed_pct = self.buffer.prev_speed_pct()
        if prev_speed_pct is None:
            prev_speed_pct = current_speed_pct

        self.buffer.push(req.current_oee, current_speed_pct)

        lag = compute_lag_features(self.buffer.store[:-1], req.current_oee)
        fields = self._build_req_fields(req)
        fields["prev_speed_pct"] = prev_speed_pct

        # M1 at current speed — temporal prediction for 1 minute ahead
        pred_oee, confidence_pct, _ = self._run_m1(fields, lag)

        # Physics-based speed optimizer: try all 10 levels, pick best OEE
        speed_level, best_oee = _find_best_speed(
            req.availability, req.performance, req.quality,
            req.current_speed,
        )
        recommended_speed = speed_level * 10

        # pred_oee_1m_applied = OEE at the physics-optimal speed (rounded for display)
        pred_applied = int(round(best_oee))

        # Confidence in speed recommendation
        current_speed_oee = estimate_oee_at_speed(
            req.availability, req.performance, req.quality,
            req.current_speed, req.current_speed,
        )
        improvement = best_oee - current_speed_oee
        if improvement <= 0:
            speed_conf = 100.0
        else:
            speed_conf = min(100, improvement * 25)

        change = "increase" if recommended_speed > current_speed_pct else (
            "decrease" if recommended_speed < current_speed_pct else "none"
        )

        return {
            "pred_oee_1m": pred_oee,
            "pred_oee_1m_applied": pred_applied,
            "confidence_pct": confidence_pct,
            "recommended_speed": recommended_speed,
            "confidence_speed_pct": round(speed_conf, 1),
            "speed_change": change,
            "batch_position": f"{req.part_slno}/{req.total_batch_size}",
        }


predictor = PredictorService()
