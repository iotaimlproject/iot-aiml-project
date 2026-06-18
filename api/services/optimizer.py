import numpy as np
import joblib
from tensorflow.keras.models import load_model

from api.config import settings


class OptimizerService:
    def __init__(self):
        self.models: dict[str, any] = {}
        self.scaler = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.scaler = joblib.load(settings.scaler_path)
        for horizon in settings.horizons:
            path = f"{settings.model_dir}/ann_oee_{horizon}.keras"
            self.models[horizon] = load_model(path)
        self._loaded = True

    def optimize(
        self,
        target_oee: float | None,
        target_oee_percentage: float | None,
        horizon: str,
        current_rpm: int,
        current_oee: float,
        availability: float,
        performance: float,
        quality: float,
        downtime_minutes: float,
    ) -> dict:
        if target_oee is not None:
            target = target_oee
        elif target_oee_percentage is not None:
            target = current_oee + target_oee_percentage
        else:
            target = current_oee + 10.0

        target = min(target, 100.0)

        best_rpm = current_rpm
        best_pred = current_oee
        feasible = False

        rpm_candidates = list(range(
            settings.rpm_min,
            settings.rpm_max + 1,
            settings.rpm_step,
        ))

        for rpm in rpm_candidates:
            features = np.array([[
                rpm, availability, performance, quality,
                current_oee, downtime_minutes
            ]])
            features_scaled = self.scaler.transform(features)
            pred = self.models[horizon].predict(features_scaled, verbose=0)
            pred_val = float(pred[0][0])

            if pred_val >= target and pred_val > best_pred:
                best_rpm = rpm
                best_pred = pred_val
                feasible = True

        return {
            "current_oee": current_oee,
            "target_oee": target,
            "optimal_rpm": best_rpm,
            "optimal_hz": round(best_rpm / 10, 1),
            "predicted_oee": round(best_pred, 2),
            "feasible": feasible,
            "search_range": {
                "min_rpm": settings.rpm_min,
                "max_rpm": settings.rpm_max,
            },
        }


optimizer = OptimizerService()
