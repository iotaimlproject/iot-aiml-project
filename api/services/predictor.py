import numpy as np
import joblib
from tensorflow.keras.models import load_model

from api.config import settings


class PredictorService:
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

    def predict_all(
        self,
        rpm: int,
        availability: float,
        performance: float,
        quality: float,
        current_oee: float,
        downtime_minutes: float,
    ) -> dict[str, float]:
        features = np.array([[rpm, availability, performance, quality,
                              current_oee, downtime_minutes]])
        features_scaled = self.scaler.transform(features)
        predictions = {}
        for horizon in settings.horizons:
            pred = self.models[horizon].predict(features_scaled, verbose=0)
            predictions[f"pred_{horizon}"] = float(np.clip(pred[0][0], 0, 100))
        return predictions


predictor = PredictorService()
