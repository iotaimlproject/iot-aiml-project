import logging
from pathlib import Path

import numpy as np
import torch

from ml_model.timefm.config import HEAD_PATH, FORECAST_HORIZON, INV_SPEED_MAP
from ml_model.timefm.encoder import forecaster, OEEHead, SpeedHead

logger = logging.getLogger(__name__)


class TimeFMPredictor:
    """Inference pipeline: load saved heads, run forecast + heads, return results."""

    def __init__(self):
        self.oee_head = None
        self.speed_head = None
        self.cov_mean = None
        self.cov_std = None
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return
        head_path = Path(HEAD_PATH)
        if not head_path.exists():
            logger.warning(f"TimeFM head not found at {HEAD_PATH} — run training first")
            return

        forecaster.load()

        state = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)

        self.oee_head = OEEHead()
        self.speed_head = SpeedHead()
        self.oee_head.load_state_dict(state["oee_head"])
        self.speed_head.load_state_dict(state["speed_head"])
        self.oee_head.eval()
        self.speed_head.eval()

        self.cov_mean = state["cov_mean"]
        self.cov_std = state["cov_std"]
        self._loaded = True
        logger.info(f"TimeFM predictor loaded (val R²={state['val_r2']:.4f}, acc={state['val_acc']:.4f})")

    @torch.no_grad()
    def predict(self, oee_history: list[float], covariates: list[list[float]]) -> dict:
        """Takes OEE history and covariate sequences, returns forecast + speed.

        Args:
            oee_history: list of float OEE values, length = CONTEXT_WINDOW
            covariates: list of [Avail, Perf, Qual, Speed_pct, pos_ratio] per timestep
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        oee_arr = np.array(oee_history, dtype=np.float32)
        cov_arr = np.array(covariates, dtype=np.float32)
        cov_last = cov_arr[-1:].astype(np.float32)

        forecast = forecaster.forecast(oee_arr, horizon=FORECAST_HORIZON)

        f_t = torch.tensor(forecast[None, :])
        c_t = torch.tensor((cov_last - self.cov_mean) / self.cov_std)

        pred_oee_raw = float(self.oee_head(f_t, c_t).item())
        pred_oee = max(0.0, min(100.0, round(pred_oee_raw, 1)))

        speed_logits = self.speed_head(f_t, c_t)
        speed_class = int(speed_logits.argmax(1).item())
        recommended_speed = INV_SPEED_MAP[speed_class]

        probs = torch.softmax(speed_logits, dim=-1)[0]
        speed_conf = float(probs[speed_class].item())

        f_std = float(np.std(forecast))
        confidence_pct = round(max(0.0, min(100.0, 100.0 - f_std * 10)), 1)

        return {
            "pred_oee_10m": pred_oee,
            "forecasted_oee": [round(v, 2) for v in forecast.tolist()],
            "confidence_pct": confidence_pct,
            "recommended_speed": recommended_speed,
            "confidence_speed_pct": round(speed_conf * 100, 1),
            "speed_change": "increase" if recommended_speed > cov_last[0, 3] else (
                "decrease" if recommended_speed < cov_last[0, 3] else "none"
            ),
        }


predictor = TimeFMPredictor()
