import logging

import numpy as np
import torch
import torch.nn as nn

from ml_model.timefm.config import CONTEXT_WINDOW, N_SPEED_CLASSES, HEAD_HIDDEN, HEAD_DROPOUT

logger = logging.getLogger(__name__)


class TimeFMForecaster:
    """Wraps TimesFM 2.5 for forecast extraction. No fine-tuning — the forecast
    from a frozen foundation model already captures temporal dynamics far beyond
    engineered lag features. A lightweight head adapts the forecast to our domain."""

    def __init__(self):
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        import timesfm

        torch.set_float32_matmul_precision("high")
        logger.info("Loading TimesFM 2.5 backbone (frozen)...")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", force_download=False
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=CONTEXT_WINDOW * 2,
                max_horizon=32,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._loaded = True
        logger.info("TimesFM 2.5 loaded and frozen")

    @torch.no_grad()
    def forecast(self, oee_history: np.ndarray, horizon: int = 10) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("TimesFM not loaded")
        inputs = [oee_history.astype(np.float32)]
        point, _ = self.model.forecast(horizon=horizon, inputs=inputs)
        return point[0]

    @torch.no_grad()
    def forecast_batch(self, oee_sequences: np.ndarray, horizon: int = 10) -> np.ndarray:
        """Batch forecast: oee_sequences shape (N, T). Returns (N, horizon)."""
        if not self._loaded:
            raise RuntimeError("TimesFM not loaded")
        inputs = [oee_sequences[i].astype(np.float32) for i in range(len(oee_sequences))]
        point, _ = self.model.forecast(horizon=horizon, inputs=inputs)
        return np.stack(point)


class OEEHead(nn.Module):
    """Adapts TimeFM's forecast + covariates to our OEE distribution."""

    def __init__(self, in_features: int = 15, hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(hidden, 1),
        )

    def forward(self, forecast, covariates):
        x = torch.cat([forecast, covariates], dim=-1)
        return self.net(x).squeeze(-1)


class SpeedHead(nn.Module):
    """Predicts optimal speed from the forecasted OEE trajectory."""

    def __init__(self, in_features: int = 15, hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(hidden, N_SPEED_CLASSES),
        )

    def forward(self, forecast, covariates):
        x = torch.cat([forecast, covariates], dim=-1)
        return self.net(x)


forecaster = TimeFMForecaster()
