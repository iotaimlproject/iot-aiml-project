import logging
from collections import deque

from ml_model.timefm.config import CONTEXT_WINDOW
from ml_model.timefm.predict import predictor as tfm_predictor

logger = logging.getLogger(__name__)


class TimeFMAPIService:
    """In-memory buffered TimeFM predictor. Stores OEE + covariate history per
    batch and feeds the last CONTEXT_WINDOW steps to TimeFM on each call."""

    def __init__(self):
        self._batches: dict[str, deque] = {}
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return
        tfm_predictor.load()
        self._loaded = tfm_predictor.loaded
        if self._loaded:
            logger.info("TimeFM API service ready")

    def build_sequences(
        self, batch_id: str, oee: float, avail: float, perf: float, qual: float,
        speed_pct: float, pos_ratio: float,
    ) -> tuple[list[float], list[list[float]]]:
        """Append reading to in-memory buffer and build context window."""

        if batch_id not in self._batches:
            self._batches[batch_id] = deque(maxlen=CONTEXT_WINDOW)

        buf = self._batches[batch_id]
        buf.append({
            "oee": oee,
            "cov": [avail, perf, qual, speed_pct, pos_ratio],
        })

        if len(buf) < 5:
            return None

        oee_hist = [x["oee"] for x in buf]
        cov_hist = [x["cov"] for x in buf]

        if len(oee_hist) < CONTEXT_WINDOW:
            pad_len = CONTEXT_WINDOW - len(oee_hist)
            oee_hist = [oee_hist[0]] * pad_len + oee_hist
            cov_hist = [cov_hist[0]] * pad_len + cov_hist

        return oee_hist[-CONTEXT_WINDOW:], cov_hist[-CONTEXT_WINDOW:]

    def predict(self, req) -> dict:
        """Build context from in-memory buffer, run TimeFM forecast + heads."""
        batch_id = req.batch_part_no

        seq = self.build_sequences(
            batch_id,
            oee=float(req.current_oee),
            avail=float(req.availability),
            perf=float(req.performance),
            qual=float(req.quality),
            speed_pct=float(req.current_speed) * 10,
            pos_ratio=req.part_slno / max(req.total_batch_size, 1),
        )

        if seq is None:
            return {
                "pred_oee_10m": float(req.current_oee),
                "forecasted_oee": [float(req.current_oee)] * 10,
                "confidence_pct": 0.0,
                "recommended_speed": float(req.current_speed) * 10,
                "confidence_speed_pct": 0.0,
                "speed_change": "none",
            }

        return tfm_predictor.predict(*seq)


timefm_api = TimeFMAPIService()
