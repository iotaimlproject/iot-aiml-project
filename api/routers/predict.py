import logging

from fastapi import APIRouter

from api.schemas import PredictRequest, PredictResponse, TimeFMPredictRequest, TimeFMPredictResponse
from api.services.predictor import predictor
from api.services.timefm_predictor import timefm_api
from api.database import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/predict-oee", response_model=PredictResponse)
async def predict_oee(req: PredictRequest):
    logger.info(
        f"POST /predict-oee batch={req.batch_part_no} "
        f"part={req.part_slno}/{req.total_batch_size} "
        f"oee={req.current_oee} speed={req.current_speed}"
    )
    result = predictor.predict(req)

    try:
        await log_prediction(req, result)
    except Exception:
        pass

    return PredictResponse(**result)


@router.post("/predict-oee-v2", response_model=TimeFMPredictResponse)
async def predict_oee_v2(req: TimeFMPredictRequest):
    logger.info(
        f"POST /predict-oee-v2 batch={req.batch_part_no} "
        f"part={req.part_slno}/{req.total_batch_size}"
    )
    result = timefm_api.predict(req)
    result["batch_position"] = f"{req.part_slno}/{req.total_batch_size}"
    return TimeFMPredictResponse(**result)
