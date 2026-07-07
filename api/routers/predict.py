import logging

from fastapi import APIRouter

from api.schemas import PredictRequest, PredictResponse
from api.services.predictor import predictor
from api.database import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/predict-oee", response_model=PredictResponse)
async def predict_oee(req: PredictRequest):
    logger.info(
        f"POST /predict-oee batch={req.batch_part_no} "
        f"part={req.part_slno}/{req.total_batch_size} "
        f"oee={req.current_oee} speed={req.current_speed} state={req.state}"
    )
    result = predictor.predict(req)

    # best-effort async logging
    try:
        await log_prediction(req, result)
    except Exception:
        pass

    return PredictResponse(**result)
