import logging

from fastapi import APIRouter, HTTPException
from starlette import status

from api.schemas import PredictRequest, PredictResponse, HorizonPredictions
from api.services.predictor import predictor
from api.database import database, fetch_latest_row

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/predict-oee", response_model=PredictResponse)
async def predict_oee(req: PredictRequest):
    logger.info(
        f"POST /predict-oee rpm={req.current_rpm} oee={req.current_oee}"
        f" avail={req.availability} perf={req.performance}"
        f" qual={req.quality} downtime={req.downtime_minutes}"
    )
    avail = req.availability
    perf = req.performance
    qual = req.quality
    downtime = req.downtime_minutes

    if any(v is None for v in [avail, perf, qual, downtime]):
        try:
            await database.connect()
            row = await fetch_latest_row()
            if row:
                avail = avail or row["availability"]
                perf = perf or row["performance"]
                qual = qual or row["quality"]
                downtime = downtime or row["downtime_minutes"]
                logger.debug(f"Filled missing fields from DB: {avail=} {perf=} {qual=} {downtime=}")
            await database.disconnect()
        except Exception as e:
            logger.error(f"DB fetch failed for missing fields: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot fetch from database and incomplete request",
            )

    predictions = predictor.predict_all(
        rpm=req.current_rpm,
        availability=avail,
        performance=perf,
        quality=qual,
        current_oee=req.current_oee,
        downtime_minutes=downtime,
    )

    logger.info(f"Predictions: {predictions}")
    return PredictResponse(
        current_rpm=req.current_rpm,
        current_oee=req.current_oee,
        predictions=HorizonPredictions(**predictions),
    )
