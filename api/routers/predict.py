from fastapi import APIRouter, HTTPException
from starlette import status

from api.schemas import PredictRequest, PredictResponse, HorizonPredictions
from api.services.predictor import predictor
from api.database import database, fetch_latest_row

router = APIRouter(tags=["Prediction"])


@router.post("/predict-oee", response_model=PredictResponse)
async def predict_oee(req: PredictRequest):
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
            await database.disconnect()
        except Exception:
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

    return PredictResponse(
        current_rpm=req.current_rpm,
        current_oee=req.current_oee,
        predictions=HorizonPredictions(**predictions),
    )
