from fastapi import APIRouter, HTTPException
from starlette import status

from api.schemas import OptimizeRequest, OptimizeResponse
from api.services.optimizer import optimizer
from api.database import database, fetch_latest_row

router = APIRouter(tags=["Optimization"])


@router.post("/optimize-oee", response_model=OptimizeResponse)
async def optimize_oee(req: OptimizeRequest):
    if req.target_oee is None and req.target_oee_percentage is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either target_oee or target_oee_percentage",
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
            await database.disconnect()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot fetch from database and incomplete request",
            )

    result = optimizer.optimize(
        target_oee=req.target_oee,
        target_oee_percentage=req.target_oee_percentage,
        horizon=req.horizon,
        current_rpm=req.current_rpm,
        current_oee=req.current_oee,
        availability=avail,
        performance=perf,
        quality=qual,
        downtime_minutes=downtime,
    )

    return OptimizeResponse(**result)
