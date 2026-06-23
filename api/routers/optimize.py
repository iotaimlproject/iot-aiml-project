import logging

from fastapi import APIRouter, HTTPException
from starlette import status

from api.schemas import OptimizeRequest, OptimizeResponse
from api.services.optimizer import optimizer
from api.database import database, fetch_latest_row

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Optimization"])


@router.post("/optimize-oee", response_model=OptimizeResponse)
async def optimize_oee(req: OptimizeRequest):
    logger.info(
        f"POST /optimize-oee rpm={req.current_rpm} oee={req.current_oee}"
        f" target_oee={req.target_oee} target_pct={req.target_oee_percentage}"
        f" horizon={req.horizon}"
    )

    if req.target_oee is None and req.target_oee_percentage is None:
        logger.error("Neither target_oee nor target_oee_percentage provided")
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
                logger.debug(f"Filled missing fields from DB: {avail=} {perf=} {qual=} {downtime=}")
            await database.disconnect()
        except Exception as e:
            logger.error(f"DB fetch failed for missing fields: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot fetch from database and incomplete request",
            )

    logger.debug(f"Starting optimization with horizon={req.horizon} rpm_range=300-500")
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

    logger.info(f"Optimize result: feasible={result['feasible']} optimal_rpm={result['optimal_rpm']} predicted_oee={result['predicted_oee']}")
    return OptimizeResponse(**result)
