import logging

from databases import Database

from api.config import settings
from api.schemas import PredictRequest

logger = logging.getLogger(__name__)

database = Database(
    f"mysql+asyncmy://{settings.db_user}:{settings.db_password}@"
    f"{settings.db_host}:{settings.db_port}/{settings.db_database}"
)


async def log_prediction(req: PredictRequest, result: dict) -> None:
    try:
        await database.connect()
        pos_ratio = round(req.part_slno / req.total_batch_size, 4)
        query = """
            INSERT INTO predictions_log
                (batch_part_no, part_slno, pos_ratio, current_oee, current_speed,
                 pred_oee_10m, confidence_pct, recommended_speed,
                 speed_confidence_pct, change_needed)
            VALUES (:batch_part_no, :part_slno, :pos_ratio, :current_oee, :current_speed,
                    :pred_oee_10m, :confidence_pct, :recommended_speed,
                    :speed_confidence_pct, :change_needed)
        """
        await database.execute(query, {
            "batch_part_no": req.batch_part_no,
            "part_slno": req.part_slno,
            "pos_ratio": pos_ratio,
            "current_oee": req.current_oee,
            "current_speed": req.current_speed,
            "pred_oee_10m": result["pred_oee_10m"],
            "confidence_pct": result["confidence_pct"],
            "recommended_speed": result["recommended_speed"],
            "speed_confidence_pct": result["confidence_speed_pct"],
            "change_needed": int(result["change_needed"]),
        })
    except Exception as e:
        logger.warning(f"MySQL logging failed (non-fatal): {e}")
    finally:
        try:
            await database.disconnect()
        except Exception:
            pass
