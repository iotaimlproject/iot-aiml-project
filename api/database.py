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
                (batch_part_no, part_slno, total_batch_size, pos_ratio,
                 availability, performance, quality, current_oee, current_speed,
                 down_time_sec, oee_delta,
                 planned_prod_duration, production_duration, production_delay_sec,
                 pred_oee_1m, pred_oee_1m_applied, confidence_pct,
                 recommended_speed, speed_confidence_pct, speed_change)
            VALUES
                (:batch_part_no, :part_slno, :total_batch_size, :pos_ratio,
                 :availability, :performance, :quality, :current_oee, :current_speed,
                 :down_time_sec, :oee_delta,
                 :planned_prod_duration, :production_duration, :production_delay_sec,
                 :pred_oee_1m, :pred_oee_1m_applied, :confidence_pct,
                 :recommended_speed, :speed_confidence_pct, :speed_change)
        """
        await database.execute(query, {
            "batch_part_no": req.batch_part_no,
            "part_slno": req.part_slno,
            "total_batch_size": req.total_batch_size,
            "pos_ratio": pos_ratio,
            "availability": req.availability,
            "performance": req.performance,
            "quality": req.quality,
            "current_oee": req.current_oee,
            "current_speed": req.current_speed,
            "down_time_sec": req.downtime_sec,
            "oee_delta": req.oee_delta,
            "planned_prod_duration": req.planned_prod_duration,
            "production_duration": req.production_duration,
            "production_delay_sec": req.production_delay_sec,
            "pred_oee_1m": result["pred_oee_1m"],
            "pred_oee_1m_applied": result["pred_oee_1m_applied"],
            "confidence_pct": result["confidence_pct"],
            "recommended_speed": result["recommended_speed"],
            "speed_confidence_pct": result["confidence_speed_pct"],
            "speed_change": result["speed_change"],
        })
    except Exception as e:
        logger.warning(f"MySQL logging failed (non-fatal): {e}")
    finally:
        try:
            await database.disconnect()
        except Exception:
            pass
