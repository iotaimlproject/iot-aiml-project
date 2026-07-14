from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    batch_part_no: str = Field(..., description="Batch identifier, e.g. WM-23-A-1_b7")
    part_slno: int = Field(..., ge=1, description="Current part index within batch (1-based)")
    total_batch_size: int = Field(..., ge=1, description="Total parts planned for this batch")
    current_speed: int = Field(..., ge=1, le=10)
    availability: int = Field(..., ge=0, le=100)
    performance: int = Field(..., ge=0, le=100)
    quality: int = Field(..., ge=0, le=100)
    current_oee: int = Field(..., ge=0, le=100)
    downtime_sec: int = Field(default=0, ge=0)
    oee_delta: int = Field(default=0, ge=-100, le=100)
    planned_prod_duration: int = Field(default=0, ge=0, description="Cumulative planned production duration (seconds)")
    production_duration: int = Field(default=0, ge=0, description="Cumulative actual production duration (seconds)")
    production_delay_sec: int = Field(default=0, ge=0, description="Cumulative delay = actual - planned (seconds)")


class PredictResponse(BaseModel):
    pred_oee_1m: float = Field(..., description="Predicted OEE 1 minute ahead at current speed (0-100)")
    pred_oee_1m_applied: float = Field(..., description="Predicted OEE if recommended speed is applied (0-100)")
    confidence_pct: float = Field(..., description="Confidence in prediction (0-100%)")
    recommended_speed: int = Field(..., ge=10, le=100, description="Optimal conveyor speed (10-100%)")
    confidence_speed_pct: float = Field(..., description="Confidence in speed recommendation (0-100%)")
    speed_change: str = Field(..., description="increase | decrease | none")
    batch_position: str = Field(..., description="e.g. '5/12'")


class TimeFMPredictRequest(BaseModel):
    batch_part_no: str = Field(..., description="Batch identifier, e.g. WM-23-A-1_b7")
    part_slno: int = Field(..., ge=1, description="Current part index within batch (1-based)")
    total_batch_size: int = Field(..., ge=1, description="Total parts planned for this batch")
    current_speed: int = Field(..., ge=1, le=10)
    availability: int = Field(..., ge=0, le=100)
    performance: int = Field(..., ge=0, le=100)
    quality: int = Field(..., ge=0, le=100)
    current_oee: int = Field(..., ge=0, le=100)
    downtime_sec: int = Field(default=0, ge=0)
    oee_delta: int = Field(default=0, ge=-100, le=100)


class TimeFMPredictResponse(BaseModel):
    pred_oee_10m: float = Field(..., description="Predicted OEE at horizon")
    forecasted_oee: list[float] = Field(..., description="Full forecast OEE trajectory")
    confidence_pct: float = Field(..., description="OEE prediction confidence (0-100%)")
    recommended_speed: int = Field(..., ge=10, le=100, description="Optimal conveyor speed")
    confidence_speed_pct: float = Field(..., description="Speed recommendation confidence (0-100%)")
    speed_change: str = Field(..., description="increase | decrease | none")
    batch_position: str = Field(..., description="e.g. '5/12'")
