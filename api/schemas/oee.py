from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    batch_part_no: str = Field(..., description="Batch identifier, e.g. WM-23-A-1_b7")
    part_slno: int = Field(..., ge=1, description="Current part index within batch (1-based)")
    total_batch_size: int = Field(..., ge=1, description="Total parts planned for this batch")
    current_speed: int = Field(..., ge=20, le=100)
    availability: int = Field(..., ge=0, le=100)
    performance: int = Field(..., ge=0, le=100)
    quality: int = Field(..., ge=0, le=100)
    current_oee: int = Field(..., ge=0, le=100)
    downtime_sec: int = Field(default=0, ge=0)
    oee_delta: int = Field(default=0, ge=-100, le=100)
    state: str = Field(default="NORMAL", description="Machine state")


class PredictResponse(BaseModel):
    pred_oee_10m: float = Field(..., description="Predicted OEE 10 minutes ahead (0-100)")
    confidence_pct: float = Field(..., description="Confidence in prediction (0-100%)")
    recommended_speed: int = Field(..., description="Optimal conveyor speed (%)")
    confidence_speed_pct: float = Field(..., description="Confidence in speed recommendation (0-100%)")
    change_needed: bool = Field(..., description="True if recommended speed differs from current")
    batch_position: str = Field(..., description="e.g. '5/12'")
