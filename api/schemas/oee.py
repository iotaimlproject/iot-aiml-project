from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    current_time: str | None = Field(None, description="ISO timestamp")
    current_rpm: int = Field(..., ge=300, le=500)
    current_oee: float = Field(..., ge=0, le=100)
    availability: float | None = Field(None, ge=0, le=100)
    performance: float | None = Field(None, ge=0, le=100)
    quality: float | None = Field(None, ge=0, le=100)
    downtime_minutes: float | None = Field(None, ge=0)


class HorizonPredictions(BaseModel):
    pred_30m: float
    pred_1h: float
    pred_2h: float
    pred_6h: float
    pred_8h: float


class PredictResponse(BaseModel):
    current_rpm: int
    current_oee: float
    predictions: HorizonPredictions


class OptimizeRequest(BaseModel):
    target_oee: float | None = Field(None, ge=0, le=100)
    target_oee_percentage: float | None = Field(None, ge=0, le=100)
    horizon: str = Field(default="1h", pattern=r"^(30m|1h|2h|6h|8h)$")
    current_rpm: int = Field(..., ge=300, le=500)
    current_oee: float = Field(..., ge=0, le=100)
    availability: float | None = Field(None, ge=0, le=100)
    performance: float | None = Field(None, ge=0, le=100)
    quality: float | None = Field(None, ge=0, le=100)
    downtime_minutes: float | None = Field(None, ge=0)


class OptimizeResponse(BaseModel):
    current_oee: float
    target_oee: float
    optimal_rpm: int
    optimal_hz: float
    predicted_oee: float
    feasible: bool
    search_range: dict[str, int]
