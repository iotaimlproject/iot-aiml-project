from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "@iotaimlproject999"
    db_database: str = "lathe"
    db_table: str = "manufacture_ai_data"

    model_dir: str = "ml_model/models"
    scaler_path: str = "ml_model/models/scaler.pkl"

    horizons: list[str] = ["30m", "1h", "2h", "6h", "8h"]
    horizon_minutes: dict[str, int] = {
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "6h": 360,
        "8h": 480,
    }

    rpm_min: int = 300
    rpm_max: int = 500
    rpm_step: int = 5

    api_title: str = "ANN OEE Horizon Prediction API"
    api_version: str = "1.0.0"

    class Config:
        env_file = ".env"


settings = Settings()
