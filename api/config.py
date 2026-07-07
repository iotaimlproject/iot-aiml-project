from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "@iotaimlproject999"
    db_database: str = "lathe"

    model_dir: str = "ml_model/models"
    scaler_m1_path: str = "ml_model/models/scaler_m1.pkl"
    scaler_m2_path: str = "ml_model/models/scaler_m2.pkl"

    speed_levels: list[int] = [20, 40, 60, 80, 100]
    speed_map: dict[int, int] = {20: 0, 40: 1, 60: 2, 80: 3, 100: 4}
    inv_speed_map: dict[int, int] = {0: 20, 1: 40, 2: 60, 3: 80, 4: 100}

    api_title: str = "Physical AI OEE Prediction & Speed Optimization API"
    api_version: str = "2.0.0"

    class Config:
        env_file = ".env"


settings = Settings()
