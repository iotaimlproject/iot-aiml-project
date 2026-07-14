from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "@iotaimlproject999"
    db_database: str = "lathe"

    model_dir: str = "ml_model/models"

    api_title: str = "Physical AI OEE Prediction & Speed Optimization API"
    api_version: str = "2.1.0"

    class Config:
        env_file = ".env"


settings = Settings()
