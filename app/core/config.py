import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Tennis Vision API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Directories
    UPLOAD_DIR: str = "data/uploads"
    RESULTS_DIR: str = "data/results"
    MODELS_DIR: str = "models"

    # Model & Vision
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    BALL_CONF_THRESHOLD: float = 0.20
    TRACKER_BUFFER_SIZE: int = 30

    # ITF Standard Court Dimensions (in meters)
    # Origin (0,0) is at the center of the court (at the net)
    COURT_LENGTH: float = 23.77       # Total length: 78 feet
    COURT_SINGLES_WIDTH: float = 8.23  # Singles width: 27 feet
    COURT_DOUBLES_WIDTH: float = 10.97 # Doubles width: 36 feet
    SERVICE_LINE_DIST: float = 6.40    # Distance from net to service line: 21 feet
    NET_HEIGHT_CENTER: float = 0.914   # 3 feet
    BALL_RADIUS_METERS: float = 0.033  # ~3.3 cm (ITF standard diameter ~6.54-6.86 cm)

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "*"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
