import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Aadhaar Face Verification API"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Security Settings
    # 10 MB in bytes
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    ALLOWED_MIME_TYPES: list[str] = ["image/jpeg", "image/png", "image/jpg"]
    ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png"]
    CORS_ORIGINS: list[str] = ["*"]

    # Encryption & Auth Settings
    ENCRYPTION_KEY: str = "gS8S-jOPh2B895p1c_h0l3t0k3n_v3ry_s3cur3_k3y_g3n="  # Can be overridden by env variable
    JWT_SECRET_KEY: str = "supersecretjwtkeyforadminlogindashboardtoken"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # Webhook callback
    WEBHOOK_URL: str = ""

    # Database Settings
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/verification.db"

    # Model Configurations
    YOLO_MODEL_PATH: str = str(BASE_DIR / "app" / "models" / "best.pt")
    # Face Matcher Threshold (Cosine Similarity)
    # Lowered slightly to 0.35 to account for scan patterns on passports/watermarks
    FACE_SIMILARITY_THRESHOLD: float = 0.35

    # Directory Configurations
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # PaddleOCR configuration
    OCR_LANG: str = "en"
    # Set to True if you want PaddleOCR to show debug logs
    OCR_DEBUG: bool = False

    # model_config for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "app" / "models").mkdir(parents=True, exist_ok=True)
