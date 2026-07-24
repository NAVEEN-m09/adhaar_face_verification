import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Aadhaar Face Verification API"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    ALLOWED_MIME_TYPES: list[str] = ["image/jpeg", "image/png", "image/jpg"]
    ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png"]
    CORS_ORIGINS: list[str] = ["*"]

    ENCRYPTION_KEY: str = "gS8S-jOPh2B895p1c_h0l3t0k3n_v3ry_s3cur3_k3y_g3n="
    JWT_SECRET_KEY: str = "supersecretjwtkeyforadminlogindashboardtoken"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    WEBHOOK_URL: str = ""

    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/verification.db"

    YOLO_MODEL_PATH: str = str(BASE_DIR / "app" / "models" / "best.pt")
    FACE_SIMILARITY_THRESHOLD: float = 0.35
    LIVENESS_THRESHOLD: float = 0.90

    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    OCR_LANG: str = "en"
    OCR_DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Validate that secrets are overridden in production environments
if not settings.DEBUG:
    import warnings
    if settings.ENCRYPTION_KEY == "gS8S-jOPh2B895p1c_h0l3t0k3n_v3ry_s3cur3_k3y_g3n=":
        warnings.warn("Security Warning: Using default hardcoded fallback value for ENCRYPTION_KEY in production mode!")
    if settings.JWT_SECRET_KEY == "supersecretjwtkeyforadminlogindashboardtoken":
        warnings.warn("Security Warning: Using default hardcoded fallback value for JWT_SECRET_KEY in production mode!")

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "app" / "models").mkdir(parents=True, exist_ok=True)
