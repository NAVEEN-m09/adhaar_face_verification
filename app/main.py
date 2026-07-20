from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from app.config import settings
from app.utils.logger import logger
from app.database import engine, Base, SessionLocal
from app.models.db_models import AdminUser
from app.utils.security_utils import hash_password
from app.api.verify import router as verify_router
from app.api.dashboard import router as dashboard_router
from app.middleware.limit_upload import LimitUploadSizeMiddleware
from app.services.detector import AadhaarDetector
from app.services.perspective import PerspectiveCorrector
from app.services.photo_cropper import PhotoCropper
from app.services.ocr import AadhaarOCR
from app.services.regex_validator import RegexValidator
from app.services.face_matcher import FaceMatcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event to configure database tables, seed default credentials,
    load heavy machine learning models, and execute pre-warming passes.
    """
    logger.info("Initializing database schemas...")
    try:
        # Create all tables in SQLite
        Base.metadata.create_all(bind=engine)
        
        # Seed default admin user if none exists
        db = SessionLocal()
        try:
            admin_user = db.query(AdminUser).filter(AdminUser.username == "admin").first()
            if not admin_user:
                logger.info("Seeding default administrator credentials ('admin' / 'admin123')...")
                hashed_pw = hash_password("admin123")
                new_admin = AdminUser(username="admin", hashed_password=hashed_pw)
                db.add(new_admin)
                db.commit()
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

    logger.info("Starting up server and loading models...")
    try:
        # Load and cache models in app.state
        app.state.detector = AadhaarDetector()
        app.state.perspective = PerspectiveCorrector()
        app.state.photo_cropper = PhotoCropper()
        app.state.ocr = AadhaarOCR()
        app.state.regex = RegexValidator()
        app.state.face_matcher = FaceMatcher()
        
        # Warm up models to eliminate first-request cold-start latency
        logger.info("Pre-warming models to eliminate cold-start latency...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Warm YOLO Card detector
        _, _, _ = app.state.detector.detect(dummy_img)
        # Warm PaddleOCR text extractor
        _ = app.state.ocr.extract_text(dummy_img)
        # Warm InsightFace embeddings
        _, _ = app.state.face_matcher.get_embedding(dummy_img, "dummy")
        
        logger.info("All models loaded and pre-warmed successfully. Server ready.")
    except Exception as e:
        logger.critical(f"Failed to load models during startup: {str(e)}", exc_info=True)
        
    yield
    logger.info("Shutting down server and releasing models...")

app = FastAPI(
    title="Aadhaar & KYC Face Verification API",
    description="Production-grade secure identity verification and document matching API.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Size limiting middleware
app.add_middleware(LimitUploadSizeMiddleware)

# Register routes
app.include_router(verify_router)
app.include_router(dashboard_router)

# Serve SPA Frontend
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
