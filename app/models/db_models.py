import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, DateTime
from app.database import Base

class AdminUser(Base):
    """
    Schema representing dashboard administrators.
    """
    __tablename__ = "admin_users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class VerificationRecord(Base):
    """
    Schema for storing encrypted verification results, similarities, and file reference paths.
    """
    __tablename__ = "verification_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Encrypted fields
    provided_aadhaar = Column(String(255), nullable=False)
    extracted_aadhaar = Column(String(255), nullable=True)
    extracted_name = Column(String(255), nullable=True)
    third_doc_name = Column(String(255), nullable=True) # Name extracted from 3rd document
    
    # Match flags
    aadhaar_matched = Column(Boolean, default=False)
    third_doc_name_matched = Column(Boolean, default=None, nullable=True)
    
    # Path to local encrypted files
    selfie_path = Column(String(255), nullable=False)
    aadhaar_path = Column(String(255), nullable=False)
    third_doc_path = Column(String(255), nullable=True) # Optional 3rd document path (Passbook/Passport)
    
    # Similarity Metrics
    selfie_similarity = Column(Float, nullable=True)      # Cosine similarity between selfie and Aadhaar photo
    third_doc_similarity = Column(Float, nullable=True)   # Cosine similarity between selfie and 3rd document photo
    
    # State and logs
    status = Column(String(50), default="Pending")       # Success, Failed, Error
    error_message = Column(String(255), nullable=True)
    
    # Webhook callback log details
    webhook_status = Column(String(50), default="Pending") # Pending, Sent, Failed, Skipped
    webhook_response = Column(Float, nullable=True)        # Response status code
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
