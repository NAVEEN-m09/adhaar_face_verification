import io
import os
from typing import Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.db_models import AdminUser, VerificationRecord
from app.utils.security_utils import (
    verify_password,
    create_access_token,
    verify_access_token,
    decrypt_text,
    decrypt_file
)
from app.utils.logger import logger
from pydantic import BaseModel

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

def get_current_admin(
    request: Request,
    token: Optional[str] = None,
    db: Session = Depends(get_db)
) -> AdminUser:
    """
    Dependency to authenticate dashboard users.
    Supports standard Bearer Token headers or direct 'token' query parameters.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]

    if not token:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing."
        )

    payload = verify_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token."
        )

    user = db.query(AdminUser).filter(AdminUser.username == payload["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User session is invalid."
        )
    return user

@router.post("/api/login", response_model=LoginResponse)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    API route to authenticate administrative users and issue JWT tokens.
    """
    user = db.query(AdminUser).filter(AdminUser.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/api/records")
def get_records(db: Session = Depends(get_db), current_user: AdminUser = Depends(get_current_admin)):
    """
    API route to list all database records with decrypted sensitive values.
    """
    records = db.query(VerificationRecord).order_by(VerificationRecord.created_at.desc()).all()

    result = []
    for r in records:
        result.append({
            "id": r.id,
            "provided_aadhaar": decrypt_text(r.provided_aadhaar),
            "extracted_aadhaar": decrypt_text(r.extracted_aadhaar) if r.extracted_aadhaar else "N/A",
            "extracted_name": decrypt_text(r.extracted_name) if r.extracted_name else "N/A",
            "third_doc_name": decrypt_text(r.third_doc_name) if r.third_doc_name else "N/A",
            "passbook_acc_num": decrypt_text(r.passbook_acc_num) if r.passbook_acc_num else "N/A",
            "passbook_ifsc": decrypt_text(r.passbook_ifsc) if r.passbook_ifsc else "N/A",
            "passbook_address": decrypt_text(r.passbook_address) if r.passbook_address else "N/A",
            "aadhaar_matched": r.aadhaar_matched,
            "third_doc_name_matched": r.third_doc_name_matched,
            "selfie_similarity": r.selfie_similarity,
            "third_doc_similarity": r.third_doc_similarity,
            "face_matched": r.selfie_similarity >= settings.FACE_SIMILARITY_THRESHOLD if r.selfie_similarity else False,
            "third_doc_matched": r.third_doc_similarity >= settings.FACE_SIMILARITY_THRESHOLD if r.third_doc_similarity else False,
            "status": r.status,
            "error_message": r.error_message,
            "webhook_status": r.webhook_status,
            "webhook_response": r.webhook_response,
            "has_third_doc": bool(r.third_doc_path),
            "created_at": r.created_at.isoformat() if r.created_at else None
        })

    return result

@router.get("/api/records/export")
def export_records(db: Session = Depends(get_db), current_user: AdminUser = Depends(get_current_admin)):
    """
    Export all verification logs as a decrypted Excel sheet.
    """
    records = db.query(VerificationRecord).order_by(VerificationRecord.created_at.desc()).all()

    data = []
    for r in records:
        data.append({
            "Record ID": r.id,
            "Provided Aadhaar": decrypt_text(r.provided_aadhaar),
            "Extracted Aadhaar": decrypt_text(r.extracted_aadhaar) if r.extracted_aadhaar else "N/A",
            "Aadhaar Matched": "MATCHED" if r.aadhaar_matched else "MISMATCH",
            "Extracted Name (Aadhaar)": decrypt_text(r.extracted_name) if r.extracted_name else "N/A",
            "3rd Doc Name": decrypt_text(r.third_doc_name) if r.third_doc_name else "N/A",
            "Passbook Acc No": decrypt_text(r.passbook_acc_num) if r.passbook_acc_num else "N/A",
            "Passbook IFSC": decrypt_text(r.passbook_ifsc) if r.passbook_ifsc else "N/A",
            "Passbook Address": decrypt_text(r.passbook_address) if r.passbook_address else "N/A",
            "3rd Doc Name Matched": "MATCHED" if r.third_doc_name_matched else "MISMATCH" if r.third_doc_name_matched is False else "N/A",
            "Selfie Similarity (%)": f"{r.selfie_similarity:.1f}%" if r.selfie_similarity else "N/A",
            "3rd Doc Similarity (%)": f"{r.third_doc_similarity:.1f}%" if r.third_doc_similarity else "N/A",
            "Webhook Status": r.webhook_status,
            "Webhook Response": r.webhook_response or "None",
            "Overall Status": r.status,
            "Created At": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "N/A"
        })

    df = pd.DataFrame(data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Verification Logs')

    output.seek(0)

    headers = {
        'Content-Disposition': 'attachment; filename="verification_logs.xlsx"',
        'Access-Control-Expose-Headers': 'Content-Disposition'
    }
    return StreamingResponse(
        output,
        headers=headers,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@router.get("/api/records/{record_id}/image/{img_type}")
def get_decrypted_image(
    record_id: str,
    img_type: str,
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_admin)
):
    """
    Proxy endpoint that reads encrypted image files from disk,
    decrypts them in memory, and streams them securely to authenticated clients.
    """
    record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    file_path = None
    if img_type == "selfie":
        file_path = record.selfie_path
    elif img_type == "aadhaar":
        file_path = record.aadhaar_path
    elif img_type == "third":
        file_path = record.third_doc_path
    else:
        raise HTTPException(status_code=400, detail="Invalid image type specifier")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        with open(file_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = decrypt_file(encrypted_data)

        media_type = "image/jpeg"
        if file_path.endswith(".png"):
            media_type = "image/png"

        return Response(content=decrypted_data, media_type=media_type)
    except Exception as e:
        logger.error(f"Error serving decrypted image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error decrypting image resource")

class ReviewActionRequest(BaseModel):
    action: str  # "Approve" or "Reject"

@router.post("/api/records/{record_id}/review")
def review_record(
    record_id: str,
    payload: ReviewActionRequest,
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends(get_current_admin)
):
    """
    API route to manually resolve borderline Amber/Review cases.
    Updates database record status and logs to outputs/manual_reviews.json for reinforcement.
    """
    record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    action = payload.action.strip().capitalize()
    if action not in ["Approve", "Reject"]:
        raise HTTPException(status_code=400, detail="Invalid action. Action must be 'Approve' or 'Reject'.")

    # Update record status in DB
    record.status = "Success" if action == "Approve" else "Failed"
    db.commit()

    # Reinforcement Logging: write to outputs/manual_reviews.json
    import json
    from datetime import datetime
    
    log_file = settings.OUTPUT_DIR / "manual_reviews.json"
    log_entry = {
        "record_id": record_id,
        "action": action,
        "reviewed_by": current_user.username,
        "timestamp": datetime.now().isoformat(),
        "selfie_similarity": record.selfie_similarity,
        "aadhaar_matched": record.aadhaar_matched
    }

    try:
        reviews = []
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    reviews = json.load(f)
                except Exception:
                    reviews = []
        
        reviews.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(reviews, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save manual review reinforcement log: {str(e)}")

    logger.info(f"HITL: Record {record_id} manual review completed. Resolution: {record.status} by {current_user.username}")
    return {"success": True, "status": record.status}

