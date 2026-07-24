from app.utils.gpu_setup import setup_gpu_dlls
setup_gpu_dlls()

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import numpy as np
import cv2

from app.main import app
from app.config import settings

_, encoded_img = cv2.imencode('.png', np.zeros((10, 10, 3), dtype=np.uint8))
VALID_PNG_BYTES = encoded_img.tobytes()

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    from sqlalchemy import text
    from app.database import engine, Base
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        for col in ["passbook_acc_num", "passbook_ifsc", "passbook_address"]:
            try:
                conn.execute(text(f"ALTER TABLE verification_records ADD COLUMN {col} VARCHAR(255)"))
                conn.commit()
            except Exception:
                pass
    yield

@pytest.fixture
def mock_app_state():
    """
    Mock the machine learning services stored in app.state to avoid loading
    massive models during unit testing.
    """
    mock_detector = MagicMock()
    mock_perspective = MagicMock()
    mock_photo_cropper = MagicMock()
    mock_ocr = MagicMock()
    mock_regex = MagicMock()
    mock_face_matcher = MagicMock()

    mock_detector.detect.return_value = (
        np.zeros((100, 200, 3), dtype=np.uint8),
        np.zeros((30, 30, 3), dtype=np.uint8),
        {"aadhaar_card_detected": True, "aadhaar_photo_detected": True, "fallback_used": False}
    )
    mock_perspective.correct.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_photo_cropper.crop_photo.return_value = np.zeros((30, 30, 3), dtype=np.uint8)
    mock_ocr.extract_text.return_value = ["Government of India", "3662 1019 8051"]
    mock_regex.extract_aadhaar_number.return_value = "366210198051"
    mock_regex.verify_match.return_value = (True, "Matched")
    mock_face_matcher.match_faces.return_value = {
        "success": True,
        "similarity": 88.50,
        "cosine_similarity": 0.77,
        "confidence": 97.00,
        "matched": True,
        "selfie_face_detected": True,
        "aadhaar_face_detected": True,
        "selfie_age": 28.0,
        "card_photo_age": 25.0
    }

    mock_liveness = MagicMock()
    mock_liveness.check_liveness.return_value = (True, 0.95)

    mock_layout_classifier = MagicMock()
    mock_layout_classifier.classify.return_value = "front"

    mock_qr_decoder = MagicMock()
    mock_qr_decoder.decode.return_value = None

    app.state.detector = mock_detector
    app.state.perspective = mock_perspective
    app.state.photo_cropper = mock_photo_cropper
    app.state.ocr = mock_ocr
    app.state.regex = mock_regex
    app.state.face_matcher = mock_face_matcher
    app.state.liveness = mock_liveness
    app.state.layout_classifier = mock_layout_classifier
    app.state.qr_decoder = mock_qr_decoder

    return {
        "detector": mock_detector,
        "perspective": mock_perspective,
        "photo_cropper": mock_photo_cropper,
        "ocr": mock_ocr,
        "regex": mock_regex,
        "face_matcher": mock_face_matcher,
        "liveness": mock_liveness,
        "layout_classifier": mock_layout_classifier,
        "qr_decoder": mock_qr_decoder
    }

def test_root_route():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Aadhaar Identity Verification" in response.text

def test_verify_endpoint_success(mock_app_state):
    client = TestClient(app)

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert res_json["face_match"]["matched"] is True
    assert res_json["face_match"]["similarity"] == 88.50
    assert res_json["aadhaar"]["provided"] == "3662 1019 8051"
    assert res_json["aadhaar"]["extracted"] == "366210198051"
    assert res_json["aadhaar"]["matched"] is True

def test_verify_endpoint_liveness_failure(mock_app_state):
    client = TestClient(app)
    mock_app_state["liveness"].check_liveness.return_value = (False, 0.45)

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 400
    assert "Liveness check failed" in response.json()["error"]

def test_verify_endpoint_invalid_file_type(mock_app_state):
    client = TestClient(app)

    files = {
        "selfie_image": ("selfie.txt", b"fake_selfie_data", "text/plain"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 415
    assert "unsupported" in response.json()["detail"].lower()

def test_verify_endpoint_request_size_limit(mock_app_state):
    client = TestClient(app)

    large_data = b"0" * (settings.MAX_FILE_SIZE + 1024)

    files = {
        "selfie_image": ("selfie.jpg", large_data, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 413
    assert response.json()["success"] is False
    assert "exceeds" in response.json()["error"].lower()

def test_admin_login_success():
    client = TestClient(app)
    from app.database import SessionLocal
    from app.models.db_models import AdminUser
    from app.utils.security_utils import hash_password

    db = SessionLocal()
    try:
        user = db.query(AdminUser).filter(AdminUser.username == "test_admin").first()
        if not user:
            user = AdminUser(username="test_admin", hashed_password=hash_password("password123"))
            db.add(user)
            db.commit()
    finally:
        db.close()

    response = client.post("/api/login", json={"username": "test_admin", "password": "password123"})
    assert response.status_code == 200
    res_json = response.json()
    assert "access_token" in res_json
    assert res_json["token_type"] == "bearer"

def test_admin_login_fail():
    client = TestClient(app)
    response = client.post("/api/login", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401

def test_get_records_unauthorized():
    client = TestClient(app)
    response = client.get("/api/records")
    assert response.status_code == 401

def test_get_records_authorized():
    client = TestClient(app)
    login_res = client.post("/api/login", json={"username": "test_admin", "password": "password123"})
    token = login_res.json()["access_token"]

    from app.database import SessionLocal
    from app.models.db_models import VerificationRecord
    from app.utils.security_utils import encrypt_text
    import uuid

    record_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        rec = VerificationRecord(
            id=record_id,
            provided_aadhaar=encrypt_text("123456789012"),
            extracted_aadhaar=encrypt_text("123456789012"),
            extracted_name=encrypt_text("John Doe"),
            selfie_path="selfie.bin",
            aadhaar_path="aadhaar.bin",
            status="Success",
            webhook_status="Skipped"
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/records", headers=headers)
    assert response.status_code == 200
    records = response.json()

    target_rec = next((r for r in records if r["id"] == record_id), None)
    assert target_rec is not None
    assert target_rec["provided_aadhaar"] == "123456789012"
    assert target_rec["extracted_name"] == "John Doe"

def test_verify_identity_async(mock_app_state):
    """
    Verifies that the /verify-async endpoint enqueues verification successfully,
    returning a 202 Accepted response with state Pending.
    """
    client = TestClient(app)
    files = {
        "selfie_image": ("selfie.png", VALID_PNG_BYTES, "image/png"),
        "aadhaar_image": ("aadhaar.png", VALID_PNG_BYTES, "image/png")
    }
    data = {
        "aadhaar_number": "366210198051",
        "callback_url": "http://example.com/callback"
    }

    response = client.post("/verify-async", files=files, data=data)
    assert response.status_code == 202
    res_data = response.json()
    assert res_data["success"] is True
    assert "record_id" in res_data
    assert res_data["status"] == "Pending"

def test_export_records(mock_app_state):
    """
    Verifies that authenticated clients can retrieve decrypted Excel exports of logs.
    """
    client = TestClient(app)

    response = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    token = response.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/records/export", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert "attachment" in response.headers["content-disposition"]
    assert "verification_logs.xlsx" in response.headers["content-disposition"]

def test_verify_identity_back_layout_rejection(mock_app_state):
    client = TestClient(app)
    mock_app_state["layout_classifier"].classify.return_value = "back"

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 400
    assert "Back side of Aadhaar card detected" in response.json()["error"]

def test_verify_identity_qr_fallback(mock_app_state):
    client = TestClient(app)
    mock_app_state["qr_decoder"].decode.return_value = {
        "aadhaar_number": "548984365730",
        "name": "TEST USER QR"
    }

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "5489 8436 5730"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["aadhaar"]["extracted"] == "548984365730"
    assert res_json["aadhaar"]["extracted_name"] == "TEST USER QR"

def test_verify_identity_childhood_photo_calibration(mock_app_state):
    client = TestClient(app)
    mock_app_state["face_matcher"].match_faces.return_value = {
        "success": True,
        "similarity": 65.0,
        "cosine_similarity": 0.30,
        "confidence": 95.0,
        "matched": False, # Raw matched is false since 0.30 < 0.35 threshold
        "selfie_face_detected": True,
        "aadhaar_face_detected": True,
        "selfie_age": 28.0,
        "card_photo_age": 8.0
    }

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    # The output matches because threshold was dynamically adjusted to 0.28, and 0.30 >= 0.28!
    assert res_json["secondary_id_required"] is True

def test_verify_identity_childhood_photo_borderline_secondary_id(mock_app_state):
    client = TestClient(app)
    mock_app_state["face_matcher"].match_faces.return_value = {
        "success": True,
        "similarity": 63.0,
        "cosine_similarity": 0.26,
        "confidence": 95.0,
        "matched": False,
        "selfie_face_detected": True,
        "aadhaar_face_detected": True,
        "selfie_age": 28.0,
        "card_photo_age": 8.0
    }

    files = {
        "selfie_image": ("selfie.jpg", VALID_PNG_BYTES, "image/jpeg"),
        "aadhaar_image": ("aadhaar.jpg", VALID_PNG_BYTES, "image/jpeg"),
    }
    data = {
        "aadhaar_number": "3662 1019 8051"
    }

    response = client.post("/verify", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["secondary_id_required"] is True

def test_manual_adjudication_route():
    from app.database import SessionLocal
    from app.models.db_models import VerificationRecord
    import uuid

    client = TestClient(app)
    db = SessionLocal()

    # 1. Create a borderline record in Review status
    rec_id = str(uuid.uuid4())
    record = VerificationRecord(
        id=rec_id,
        provided_aadhaar="encrypted",
        extracted_aadhaar="encrypted",
        status="Review",
        selfie_similarity=33.0,
        aadhaar_matched=True,
        selfie_path="dummy_selfie.bin",
        aadhaar_path="dummy_aadhaar.bin"
    )
    db.add(record)
    db.commit()
    db.close()

    # 2. Log in as admin
    login_resp = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    # 3. Adjudicate review
    review_resp = client.post(
        f"/api/records/{rec_id}/review",
        json={"action": "Approve"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["success"] is True

    # 4. Check db update
    db = SessionLocal()
    updated = db.query(VerificationRecord).filter(VerificationRecord.id == rec_id).first()
    assert updated.status == "Success"
    db.delete(updated)
    db.commit()
    db.close()





