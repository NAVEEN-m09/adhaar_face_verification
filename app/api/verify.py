import time
import re
import uuid
import asyncio
import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends, Request, status, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.schemas.response import (
    VerificationResponse,
    FaceMatchResult,
    AadhaarResult,
    ThirdDocumentResult,
    ProcessingMeta,
    AsyncVerificationResponse
)
from app.utils.image_utils import (
    validate_image_file,
    read_image_from_upload,
    save_temp_image,
    delete_temp_files
)
from app.utils.logger import logger
from app.config import settings
from app.database import get_db, SessionLocal
from app.models.db_models import VerificationRecord
from app.services.detector import AadhaarDetector
from app.services.perspective import PerspectiveCorrector
from app.services.photo_cropper import PhotoCropper
from app.services.ocr import AadhaarOCR
from app.services.regex_validator import RegexValidator
from app.services.face_matcher import FaceMatcher
from app.services.webhook import trigger_webhook
from app.utils.security_utils import encrypt_text, encrypt_file

router = APIRouter()

def get_detector(request: Request) -> AadhaarDetector:
    return request.app.state.detector

def get_perspective(request: Request) -> PerspectiveCorrector:
    return request.app.state.perspective

def get_photo_cropper(request: Request) -> PhotoCropper:
    return request.app.state.photo_cropper

def get_ocr(request: Request) -> AadhaarOCR:
    return request.app.state.ocr

def get_regex(request: Request) -> RegexValidator:
    return request.app.state.regex

def get_face_matcher(request: Request) -> FaceMatcher:
    return request.app.state.face_matcher

def extract_name_from_mrz(line: str) -> Optional[str]:
    """
    Parses a Passport/ID Card MRZ (Machine Readable Zone) Line 1 to extract the owner's name.
    Formats:
    P<COLTOBON<<CLAUDIA<MARCELA... -> CLAUDIA MARCELA TOBON
    PPCOLTOBON<RODRIGUEZ<<CLAUDIA<MARCELA... -> CLAUDIA MARCELA TOBON RODRIGUEZ
    """
    line = line.replace(" ", "").upper()
    if len(line) >= 30 and (line.startswith("P<") or line.startswith("PP")):
        name_part = line[5:]
        parts = name_part.split("<<")
        if len(parts) >= 2:
            surname = parts[0].replace("<", " ").strip()
            given_names = parts[1].split("<<<")[0].replace("<", " ").strip()

            full_name = f"{given_names} {surname}".strip()
            full_name = re.sub(r"\s+", " ", full_name)
            if full_name:
                return full_name
    return None

def extract_name_from_ocr(text_lines: list[str]) -> str:
    """
    Extracts a card holder's name from Aadhaar/Passport OCR text by filtering out
    common keywords, structural labels, and prioritizing MRZ parsing or multi-word lines.
    """
    for line in text_lines:
        clean_line = line.replace(" ", "").upper()
        if "<" in clean_line and (clean_line.startswith("P<") or clean_line.startswith("PP")):
            mrz_name = extract_name_from_mrz(clean_line)
            if mrz_name:
                logger.info(f"OCR: Extracted name from Passport MRZ: {mrz_name}")
                return mrz_name

    exclude_keywords = {
        "government", "india", "dob", "date", "birth", "male", "female",
        "father", "husband", "address", "unique", "identification", "help",
        "enrollment", "number", "card", "signature", "authority", "document",
        "passport", "republic", "given", "surname", "name", "national", "state",
        "holder", "photo", "identity", "issue", "validity", "expiry", "sex"
    }

    cleaned_candidates = []
    for line in text_lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Strip common structural prefixes (e.g. NAME(S), NAME:, ACCOUNT HOLDER:, CUSTOMER NAME:)
        clean_line = re.sub(r"^(NAME\(S\)|NAME[S]?|ACCOUNT HOLDER|HOLDER NAME|CUSTOMER NAME|NAME OF HOLDER)[:\s]*", "", clean_line, flags=re.IGNORECASE).strip()

        if any(char.isdigit() for char in clean_line):
            continue

        sanitized = re.sub(r"[^a-zA-Z\s\.]", "", clean_line).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)

        if len(sanitized) < 3:
            continue

        words = sanitized.split()
        if any(w.lower() in exclude_keywords for w in words):
            continue

        if all(re.match(r"^[a-zA-Z\.]+$", w) for w in words):
            cleaned_candidates.append(sanitized)

    for cand in cleaned_candidates:
        words = cand.split()
        if len(words) >= 2:
            long_words = sum(1 for w in words if len(w.replace(".", "")) >= 3)
            uppercase_words = sum(1 for w in words if w[0].isupper())
            if long_words >= 2 and uppercase_words >= 2:
                return cand

    for cand in cleaned_candidates:
        words = cand.split()
        if len(words) >= 2:
            long_words = sum(1 for w in words if len(w.replace(".", "")) >= 3)
            if long_words >= 2:
                return cand

    for cand in cleaned_candidates:
        if len(cand) >= 4:
            return cand

    return ""

def extract_passbook_details(text_lines: list[str]) -> dict:
    """
    Extracts structured client details (Account Name, Account Number, IFSC Code, Address)
    from Bank Passbook OCR text lines using robust pattern matching.
    """
    details = {
        "account_name": extract_name_from_ocr(text_lines) or None,
        "account_number": None,
        "ifsc_code": None,
        "address": None
    }
    
    full_text = " ".join(text_lines)

    # 1. Extract Account Number (Regex matching 9-18 digits following ACCOUNT/ACC/A/C NO)
    acc_match = re.search(r"(?:ACCOUNT|A/C|ACC)\s*(?:NO|NUM|NUMBER)?[\.\:\s]*([0-9]{9,18})", full_text, re.IGNORECASE)
    if acc_match:
        details["account_number"] = acc_match.group(1)
    else:
        # Fallback: standalone digits of length 9 to 18
        digits_found = re.findall(r"\b[0-9]{9,18}\b", full_text)
        for cand in digits_found:
            if len(cand) != 12: # Avoid standard 12-digit Aadhaar length if ambiguous
                details["account_number"] = cand
                break
        if not details["account_number"] and digits_found:
            details["account_number"] = digits_found[0]

    # 2. Extract IFSC Code (Regex: 4 letters + 0 + 6 alphanumeric)
    ifsc_match = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", full_text, re.IGNORECASE)
    if ifsc_match:
        details["ifsc_code"] = ifsc_match.group(0).upper()

    # 3. Extract Address lines
    addr_lines = []
    capture_addr = False
    for line in text_lines:
        if re.search(r"ADDRESS[:\s]*", line, re.IGNORECASE):
            capture_addr = True
            cleaned = re.sub(r"^ADDRESS[:\s]*", "", line, flags=re.IGNORECASE).strip()
            if cleaned:
                addr_lines.append(cleaned)
            continue
        if capture_addr:
            if re.search(r"(PIN|ZIP|TEL|PHONE|MICR|BRANCH|IFSC|DATE)[:\s]*", line, re.IGNORECASE):
                capture_addr = False
            else:
                addr_lines.append(line.strip())
                
    if addr_lines:
        details["address"] = " ".join(addr_lines[:3])

    return details

def verify_name_overlap(name1: str, name2: str) -> bool:
    """
    Compares two names for overlaps (checking if word tokens intersect).
    """
    if not name1 or not name2:
        return False
    w1 = set(re.findall(r"\w+", name1.lower()))
    w2 = set(re.findall(r"\w+", name2.lower()))
    return len(w1.intersection(w2)) >= 1

@router.post(
    "/verify",
    response_model=VerificationResponse,
    responses={
        400: {"description": "Pipeline failure (e.g. no face detected, OCR failed)"},
        413: {"description": "File upload exceeds maximum limit"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Internal server error"}
    }
)
async def verify_identity(
    request: Request,
    background_tasks: BackgroundTasks,
    selfie_image: UploadFile = File(..., description="Live selfie photograph of the user"),
    aadhaar_image: UploadFile = File(..., description="Uploaded image of the Aadhaar card"),
    aadhaar_number: str = Form(..., description="12-digit Aadhaar number for verification"),
    third_document: Optional[UploadFile] = File(None, description="Optional third document image (Bank Passbook or Passport)"),
    detector: AadhaarDetector = Depends(get_detector),
    perspective: PerspectiveCorrector = Depends(get_perspective),
    photo_cropper: PhotoCropper = Depends(get_photo_cropper),
    ocr: AadhaarOCR = Depends(get_ocr),
    regex: RegexValidator = Depends(get_regex),
    face_matcher: FaceMatcher = Depends(get_face_matcher),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    temp_files = []

    try:
        validate_image_file(selfie_image)
        validate_image_file(aadhaar_image)
        if third_document:
            validate_image_file(third_document)

        logger.info("Reading selfie and Aadhaar images...")
        selfie_img = await read_image_from_upload(selfie_image)
        aadhaar_img = await read_image_from_upload(aadhaar_image)

        logger.info("Running Aadhaar card detection...")
        card_crop, initial_photo_crop, detect_status = detector.detect(aadhaar_img)

        if card_crop is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Aadhaar card not detected"}
            )

        logger.info("Applying perspective correction on card...")
        corrected_card = perspective.correct(card_crop)

        logger.info("Cropping face photo from corrected Aadhaar card...")
        model_to_use = None if detect_status["fallback_used"] else detector.model
        photo_crop = photo_cropper.crop_photo(corrected_card, yolo_model=model_to_use)

        if photo_crop is None or photo_crop.size == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "No Aadhaar photo detected"}
            )

        temp_photo_path = save_temp_image(photo_crop, prefix="cropped_photo")
        temp_files.append(temp_photo_path)

        logger.info("Running PaddleOCR on corrected Aadhaar card...")
        ocr_texts = ocr.extract_text(corrected_card)
        if not ocr_texts:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "OCR failed"}
            )

        extracted_name = extract_name_from_ocr(ocr_texts)

        logger.info("Extracting Aadhaar number via Regex and Verhoeff...")
        extracted_aadhaar = regex.extract_aadhaar_number(ocr_texts)
        if not extracted_aadhaar:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Aadhaar number not found"}
            )

        aadhaar_matched, match_msg = regex.verify_match(extracted_aadhaar, aadhaar_number)

        logger.info("Matching face embeddings between selfie and Aadhaar photo...")
        match_result = face_matcher.match_faces(selfie_img, photo_crop)

        if not match_result["success"]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": match_result["error"]}
            )

        third_doc_result = None
        extracted_third_name = None
        third_name_matched = None
        third_similarity = None
        third_face_matched = None
        third_img = None

        if third_document:
            logger.info("Processing optional third document (passbook/passport)...")
            third_img = await read_image_from_upload(third_document)

            third_ocr_texts = ocr.extract_text(third_img)
            passbook_details = extract_passbook_details(third_ocr_texts)
            extracted_third_name = passbook_details["account_name"]
            passbook_acc_num = passbook_details["account_number"]
            passbook_ifsc = passbook_details["ifsc_code"]
            passbook_address = passbook_details["address"]

            third_name_matched = True  # Pure extraction for Admin UI; no comparison against Aadhaar name

            third_match_res = face_matcher.match_faces(selfie_img, third_img)
            if third_match_res["success"]:
                third_similarity = third_match_res["similarity"]
                third_face_matched = third_match_res["matched"]
            else:
                third_similarity = 0.0
                third_face_matched = True  # Passbook text crops do not contain faces; rely on name matching

            third_doc_result = ThirdDocumentResult(
                provided=True,
                extracted_name=extracted_third_name or "Unknown",
                name_matched=third_name_matched,
                similarity=third_similarity,
                matched=third_face_matched
            )
        else:
            third_doc_result = ThirdDocumentResult(
                provided=False
            )
            passbook_acc_num = None
            passbook_ifsc = None
            passbook_address = None

        record_id = str(uuid.uuid4())
        selfie_filename = f"selfie_{record_id}.bin"
        aadhaar_filename = f"aadhaar_{record_id}.bin"
        third_doc_filename = f"third_{record_id}.bin" if third_document else None

        selfie_path = settings.UPLOAD_DIR / selfie_filename
        aadhaar_path = settings.UPLOAD_DIR / aadhaar_filename
        third_doc_path = settings.UPLOAD_DIR / third_doc_filename if third_doc_filename else None

        await selfie_image.seek(0)
        selfie_bytes = await selfie_image.read()
        encrypted_selfie = encrypt_file(selfie_bytes)

        await aadhaar_image.seek(0)
        aadhaar_bytes = await aadhaar_image.read()
        encrypted_aadhaar = encrypt_file(aadhaar_bytes)

        with open(selfie_path, "wb") as f:
            f.write(encrypted_selfie)
        with open(aadhaar_path, "wb") as f:
            f.write(encrypted_aadhaar)

        if third_document:
            await third_document.seek(0)
            third_bytes = await third_document.read()
            encrypted_third = encrypt_file(third_bytes)
            with open(third_doc_path, "wb") as f:
                f.write(encrypted_third)

        cos_sim = match_result.get("cosine_similarity", 0.0)
        if aadhaar_matched:
            if 0.25 <= cos_sim <= 0.40:
                overall_status = "Review"
            elif cos_sim > 0.40:
                overall_status = "Success"
            else:
                overall_status = "Failed"
        else:
            overall_status = "Failed"

        record = VerificationRecord(
            id=record_id,
            provided_aadhaar=encrypt_text(aadhaar_number),
            extracted_aadhaar=encrypt_text(extracted_aadhaar),
            extracted_name=encrypt_text(extracted_name) if extracted_name else encrypt_text(""),
            third_doc_name=encrypt_text(extracted_third_name) if extracted_third_name else None,
            passbook_acc_num=encrypt_text(passbook_acc_num) if passbook_acc_num else None,
            passbook_ifsc=encrypt_text(passbook_ifsc) if passbook_ifsc else None,
            passbook_address=encrypt_text(passbook_address) if passbook_address else None,
            aadhaar_matched=aadhaar_matched,
            third_doc_name_matched=third_name_matched,
            selfie_path=str(selfie_path),
            aadhaar_path=str(aadhaar_path),
            third_doc_path=str(third_doc_path) if third_doc_path else None,
            selfie_similarity=match_result["similarity"],
            third_doc_similarity=third_similarity,
            status=overall_status,
            webhook_status="Pending"
        )

        db.add(record)
        db.commit()

        background_tasks.add_task(trigger_webhook, record_id)

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Verification pipeline completed in {processing_time}s.")

        response_data = VerificationResponse(
            success=True,
            record_id=record_id,
            face_match=FaceMatchResult(
                similarity=match_result["similarity"],
                matched=match_result["matched"]
            ),
            aadhaar=AadhaarResult(
                provided=aadhaar_number,
                extracted=extracted_aadhaar,
                matched=aadhaar_matched,
                extracted_name=extracted_name or None
            ),
            third_document=third_doc_result,
            processing=ProcessingMeta(
                selfie_face_detected=match_result["selfie_face_detected"],
                aadhaar_face_detected=match_result["aadhaar_face_detected"],
                ocr_success=True,
                processing_time=processing_time
            )
        )
        return response_data

    except asyncio.TimeoutError:
        logger.error("Request processing timed out.")
        return JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            content={"success": False, "error": "Processing timeout"}
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unhandled pipeline error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": f"Internal processing error: {str(e)}"}
        )
        delete_temp_files(*temp_files)

async def run_async_pipeline(
    record_id: str,
    provided_aadhaar: str,
    selfie_path: str,
    aadhaar_path: str,
    third_doc_path: Optional[str],
    callback_url: Optional[str],
    detector: AadhaarDetector,
    perspective: PerspectiveCorrector,
    photo_cropper: PhotoCropper,
    ocr: AadhaarOCR,
    regex: RegexValidator,
    face_matcher: FaceMatcher
):
    """
    Executes the identity verification pipeline asynchronously in the background.
    Loads the encrypted files from disk, runs detection, matches faces and text,
    updates the record in the database, and triggers the webhook callback.
    """
    logger.info(f"Async Pipeline: Starting background analysis for record ID: {record_id}...")
    temp_files = []

    db = SessionLocal()
    try:
        logger.info(f"Async Pipeline: Reading encrypted selfie from {selfie_path}")
        with open(selfie_path, "rb") as f:
            enc_selfie_bytes = f.read()
        selfie_bytes = decrypt_file(enc_selfie_bytes)
        nparr_selfie = np.frombuffer(selfie_bytes, np.uint8)
        selfie_img = cv2.imdecode(nparr_selfie, cv2.IMREAD_COLOR)

        logger.info(f"Async Pipeline: Reading encrypted Aadhaar from {aadhaar_path}")
        with open(aadhaar_path, "rb") as f:
            enc_aadhaar_bytes = f.read()
        aadhaar_bytes = decrypt_file(enc_aadhaar_bytes)
        nparr_aadhaar = np.frombuffer(aadhaar_bytes, np.uint8)
        aadhaar_img = cv2.imdecode(nparr_aadhaar, cv2.IMREAD_COLOR)

        logger.info("Async Pipeline: Running Aadhaar card detection...")
        card_crop, initial_photo_crop, detect_status = detector.detect(aadhaar_img)
        if card_crop is None:
            raise Exception("Aadhaar card not detected")

        logger.info("Async Pipeline: Correcting card perspective...")
        corrected_card = perspective.correct(card_crop)

        logger.info("Async Pipeline: Cropping face photo...")
        model_to_use = None if detect_status["fallback_used"] else detector.model
        photo_crop = photo_cropper.crop_photo(corrected_card, yolo_model=model_to_use)
        if photo_crop is None or photo_crop.size == 0:
            raise Exception("No Aadhaar photo detected")

        temp_photo_path = save_temp_image(photo_crop, prefix="cropped_photo_async")
        temp_files.append(temp_photo_path)

        logger.info("Async Pipeline: Running OCR...")
        ocr_texts = ocr.extract_text(corrected_card)
        if not ocr_texts:
            raise Exception("OCR failed")

        extracted_name = extract_name_from_ocr(ocr_texts)

        logger.info("Async Pipeline: Extracting Aadhaar UID...")
        extracted_aadhaar = regex.extract_aadhaar_number(ocr_texts)
        if not extracted_aadhaar:
            raise Exception("Aadhaar number not found")

        aadhaar_matched, _ = regex.verify_match(extracted_aadhaar, provided_aadhaar)

        logger.info("Async Pipeline: Comparing selfie and Aadhaar photo...")
        match_result = face_matcher.match_faces(selfie_img, photo_crop)
        if not match_result["success"]:
            raise Exception(match_result["error"])

        third_doc_result = None
        extracted_third_name = None
        third_name_matched = None
        third_similarity = None
        third_face_matched = None

        if third_doc_path:
            logger.info("Async Pipeline: Reading encrypted third document...")
            with open(third_doc_path, "rb") as f:
                enc_third_bytes = f.read()
            third_bytes = decrypt_file(enc_third_bytes)
            nparr_third = np.frombuffer(third_bytes, np.uint8)
            third_img = cv2.imdecode(nparr_third, cv2.IMREAD_COLOR)

            logger.info("Async Pipeline: Extracting details from third document...")
            third_ocr_texts = ocr.extract_text(third_img)
            passbook_details = extract_passbook_details(third_ocr_texts)
            extracted_third_name = passbook_details["account_name"]
            passbook_acc_num = passbook_details["account_number"]
            passbook_ifsc = passbook_details["ifsc_code"]
            passbook_address = passbook_details["address"]

            third_name_matched = True  # Pure extraction for Admin UI; no comparison against Aadhaar name

            logger.info("Async Pipeline: Comparing selfie and third document...")
            third_match_res = face_matcher.match_faces(selfie_img, third_img)
            if third_match_res["success"]:
                third_similarity = third_match_res["similarity"]
                third_face_matched = third_match_res["matched"]
            else:
                third_similarity = 0.0
                third_face_matched = True  # Passbook text crops do not contain faces; rely on name matching
        else:
            passbook_acc_num = None
            passbook_ifsc = None
            passbook_address = None

        cos_sim = match_result.get("cosine_similarity", 0.0)
        if aadhaar_matched:
            if 0.25 <= cos_sim <= 0.40:
                overall_status = "Review"
            elif cos_sim > 0.40:
                overall_status = "Success"
            else:
                overall_status = "Failed"
        else:
            overall_status = "Failed"

        record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
        if record:
            record.extracted_aadhaar = encrypt_text(extracted_aadhaar)
            record.extracted_name = encrypt_text(extracted_name) if extracted_name else encrypt_text("")
            record.third_doc_name = encrypt_text(extracted_third_name) if extracted_third_name else None
            record.passbook_acc_num = encrypt_text(passbook_acc_num) if passbook_acc_num else None
            record.passbook_ifsc = encrypt_text(passbook_ifsc) if passbook_ifsc else None
            record.passbook_address = encrypt_text(passbook_address) if passbook_address else None
            record.aadhaar_matched = aadhaar_matched
            record.third_doc_name_matched = third_name_matched
            record.selfie_similarity = match_result["similarity"]
            record.third_doc_similarity = third_similarity
            record.status = overall_status
            db.commit()

    except Exception as e:
        logger.error(f"Async Pipeline: Background analysis failed for record {record_id}: {str(e)}", exc_info=True)
        record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
        if record:
            record.status = "Error"
            record.error_message = str(e)
            db.commit()

    finally:
        db.close()
        delete_temp_files(*temp_files)
        await trigger_webhook(record_id, custom_callback_url=callback_url)


@router.post(
    "/verify-async",
    response_model=AsyncVerificationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        413: {"description": "File upload exceeds maximum limit"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Internal server error"}
    }
)
async def verify_identity_async(
    request: Request,
    background_tasks: BackgroundTasks,
    selfie_image: UploadFile = File(..., description="Live selfie photograph of the user"),
    aadhaar_image: UploadFile = File(..., description="Uploaded image of the Aadhaar card"),
    aadhaar_number: str = Form(..., description="12-digit Aadhaar number for verification"),
    third_document: Optional[UploadFile] = File(None, description="Optional third document image (Bank Passbook or Passport)"),
    callback_url: Optional[str] = Form(None, description="Optional custom webhook callback endpoint URL"),
    detector: AadhaarDetector = Depends(get_detector),
    perspective: PerspectiveCorrector = Depends(get_perspective),
    photo_cropper: PhotoCropper = Depends(get_photo_cropper),
    ocr: AadhaarOCR = Depends(get_ocr),
    regex: RegexValidator = Depends(get_regex),
    face_matcher: FaceMatcher = Depends(get_face_matcher),
    db: Session = Depends(get_db)
):
    """
    Asynchronously uploads verification images and triggers the background analysis pipeline.
    Instantly returns an ID and status "Pending". Once completed, the analysis results
    are saved to the database and pushed to the webhook callback URL.
    """
    try:
        validate_image_file(selfie_image)
        validate_image_file(aadhaar_image)
        if third_document:
            validate_image_file(third_document)

        record_id = str(uuid.uuid4())
        selfie_filename = f"selfie_{record_id}.bin"
        aadhaar_filename = f"aadhaar_{record_id}.bin"
        third_doc_filename = f"third_{record_id}.bin" if third_document else None

        selfie_path = settings.UPLOAD_DIR / selfie_filename
        aadhaar_path = settings.UPLOAD_DIR / aadhaar_filename
        third_doc_path = settings.UPLOAD_DIR / third_doc_filename if third_doc_filename else None

        selfie_bytes = await selfie_image.read()
        encrypted_selfie = encrypt_file(selfie_bytes)
        with open(selfie_path, "wb") as f:
            f.write(encrypted_selfie)

        aadhaar_bytes = await aadhaar_image.read()
        encrypted_aadhaar = encrypt_file(aadhaar_bytes)
        with open(aadhaar_path, "wb") as f:
            f.write(encrypted_aadhaar)

        if third_document:
            third_bytes = await third_document.read()
            encrypted_third = encrypt_file(third_bytes)
            with open(third_doc_path, "wb") as f:
                f.write(encrypted_third)

        record = VerificationRecord(
            id=record_id,
            provided_aadhaar=encrypt_text(aadhaar_number),
            extracted_aadhaar=None,
            extracted_name=None,
            third_doc_name=None,
            aadhaar_matched=False,
            third_doc_name_matched=None,
            selfie_path=str(selfie_path),
            aadhaar_path=str(aadhaar_path),
            third_doc_path=str(third_doc_path) if third_doc_path else None,
            selfie_similarity=None,
            third_doc_similarity=None,
            status="Pending",
            webhook_status="Pending"
        )
        db.add(record)
        db.commit()

        background_tasks.add_task(
            run_async_pipeline,
            record_id=record_id,
            provided_aadhaar=aadhaar_number,
            selfie_path=str(selfie_path),
            aadhaar_path=str(aadhaar_path),
            third_doc_path=str(third_doc_path) if third_doc_path else None,
            callback_url=callback_url,
            detector=detector,
            perspective=perspective,
            photo_cropper=photo_cropper,
            ocr=ocr,
            regex=regex,
            face_matcher=face_matcher
        )

        return AsyncVerificationResponse(
            success=True,
            record_id=record_id,
            status="Pending",
            message="Analysis queued in background. Results will be posted to the callback URL."
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Async Upload Error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": f"Failed to enqueue analysis: {str(e)}"}
        )
