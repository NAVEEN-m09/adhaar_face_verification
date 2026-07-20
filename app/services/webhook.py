import httpx
import asyncio
from datetime import datetime, timezone
from app.config import settings
from app.utils.logger import logger
from app.database import SessionLocal
from app.models.db_models import VerificationRecord
from app.utils.security_utils import decrypt_text

from typing import Optional

async def trigger_webhook(record_id: str, custom_callback_url: Optional[str] = None):
    """
    Sends a HTTP POST callback to the specified custom_callback_url or settings.WEBHOOK_URL
    containing decrypted metadata results, and logs the API status response code.
    Runs asynchronously as a background task.
    """
    url = custom_callback_url or settings.WEBHOOK_URL
    if not url:
        logger.info("Webhook: Skipping trigger (no webhook/callback URL configured).")
        db = SessionLocal()
        try:
            record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
            if record:
                record.webhook_status = "Skipped"
                db.commit()
        finally:
            db.close()
        return

    logger.info(f"Webhook: Triggering callback for record ID: {record_id}...")
    db = SessionLocal()

    try:
        record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
        if not record:
            logger.error(f"Webhook: Record {record_id} not found in database.")
            return

        payload = {
            "record_id": record.id,
            "status": record.status,
            "error_message": record.error_message,
            "timestamp": record.created_at.isoformat() if record.created_at else datetime.now(timezone.utc).isoformat(),
            "aadhaar": {
                "provided": decrypt_text(record.provided_aadhaar),
                "extracted": decrypt_text(record.extracted_aadhaar) if record.extracted_aadhaar else None,
                "matched": record.aadhaar_matched,
                "extracted_name": decrypt_text(record.extracted_name) if record.extracted_name else None
            },
            "face_match": {
                "similarity": record.selfie_similarity,
                "matched": record.selfie_similarity >= settings.FACE_SIMILARITY_THRESHOLD if record.selfie_similarity else False
            },
            "third_document": {
                "provided": bool(record.third_doc_path),
                "extracted_name": decrypt_text(record.third_doc_name) if record.third_doc_name else None,
                "name_matched": record.third_doc_name_matched,
                "selfie_similarity": record.third_doc_similarity,
                "face_matched": record.third_doc_similarity >= settings.FACE_SIMILARITY_THRESHOLD if record.third_doc_similarity else False
            }
        }

        attempts = 3
        backoff = 2.0
        success = False
        last_status_code = None

        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"Webhook: Sending callback to {url} (Attempt {attempt}/{attempts})...")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=10.0
                    )
                    last_status_code = response.status_code
                    if response.is_success:
                        success = True
                        logger.info(f"Webhook: Callback succeeded on attempt {attempt}. Status Code: {response.status_code}")
                        break
                    else:
                        logger.warning(f"Webhook: Attempt {attempt} returned status {response.status_code}")
            except Exception as exc:
                logger.warning(f"Webhook: Attempt {attempt} failed with exception: {str(exc)}")

            if attempt < attempts:
                logger.info(f"Webhook: Sleeping {backoff} seconds before retry...")
                await asyncio.sleep(backoff)
                backoff *= 2.0

        record.webhook_status = "Sent" if success else "Failed"
        record.webhook_response = last_status_code

    except Exception as e:
        logger.error(f"Webhook: Failed dispatching callback event: {str(e)}")
        if record:
            record.webhook_status = "Failed"

    finally:
        db.commit()
        db.close()
