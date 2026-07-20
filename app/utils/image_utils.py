import os
import uuid
import mimetypes
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
import cv2
import numpy as np
from app.config import settings
from app.utils.logger import logger

def validate_image_file(file: UploadFile) -> None:
    """
    Validates the uploaded file's size, file extension, and MIME type.
    Raises HTTPException if validation fails.
    """
    # 1. Validate File Size
    # Read a chunk to see if it exceeds maximum size without reading everything into memory at once
    # FastAPI UploadFile file is a SpooledTemporaryFile
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0) # Reset to beginning
    
    if file_size > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
        logger.warning(f"File upload rejected: {file.filename} exceeds limit of {max_mb}MB. Size: {file_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {max_mb} MB."
        )

    # 2. Validate Extension
    filename = file.filename or ""
    file_ext = Path(filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected: {file.filename} has unsupported extension {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension {file_ext}. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # 3. Validate MIME Type
    # Guess mime type from filename
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = file.content_type
        
    if mime_type not in settings.ALLOWED_MIME_TYPES:
        logger.warning(f"File upload rejected: {file.filename} has unsupported MIME type {mime_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported MIME type {mime_type}. Allowed: {', '.join(settings.ALLOWED_MIME_TYPES)}"
        )

async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Reads a FastAPI UploadFile and decodes it into a NumPy array (OpenCV image).
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        return img
    except Exception as e:
        logger.error(f"Error decoding image {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted image file."
        )
    finally:
        # Reset file pointer so it can be read again if needed
        await file.seek(0)

def save_temp_image(img: np.ndarray, prefix: str = "temp") -> Path:
    """
    Saves an OpenCV image to a temporary file with a secure random name
    in the upload directory.
    """
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    filepath = settings.UPLOAD_DIR / filename
    cv2.imwrite(str(filepath), img)
    return filepath

def delete_temp_files(*filepaths: Path) -> None:
    """
    Safely deletes specified files if they exist.
    """
    for path in filepaths:
        try:
            if path and path.exists():
                os.remove(path)
                logger.info(f"Deleted temporary file: {path.name}")
        except Exception as e:
            logger.error(f"Failed to delete temporary file {path}: {str(e)}")
