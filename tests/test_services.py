import pytest
import numpy as np
from app.services.regex_validator import RegexValidator
from app.services.perspective import PerspectiveCorrector
from app.services.photo_cropper import PhotoCropper

def test_verhoeff_validator():
    # Valid Aadhaar numbers with correct Verhoeff checksums
    assert RegexValidator.validate_verhoeff("366210198051") is True
    assert RegexValidator.validate_verhoeff("548984365730") is True
    
    # Invalid Aadhaar numbers or wrong checksums
    assert RegexValidator.validate_verhoeff("366210198054") is False
    assert RegexValidator.validate_verhoeff("123456789012") is False
    assert RegexValidator.validate_verhoeff("invalidnumber") is False
    assert RegexValidator.validate_verhoeff("123456") is False

def test_aadhaar_extraction():
    validator = RegexValidator()
    
    # Case 1: Valid space-separated number
    ocr_lines = [
        "Government of India",
        "Manoj Kumar",
        "DOB: 12/05/1990",
        "3662 1019 8051",
        "Male"
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    # Case 2: Valid dash-separated number
    ocr_lines = [
        "To",
        "3662-1019-8051",
        "Address lines..."
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    # Case 3: Valid continuous number
    ocr_lines = [
        "AADHAAR NUMBER: 366210198051"
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    # Case 4: No valid Aadhaar number (or fails Verhoeff)
    ocr_lines = [
        "Phone: 9876543210",
        "1234 5678 9012"  # Fails Verhoeff checksum
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted is None

def test_regex_matching():
    validator = RegexValidator()
    
    # Matching correct numbers
    matched, msg = validator.verify_match("366210198051", "3662 1019 8051")
    assert matched is True
    assert msg == "Matched"
    
    # Mismatch
    matched, msg = validator.verify_match("366210198051", "5489 8436 5730")
    assert matched is False
    assert msg == "Not Matched"

    # Invalid input formatting
    matched, msg = validator.verify_match("366210198051", "invalid")
    assert matched is False
    assert "invalid" in msg.lower()

def test_perspective_and_cropper_fallbacks():
    # Verify that the algorithms handle empty or uniform test arrays without crashing
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    corrector = PerspectiveCorrector()
    corrected = corrector.correct(dummy_img, target_width=85, target_height=50)
    
    # Corrected should resize to 85x50
    assert corrected.shape[1] == 85
    assert corrected.shape[0] == 50

    cropper = PhotoCropper()
    # Crop with no YOLO model (triggers fallback)
    photo = cropper.crop_photo(corrected)
    assert photo.shape[1] > 0
    assert photo.shape[0] > 0
