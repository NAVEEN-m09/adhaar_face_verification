import pytest
import numpy as np
from app.services.regex_validator import RegexValidator
from app.services.perspective import PerspectiveCorrector
from app.services.photo_cropper import PhotoCropper

def test_verhoeff_validator():
    assert RegexValidator.validate_verhoeff("366210198051") is True
    assert RegexValidator.validate_verhoeff("548984365730") is True

    assert RegexValidator.validate_verhoeff("366210198054") is False
    assert RegexValidator.validate_verhoeff("123456789012") is False
    assert RegexValidator.validate_verhoeff("invalidnumber") is False
    assert RegexValidator.validate_verhoeff("123456") is False

def test_aadhaar_extraction():
    validator = RegexValidator()

    ocr_lines = [
        "Government of India",
        "Manoj Kumar",
        "DOB: 12/05/1990",
        "3662 1019 8051",
        "Male"
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    ocr_lines = [
        "To",
        "3662-1019-8051",
        "Address lines..."
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    ocr_lines = [
        "AADHAAR NUMBER: 366210198051"
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted == "366210198051"

    ocr_lines = [
        "Phone: 9876543210",
        "1234 5678 9012"
    ]
    extracted = validator.extract_aadhaar_number(ocr_lines)
    assert extracted is None

def test_regex_matching():
    validator = RegexValidator()

    matched, msg = validator.verify_match("366210198051", "3662 1019 8051")
    assert matched is True
    assert msg == "Matched"

    matched, msg = validator.verify_match("366210198051", "5489 8436 5730")
    assert matched is False
    assert msg == "Not Matched"

    matched, msg = validator.verify_match("366210198051", "invalid")
    assert matched is False
    assert "invalid" in msg.lower()

def test_perspective_and_cropper_fallbacks():
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)

    corrector = PerspectiveCorrector()
    corrected = corrector.correct(dummy_img, target_width=85, target_height=50)

    assert corrected.shape[1] == 85
    assert corrected.shape[0] == 50

    cropper = PhotoCropper()
    photo = cropper.crop_photo(corrected)
    assert photo.shape[1] > 0
    assert photo.shape[0] > 0

def test_benchmark_metrics(tmp_path):
    from app.utils.benchmark import calculate_iou, levenshtein_distance, calculate_cer_wer, run_benchmark
    import json

    # 1. Test IoU
    box1 = [10.0, 10.0, 50.0, 50.0]
    box2 = [20.0, 20.0, 60.0, 60.0]
    iou = calculate_iou(box1, box2)
    assert 0.38 <= iou <= 0.40

    # 2. Test Levenshtein distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("NAVEEN UNKAL", "NAVEEN M UNKAL") == 2

    # 3. Test CER/WER
    cer, wer = calculate_cer_wer("NAVEEN UNKAL", "NAVEEN UNKAL")
    assert cer == 0.0
    assert wer == 0.0

    # 4. Test run_benchmark
    gt_data = [
        {
            "image_id": "test_1",
            "expected_card_box": [0, 0, 100, 100],
            "predicted_card_box": [0, 0, 100, 100],
            "expected_name": "JOHN DOE",
            "predicted_name": "JOHN DOE",
            "similarity_percentage": 85.0,
            "is_same_person": True
        },
        {
            "image_id": "test_2",
            "expected_card_box": [0, 0, 100, 100],
            "predicted_card_box": [0, 0, 100, 100],
            "expected_name": "MOCK USER",
            "predicted_name": "MOCK USER",
            "similarity_percentage": 10.0,
            "is_same_person": False
        }
    ]
    gt_file = tmp_path / "gt.json"
    with open(gt_file, "w") as f:
        json.dump(gt_data, f)

    results = run_benchmark(str(tmp_path), str(gt_file))
    assert results["card_detection"]["average_iou"] == 1.0
    assert results["ocr"]["average_cer"] == 0.0
    assert results["threshold_calibration"]["recommended_percentage_threshold"] > 10.0

def test_liveness_detector():
    from app.services.liveness import LivenessDetector
    detector = LivenessDetector()

    # Test with valid image dimensions
    high_texture_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    is_live, score = detector.check_liveness(high_texture_img)
    # Randomly generated pixel noise has extremely high variance and no display screen line frequency spikes
    assert score > 0.50

    # Test flat image (spoofing printout mockup with 0 Laplacian variance)
    flat_img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    is_live_flat, score_flat = detector.check_liveness(flat_img)
    assert is_live_flat is False
    assert score_flat < 0.90

def test_layout_classifier():
    from app.services.layout_classifier import DocumentLayoutClassifier
    classifier = DocumentLayoutClassifier()

    # 1. Long letter (aspect ratio > 1.25)
    tall_img = np.zeros((400, 200, 3), dtype=np.uint8)
    assert classifier.classify(tall_img, face_detected=False, ocr_texts=[]) == "long_letter"

    # 2. Back side classification
    std_img = np.zeros((200, 300, 3), dtype=np.uint8)
    assert classifier.classify(std_img, face_detected=False, ocr_texts=["Address: 123 Street", "Pin Code: 560001"]) == "back"

    # 3. Front side
    assert classifier.classify(std_img, face_detected=True, ocr_texts=["Government of India"]) == "front"

def test_qr_decoder():
    from app.services.qr_decoder import AadhaarQRDecoder
    decoder = AadhaarQRDecoder()

    # Empty image should return None
    assert decoder.decode(None) is None
    assert decoder.decode(np.zeros((10, 10, 3), dtype=np.uint8)) is None

def test_face_matcher_age_estimation():
    from app.services.face_matcher import FaceMatcher
    from unittest.mock import MagicMock
    matcher = FaceMatcher()
    
    # Mock self.app.get to return a mock face object with age
    mock_face = MagicMock()
    mock_face.normed_embedding = np.ones((512,), dtype=np.float32)
    mock_face.age = 25.0
    matcher.app.get = MagicMock(return_value=[mock_face])
    
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = matcher.match_faces(dummy_img, dummy_img)
    assert res["success"] is True
    assert res["selfie_age"] == 25.0
    assert res["card_photo_age"] == 25.0


