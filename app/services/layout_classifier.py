import cv2
import numpy as np
import re
from app.utils.logger import logger

class DocumentLayoutClassifier:
    """
    Classifies Aadhaar card uploads into layouts: front, back, long_letter, or digital_pdf.
    """
    def __init__(self):
        logger.info("DocumentLayoutClassifier initialized successfully.")

    def classify(self, img: np.ndarray, face_detected: bool, ocr_texts: list[str]) -> str:
        """
        Classifies layout based on image shape, face detection, and extracted text.
        """
        try:
            if img is None or img.size == 0:
                return "front"

            h, w, _ = img.shape
            aspect_ratio = h / w

            # 1. Height-to-width ratio check for vertical long letters (A4 uncut format)
            if aspect_ratio > 1.25:
                logger.info(f"LayoutClassifier: Classified as 'long_letter' (AspectRatio={aspect_ratio:.2f})")
                return "long_letter"

            # 2. Check for digital PDF vs scan/photograph
            # Digital PDFs are extremely sharp, high contrast, and clean (high laplacian variance, no background noise)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Combine text lines to search for rear-side address keywords
            combined_text = " ".join(ocr_texts).lower()
            
            has_back_keywords = any(k in combined_text for k in ["address", "pin code", "father", "husband", "spouse", "w/o", "s/o", "d/o"])
            has_front_keywords = any(k in combined_text for k in ["government of india", "unique identification", "enrollment", "dob", "year of birth"])

            # 3. Classify Back side
            # Lacks a main face photo, contains address keywords, and may contain "address" text labels.
            if not face_detected and has_back_keywords and not has_front_keywords:
                logger.info("LayoutClassifier: Classified as 'back' side (No face, back keywords present)")
                return "back"

            # 4. Classify Digital PDF
            # Highly uniform backgrounds with very clean, sharp text and distinct high Laplacian variance (> 800)
            # without camera lens distortion/frequency patterns.
            if laplacian_var > 600.0 and len(combined_text) > 100:
                logger.info(f"LayoutClassifier: Classified as 'digital_pdf' (LaplacianVar={laplacian_var:.1f})")
                return "digital_pdf"

            # Default to front side
            logger.info(f"LayoutClassifier: Classified as 'front' side (FaceDetected={face_detected}, LaplacianVar={laplacian_var:.1f})")
            return "front"

        except Exception as e:
            logger.error(f"LayoutClassifier error: {str(e)}", exc_info=True)
            return "front"

def get_layout_classifier() -> DocumentLayoutClassifier:
    return DocumentLayoutClassifier()
