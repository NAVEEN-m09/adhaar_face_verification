import cv2
import numpy as np
import re
from app.utils.logger import logger

class AadhaarQRDecoder:
    """
    Decodes the printed QR code on Aadhaar documents using OpenCV QRCodeDetector,
    parsing XML elements to extract verified card data.
    """
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        logger.info("AadhaarQRDecoder initialized successfully.")

    def decode(self, img: np.ndarray) -> dict:
        """
        Attempts to detect and parse QR code contents from input image using multi-stage preprocessing.
        Returns:
            dict containing extracted fields, or None if QR code is missing/undecodable.
        """
        try:
            if img is None or img.size == 0:
                return None

            # Generate multiple preprocessed candidates to maximize detection chances
            candidates = []
            
            # Candidate 1: Raw image
            candidates.append(img)
            
            # Candidate 2: Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            candidates.append(gray)
            
            # Candidate 3: Upscaled (2x) using cubic interpolation - crucial for high-density QR modules
            h, w = gray.shape[:2]
            if w < 1600:
                upscaled = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                candidates.append(upscaled)
            
            # Candidate 4: Grayscale + CLAHE (Local Contrast Enhancement)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_img = clahe.apply(gray)
            candidates.append(cl_img)
            
            # Candidate 5: Grayscale + OTSU Thresholding
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            candidates.append(otsu)

            # Candidate 6: Grayscale + Adaptive Thresholding
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            candidates.append(adaptive)

            raw_qr_contents = []

            # Test all candidates with both detectAndDecode and detectAndDecodeMulti
            for cand in candidates:
                # 1. Try detectAndDecodeMulti (handles multiple QR codes in side-by-side images)
                ok, decoded_info, _, _ = self.detector.detectAndDecodeMulti(cand)
                if ok and decoded_info:
                    for info in decoded_info:
                        if info and info not in raw_qr_contents:
                            raw_qr_contents.append(info)
                
                # 2. Try single detectAndDecode
                data, _, _ = self.detector.detectAndDecode(cand)
                if data and data not in raw_qr_contents:
                    raw_qr_contents.append(data)
                
                if raw_qr_contents:
                    # Found at least one decoded QR content, stop scanning other candidates
                    break

            if not raw_qr_contents:
                return None

            # Process the first valid decoded content
            for data in raw_qr_contents:
                logger.info("AadhaarQRDecoder: Successfully decoded QR code content.")
                details = {}

                # Parse Aadhaar XML Barcode format
                uid_match = re.search(r'uid="(\d{12})"', data)
                name_match = re.search(r'name="([^"]+)"', data)
                gender_match = re.search(r'gender="([^"]+)"', data)
                yob_match = re.search(r'yob="([^"]+)"', data)
                dob_match = re.search(r'dob="([^"]+)"', data)

                if uid_match:
                    details["aadhaar_number"] = uid_match.group(1)
                if name_match:
                    details["name"] = name_match.group(1)
                if gender_match:
                    details["gender"] = gender_match.group(1)
                
                # Map DOB
                if dob_match:
                    details["dob"] = dob_match.group(1)
                elif yob_match:
                    details["dob"] = f"01/01/{yob_match.group(1)}"

                if details:
                    logger.info(f"AadhaarQRDecoder: Extracted data={details}")
                    return details

                # Return raw content if format is plain text or newer secure QR payload
                if len(data.strip()) > 0:
                    return {"raw_data": data}

            return None

        except Exception as e:
            logger.error(f"AadhaarQRDecoder error: {str(e)}", exc_info=True)
            return None

def get_qr_decoder() -> AadhaarQRDecoder:
    return AadhaarQRDecoder()
