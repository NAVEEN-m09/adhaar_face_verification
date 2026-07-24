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
        Attempts to detect and parse QR code contents from input image.
        Returns:
            dict containing extracted fields, or None if QR code is missing/undecodable.
        """
        try:
            if img is None or img.size == 0:
                return None

            # Run detection & decoding
            data, _, _ = self.detector.detectAndDecode(img)
            if not data:
                # Retry by converting to grayscale and thresholding to improve QR detection on low contrast scans
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                data, _, _ = self.detector.detectAndDecode(thresh)
                
            if not data:
                return None

            logger.info("AadhaarQRDecoder: Successfully decoded QR code.")
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

            # Fallback for plain text format or other versions
            return {"raw_data": data}

        except Exception as e:
            logger.error(f"AadhaarQRDecoder error: {str(e)}", exc_info=True)
            return None

def get_qr_decoder() -> AadhaarQRDecoder:
    return AadhaarQRDecoder()
