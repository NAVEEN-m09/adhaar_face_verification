from paddleocr import PaddleOCR
import numpy as np
from app.config import settings
from app.utils.logger import logger

class AadhaarOCR:
    """
    Service to run PaddleOCR on the corrected Aadhaar card and extract text.
    Initialized once at startup.
    """
    def __init__(self):
        try:
            logger.info("Initializing PaddleOCR model...")
            # Initialize PaddleOCR with english language, angle classifier enabled
            self.ocr_client = PaddleOCR(
                use_angle_cls=True,
                lang=settings.OCR_LANG,
                enable_mkldnn=False
            )
            logger.info("PaddleOCR model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise e

    def extract_text(self, card_img: np.ndarray) -> list[str]:
        """
        Runs OCR on the card image and returns list of extracted text lines.
        """
        try:
            result = self.ocr_client.ocr(card_img)
            
            extracted_lines = []
            if result and isinstance(result, list):
                for item in result:
                    if not item:
                        continue
                    # 1. New PaddleX v3 dictionary format support
                    if isinstance(item, dict) and "rec_texts" in item:
                        texts = item["rec_texts"]
                        if isinstance(texts, list):
                            for txt in texts:
                                if isinstance(txt, str):
                                    extracted_lines.append(txt.strip())
                    # 2. Legacy nested list format support
                    elif isinstance(item, list):
                        for res in item:
                            try:
                                if len(res) == 2 and len(res[1]) == 2:
                                    text_str = res[1][0]
                                    if isinstance(text_str, str):
                                        extracted_lines.append(text_str.strip())
                            except Exception:
                                continue
                            
            logger.info(f"OCR: Extracted {len(extracted_lines)} lines of text.")
            return extracted_lines
        except Exception as e:
            logger.error(f"OCR: Error during text extraction: {str(e)}")
            return []
