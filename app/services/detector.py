import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from app.config import settings
from app.utils.logger import logger

class AadhaarDetector:
    """
    Service to load YOLOv11 and detect the Aadhaar card and the photo printed on it.
    Supports fallback to default YOLO11 nano model or heuristic-based cropping if
    custom weights are missing.
    """
    def __init__(self):
        self.model_path = settings.YOLO_MODEL_PATH
        self.is_fallback_mode = False

        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading custom YOLOv11 model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.warning(
                    f"Custom YOLOv11 model not found at {self.model_path}. "
                    "Falling back to standard yolo11n.pt for model initialization."
                )
                self.model = YOLO("yolo11n.pt")
                self.is_fallback_mode = True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise e

    def detect(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, bool]]:
        """
        Detects Aadhaar card and photo in the image.
        Returns:
            - cropped_card: The cropped and corrected Aadhaar card image, or None
            - cropped_photo: The cropped passport-size photo from the card, or None
            - status: Dictionary indicating detection success of each component
        """
        h, w = img.shape[:2]
        status = {
            "aadhaar_card_detected": False,
            "aadhaar_photo_detected": False,
            "fallback_used": False
        }

        if not self.is_fallback_mode:
            try:
                results = self.model(img, imgsz=320, verbose=False)
                card_box = None
                photo_box = None

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = self.model.names

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, "")
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].item())

                        if cls_name == "aadhaar_card" or "card" in cls_name.lower():
                            if card_box is None or conf > card_box[1]:
                                card_box = (xyxy, conf)
                        elif cls_name == "photo" or "photo" in cls_name.lower():
                            if photo_box is None or conf > photo_box[1]:
                                photo_box = (xyxy, conf)

                cropped_card = None
                cropped_photo = None

                if card_box is not None:
                    xyxy, _ = card_box
                    x1, y1, x2, y2 = clip_coords(xyxy, w, h)
                    cropped_card = img[y1:y2, x1:x2]
                    status["aadhaar_card_detected"] = True
                    logger.info("YOLO: Successfully detected Aadhaar card.")

                if photo_box is not None:
                    xyxy, _ = photo_box
                    x1, y1, x2, y2 = clip_coords(xyxy, w, h)
                    cropped_photo = img[y1:y2, x1:x2]
                    status["aadhaar_photo_detected"] = True
                    logger.info("YOLO: Successfully detected Photo inside card.")

                if cropped_card is not None:
                    return cropped_card, cropped_photo, status
            except Exception as e:
                logger.error(f"Error during custom YOLO inference: {str(e)}")

        logger.warning("YOLO detection failed or in fallback mode. Applying image heuristics.")
        status["fallback_used"] = True

        cropped_card = self._heuristic_crop_card(img)
        status["aadhaar_card_detected"] = True

        card_h, card_w = cropped_card.shape[:2]
        px1 = int(card_w * 0.05)
        py1 = int(card_h * 0.15)
        px2 = int(card_w * 0.40)
        py2 = int(card_h * 0.70)

        cropped_photo = cropped_card[py1:py2, px1:px2]
        status["aadhaar_photo_detected"] = True
        logger.info("Heuristics: Applied layouts to extract Aadhaar card and photo.")

        return cropped_card, cropped_photo, status

    def _heuristic_crop_card(self, img: np.ndarray) -> np.ndarray:
        """
        Uses OpenCV contours to find the Aadhaar card shape, or returns a 95% crop
        if no distinct rectangular contour of suitable size is found.
        """
        h, w = img.shape[:2]

        max_dim = max(w, h)
        min_dim = min(w, h)
        input_ar = max_dim / min_dim if min_dim > 0 else 0
        if 1.4 <= input_ar <= 1.7:
            logger.info(f"OpenCV Heuristics: Input image aspect ratio is {input_ar:.2f} (already card-shaped). Bypassing contour crop.")
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.15):
                x, y, cw, ch = cv2.boundingRect(approx)
                max_dim = max(cw, ch)
                min_dim = min(cw, ch)
                aspect_ratio = max_dim / min_dim if min_dim > 0 else 0
                if 1.1 <= aspect_ratio <= 2.0:
                    logger.info(f"OpenCV Heuristics: Detected rectangular contour of size {cw}x{ch} with aspect ratio {aspect_ratio:.2f}")
                    return img[y:y+ch, x:x+cw]

        logger.info("OpenCV Heuristics: No document contour found. Using center crop.")
        cx1 = int(w * 0.025)
        cy1 = int(h * 0.025)
        cx2 = int(w * 0.975)
        cy2 = int(h * 0.975)
        return img[cy1:cy2, cx1:cx2]

def clip_coords(xyxy: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clips coordinates to ensure they remain inside image boundaries."""
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(width, int(xyxy[2]))
    y2 = min(height, int(xyxy[3]))
    return x1, y1, x2, y2
