import cv2
import numpy as np
from app.utils.logger import logger

class PhotoCropper:
    """
    Crops the passport-size photo from the perspective-corrected Aadhaar card.
    Uses either the detection coordinates or layout-based heuristic crop.
    """
    def crop_photo(self, card_img: np.ndarray, yolo_model=None) -> np.ndarray:
        """
        Crops the passport-size photo from the corrected Aadhaar card.
        If a YOLO model is provided, it tries to detect the photo inside the card.
        Otherwise, it falls back to layout heuristics.
        """
        h, w = card_img.shape[:2]

        if yolo_model is not None:
            try:
                results = yolo_model(card_img, imgsz=320, verbose=False)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = yolo_model.names

                    best_box = None
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, "")
                        conf = float(box.conf[0].item())

                        if cls_name == "photo" or "photo" in cls_name.lower():
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            if best_box is None or conf > best_box[1]:
                                best_box = (xyxy, conf)

                    if best_box is not None:
                        xyxy, _ = best_box
                        x1 = max(0, xyxy[0])
                        y1 = max(0, xyxy[1])
                        x2 = min(w, xyxy[2])
                        y2 = min(h, xyxy[3])
                        logger.info("PhotoCropper: YOLO detected photo inside warped Aadhaar card.")
                        return card_img[y1:y2, x1:x2]
            except Exception as e:
                logger.error(f"PhotoCropper: Error during YOLO photo detection: {str(e)}")

        logger.info("PhotoCropper: Using layout-based heuristics to crop photo.")
        x1 = int(w * 0.05)
        y1 = int(h * 0.15)
        x2 = int(w * 0.38)
        y2 = int(h * 0.68)

        return card_img[y1:y2, x1:x2]
