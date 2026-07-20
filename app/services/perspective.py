import cv2
import numpy as np
from app.utils.logger import logger

class PerspectiveCorrector:
    """
    Corrects perspective distortion of Aadhaar card images.
    Tries to find the 4 corners of the card and warps it to a flat front-facing rectangular view.
    """
    def correct(self, img: np.ndarray, target_width: int = 856, target_height: int = 540) -> np.ndarray:
        """
        Applies warp perspective transform to straighten the card.
        If corners are not detected reliably, it standardizes the image size and returns it.
        """
        h, w = img.shape[:2]
        
        # If the input image is already a clean card crop (aspect ratio between 1.4 and 1.7),
        # skip contour detection/perspective warping and resize it directly.
        max_dim = max(w, h)
        min_dim = min(w, h)
        input_ar = max_dim / min_dim if min_dim > 0 else 0
        
        if 1.4 <= input_ar <= 1.7:
            logger.info(f"PerspectiveCorrector: Input image aspect ratio is {input_ar:.2f} (already card-shaped). Bypassing warp perspective.")
            if h > w:
                logger.info("PerspectiveCorrector: Rotating portrait card to landscape.")
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return cv2.resize(img, (target_width, target_height))
            
        # 1. Convert to gray and threshold/edge detect
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Otsu's thresholding and Canny edge detection
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged = cv2.Canny(blurred, 30, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours[:3]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If the approximated contour has 4 points and covers a significant area
            if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.15):
                # Calculate aspect ratio of bounding box
                x, y, cw, ch = cv2.boundingRect(approx)
                max_dim = max(cw, ch)
                min_dim = min(cw, ch)
                aspect_ratio = max_dim / min_dim if min_dim > 0 else 0
                
                # Aadhaar card standard aspect ratio is ~1.55. Ignore boxes outside 1.1 to 2.0.
                if 1.1 <= aspect_ratio <= 2.0:
                    logger.info(f"PerspectiveCorrector: Found 4 corners for warp perspective with aspect ratio {aspect_ratio:.2f}.")
                    # Reshape approx points to (4, 2)
                    pts = approx.reshape(4, 2)
                    
                    # Order the points: top-left, top-right, bottom-right, bottom-left
                    ordered_pts = self._order_points(pts)
                
                    # Define destination points (flat rectangular image)
                    dst = np.array([
                        [0, 0],
                        [target_width - 1, 0],
                        [target_width - 1, target_height - 1],
                        [0, target_height - 1]
                    ], dtype="float32")
                    
                    # Compute transform matrix and warp
                    M = cv2.getPerspectiveTransform(ordered_pts, dst)
                    warped = cv2.warpPerspective(img, M, (target_width, target_height))
                    return warped
                
        # If we failed to find 4 corners, resize image to the target dimensions
        logger.info("PerspectiveCorrector: 4 corners not found. Resizing to standard dimensions.")
        return cv2.resize(img, (target_width, target_height))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates as: [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # top-left has the smallest sum, bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # top-right has the smallest difference, bottom-left has the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
