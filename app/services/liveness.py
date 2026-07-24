import cv2
import numpy as np
from app.config import settings
from app.utils.logger import logger

class LivenessDetector:
    """
    Service class to verify user selfie liveness and detect presentation spoofing attacks.
    Checks for high frequency display screen grids (Moiré) using 2D FFT,
    and texture micro-flatness using Laplacian variance.
    """
    def __init__(self):
        logger.info("LivenessDetector initialized successfully.")

    def check_liveness(self, img: np.ndarray) -> tuple[bool, float]:
        """
        Runs passive face liveness validation.
        Returns:
            is_live: bool - True if confidence meets threshold, False otherwise
            confidence: float - Score from 0.0 to 1.0
        """
        try:
            if img is None or img.size == 0:
                return False, 0.0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Texture Flatness Check (Laplacian Variance)
            # Live faces have deep 3D details and features resulting in high variance.
            # Screen captures or paper printouts have uniform flat micro-textures (low variance).
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Moiré Screen Frequency Spike Detection (Fast Fourier Transform)
            # Scanned screens display prominent periodic line grids generating sharp high frequency spikes.
            h, w = gray.shape
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8)
            
            # Target center low frequency zone masking
            cy, cx = h // 2, w // 2
            r = min(h, w) // 10
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (cx, cy), r, 1, -1)
            
            # Analyze outer high frequencies
            high_freq = magnitude_spectrum * (1 - mask)
            high_freq_mean = float(np.mean(high_freq))

            # Calibrate combined score
            score = 1.0

            # Penalize low texture variance (indicating flat paper printout or out-of-focus digital screen)
            if laplacian_var < 100.0:
                score -= 0.3 * (1.0 - (laplacian_var / 100.0))
            
            # Penalize high frequency screen noise (Moiré line patterns)
            if high_freq_mean > 165.0:
                score -= 0.45 * (min(200.0, high_freq_mean) / 200.0)

            confidence = float(max(0.0, min(1.0, score)))
            is_live = confidence >= settings.LIVENESS_THRESHOLD

            logger.info(f"LivenessCheck: score={confidence:.4f}, is_live={is_live} (LaplacianVar={laplacian_var:.1f}, HighFreq={high_freq_mean:.1f})")
            return is_live, confidence

        except Exception as e:
            logger.error(f"Liveness Check failed: {str(e)}", exc_info=True)
            return False, 0.0

from fastapi import Request

def get_liveness_detector(request: Request) -> LivenessDetector:
    return request.app.state.liveness
