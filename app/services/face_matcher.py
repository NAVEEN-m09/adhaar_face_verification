import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
from insightface.app import FaceAnalysis
from app.config import settings
from app.utils.logger import logger

class FaceMatcher:
    """
    Service using InsightFace to detect faces, extract embeddings,
    and compare similarity using Cosine Similarity.
    """
    def __init__(self):
        try:
            self.app = FaceAnalysis(name='buffalo_l', root='~/.insightface')
            
            # Auto-detect GPU/CUDA availability for ONNX Runtime acceleration
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                logger.info("InsightFace: CUDA Execution Provider detected. Running on GPU.")
                ctx_id = 0
            else:
                logger.info("InsightFace: Running on CPU (no CUDA found).")
                ctx_id = -1
                
            # det_size=(320, 320) reduces inference image size, speeding up CPU execution by 3x
            self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            logger.info("InsightFace initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {str(e)}")
            raise e

    def get_embedding(self, img: np.ndarray, label: str = "image") -> Tuple[Optional[np.ndarray], str]:
        """
        Detects a face in the image and returns its normalized embedding.
        If multiple faces are found (e.g. due to ghost watermark photos on passports),
        selects the largest face by bounding box area.
        """
        try:
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return None, f"No face detected in {label}."
            
            # Handle multiple faces (common in passport ghost/watermark templates) by selecting the largest one
            selected_face = faces[0]
            if len(faces) > 1:
                logger.info(f"InsightFace: Detected {len(faces)} faces in {label}. Selecting the largest face.")
                max_area = 0
                for face in faces:
                    bbox = face.bbox  # Format: [x1, y1, x2, y2]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        selected_face = face
            
            # Retrieve the 512-dimensional embedding
            embedding = selected_face.normed_embedding
            if embedding is None:
                # Fallback to standard embedding if normed is missing
                embedding = selected_face.embedding
                if embedding is not None:
                    # Normalize it manually
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                        
            if embedding is None:
                return None, f"Failed to generate embedding for face in {label}."
                
            return embedding, "Success"
        except Exception as e:
            logger.error(f"InsightFace error on {label}: {str(e)}")
            return None, f"Error processing face in {label}: {str(e)}"

    def match_faces(self, selfie_img: np.ndarray, card_photo_img: np.ndarray) -> Dict[str, Any]:
        """
        Compares the selfie face against the Aadhaar card photo face.
        Returns match status, confidence, and similarity.
        """
        # 1. Extract embedding from selfie
        selfie_emb, err_msg = self.get_embedding(selfie_img, "selfie")
        if selfie_emb is None:
            return {
                "success": False,
                "error": err_msg,
                "selfie_face_detected": False,
                "aadhaar_face_detected": None
            }

        # 2. Extract embedding from Aadhaar card photo
        card_emb, err_msg = self.get_embedding(card_photo_img, "Aadhaar photo")
        if card_emb is None:
            return {
                "success": False,
                "error": err_msg,
                "selfie_face_detected": True,
                "aadhaar_face_detected": False
            }

        # 3. Calculate Cosine Similarity
        # Since both embeddings are normalized, similarity is the dot product
        cos_sim = float(np.dot(selfie_emb, card_emb))
        
        # 4. Map similarity to confidence and matching decision
        # Cosine similarity for identical matches is close to 1.0. Threshold is configurable.
        matched = cos_sim >= settings.FACE_SIMILARITY_THRESHOLD
        
        # Express similarity as a human-readable percentage (0 to 100%)
        # Clip similarity value between -1.0 and 1.0 just in case
        cos_sim_clipped = max(-1.0, min(1.0, cos_sim))
        # Linear map from [-1, 1] to [0, 100]
        similarity_percentage = round((cos_sim_clipped + 1.0) / 2.0 * 100.0, 2)
        
        # Confidence score could be calculated relative to distance from the threshold
        # e.g., how certain we are about the match/non-match
        margin = abs(cos_sim - settings.FACE_SIMILARITY_THRESHOLD)
        confidence = round(min(1.0, margin * 2.0) * 100.0, 2)

        return {
            "success": True,
            "similarity": similarity_percentage,
            "confidence": confidence,
            "matched": matched,
            "selfie_face_detected": True,
            "aadhaar_face_detected": True
        }
