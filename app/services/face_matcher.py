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

            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                logger.info("InsightFace: CUDA Execution Provider detected. Running on GPU.")
                ctx_id = 0
            else:
                logger.info("InsightFace: Running on CPU (no CUDA found).")
                ctx_id = -1

            self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            logger.info("InsightFace initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {str(e)}")
            raise e

    def get_embedding(self, img: np.ndarray, label: str = "image") -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """
        Detects a face in the image and returns its normalized embedding along with its estimated age.
        """
        try:
            faces = self.app.get(img)

            if len(faces) == 0:
                return None, None, f"No face detected in {label}."

            selected_face = faces[0]
            if len(faces) > 1:
                logger.info(f"InsightFace: Detected {len(faces)} faces in {label}. Selecting the largest face.")
                max_area = 0
                for face in faces:
                    bbox = face.bbox
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        selected_face = face

            embedding = selected_face.normed_embedding
            if embedding is None:
                embedding = selected_face.embedding
                if embedding is not None:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

            if embedding is None:
                return None, None, f"Failed to generate embedding for face in {label}."

            # Extract estimated face age
            age = float(selected_face.age) if hasattr(selected_face, "age") and selected_face.age is not None else None
            return embedding, age, "Success"
        except Exception as e:
            logger.error(f"InsightFace error on {label}: {str(e)}")
            return None, None, f"Error processing face in {label}: {str(e)}"

    def match_faces(self, selfie_img: np.ndarray, card_photo_img: np.ndarray) -> Dict[str, Any]:
        """
        Compares the selfie face against the Aadhaar card photo face.
        Returns match status, confidence, and similarity.
        """
        selfie_emb, selfie_age, err_msg = self.get_embedding(selfie_img, "selfie")
        if selfie_emb is None:
            return {
                "success": False,
                "error": err_msg,
                "selfie_face_detected": False,
                "aadhaar_face_detected": None
            }

        card_emb, card_age, err_msg = self.get_embedding(card_photo_img, "Aadhaar photo")
        if card_emb is None:
            return {
                "success": False,
                "error": err_msg,
                "selfie_face_detected": True,
                "aadhaar_face_detected": False
            }

        cos_sim = float(np.dot(selfie_emb, card_emb))

        matched = cos_sim >= settings.FACE_SIMILARITY_THRESHOLD

        cos_sim_clipped = max(-1.0, min(1.0, cos_sim))
        similarity_percentage = round((cos_sim_clipped + 1.0) / 2.0 * 100.0, 2)

        margin = abs(cos_sim - settings.FACE_SIMILARITY_THRESHOLD)
        confidence = round(min(1.0, margin * 2.0) * 100.0, 2)

        return {
            "success": True,
            "similarity": similarity_percentage,
            "cosine_similarity": cos_sim,
            "confidence": confidence,
            "matched": matched,
            "selfie_face_detected": True,
            "aadhaar_face_detected": True,
            "selfie_age": selfie_age,
            "card_photo_age": card_age
        }
