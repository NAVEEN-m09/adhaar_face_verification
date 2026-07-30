import os
import json
import numpy as np
from sqlalchemy.orm import Session
from app.utils.logger import logger
from app.models.db_models import VerificationRecord
from sklearn.linear_model import LogisticRegression

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "kyc_classifier_weights.json")

# Default heuristic weights (representing Option A combined with Option B logic)
DEFAULT_WEIGHTS = {
    "intercept": -5.0,
    "coef_similarity": 0.12,
    "coef_age_gap": -0.05,
    "coef_liveness": 2.0
}

class KYCClassifier:
    def __init__(self):
        self.weights = self.load_weights()

    def load_weights(self) -> dict:
        """
        Loads custom trained Logistic Regression weights from disk,
        falling back to default heuristic weights if not present.
        """
        if os.path.exists(WEIGHTS_PATH):
            try:
                with open(WEIGHTS_PATH, "r") as f:
                    weights = json.load(f)
                    logger.info("Loaded custom trained KYC classifier weights from disk.")
                    return weights
            except Exception as e:
                logger.error(f"Failed to load custom KYC weights: {str(e)}. Using defaults.")
        
        return DEFAULT_WEIGHTS

    def save_weights(self, weights: dict):
        """
        Saves updated weights to JSON file on disk.
        """
        try:
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            with open(WEIGHTS_PATH, "w") as f:
                json.dump(weights, f, indent=4)
            logger.info("Saved updated KYC classifier weights to disk.")
        except Exception as e:
            logger.error(f"Failed to save KYC weights to disk: {str(e)}")

    def predict_match(self, similarity: float, age_gap: float, liveness: float) -> float:
        """
        Calculates probability of being a valid match using Logistic Regression.
        """
        intercept = self.weights.get("intercept", -5.0)
        c_sim = self.weights.get("coef_similarity", 0.12)
        c_age = self.weights.get("coef_age_gap", -0.05)
        c_live = self.weights.get("coef_liveness", 2.0)

        # Logit calculation: z = w^T x + b
        z = intercept + (c_sim * similarity) + (c_age * age_gap) + (c_live * liveness)
        
        # Sigmoid: 1 / (1 + exp(-z))
        probability = 1.0 / (1.0 + np.exp(-z))
        return float(probability)

    def retrain_model(self, db: Session) -> dict:
        """
        Queries all adjudicated records from the database, trains a Logistic Regression classifier,
        and saves updated weights to disk. Returns accuracy statistics.
        """
        # Fetch finalized records ('Success' or 'Failed')
        records = db.query(VerificationRecord).filter(
            VerificationRecord.status.in_(["Success", "Failed"])
        ).all()

        X = []
        y = []

        for r in records:
            # Skip if critical features are missing
            if r.selfie_similarity is None or r.liveness_score is None:
                continue

            # Calculate age gap if ages are estimated
            selfie_age = r.selfie_age
            card_age = r.card_photo_age
            age_gap = abs(selfie_age - card_age) if (selfie_age is not None and card_age is not None) else 0.0

            # Features: [similarity, age_gap, liveness]
            X.append([r.selfie_similarity, age_gap, r.liveness_score])
            y.append(1 if r.status == "Success" else 0)

        if len(X) < 10:
            logger.info(f"Insufficient training records ({len(X)}/10 minimum). Skipping model retraining.")
            return {"status": "skipped", "reason": "insufficient_records"}

        # Ensure we have both classes represented in training set
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.info("Only one class represented in training set. Skipping model retraining.")
            return {"status": "skipped", "reason": "single_class_only"}

        try:
            X_arr = np.array(X)
            y_arr = np.array(y)

            # Fit Logistic Regression
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_arr, y_arr)

            accuracy = clf.score(X_arr, y_arr)

            # Extract weights
            updated_weights = {
                "intercept": float(clf.intercept_[0]),
                "coef_similarity": float(clf.coef_[0][0]),
                "coef_age_gap": float(clf.coef_[0][1]),
                "coef_liveness": float(clf.coef_[0][2])
            }

            self.weights = updated_weights
            self.save_weights(updated_weights)

            logger.info(f"KYC Classifier retrained successfully. Accuracy: {accuracy:.4f}")
            return {
                "status": "success",
                "accuracy": accuracy,
                "records_trained": len(X),
                "weights": updated_weights
            }

        except Exception as e:
            logger.error(f"Failed to retrain KYC model: {str(e)}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

kyc_classifier = KYCClassifier()
