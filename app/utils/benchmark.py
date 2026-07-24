import json
import os
from pathlib import Path
import numpy as np

def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return float(intersection_area / union_area)

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates the minimum edit distance (Levenshtein) between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_cer_wer(target: str, predicted: str) -> tuple[float, float]:
    """
    Calculates Character Error Rate (CER) and Word Error Rate (WER).
    """
    target = target.strip()
    predicted = predicted.strip()

    if not target:
        return (0.0, 0.0) if not predicted else (1.0, 1.0)

    # CER
    char_distance = levenshtein_distance(target, predicted)
    cer = char_distance / max(1, len(target))

    # WER
    target_words = target.split()
    pred_words = predicted.split()
    
    if not target_words:
        return cer, (0.0 if not pred_words else 1.0)

    word_distance = levenshtein_distance(target_words, pred_words)
    wer = word_distance / max(1, len(target_words))

    return float(cer), float(wer)

def run_benchmark(dataset_dir: str, ground_truth_file: str) -> dict:
    """
    Executes the benchmarking pipeline over the golden testing dataset.
    Generates detailed metrics for Card Detector, OCR, and Face Matcher.
    """
    dataset_path = Path(dataset_dir)
    gt_path = Path(ground_truth_file)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_file}")

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    results = {
        "card_detection": {"iou_scores": [], "success_rate": 0.0, "average_iou": 0.0},
        "ocr": {"cer_scores": [], "wer_scores": [], "average_cer": 0.0, "average_wer": 0.0},
        "face_matching": {"true_positives": [], "false_positives": [], "true_negatives": [], "false_negatives": []}
    }

    # Evaluate each entry in the ground truth dataset
    for entry in gt_data:
        # Bounding box evaluations
        expected_box = entry.get("expected_card_box")
        predicted_box = entry.get("predicted_card_box")
        if expected_box and predicted_box:
            iou = calculate_iou(expected_box, predicted_box)
            results["card_detection"]["iou_scores"].append(iou)

        # OCR text evaluation
        expected_name = entry.get("expected_name", "")
        predicted_name = entry.get("predicted_name", "")
        cer, wer = calculate_cer_wer(expected_name, predicted_name)
        results["ocr"]["cer_scores"].append(cer)
        results["ocr"]["wer_scores"].append(wer)

        # Face similarity values
        similarity = entry.get("similarity_percentage", 0.0) # percentage scale (0 to 100)
        is_same_person = entry.get("is_same_person", True)
        results["face_matching"]["true_positives" if is_same_person else "false_positives"].append(similarity)

    # Compute aggregation summaries
    if results["card_detection"]["iou_scores"]:
        results["card_detection"]["average_iou"] = float(np.mean(results["card_detection"]["iou_scores"]))
        results["card_detection"]["success_rate"] = float(np.sum(np.array(results["card_detection"]["iou_scores"]) >= 0.5) / len(results["card_detection"]["iou_scores"]) * 100.0)

    if results["ocr"]["cer_scores"]:
        results["ocr"]["average_cer"] = float(np.mean(results["ocr"]["cer_scores"]))
        results["ocr"]["average_wer"] = float(np.mean(results["ocr"]["wer_scores"]))

    # Threshold calibration tuning curve calculations
    best_threshold = 35.0
    optimal_metrics = {"far": 100.0, "frr": 100.0}
    thresholds = list(range(0, 101))
    
    same_scores = np.array(results["face_matching"]["true_positives"])
    diff_scores = np.array(results["face_matching"]["false_positives"])

    for t in thresholds:
        # FAR: False Acceptances (different people accepted as matches)
        far = float(np.sum(diff_scores >= t) / len(diff_scores) * 100.0) if len(diff_scores) > 0 else 0.0
        # FRR: False Rejections (same people rejected as non-matches)
        frr = float(np.sum(same_scores < t) / len(same_scores) * 100.0) if len(same_scores) > 0 else 0.0

        if far < 0.1 and (far < optimal_metrics["far"] or frr < optimal_metrics["frr"]):
            best_threshold = t
            optimal_metrics = {"far": far, "frr": frr}

    results["threshold_calibration"] = {
        "recommended_percentage_threshold": best_threshold,
        "recommended_cosine_threshold": round((best_threshold / 100.0) * 2.0 - 1.0, 4),
        "target_far": optimal_metrics["far"],
        "target_frr": optimal_metrics["frr"]
    }

    return results

if __name__ == "__main__":
    # Create sample ground truth structure for demonstration/first runs
    sample_gt = [
        {
            "image_id": "test_001",
            "expected_card_box": [50.0, 50.0, 450.0, 300.0],
            "predicted_card_box": [52.0, 48.0, 448.0, 302.0],
            "expected_name": "NAVEEN UNKAL",
            "predicted_name": "NAVEEN UNKAL",
            "similarity_percentage": 78.45,
            "is_same_person": True
        },
        {
            "image_id": "test_002",
            "expected_card_box": [10.0, 10.0, 200.0, 150.0],
            "predicted_card_box": [10.0, 10.0, 200.0, 150.0],
            "expected_name": "JOHN DOE",
            "predicted_name": "J0HN D0E",
            "similarity_percentage": 12.14,
            "is_same_person": False
        }
    ]
    
    mock_dir = Path("outputs")
    mock_dir.mkdir(exist_ok=True)
    gt_file = mock_dir / "mock_ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(sample_gt, f, indent=4)
        
    print("Running initial calibration benchmark...")
    summary = run_benchmark(str(mock_dir), str(gt_file))
    print(json.dumps(summary, indent=4))
