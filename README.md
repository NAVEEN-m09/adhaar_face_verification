# Aadhaar Face Verification REST API

A production-grade, secure, and highly optimized REST API built with **FastAPI** to verify if a user's live selfie matches the face printed on an uploaded Aadhaar card, extract the Aadhaar number from the card using OCR, and validate it against the user's provided number.

## 🚀 Key Features

- **Asynchronous FastAPI Architecture**: High concurrency support and modern Pydantic request/response validation.
- **YOLOv11 Bounding Box Detection**: Detects the Aadhaar card and the face photo on it. (Runs with automatic heuristic fallback if custom weights are missing).
- **OpenCV Perspective Warp**: Automatically straightens, crop, and normalizes skewed or distorted Aadhaar card uploads.
- **PaddleOCR Engine**: Extracts Aadhaar card text with robust orientation-aware character recognition.
- **Verhoeff Checksum & Regex Validation**: Validates extracted 12-digit Aadhaar numbers and verifies them against user inputs.
- **InsightFace (ArcFace Embeddings)**: Compares the selfie and Aadhaar photo faces using 512-dimensional cosine similarity.
- **Live Camera & File Upload GUI**: Interactive Web GUI at `/` that allows taking a live selfie via webcam or uploading photos to verify identity.
- **Privacy & Security Compliance**: Never logs raw Aadhaar numbers (uses strict log masking regex) or face embeddings. All temporary crops and files are automatically deleted after processing.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic
- **Computer Vision**: OpenCV, NumPy, Pillow
- **AI Models**: Ultralytics YOLOv11, PaddleOCR (PaddlePaddle), InsightFace (RetinaFace & ArcFace)

---

## 📁 Directory Structure

```
aadhaar-face-verification/
├── app/
│   ├── main.py                # FastAPI Application Entry
│   ├── config.py              # Settings & Path Management
│   ├── api/
│   │   └── verify.py          # /verify endpoint routing & execution
│   ├── services/
│   │   ├── detector.py        # YOLOv11 Aadhaar / Photo Detector
│   │   ├── perspective.py     # OpenCV Perspective Correction
│   │   ├── photo_cropper.py   # Crop Photo from corrected document
│   │   ├── ocr.py             # PaddleOCR Text Reading
│   │   ├── regex_validator.py # Aadhaar Regex extraction & Verhoeff check
│   │   └── face_matcher.py    # InsightFace ArcFace Cosine Similarity
│   ├── utils/
│   │   ├── image_utils.py     # File validation & decoding helpers
│   │   └── logger.py          # Masked JSON Logger
│   ├── schemas/
│   │   └── response.py        # Pydantic Response schemas
│   ├── middleware/
│   │   └── limit_upload.py    # Upload Size Limit Middleware
│   └── static/
│       └── index.html         # Single Page App (Web GUI)
├── Dockerfile                 # Container configurations
├── docker-compose.yml         # Container compositions
├── requirements.txt           # Package dependencies
├── tests/
│   ├── test_api.py            # Route & Middleware integration tests
│   └── test_services.py       # Helper & Regex unit tests
└── README.md                  # Documentation
```

---

## 🚀 Running the API Locally

### 1. Prerequisite Requirements
Ensure you have C++ Build Tools installed on your system if you compile `insightface` from source.

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Clone/Open project directory
cd aadhaar-face-verification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required python packages
pip install -r requirements.txt
```

### 3. Place YOLOv11 custom weights (Optional)
Place your custom-trained YOLOv11 card detection weights inside `app/models/best.pt`.
> [!NOTE]
> If `app/models/best.pt` is missing, the application will fallback gracefully to default `yolo11n.pt` and use layout-based bounding box heuristics to extract document crops and photos. This allows complete end-to-end testing immediately out-of-the-box!

### 4. Start the Application
Run the dev server using Uvicorn:
```bash
uvicorn app.main:app --reload
```
Once started:
- Access the **Interactive Web GUI** at: `http://localhost:8000/`
- Access the **FastAPI Swagger Docs** at: `http://localhost:8000/docs`

---

## 🐳 Deployment with Docker

The Dockerfile is highly optimized and pre-caches the YOLO, PaddleOCR, and InsightFace models during build time, preventing long downloads when first processing an API request.

### Build and Run with Docker Compose
```bash
docker-compose up --build
```
This maps the port to `http://localhost:8000` on the host machine.

---

## 🔌 API Specifications

### POST `/verify`
Validates identity by comparing files and checking the Aadhaar number.

- **Content-Type**: `multipart/form-data`
- **Request Form Data**:
  - `selfie_image`: File (PNG, JPG, JPEG) - Live webcam selfie or upload photo.
  - `aadhaar_image`: File (PNG, JPG, JPEG) - Uploaded Aadhaar card image.
  - `aadhaar_number`: String (12 digits, with or without spaces/dashes).

#### Successful Response (200 OK)
```json
{
  "success": true,
  "face_match": {
    "similarity": 89.42,
    "matched": true
  },
  "aadhaar": {
    "provided": "366210198051",
    "extracted": "366210198051",
    "matched": true
  },
  "processing": {
    "selfie_face_detected": true,
    "aadhaar_face_detected": true,
    "ocr_success": true,
    "processing_time": 1.45
  }
}
```

#### Error Response Example (400 Bad Request / 413 Payload Too Large / 415 Unsupported)
```json
{
  "success": false,
  "error": "No face detected in selfie."
}
```

---

## 🧪 Testing

Run unit and integration tests using `pytest`:
```bash
pytest -v
```
This tests the endpoints (using mocked ML models to prevent heavy load) and validates the OCR regex extraction and Verhoeff algorithm.
