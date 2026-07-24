from pydantic import BaseModel, Field
from typing import Optional

class FaceMatchResult(BaseModel):
    similarity: float = Field(..., description="Similarity percentage between selfie and Aadhaar photo (0.0 to 100.0)")
    matched: bool = Field(..., description="True if similarity meets or exceeds threshold, False otherwise")

class AadhaarResult(BaseModel):
    provided: str = Field(..., description="The user-supplied 12-digit Aadhaar number")
    extracted: Optional[str] = Field(..., description="The 12-digit Aadhaar number extracted from the card, or null if OCR failed")
    matched: bool = Field(..., description="True if provided and extracted numbers match, False otherwise")
    extracted_name: Optional[str] = Field(None, description="The card holder's name extracted from the Aadhaar card")

class ThirdDocumentResult(BaseModel):
    provided: bool = Field(..., description="Whether a third document was uploaded")
    extracted_name: Optional[str] = Field(None, description="The name extracted from the third document")
    name_matched: Optional[bool] = Field(None, description="True if name matches Aadhaar name, False otherwise")
    similarity: Optional[float] = Field(None, description="Similarity percentage between selfie and third document photo")
    matched: Optional[bool] = Field(None, description="True if similarity meets threshold, False otherwise")

class ProcessingMeta(BaseModel):
    selfie_face_detected: bool = Field(..., description="Whether a face was detected in the selfie")
    aadhaar_face_detected: bool = Field(..., description="Whether a face was detected in the Aadhaar card photo")
    ocr_success: bool = Field(..., description="Whether the Aadhaar number was successfully extracted via OCR")
    liveness_score: Optional[float] = Field(None, description="Passive liveness anti-spoofing confidence score (0.0 to 1.0)")
    is_live: Optional[bool] = Field(None, description="True if selfie meets liveness thresholds, False otherwise")
    layout_type: Optional[str] = Field(None, description="Classified layout type (front, back, long_letter, digital_pdf)")
    qr_decoded: Optional[bool] = Field(None, description="Whether the secure QR code was successfully decoded")
    processing_time: float = Field(..., description="Processing time in seconds")

class VerificationResponse(BaseModel):
    success: bool = Field(True, description="Whether the request was processed successfully without pipeline errors")
    record_id: str = Field(..., description="The unique database tracking ID for this verification")
    face_match: FaceMatchResult = Field(..., description="Results of the face matching component")
    aadhaar: AadhaarResult = Field(..., description="Results of the Aadhaar OCR verification")
    third_document: Optional[ThirdDocumentResult] = Field(None, description="Results of the third document verification")
    processing: ProcessingMeta = Field(..., description="Metadata detailing pipeline executions")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always False for error responses")
    error: str = Field(..., description="Descriptive error message indicating the failure reason")

class AsyncVerificationResponse(BaseModel):
    success: bool = Field(True, description="True if background analysis request was accepted")
    record_id: str = Field(..., description="The unique database tracking ID for this verification")
    status: str = Field("Pending", description="The initial status of the verification")
    message: str = Field(..., description="Instructional details about the async queue status")
