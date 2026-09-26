from typing import Optional
from pydantic import BaseModel


class GradCamResult(BaseModel):
    original: str   # base64 data URI
    heatmap: str    # base64 data URI
    overlay: str    # base64 data URI


class MetadataResult(BaseModel):
    filename: str
    format: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    color_mode: Optional[str] = None
    channels: Optional[int] = None
    exif_available: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    orientation: Optional[int] = None
    gps_available: bool = False
    observations: list[str] = []


class ForensicResult(BaseModel):
    ela_mean_error: Optional[float] = None
    ela_note: Optional[str] = None
    fft_high_low_ratio: Optional[float] = None
    fft_note: Optional[str] = None
    noise_residual_variance: Optional[float] = None
    noise_note: Optional[str] = None
    edge_sharpness_variance: Optional[float] = None
    edge_note: Optional[str] = None


class CameraDetails(BaseModel):
    camera_make: str
    camera_model: str
    device_category: str
    lens_model: str
    focal_length_aperture: str
    exposure_iso: str
    capture_software: str


class CameraProfileResult(BaseModel):
    capture_source: str
    sensor_profile: str
    dimensions_size: str
    color_gamut_depth: str
    gps_location: str
    camera_details: Optional[CameraDetails] = None
    no_exif_note: Optional[str] = None


class AnalyzeResponse(BaseModel):
    prediction: str
    confidence: float
    probability_fake: float
    authenticity_assessment: str
    explanation: str
    gradcam: GradCamResult
    metadata: MetadataResult
    forensics: ForensicResult
    camera_profile: CameraProfileResult
    model_source: str
    disclaimer: str
    report_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
