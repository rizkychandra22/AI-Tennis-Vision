from pydantic import BaseModel
from typing import List, Optional
from app.schemas.analysis import Point2D, BounceEvent


class LiveFrameTelemetry(BaseModel):
    frame_idx: int
    timestamp: float
    ball_detected: bool
    current_speed_kmh: float
    bounce_detected: bool
    last_bounce: Optional[BounceEvent] = None
    recent_trajectory: List[Point2D] = []


class CalibrationRequest(BaseModel):
    """Four court corner points in pixel coordinates:
    1. Top-Left (far baseline left)
    2. Top-Right (far baseline right)
    3. Bottom-Right (near baseline right)
    4. Bottom-Left (near baseline left)
    """
    corners: List[Point2D]


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    is_active: bool
    calibrated: bool
    total_frames_processed: int = 0
    total_bounces_detected: int = 0
