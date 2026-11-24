from pydantic import BaseModel
from typing import List, Optional


class Point2D(BaseModel):
    x: float
    y: float


class TrajectoryPoint(BaseModel):
    frame_idx: int
    timestamp: float
    pixel: Point2D
    court: Optional[Point2D] = None
    confidence: float = 1.0


class BounceEvent(BaseModel):
    frame_idx: int
    timestamp: float
    pixel_coord: Point2D
    court_coord: Point2D
    speed_kmh: float
    is_in: bool
    distance_to_line_cm: float
    bounce_index: int


class ShotMetric(BaseModel):
    shot_number: int
    speed_kmh: float
    is_in: bool
    bounce: Optional[BounceEvent] = None


class AnalysisSummary(BaseModel):
    total_shots: int
    total_bounces: int
    in_count: int
    out_count: int
    in_percentage: float
    max_speed_kmh: float
    avg_speed_kmh: float


class VideoAnalysisResponse(BaseModel):
    video_id: str
    filename: str
    fps: float
    total_frames: int
    duration_seconds: float
    bounces: List[BounceEvent]
    summary: AnalysisSummary
    trajectory_sample: List[TrajectoryPoint]
