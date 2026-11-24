import os
import shutil
import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.core.config import settings
from app.schemas.analysis import VideoAnalysisResponse, BounceEvent, AnalysisSummary, Point2D
from app.services.video_service import video_service
from app.services.analysis_service import analysis_service

router = APIRouter()


@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    calibration_corners: Optional[str] = Form(None)
):
    """Upload a tennis match video file (.mp4, .mov, .avi) for automated analysis.
    Optional calibration_corners: JSON string of 4 corner points [[x,y], ...].
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please upload MP4, MOV, or AVI."
        )

    corners = None
    if calibration_corners:
        try:
            parsed = json.loads(calibration_corners)
            if isinstance(parsed, list) and len(parsed) == 4:
                corners = [(float(pt[0]), float(pt[1])) for pt in parsed]
        except Exception:
            pass

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = video_service.process_video(temp_path, custom_corners=corners)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/result/{video_id}", response_model=VideoAnalysisResponse)
def get_analysis_result(video_id: str):
    """Retrieve saved analysis result by video_id."""
    result = analysis_service.get_result(video_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Analysis for video {video_id} not found.")
    return result


@router.get("/sample", response_model=VideoAnalysisResponse)
def get_sample_analysis():
    """Provides a realistic sample analysis response for rapid frontend testing."""
    bounces = [
        BounceEvent(
            bounce_index=1,
            frame_idx=45,
            timestamp=1.5,
            pixel_coord=Point2D(x=620.0, y=710.0),
            court_coord=Point2D(x=7.4, y=-1.8),
            speed_kmh=142.5,
            is_in=True,
            distance_to_line_cm=-32.4
        ),
        BounceEvent(
            bounce_index=2,
            frame_idx=92,
            timestamp=3.06,
            pixel_coord=Point2D(x=410.0, y=520.0),
            court_coord=Point2D(x=-6.8, y=2.9),
            speed_kmh=128.0,
            is_in=True,
            distance_to_line_cm=-18.6
        ),
        BounceEvent(
            bounce_index=3,
            frame_idx=138,
            timestamp=4.6,
            pixel_coord=Point2D(x=780.0, y=740.0),
            court_coord=Point2D(x=8.9, y=3.7),
            speed_kmh=136.2,
            is_in=True,
            distance_to_line_cm=-4.2
        ),
        BounceEvent(
            bounce_index=4,
            frame_idx=184,
            timestamp=6.13,
            pixel_coord=Point2D(x=340.0, y=490.0),
            court_coord=Point2D(x=-11.95, y=-2.1),
            speed_kmh=151.8,
            is_in=False,
            distance_to_line_cm=6.5
        )
    ]

    summary = AnalysisSummary(
        total_shots=4,
        total_bounces=4,
        in_count=3,
        out_count=1,
        in_percentage=75.0,
        max_speed_kmh=151.8,
        avg_speed_kmh=139.6
    )

    return VideoAnalysisResponse(
        video_id="sample-demo-rally-001",
        filename="tennis_rally_sample.mp4",
        fps=30.0,
        total_frames=240,
        duration_seconds=8.0,
        bounces=bounces,
        summary=summary,
        trajectory_sample=[]
    )
