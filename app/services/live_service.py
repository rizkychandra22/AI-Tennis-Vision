import time
import cv2
import numpy as np
from typing import Optional, List, Tuple
from app.schemas.live import LiveFrameTelemetry
from app.schemas.analysis import Point2D, BounceEvent
from app.vision.pipeline import VisionPipeline
from app.streaming.session_manager import session_manager


class LiveService:
    """Service to process real-time incoming video frames from mobile camera or RTSP."""

    def __init__(self):
        self.pipeline = VisionPipeline()
        self.frame_counter = 0
        self.recent_trajectory: List[Point2D] = []

    def set_calibration(self, corners: List[Tuple[float, float]]):
        self.pipeline.set_calibration(corners)

    def process_frame_bytes(self, image_bytes: bytes) -> LiveFrameTelemetry:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return self.process_frame(frame)

    def process_frame(self, frame: Optional[np.ndarray]) -> LiveFrameTelemetry:
        self.frame_counter += 1
        now = time.time()

        if frame is None:
            return LiveFrameTelemetry(
                frame_idx=self.frame_counter,
                timestamp=round(now, 3),
                ball_detected=False,
                current_speed_kmh=0.0,
                bounce_detected=False,
                last_bounce=None,
                recent_trajectory=self.recent_trajectory
            )

        ball_detected, ball_pixel, speed, bounce_event = self.pipeline.process_frame(
            frame=frame,
            frame_idx=self.frame_counter,
            timestamp=now
        )

        if ball_pixel:
            self.recent_trajectory.append(Point2D(x=round(ball_pixel[0], 1), y=round(ball_pixel[1], 1)))
            if len(self.recent_trajectory) > 20:
                self.recent_trajectory.pop(0)

        return LiveFrameTelemetry(
            frame_idx=self.frame_counter,
            timestamp=round(now, 3),
            ball_detected=ball_detected,
            current_speed_kmh=speed,
            bounce_detected=bounce_event is not None,
            last_bounce=bounce_event,
            recent_trajectory=self.recent_trajectory
        )


live_service = LiveService()
