import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from app.core.config import settings
from app.vision.ball_detector import BallDetector


class BallTracker:
    """Maintains trajectory history of tennis ball across video frames,
    smooths coordinates, and interpolates missing detections.
    """

    def __init__(
        self,
        detector: Optional[BallDetector] = None,
        buffer_size: int = settings.TRACKER_BUFFER_SIZE
    ):
        self.detector = detector or BallDetector()
        self.buffer_size = buffer_size
        self.trajectory: deque = deque(maxlen=buffer_size)  # [(frame_idx, x, y, conf)]

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Optional[Tuple[float, float, float]]:
        """Processes frame, detects ball, and updates trajectory buffer.
        Returns: (x, y, confidence) or None.
        """
        detection = self.detector.detect(frame)
        if detection is not None:
            cx, cy, conf = detection
            self.trajectory.append((frame_idx, cx, cy, conf))
            return cx, cy, conf

        # Missed-detection interpolation (constant-velocity extrapolation for 1-2 frames)
        if len(self.trajectory) >= 2:
            last_frame, last_x, last_y, last_conf = self.trajectory[-1]
            prev_frame, prev_x, prev_y, _ = self.trajectory[-2]
            frame_gap = frame_idx - last_frame

            if 1 <= frame_gap <= 2 and (last_frame - prev_frame) > 0:
                vx = (last_x - prev_x) / (last_frame - prev_frame)
                vy = (last_y - prev_y) / (last_frame - prev_frame)
                pred_x = last_x + vx * frame_gap
                pred_y = last_y + vy * frame_gap
                pred_conf = max(0.2, last_conf * 0.8)
                self.trajectory.append((frame_idx, pred_x, pred_y, pred_conf))
                return pred_x, pred_y, pred_conf

        return None

    def get_recent_points(self) -> List[Tuple[int, float, float]]:
        return [(p[0], p[1], p[2]) for p in self.trajectory]

    def reset(self):
        self.trajectory.clear()
