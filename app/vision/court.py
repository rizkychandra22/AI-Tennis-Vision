import numpy as np
import cv2
from typing import Tuple, List, Optional
from app.core.config import settings
from app.schemas.analysis import Point2D


class CourtModel:
    """Represents ITF standard tennis court geometry and manages coordinate
    transformations between 2D camera pixel coordinates and 2D metric court space.

    Standard metric court coordinate system (origin (0,0) at court center, i.e., at the net):
    - X axis: Long axis along the court length [-11.885m, +11.885m]
    - Y axis: Lateral axis across the court width [-4.115m, +4.115m] (singles)
    """

    def __init__(self, homography_matrix: Optional[np.ndarray] = None):
        self.length = settings.COURT_LENGTH              # 23.77 m
        self.singles_width = settings.COURT_SINGLES_WIDTH # 8.23 m
        self.doubles_width = settings.COURT_DOUBLES_WIDTH # 10.97 m
        self.service_dist = settings.SERVICE_LINE_DIST    # 6.40 m
        self.ball_radius = settings.BALL_RADIUS_METERS    # 0.033 m

        self.half_len = self.length / 2.0                 # 11.885 m
        self.half_width = self.singles_width / 2.0        # 4.115 m
        self.half_doubles = self.doubles_width / 2.0      # 5.485 m

        self.H_pixel_to_court: Optional[np.ndarray] = homography_matrix
        self.H_court_to_pixel: Optional[np.ndarray] = None
        if homography_matrix is not None:
            self._update_inverse_homography()

    def set_homography_from_corners(self, pixel_corners: List[Tuple[float, float]]):
        """Calculates homography matrix given 4 court corners in camera image:
        0: Top-Left (far-left baseline intersection) -> (-half_len, -half_width)
        1: Top-Right (far-right baseline intersection) -> (-half_len, +half_width)
        2: Bottom-Right (near-right baseline intersection) -> (+half_len, +half_width)
        3: Bottom-Left (near-left baseline intersection) -> (+half_len, -half_width)
        """
        if len(pixel_corners) != 4:
            raise ValueError("Exactly 4 corner points required for homography calibration.")

        src_pts = np.array(pixel_corners, dtype=np.float32)
        dst_pts = np.array([
            [-self.half_len, -self.half_width],
            [-self.half_len, self.half_width],
            [self.half_len, self.half_width],
            [self.half_len, -self.half_width]
        ], dtype=np.float32)

        self.H_pixel_to_court, _ = cv2.findHomography(src_pts, dst_pts)
        self._update_inverse_homography()

    def set_default_perspective(self, frame_width: int, frame_height: int):
        """Sets heuristic default 4-point perspective for standard behind-baseline camera view."""
        w_f, h_f = float(frame_width), float(frame_height)
        default_corners = [
            (w_f * 0.28, h_f * 0.42),
            (w_f * 0.72, h_f * 0.42),
            (w_f * 0.88, h_f * 0.85),
            (w_f * 0.12, h_f * 0.85),
        ]
        self.set_homography_from_corners(default_corners)

    def _update_inverse_homography(self):
        if self.H_pixel_to_court is not None:
            self.H_court_to_pixel = np.linalg.inv(self.H_pixel_to_court)

    def pixel_to_court(self, px: float, py: float) -> Tuple[float, float]:
        """Converts camera pixel (px, py) to metric court coordinates (X, Y) in meters."""
        if self.H_pixel_to_court is None:
            return (0.0, 0.0)

        vec = np.array([px, py, 1.0], dtype=np.float32).reshape(3, 1)
        res = np.dot(self.H_pixel_to_court, vec)
        w = res[2, 0]
        if abs(w) < 1e-6:
            return (0.0, 0.0)
        return float(res[0, 0] / w), float(res[1, 0] / w)

    def court_to_pixel(self, cx: float, cy: float) -> Tuple[float, float]:
        """Converts metric court coordinates (cx, cy) to camera pixel coordinates."""
        if self.H_court_to_pixel is None:
            return (0.0, 0.0)

        vec = np.array([cx, cy, 1.0], dtype=np.float32).reshape(3, 1)
        res = np.dot(self.H_court_to_pixel, vec)
        w = res[2, 0]
        if abs(w) < 1e-6:
            return (0.0, 0.0)
        return float(res[0, 0] / w), float(res[1, 0] / w)

    def is_calibrated(self) -> bool:
        return self.H_pixel_to_court is not None
