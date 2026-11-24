from typing import List, Tuple, Optional


class BounceDetector:
    """Detects tennis ball bounces from trajectory history using vertical velocity reversal
    and inflection point analysis.
    """

    def __init__(self, min_frame_gap: int = 12, min_trajectory_len: int = 5):
        self.min_frame_gap = min_frame_gap
        self.min_trajectory_len = min_trajectory_len
        self.last_bounce_frame = -min_frame_gap

    def detect_in_stream(
        self,
        recent_points: List[Tuple[int, float, float]]
    ) -> Optional[Tuple[int, float, float]]:
        """Detects if a bounce occurred in recent trajectory points [(frame, x, y)]."""
        if len(recent_points) < self.min_trajectory_len:
            return None

        latest_frame = recent_points[-1][0]
        if latest_frame - self.last_bounce_frame < self.min_frame_gap:
            return None

        pts = recent_points[-5:]
        center_idx = 2
        cand_frame, cand_x, cand_y = pts[center_idx]

        # In image coordinates, y increases downward.
        # Bounce = local peak in y (deepest ground contact)
        is_local_max = (
            cand_y > pts[center_idx - 1][2] and
            cand_y > pts[center_idx - 2][2] and
            cand_y >= pts[center_idx + 1][2] and
            cand_y > pts[center_idx + 2][2]
        )

        if is_local_max and (cand_frame - self.last_bounce_frame >= self.min_frame_gap):
            self.last_bounce_frame = cand_frame
            return cand_frame, cand_x, cand_y

        return None

    def detect_in_batch(
        self,
        trajectory: List[Tuple[int, float, float]]
    ) -> List[Tuple[int, float, float]]:
        """Analyzes full trajectory [(frame, x, y), ...] and returns all detected bounces."""
        bounces: List[Tuple[int, float, float]] = []
        if len(trajectory) < self.min_trajectory_len:
            return bounces

        last_bounce = -self.min_frame_gap
        for i in range(2, len(trajectory) - 2):
            frame, x, y = trajectory[i]
            if frame - last_bounce < self.min_frame_gap:
                continue

            y_prev2 = trajectory[i - 2][2]
            y_prev1 = trajectory[i - 1][2]
            y_curr = y
            y_next1 = trajectory[i + 1][2]
            y_next2 = trajectory[i + 2][2]

            if (y_curr > y_prev1 and y_prev1 >= y_prev2 and
                y_curr >= y_next1 and y_next1 > y_next2):
                bounces.append((frame, x, y))
                last_bounce = frame

        return bounces
