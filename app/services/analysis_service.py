import os
import json
from typing import Optional, Dict
from app.core.config import settings
from app.schemas.analysis import VideoAnalysisResponse


class AnalysisService:
    """Service for storing, retrieving, and managing match analysis records."""

    def __init__(self):
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        self._memory_cache: Dict[str, VideoAnalysisResponse] = {}

    def save_result(self, result: VideoAnalysisResponse):
        self._memory_cache[result.video_id] = result
        file_path = os.path.join(settings.RESULTS_DIR, f"{result.video_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"[AnalysisService] Error saving result to disk: {e}")

    def get_result(self, video_id: str) -> Optional[VideoAnalysisResponse]:
        if video_id in self._memory_cache:
            return self._memory_cache[video_id]

        file_path = os.path.join(settings.RESULTS_DIR, f"{video_id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = VideoAnalysisResponse.model_validate(data)
                    self._memory_cache[video_id] = res
                    return res
            except Exception:
                pass
        return None


analysis_service = AnalysisService()
