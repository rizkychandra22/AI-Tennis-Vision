import time
import uuid
from typing import Dict, Optional, List, Tuple
from app.schemas.live import SessionInfo
from app.vision.court import CourtModel


class LiveSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.is_active = True
        self.calibrated = False
        self.corners: Optional[List[Tuple[float, float]]] = None
        self.total_frames = 0
        self.total_bounces = 0

    def to_info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            is_active=self.is_active,
            calibrated=self.calibrated,
            total_frames_processed=self.total_frames,
            total_bounces_detected=self.total_bounces
        )


class SessionManager:
    """Manages active live streaming sessions across WebSockets and WebRTC connections."""

    def __init__(self):
        self.sessions: Dict[str, LiveSession] = {}

    def create_session(self, session_id: Optional[str] = None) -> LiveSession:
        sid = session_id or str(uuid.uuid4())
        session = LiveSession(sid)
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False

    def list_active_sessions(self) -> List[SessionInfo]:
        return [s.to_info() for s in self.sessions.values() if s.is_active]


session_manager = SessionManager()
