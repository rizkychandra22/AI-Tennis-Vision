import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
from pydantic import BaseModel
from app.schemas.live import LiveFrameTelemetry, CalibrationRequest, SessionInfo
from app.services.live_service import live_service
from app.streaming.session_manager import session_manager
from app.streaming.webrtc import webrtc_manager

router = APIRouter()


class WebRTCOfferPayload(BaseModel):
    sdp: str
    type: str


@router.post("/calibrate")
def calibrate_camera(payload: CalibrationRequest):
    """Calibrate camera homography with 4 corner points."""
    if len(payload.corners) != 4:
        return {"success": False, "error": "Exactly 4 corners are required."}

    corners = [(c.x, c.y) for c in payload.corners]
    live_service.set_calibration(corners)
    return {"success": True, "calibrated": True}


@router.get("/sessions", response_model=List[SessionInfo])
def get_active_sessions():
    """List active live streaming sessions."""
    return session_manager.list_active_sessions()


@router.post("/webrtc/offer")
async def handle_webrtc_offer(payload: WebRTCOfferPayload):
    """Handle WebRTC negotiation offer for real-time live video stream."""
    answer = await webrtc_manager.handle_offer(sdp=payload.sdp, type_=payload.type)
    return answer


@router.websocket("/ws")
async def websocket_live_stream(websocket: WebSocket):
    """Real-time WebSocket endpoint for smartphone camera or live video streaming.
    Receives JPEG binary bytes or JSON control frames, and streams back LiveFrameTelemetry.
    """
    await websocket.accept()
    session = session_manager.create_session()

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                telemetry = live_service.process_frame_bytes(data["bytes"])
                session.total_frames += 1
                if telemetry.bounce_detected:
                    session.total_bounces += 1
                await websocket.send_text(telemetry.model_dump_json())
            elif "text" in data and data["text"]:
                msg = data["text"]
                if msg == "ping":
                    await websocket.send_text("pong")
                else:
                    try:
                        parsed = json.loads(msg)
                        if parsed.get("action") == "calibrate" and "corners" in parsed:
                            pts = [(float(p["x"]), float(p["y"])) for p in parsed["corners"]]
                            live_service.set_calibration(pts)
                            session.calibrated = True
                            await websocket.send_text(json.dumps({"calibrated": True}))
                    except Exception:
                        pass
    except WebSocketDisconnect:
        session_manager.close_session(session.session_id)
    except Exception:
        session_manager.close_session(session.session_id)
