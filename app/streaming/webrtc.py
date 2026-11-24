import asyncio
from typing import Optional, Dict
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False


class WebRTCManager:
    """Manages WebRTC PeerConnections for ultra-low latency live camera streaming."""

    def __init__(self):
        self.pcs: Dict[str, any] = {}

    def is_available(self) -> bool:
        return AIORTC_AVAILABLE

    async def handle_offer(self, sdp: str, type_: str) -> Optional[dict]:
        """Handles SDP Offer from WebRTC frontend client and returns SDP Answer."""
        if not AIORTC_AVAILABLE:
            return {"error": "aiortc is not available on this host"}

        pc = RTCPeerConnection()
        self.pcs[id(pc)] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.pop(id(pc), None)

        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def close_all(self):
        for pc in list(self.pcs.values()):
            await pc.close()
        self.pcs.clear()


webrtc_manager = WebRTCManager()
