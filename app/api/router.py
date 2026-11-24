import platform
from fastapi import APIRouter
from app.core.config import settings
from app.api import analysis, live

api_router = APIRouter()


@api_router.get("/health", tags=["Health"])
def health_check():
    """Health check and AI environment diagnostics."""
    torch_version = "not installed"
    torch_device = "none"
    try:
        import torch
        torch_version = torch.__version__
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    yolo_status = "available"
    try:
        import ultralytics
    except ImportError:
        yolo_status = "unavailable"

    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": {
            "version": torch_version,
            "device": torch_device
        },
        "ultralytics": yolo_status,
        "court_standards": {
            "singles_length_m": settings.COURT_LENGTH,
            "singles_width_m": settings.COURT_SINGLES_WIDTH,
            "ball_radius_m": settings.BALL_RADIUS_METERS
        }
    }


# Include modular feature routers
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(live.router, prefix="/live", tags=["Live Stream"])
