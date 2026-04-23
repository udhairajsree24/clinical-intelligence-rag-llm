from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Clinical Intelligence Backend",
        "timestamp": datetime.utcnow().isoformat()
    }