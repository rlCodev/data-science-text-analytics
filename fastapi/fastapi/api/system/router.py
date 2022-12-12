import datetime
from fastapi import APIRouter
from loguru import logger


router = APIRouter()


@router.get("/_health")
def check_health():
    """Standard health checker"""
    response = {
        "timestamp": datetime.datetime.now().isoformat(),
        "message": "healthcheck ok!",
    }
    logger.info("Health ok!")
    return response