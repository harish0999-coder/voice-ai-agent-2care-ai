"""Health check endpoint"""
from fastapi import APIRouter
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "running",
        "service": "2Care.ai Voice AI Agent",
        "timestamp": time.time(),
        "version": "1.0.0"
    }
