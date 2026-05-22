"""
2Care.ai Voice AI Agent - Main FastAPI Application
Real-Time Multilingual Clinical Appointment Booking System
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env FIRST — before any import that reads os.getenv
from dotenv import load_dotenv
for _p in [Path(".env"), Path(__file__).parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routes import appointments, health, campaigns, websocket_handler
from backend.middleware.logging_middleware import LatencyLoggingMiddleware
from database.connection import init_db
from memory.session_memory.redis_session import init_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Voice AI Agent...")
    llm = os.getenv("LLM_PROVIDER", "openai")
    key = os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    logger.info(f"LLM_PROVIDER={llm} | API key loaded={'YES' if key else 'NO — set in .env!'}")
    await init_db()
    await init_redis()
    logger.info("All services initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="2Care.ai Voice AI Agent",
    description="Real-Time Multilingual Clinical Appointment Booking System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LatencyLoggingMiddleware)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(appointments.router, prefix="/api/appointments", tags=["Appointments"])
app.include_router(campaigns.router, prefix="/api/campaigns", tags=["Campaigns"])
app.include_router(websocket_handler.router, tags=["WebSocket"])

# Serve frontend at / so no port confusion
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(_frontend / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
