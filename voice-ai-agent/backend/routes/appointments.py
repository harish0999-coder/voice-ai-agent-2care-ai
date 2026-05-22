"""
Appointments REST API Routes
CRUD operations for appointment management
"""

from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure .env loaded in this module too
for _p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break

from database.models.appointment import AppointmentModel
from scheduler.appointment_engine.scheduler import AppointmentScheduler

router = APIRouter()
scheduler = AppointmentScheduler()


class BookAppointmentRequest(BaseModel):
    patient_id: str
    doctor_id: str
    date: str          # YYYY-MM-DD
    time_slot: str     # HH:MM
    notes: Optional[str] = None
    language: Optional[str] = "en"


class RescheduleRequest(BaseModel):
    appointment_id: str
    new_date: str
    new_time_slot: str


@router.get("/")
async def list_appointments(
    patient_id: Optional[str] = Query(None),
    doctor_id: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
):
    """List appointments with optional filters."""
    return await AppointmentModel.list_appointments(
        patient_id=patient_id,
        doctor_id=doctor_id,
        date=date
    )


@router.post("/book")
async def book_appointment(req: BookAppointmentRequest):
    """Book a new appointment."""
    result = await scheduler.book_appointment(
        patient_id=req.patient_id,
        doctor_id=req.doctor_id,
        date=req.date,
        time_slot=req.time_slot,
        notes=req.notes,
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=409, detail=result["message"])


@router.post("/reschedule")
async def reschedule_appointment(req: RescheduleRequest):
    """Reschedule an existing appointment."""
    result = await scheduler.reschedule_appointment(
        appointment_id=req.appointment_id,
        new_date=req.new_date,
        new_time_slot=req.new_time_slot,
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=409, detail=result["message"])


@router.delete("/{appointment_id}")
async def cancel_appointment(appointment_id: str):
    """Cancel an appointment."""
    result = await scheduler.cancel_appointment(appointment_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result["message"])


@router.get("/availability/{doctor_id}")
async def check_availability(doctor_id: str, date: str = Query(...)):
    """Check doctor availability for a given date."""
    slots = await scheduler.get_available_slots(doctor_id=doctor_id, date=date)
    return {"doctor_id": doctor_id, "date": date, "available_slots": slots}


class VoiceTextRequest(BaseModel):
    text: str
    session_id: str
    patient_id: Optional[str] = "patient_demo_001"
    language: Optional[str] = "en"


@router.post("/voice-text")
async def voice_text(req: VoiceTextRequest):
    """Process text input directly (bypasses STT — uses same session memory as WebSocket)."""
    import os
    from agent.reasoning.voice_agent import VoiceAgent
    from memory.session_memory.redis_session import SessionMemory
    from services.language_detection.lang_detector import LanguageDetector

    session_memory = SessionMemory(req.session_id)
    await session_memory.set("patient_id", req.patient_id)

    # Detect language from text
    detector = LanguageDetector()
    detected_lang = await detector.detect(req.text)
    language = detected_lang if detected_lang != "en" else req.language

    agent = VoiceAgent(session_id=req.session_id, session_memory=session_memory)
    result = await agent.process(transcript=req.text, language=language)
    result["detected_language"] = language
    return result


@router.get("/doctors")
async def list_doctors(specialty: Optional[str] = Query(None)):
    """List available doctors, optionally filtered by specialty."""
    from database.models.doctor import DoctorModel
    return await DoctorModel.list_doctors(specialty=specialty)
