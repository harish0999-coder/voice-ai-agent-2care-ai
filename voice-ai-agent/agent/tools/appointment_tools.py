"""
Appointment Tool Implementations
These are the actual functions called by the LLM via tool orchestration.
All tools interact with the scheduling engine — no hardcoded responses.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_date(date_str: str) -> str:
    """Resolve natural language dates to YYYY-MM-DD."""
    date_str = date_str.lower().strip()
    today = datetime.now()

    if date_str in ("today", "आज", "இன்று"):
        return today.strftime("%Y-%m-%d")
    elif date_str in ("tomorrow", "कल", "நாளை"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_str in ("day after tomorrow", "परसों", "நாளை மறுதினம்"):
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    # Try to parse as YYYY-MM-DD already
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        pass

    # Weekday parsing
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in date_str:
            current_weekday = today.weekday()
            days_until = (i - current_weekday) % 7 or 7
            return (today + timedelta(days=days_until)).strftime("%Y-%m-%d")

    # Default: return as-is and let the DB validate
    return date_str


async def check_availability_tool(doctor_id: str, date: str) -> dict:
    """Check available time slots for a doctor on a given date."""
    from scheduler.appointment_engine.scheduler import AppointmentScheduler
    scheduler = AppointmentScheduler()

    resolved_date = _resolve_date(date)
    logger.info(f"Tool: check_availability | doctor={doctor_id} | date={resolved_date}")

    slots = await scheduler.get_available_slots(doctor_id=doctor_id, date=resolved_date)

    return {
        "success": True,
        "doctor_id": doctor_id,
        "date": resolved_date,
        "available_slots": slots,
        "slot_count": len(slots),
    }


async def book_appointment_tool(
    doctor_id: str,
    date: str,
    time_slot: str,
    notes: Optional[str] = None,
    patient_id: str = "session_patient"
) -> dict:
    """Book an appointment."""
    from scheduler.appointment_engine.scheduler import AppointmentScheduler
    scheduler = AppointmentScheduler()

    resolved_date = _resolve_date(date)
    logger.info(f"Tool: book_appointment | doctor={doctor_id} | date={resolved_date} | time={time_slot}")

    result = await scheduler.book_appointment(
        patient_id=patient_id,
        doctor_id=doctor_id,
        date=resolved_date,
        time_slot=time_slot,
        notes=notes,
    )
    return result


async def cancel_appointment_tool(appointment_id: str) -> dict:
    """Cancel an existing appointment."""
    from scheduler.appointment_engine.scheduler import AppointmentScheduler
    scheduler = AppointmentScheduler()

    logger.info(f"Tool: cancel_appointment | id={appointment_id}")
    result = await scheduler.cancel_appointment(appointment_id)
    return result


async def reschedule_appointment_tool(
    appointment_id: str,
    new_date: str,
    new_time_slot: str,
) -> dict:
    """Reschedule an appointment to a new date/time."""
    from scheduler.appointment_engine.scheduler import AppointmentScheduler
    scheduler = AppointmentScheduler()

    resolved_date = _resolve_date(new_date)
    logger.info(
        f"Tool: reschedule_appointment | id={appointment_id} | "
        f"new_date={resolved_date} | new_time={new_time_slot}"
    )

    result = await scheduler.reschedule_appointment(
        appointment_id=appointment_id,
        new_date=resolved_date,
        new_time_slot=new_time_slot,
    )
    return result


async def list_doctors_tool(specialty: Optional[str] = None) -> dict:
    """List available doctors with optional specialty filter."""
    from database.models.doctor import DoctorModel
    logger.info(f"Tool: list_doctors | specialty={specialty}")

    doctors = await DoctorModel.list_doctors(specialty=specialty)
    return {
        "success": True,
        "doctors": doctors,
        "count": len(doctors),
    }
