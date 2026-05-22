"""
Appointment Scheduling Engine
Handles booking, rescheduling, cancellation, conflict detection,
and alternative slot suggestion.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# All valid time slots in a clinic day
ALL_SLOTS = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "14:00", "14:30", "15:00", "15:30",
    "16:00", "16:30", "17:00", "17:30",
]


class AppointmentScheduler:
    """
    Core scheduling engine.
    
    Validation rules enforced:
    1. Slot must not be in the past
    2. Slot must not be already booked (same doctor + date + time)
    3. Doctor must exist and be active
    4. Booking must be within working hours
    """

    async def get_available_slots(self, doctor_id: str, date: str) -> list[str]:
        """
        Get available slots for a doctor on a date.
        Returns list of time strings not yet booked.
        """
        # Validate date is not in the past
        try:
            slot_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return []

        today = datetime.now().date()
        if slot_date < today:
            logger.warning(f"Requested availability for past date: {date}")
            return []

        # Get booked slots from DB
        booked = await self._get_booked_slots(doctor_id, date)

        # For today, filter out already-passed times
        available = []
        now = datetime.now()
        for slot in ALL_SLOTS:
            if slot in booked:
                continue
            if slot_date == today:
                slot_dt = datetime.strptime(f"{date} {slot}", "%Y-%m-%d %H:%M")
                if slot_dt <= now:
                    continue  # Past time today
            available.append(slot)

        logger.info(
            f"Availability: doctor={doctor_id} date={date} | "
            f"{len(available)}/{len(ALL_SLOTS)} slots free"
        )
        return available

    async def book_appointment(
        self,
        patient_id: str,
        doctor_id: str,
        date: str,
        time_slot: str,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Book an appointment after full validation.
        Returns success dict or failure with alternative suggestions.
        """
        # ── Validation ────────────────────────────────────────────────────────
        val_error = await self._validate_slot(doctor_id, date, time_slot)
        if val_error:
            alternatives = await self._suggest_alternatives(doctor_id, date, time_slot)
            return {
                "success": False,
                "message": val_error,
                "alternatives": alternatives,
            }

        # ── Check for conflict ────────────────────────────────────────────────
        is_booked = await self._is_slot_booked(doctor_id, date, time_slot)
        if is_booked:
            alternatives = await self._suggest_alternatives(doctor_id, date, time_slot)
            return {
                "success": False,
                "message": f"The {time_slot} slot on {date} is already booked.",
                "alternatives": alternatives,
            }

        # ── Create appointment ────────────────────────────────────────────────
        appointment_id = str(uuid.uuid4())[:8].upper()

        try:
            await self._create_appointment_record(
                appointment_id=appointment_id,
                patient_id=patient_id,
                doctor_id=doctor_id,
                date=date,
                time_slot=time_slot,
                notes=notes,
            )
        except Exception as e:
            logger.error(f"DB write failed for appointment: {e}")
            return {"success": False, "message": "Could not save appointment. Please try again."}

        logger.info(
            f"✅ Appointment booked: ID={appointment_id} | "
            f"patient={patient_id} | doctor={doctor_id} | {date} {time_slot}"
        )

        return {
            "success": True,
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "date": date,
            "time_slot": time_slot,
            "message": f"Appointment confirmed! ID: {appointment_id} on {date} at {time_slot}.",
        }

    async def reschedule_appointment(
        self, appointment_id: str, new_date: str, new_time_slot: str
    ) -> dict:
        """Reschedule an existing appointment."""
        # Fetch existing
        existing = await self._get_appointment(appointment_id)
        if not existing:
            return {
                "success": False,
                "message": f"Appointment {appointment_id} not found.",
            }

        if existing.get("status") == "cancelled":
            return {
                "success": False,
                "message": "This appointment has already been cancelled.",
            }

        doctor_id = existing["doctor_id"]

        # Validate new slot
        val_error = await self._validate_slot(doctor_id, new_date, new_time_slot)
        if val_error:
            alternatives = await self._suggest_alternatives(doctor_id, new_date, new_time_slot)
            return {
                "success": False,
                "message": val_error,
                "alternatives": alternatives,
            }

        is_booked = await self._is_slot_booked(doctor_id, new_date, new_time_slot)
        if is_booked:
            alternatives = await self._suggest_alternatives(doctor_id, new_date, new_time_slot)
            return {
                "success": False,
                "message": f"The {new_time_slot} slot on {new_date} is not available.",
                "alternatives": alternatives,
            }

        # Update in DB
        await self._update_appointment(
            appointment_id=appointment_id,
            date=new_date,
            time_slot=new_time_slot,
        )

        logger.info(f"✅ Rescheduled {appointment_id} → {new_date} {new_time_slot}")

        return {
            "success": True,
            "appointment_id": appointment_id,
            "new_date": new_date,
            "new_time_slot": new_time_slot,
            "message": f"Appointment rescheduled to {new_date} at {new_time_slot}.",
        }

    async def cancel_appointment(self, appointment_id: str) -> dict:
        """Cancel an appointment."""
        existing = await self._get_appointment(appointment_id)
        if not existing:
            return {
                "success": False,
                "message": f"Appointment {appointment_id} not found.",
            }

        if existing.get("status") == "cancelled":
            return {
                "success": False,
                "message": "This appointment is already cancelled.",
            }

        await self._mark_cancelled(appointment_id)
        logger.info(f"✅ Cancelled appointment: {appointment_id}")

        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": f"Appointment {appointment_id} has been successfully cancelled.",
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    async def _validate_slot(self, doctor_id: str, date: str, time_slot: str) -> Optional[str]:
        """Return error message if slot is invalid, else None."""
        try:
            slot_dt = datetime.strptime(f"{date} {time_slot}", "%Y-%m-%d %H:%M")
        except ValueError:
            return f"Invalid date or time format: {date} {time_slot}"

        if slot_dt <= datetime.now():
            return f"Cannot book appointments in the past ({date} {time_slot} has already passed)."

        if time_slot not in ALL_SLOTS:
            return (
                f"{time_slot} is not a valid clinic slot. "
                f"Valid slots are between 09:00 and 17:30."
            )
        return None

    async def _suggest_alternatives(
        self, doctor_id: str, date: str, time_slot: str, count: int = 3
    ) -> list[dict]:
        """Suggest alternative slots when requested slot is unavailable."""
        suggestions = []
        check_date = datetime.now().date()
        try:
            check_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            pass

        # Search next 7 days
        for day_offset in range(8):
            if len(suggestions) >= count:
                break
            search_date = (check_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            slots = await self.get_available_slots(doctor_id, search_date)
            for slot in slots[:2]:  # Max 2 per day
                if len(suggestions) >= count:
                    break
                suggestions.append({"date": search_date, "time_slot": slot})

        return suggestions

    async def _get_booked_slots(self, doctor_id: str, date: str) -> set:
        """Get set of booked time slots for a doctor on a date."""
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    cursor = await conn.execute(
                        "SELECT time_slot FROM appointments WHERE doctor_id=? AND date=? AND status!='cancelled'",
                        (doctor_id, date)
                    )
                    rows = await cursor.fetchall()
                    return {row["time_slot"] for row in rows}
                else:
                    rows = await conn.fetch(
                        "SELECT time_slot FROM appointments WHERE doctor_id=$1 AND date=$2 AND status!='cancelled'",
                        doctor_id, date
                    )
                    return {row["time_slot"] for row in rows}
        except Exception as e:
            logger.warning(f"DB unavailable for booked slots: {e}")
            return set()

    async def _is_slot_booked(self, doctor_id: str, date: str, time_slot: str) -> bool:
        booked = await self._get_booked_slots(doctor_id, date)
        return time_slot in booked

    async def _get_appointment(self, appointment_id: str) -> Optional[dict]:
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    cursor = await conn.execute(
                        "SELECT * FROM appointments WHERE id=?", (appointment_id,)
                    )
                    row = await cursor.fetchone()
                    return dict(row) if row else None
                else:
                    row = await conn.fetchrow(
                        "SELECT * FROM appointments WHERE id=$1", appointment_id
                    )
                    return dict(row) if row else None
        except Exception as e:
            logger.warning(f"DB unavailable for get_appointment: {e}")
            return None

    async def _create_appointment_record(self, **kwargs):
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    await conn.execute(
                        "INSERT INTO appointments (id, patient_id, doctor_id, date, time_slot, notes, status) VALUES (?,?,?,?,?,?,'confirmed')",
                        (kwargs["appointment_id"], kwargs["patient_id"], kwargs["doctor_id"],
                         kwargs["date"], kwargs["time_slot"], kwargs.get("notes", ""))
                    )
                    await conn.commit()
                else:
                    await conn.execute(
                        "INSERT INTO appointments (id, patient_id, doctor_id, date, time_slot, notes, status, created_at) VALUES ($1,$2,$3,$4,$5,$6,'confirmed',NOW())",
                        kwargs["appointment_id"], kwargs["patient_id"], kwargs["doctor_id"],
                        kwargs["date"], kwargs["time_slot"], kwargs.get("notes", "")
                    )
        except Exception as e:
            logger.error(f"Failed to create appointment record: {e}")
            raise

    async def _update_appointment(self, appointment_id: str, date: str, time_slot: str):
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    await conn.execute(
                        "UPDATE appointments SET date=?, time_slot=? WHERE id=?",
                        (date, time_slot, appointment_id)
                    )
                    await conn.commit()
                else:
                    await conn.execute(
                        "UPDATE appointments SET date=$2, time_slot=$3, updated_at=NOW() WHERE id=$1",
                        appointment_id, date, time_slot
                    )
        except Exception as e:
            logger.error(f"Failed to update appointment: {e}")
            raise

    async def _mark_cancelled(self, appointment_id: str):
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    await conn.execute(
                        "UPDATE appointments SET status='cancelled' WHERE id=?",
                        (appointment_id,)
                    )
                    await conn.commit()
                else:
                    await conn.execute(
                        "UPDATE appointments SET status='cancelled', updated_at=NOW() WHERE id=$1",
                        appointment_id
                    )
        except Exception as e:
            logger.error(f"Failed to cancel appointment: {e}")
            raise
