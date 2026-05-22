"""
Persistent Patient Memory
Stores long-term patient context across sessions in PostgreSQL.
Falls back to JSON file if DB unavailable.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import tempfile
FALLBACK_FILE = Path(tempfile.gettempdir()) / "patient_memory_fallback.json"


class PatientMemory:
    """
    Long-term memory for returning patients.
    
    Stores:
    - preferred_language
    - preferred_doctor
    - preferred_hospital  
    - last_appointment date/doctor
    - total_appointments count
    - communication preferences
    """

    async def get_context(self, patient_id: str) -> dict:
        """Retrieve patient context for prompt injection."""
        if patient_id in ("anonymous", "session_patient", None):
            return {}

        try:
            record = await self._load(patient_id)
            if record:
                return {
                    "patient_id": patient_id,
                    "preferred_language": record.get("preferred_language", "en"),
                    "preferred_doctor": record.get("preferred_doctor"),
                    "preferred_hospital": record.get("preferred_hospital"),
                    "last_appointment": record.get("last_appointment"),
                    "total_appointments": record.get("total_appointments", 0),
                }
        except Exception as e:
            logger.warning(f"PatientMemory.get_context failed for {patient_id}: {e}")
        return {}

    async def update_context(
        self,
        patient_id: str,
        language: Optional[str] = None,
        last_intent: Optional[str] = None,
        doctor_id: Optional[str] = None,
        appointment_date: Optional[str] = None,
    ):
        """Update patient persistent context after each interaction."""
        if patient_id in ("anonymous", "session_patient", None):
            return

        try:
            record = await self._load(patient_id) or {}

            if language:
                record["preferred_language"] = language
            if doctor_id:
                record["preferred_doctor"] = doctor_id
            if appointment_date:
                record["last_appointment"] = appointment_date
                record["total_appointments"] = record.get("total_appointments", 0) + 1

            record["last_seen"] = time.time()
            record["patient_id"] = patient_id

            await self._save(patient_id, record)

        except Exception as e:
            logger.warning(f"PatientMemory.update_context failed for {patient_id}: {e}")

    # ── Storage Backend ───────────────────────────────────────────────────────

    async def _load(self, patient_id: str) -> Optional[dict]:
        """Load patient record from DB or fallback file."""
        try:
            from database.connection import get_db_connection
            async with get_db_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM patient_memory WHERE patient_id = $1",
                    patient_id
                )
                if row:
                    return json.loads(row["data"])
        except Exception:
            pass

        # Fallback: JSON file
        return self._load_fallback(patient_id)

    async def _save(self, patient_id: str, data: dict):
        """Save patient record to DB or fallback file."""
        try:
            from database.connection import get_db_connection
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO patient_memory (patient_id, data, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (patient_id) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = NOW()
                    """,
                    patient_id, json.dumps(data)
                )
                return
        except Exception:
            pass

        # Fallback: JSON file
        self._save_fallback(patient_id, data)

    def _load_fallback(self, patient_id: str) -> Optional[dict]:
        if not FALLBACK_FILE.exists():
            return None
        try:
            all_data = json.loads(FALLBACK_FILE.read_text())
            return all_data.get(patient_id)
        except Exception:
            return None

    def _save_fallback(self, patient_id: str, data: dict):
        try:
            all_data = {}
            if FALLBACK_FILE.exists():
                all_data = json.loads(FALLBACK_FILE.read_text())
            all_data[patient_id] = data
            FALLBACK_FILE.write_text(json.dumps(all_data, indent=2))
        except Exception as e:
            logger.warning(f"Fallback file save failed: {e}")
