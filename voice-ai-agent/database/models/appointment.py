"""Appointment database model"""
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class AppointmentModel:
    @staticmethod
    async def list_appointments(patient_id=None, doctor_id=None, date=None) -> list:
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                conditions, params = [], []
                if _using_sqlite:
                    if patient_id: conditions.append("patient_id=?"); params.append(patient_id)
                    if doctor_id:  conditions.append("doctor_id=?");  params.append(doctor_id)
                    if date:       conditions.append("date=?");        params.append(date)
                    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                    cursor = await conn.execute(
                        f"SELECT * FROM appointments {where} ORDER BY date, time_slot", params)
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
                else:
                    idx = 1
                    if patient_id: conditions.append(f"patient_id=${idx}"); params.append(patient_id); idx+=1
                    if doctor_id:  conditions.append(f"doctor_id=${idx}");  params.append(doctor_id);  idx+=1
                    if date:       conditions.append(f"date=${idx}");       params.append(date);       idx+=1
                    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                    rows = await conn.fetch(
                        f"SELECT * FROM appointments {where} ORDER BY date, time_slot", *params)
                    return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"list_appointments DB error: {e}")
            return []
