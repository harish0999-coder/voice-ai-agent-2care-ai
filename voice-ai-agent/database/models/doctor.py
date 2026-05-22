"""Doctor database model"""
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class DoctorModel:
    @staticmethod
    async def list_doctors(specialty: Optional[str] = None) -> list:
        try:
            from database.connection import get_db_connection, _using_sqlite
            async with get_db_connection() as conn:
                if _using_sqlite:
                    if specialty:
                        cursor = await conn.execute(
                            "SELECT * FROM doctors WHERE specialty LIKE ? AND is_active=1",
                            (f"%{specialty.lower()}%",))
                    else:
                        cursor = await conn.execute("SELECT * FROM doctors WHERE is_active=1")
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
                else:
                    if specialty:
                        rows = await conn.fetch(
                            "SELECT * FROM doctors WHERE specialty ILIKE $1 AND is_active=TRUE",
                            f"%{specialty}%")
                    else:
                        rows = await conn.fetch("SELECT * FROM doctors WHERE is_active=TRUE")
                    return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"list_doctors DB error: {e}")
            return []
