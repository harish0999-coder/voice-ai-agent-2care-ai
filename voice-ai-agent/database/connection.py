"""
Database Connection Manager
Supports PostgreSQL via asyncpg with SQLite fallback for local dev
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_db_path() -> str:
    """Return a writable SQLite path that works on Windows, Linux, and Mac."""
    env_path = os.getenv("SQLITE_PATH", "")
    if env_path:
        return env_path
    # Use the system temp directory — always writable on all platforms
    return str(Path(tempfile.gettempdir()) / "voice_agent.db")

_pool = None
_sqlite_conn = None
_using_sqlite = False


async def init_db():
    global _pool, _sqlite_conn, _using_sqlite
    db_url = os.getenv("DATABASE_URL", "")

    if db_url.startswith("postgresql"):
        try:
            import asyncpg
            _pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            logger.info("PostgreSQL connected")
            await _run_migrations_pg()
            return
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable ({e}) — falling back to SQLite")

    try:
        import aiosqlite
        _using_sqlite = True
        db_path = _default_db_path()
        _sqlite_conn = await aiosqlite.connect(db_path)
        _sqlite_conn.row_factory = aiosqlite.Row
        logger.info(f"SQLite connected: {db_path}")
        await _run_migrations_sqlite()
    except ImportError:
        logger.warning("aiosqlite not installed — DB ops will be no-ops")


@asynccontextmanager
async def get_db_connection():
    if _using_sqlite and _sqlite_conn:
        yield _sqlite_conn
    elif _pool:
        async with _pool.acquire() as conn:
            yield conn
    else:
        raise RuntimeError("No database connection available")


async def _run_migrations_pg():
    async with get_db_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id VARCHAR(20) PRIMARY KEY,
                patient_id VARCHAR(100) NOT NULL,
                doctor_id VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                time_slot VARCHAR(10) NOT NULL,
                notes TEXT DEFAULT '',
                status VARCHAR(20) DEFAULT 'confirmed',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id VARCHAR(100) PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                specialty VARCHAR(100) NOT NULL,
                hospital VARCHAR(200),
                available_days VARCHAR(50) DEFAULT 'Mon-Sat',
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS patient_memory (
                patient_id VARCHAR(100) PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await _seed_doctors_pg(conn)
        logger.info("PostgreSQL migrations complete")


async def _run_migrations_sqlite():
    await _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            doctor_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            notes TEXT DEFAULT '',
            status TEXT DEFAULT 'confirmed',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            hospital TEXT,
            available_days TEXT DEFAULT 'Mon-Sat',
            is_active INTEGER DEFAULT 1
        )
    """)
    await _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_memory (
            patient_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await _sqlite_conn.commit()
    await _seed_doctors_sqlite()
    logger.info("SQLite migrations complete")


async def _seed_doctors_pg(conn):
    for d in _get_sample_doctors():
        await conn.execute("""
            INSERT INTO doctors (id, name, specialty, hospital, available_days)
            VALUES ($1,$2,$3,$4,$5) ON CONFLICT (id) DO NOTHING
        """, d["id"], d["name"], d["specialty"], d["hospital"], d["available_days"])


async def _seed_doctors_sqlite():
    for d in _get_sample_doctors():
        await _sqlite_conn.execute("""
            INSERT OR IGNORE INTO doctors (id, name, specialty, hospital, available_days)
            VALUES (?,?,?,?,?)
        """, (d["id"], d["name"], d["specialty"], d["hospital"], d["available_days"]))
    await _sqlite_conn.commit()


def _get_sample_doctors():
    return [
        {"id": "DR001", "name": "Dr. Ananya Sharma",  "specialty": "cardiologist",    "hospital": "Apollo Hospital",        "available_days": "Mon-Sat"},
        {"id": "DR002", "name": "Dr. Rajesh Kumar",   "specialty": "dermatologist",   "hospital": "Fortis Clinic",          "available_days": "Mon-Fri"},
        {"id": "DR003", "name": "Dr. Priya Nair",     "specialty": "pediatrician",    "hospital": "Rainbow Children's",     "available_days": "Mon-Sat"},
        {"id": "DR004", "name": "Dr. Venkat Rao",     "specialty": "orthopedic",      "hospital": "Apollo Hospital",        "available_days": "Tue-Sat"},
        {"id": "DR005", "name": "Dr. Meena Iyer",     "specialty": "gynecologist",    "hospital": "Cloudnine Hospital",     "available_days": "Mon-Fri"},
        {"id": "DR006", "name": "Dr. Suresh Babu",    "specialty": "neurologist",     "hospital": "Manipal Hospital",       "available_days": "Mon-Sat"},
        {"id": "DR007", "name": "Dr. Lakshmi Devi",   "specialty": "general physician","hospital": "Primary Care Clinic",   "available_days": "Mon-Sat"},
        {"id": "DR008", "name": "Dr. Arjun Patel",    "specialty": "ophthalmologist", "hospital": "Sankara Eye",            "available_days": "Mon-Fri"},
    ]
