"""
Redis Session Memory
Stores per-session conversation history and state with TTL.
Falls back to in-memory dict if Redis is unavailable.
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_memory_fallback: dict = {}   # In-memory fallback when Redis unavailable

SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "3600"))     # 1 hour
MAX_HISTORY = int(os.getenv("MAX_HISTORY_TURNS", "20"))         # Keep last N turns


async def init_redis():
    """Initialize Redis connection at startup."""
    global _redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info(f"✅ Redis connected: {redis_url}")
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}) — using in-memory fallback")
        _redis_client = None


class SessionMemory:
    """
    Per-session memory that stores:
    - Conversation history (user + assistant turns)
    - Session variables (patient_id, preferred_language, pending intents)
    
    Keys:
    - session:{id}:history  → JSON list of message dicts
    - session:{id}:vars     → JSON dict of key-value state
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._history_key = f"session:{session_id}:history"
        self._vars_key = f"session:{session_id}:vars"

    # ── History Management ────────────────────────────────────────────────────

    async def get_history(self) -> list[dict]:
        """Return conversation history as list of {role, content} dicts."""
        raw = await self._get(self._history_key)
        if raw:
            history = json.loads(raw)
            # Trim to last N turns (each turn = 2 messages)
            if len(history) > MAX_HISTORY * 2:
                history = history[-(MAX_HISTORY * 2):]
            return history
        return []

    async def add_message(self, role: str, content: str):
        """Append a message to conversation history."""
        history = await self.get_history()
        history.append({
            "role": role,
            "content": content,
        })
        await self._set(self._history_key, json.dumps(history), ttl=SESSION_TTL)

    async def clear_history(self):
        """Clear conversation history."""
        await self._delete(self._history_key)

    # ── Variable Management ───────────────────────────────────────────────────

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a session variable."""
        raw = await self._get(self._vars_key)
        if raw:
            vars_dict = json.loads(raw)
            return vars_dict.get(key, default)
        return default

    async def set(self, key: str, value: Any):
        """Set a session variable."""
        raw = await self._get(self._vars_key)
        vars_dict = json.loads(raw) if raw else {}
        vars_dict[key] = value
        await self._set(self._vars_key, json.dumps(vars_dict), ttl=SESSION_TTL)

    async def get_all(self) -> dict:
        """Get all session variables."""
        raw = await self._get(self._vars_key)
        return json.loads(raw) if raw else {}

    async def clear(self):
        """Clear all session data."""
        await self._delete(self._history_key)
        await self._delete(self._vars_key)
        logger.info(f"Session cleared: {self.session_id}")

    async def cleanup(self):
        """Called on WebSocket disconnect — optionally extend TTL for reconnection."""
        pass  # TTL handles cleanup automatically

    # ── Redis / Fallback Backend ──────────────────────────────────────────────

    async def _get(self, key: str) -> Optional[str]:
        if _redis_client:
            try:
                return await _redis_client.get(key)
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")
        return _memory_fallback.get(key)

    async def _set(self, key: str, value: str, ttl: int = SESSION_TTL):
        if _redis_client:
            try:
                await _redis_client.setex(key, ttl, value)
                return
            except Exception as e:
                logger.warning(f"Redis SET failed: {e}")
        _memory_fallback[key] = value

    async def _delete(self, key: str):
        if _redis_client:
            try:
                await _redis_client.delete(key)
                return
            except Exception as e:
                logger.warning(f"Redis DELETE failed: {e}")
        _memory_fallback.pop(key, None)
