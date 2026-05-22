"""
WebSocket Handler - Real-time voice communication pipeline
Manages the full STT → Agent → TTS pipeline with latency measurement
"""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.reasoning.voice_agent import VoiceAgent
from services.speech_to_text.stt_service import STTService
from services.text_to_speech.tts_service import TTSService
from services.language_detection.lang_detector import LanguageDetector
from memory.session_memory.redis_session import SessionMemory

logger = logging.getLogger(__name__)
router = APIRouter()

# Minimum valid audio payload in bytes.
# Groq Whisper rejects files that are too short (truncated WebM container or
# near-silence from accidental mic taps).  3 000 B ≈ 0.2 s at 96 kbps opus.
MIN_AUDIO_BYTES = 3000


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: session={session_id}")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected: session={session_id}")

    async def send_audio(self, session_id: str, audio_data: bytes):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_bytes(audio_data)

    async def send_json(self, session_id: str, data: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    Main real-time voice pipeline WebSocket endpoint.

    Message protocol:
    - Client → Server: ONE binary message containing the complete WebM/Opus
      audio blob, followed immediately by a 4-byte end-of-speech marker
      (b"\\x00\\x00\\x00\\x00").
    - Server → Client: JSON status messages + binary audio responses.

    NOTE: The previous design streamed 100 ms chunks individually.  That caused
    a race condition — the async .arrayBuffer() promise for the last chunk could
    resolve AFTER onstop fired and sent the end-of-speech marker, so the server
    processed a truncated buffer.  The client now sends a single complete blob,
    which is both race-free and produces a fully valid WebM container.

    Latency target: < 450ms from speech end to first audio byte.
    """
    await manager.connect(websocket, session_id)

    stt = STTService()
    tts = TTSService()
    lang_detector = LanguageDetector()
    session_memory = SessionMemory(session_id)
    agent = VoiceAgent(session_id=session_id, session_memory=session_memory)

    # FIX: buffer is still kept for safety (e.g. if a caller sends multiple
    # binary frames before the marker), but the expected path is one big frame.
    audio_buffer: bytearray = bytearray()
    speech_end_time: Optional[float] = None

    try:
        await manager.send_json(session_id, {
            "type": "ready",
            "message": "Voice AI Agent connected and ready",
            "session_id": session_id
        })

        while True:
            data = await websocket.receive()

            # ── Binary frame: audio data or end-of-speech marker ──────────────
            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]

                if chunk == b"\x00\x00\x00\x00":
                    # End-of-speech marker received
                    speech_end_time = time.time()
                    logger.info(f"Speech end detected for session {session_id}")

                    if audio_buffer:
                        # FIX: snapshot & clear the buffer BEFORE awaiting the
                        # pipeline so that any stray trailing frames don't mix
                        # into the next utterance if an exception interrupts us.
                        audio_snapshot = bytes(audio_buffer)
                        audio_buffer.clear()

                        await process_voice_pipeline(
                            session_id=session_id,
                            audio_data=audio_snapshot,
                            speech_end_time=speech_end_time,
                            stt=stt,
                            tts=tts,
                            lang_detector=lang_detector,
                            agent=agent,
                            manager=manager,
                        )
                    else:
                        logger.warning(
                            f"[{session_id}] End-of-speech marker received but "
                            "audio buffer is empty — ignoring"
                        )
                else:
                    audio_buffer.extend(chunk)

            # ── Text frame: JSON control messages ─────────────────────────────
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    await handle_control_message(
                        msg, session_id, agent, manager, session_memory
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {session_id}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            logger.info(f"Client disconnected mid-receive: {session_id}")
        else:
            logger.error(f"WebSocket runtime error for {session_id}: {e}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}", exc_info=True)
        try:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                msg = (
                    "OpenAI quota exceeded. Set LLM_PROVIDER=demo in .env for "
                    "free mode, then restart the server."
                )
            else:
                msg = "An internal error occurred. Please try again."
            await manager.send_json(session_id, {"type": "error", "message": msg})
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
        await session_memory.cleanup()


async def process_voice_pipeline(
    session_id: str,
    audio_data: bytes,
    speech_end_time: float,
    stt: STTService,
    tts: TTSService,
    lang_detector: LanguageDetector,
    agent: VoiceAgent,
    manager: ConnectionManager,
):
    """
    Full pipeline: Audio → STT → LangDetect → Agent → TTS → Audio
    All stages are timed and logged for latency analysis.
    """
    pipeline_start = time.time()
    latency_log = {"session_id": session_id, "pipeline_start": pipeline_start}

    try:
        # ── Guard: reject payloads that are too small ─────────────────────────
        # Groq Whisper returns HTTP 400 for truncated or near-empty WebM files.
        # This happens when the user taps the mic accidentally or stops
        # immediately, producing a header-only container with no audio frames.
        if len(audio_data) < MIN_AUDIO_BYTES:
            logger.warning(
                f"[{session_id}] Audio payload too small ({len(audio_data)} B < "
                f"{MIN_AUDIO_BYTES} B) — skipping transcription"
            )
            await manager.send_json(session_id, {
                "type": "error",
                "message": "Recording was too short. Please hold the mic button and speak."
            })
            return

        # ── Stage 1: Speech-to-Text ───────────────────────────────────────────
        stt_start = time.time()
        transcript = await stt.transcribe(audio_data)
        stt_ms = (time.time() - stt_start) * 1000
        latency_log["stt_ms"] = round(stt_ms, 2)

        if not transcript or not transcript.strip():
            logger.info(f"[{session_id}] Empty transcript — sending hint")
            await manager.send_json(session_id, {
                "type": "agent_response",
                "intent": "conversation",
                "text": (
                    "I couldn't make out any speech. Please try again, or use "
                    "the text box below to type your request."
                ),
                "action_result": None,
                "latency_ms": 0
            })
            return

        logger.info(f"[{session_id}] STT ({stt_ms:.0f}ms): '{transcript}'")

        await manager.send_json(session_id, {
            "type": "transcript",
            "text": transcript,
            "latency_ms": round(stt_ms, 2)
        })

        # ── Stage 2: Language Detection ───────────────────────────────────────
        lang_start = time.time()
        detected_lang = await lang_detector.detect(transcript)
        lang_ms = (time.time() - lang_start) * 1000
        latency_log["lang_detect_ms"] = round(lang_ms, 2)

        logger.info(f"[{session_id}] Language ({lang_ms:.0f}ms): {detected_lang}")

        await manager.send_json(session_id, {
            "type": "language",
            "language": detected_lang,
            "latency_ms": round(lang_ms, 2)
        })

        # ── Stage 3: AI Agent Reasoning ───────────────────────────────────────
        agent_start = time.time()
        agent_response = await agent.process(
            transcript=transcript,
            language=detected_lang
        )
        agent_ms = (time.time() - agent_start) * 1000
        latency_log["agent_ms"] = round(agent_ms, 2)

        logger.info(
            f"[{session_id}] Agent ({agent_ms:.0f}ms): "
            f"intent={agent_response.get('intent')} "
            f"response='{agent_response.get('text_response', '')[:60]}...'"
        )

        await manager.send_json(session_id, {
            "type": "agent_response",
            "intent": agent_response.get("intent"),
            "text": agent_response.get("text_response"),
            "action_result": agent_response.get("action_result"),
            "latency_ms": round(agent_ms, 2)
        })

        # ── Stage 4: Text-to-Speech ───────────────────────────────────────────
        tts_start = time.time()
        audio_response = await tts.synthesize(
            text=agent_response["text_response"],
            language=detected_lang
        )
        tts_ms = (time.time() - tts_start) * 1000
        latency_log["tts_ms"] = round(tts_ms, 2)

        # ── Total Pipeline Latency ─────────────────────────────────────────────
        total_ms = (time.time() - pipeline_start) * 1000
        latency_log["total_ms"] = round(total_ms, 2)
        latency_log["target_ms"] = 450
        latency_log["within_target"] = total_ms < 450

        logger.info(
            f"[{session_id}] LATENCY BREAKDOWN: "
            f"STT={stt_ms:.0f}ms | "
            f"Lang={lang_ms:.0f}ms | "
            f"Agent={agent_ms:.0f}ms | "
            f"TTS={tts_ms:.0f}ms | "
            f"TOTAL={total_ms:.0f}ms | "
            f"{'✅ WITHIN TARGET' if total_ms < 450 else '⚠️ OVER TARGET'}"
        )

        await manager.send_json(session_id, {
            "type": "latency_metrics",
            **latency_log
        })

        # ── Send Audio Response ───────────────────────────────────────────────
        if audio_response:
            await websocket_send_audio_chunked(session_id, audio_response, manager)

    except Exception as e:
        err_str = str(e)
        logger.error(f"Pipeline error for {session_id}: {err_str}")

        if "429" in err_str or "quota" in err_str.lower() or "insufficient_quota" in err_str:
            fallback_text = (
                "Sorry, the OpenAI quota is exceeded. "
                "Please set LLM_PROVIDER=demo in your .env file and restart."
            )
            await manager.send_json(session_id, {"type": "error", "message": fallback_text})
        else:
            fallback_text = "I'm having trouble processing that. Could you please repeat?"
            await manager.send_json(session_id, {"type": "error", "message": fallback_text})
            try:
                fallback_audio = await tts.synthesize(fallback_text, language="en")
                if fallback_audio:
                    await websocket_send_audio_chunked(session_id, fallback_audio, manager)
            except Exception:
                pass


async def websocket_send_audio_chunked(
    session_id: str,
    audio_data: bytes,
    manager: ConnectionManager,
    chunk_size: int = 4096,
):
    """Stream audio in chunks for lower perceived latency."""
    for i in range(0, len(audio_data), chunk_size):
        await manager.send_audio(session_id, audio_data[i:i + chunk_size])
        await asyncio.sleep(0)  # yield to event loop

    # End-of-audio marker
    await manager.send_audio(session_id, b"\x00\x00\x00\x00")


async def handle_control_message(
    msg: dict,
    session_id: str,
    agent: VoiceAgent,
    manager: ConnectionManager,
    session_memory: SessionMemory,
):
    """Handle non-audio control messages from client."""
    msg_type = msg.get("type")

    if msg_type == "ping":
        await manager.send_json(session_id, {"type": "pong"})

    elif msg_type == "clear_session":
        await session_memory.clear()
        await manager.send_json(session_id, {
            "type": "session_cleared",
            "message": "Conversation history cleared"
        })

    elif msg_type == "set_language":
        lang = msg.get("language", "en")
        await session_memory.set("preferred_language", lang)
        await manager.send_json(session_id, {
            "type": "language_set",
            "language": lang
        })

    elif msg_type == "barge_in":
        logger.info(f"Barge-in detected for session {session_id}")
        await manager.send_json(session_id, {"type": "barge_in_acknowledged"})

    elif msg_type == "text_input":
        text = msg.get("text", "").strip()
        language = msg.get("language", "en")
        if text:
            agent_response = await agent.process(transcript=text, language=language)
            await manager.send_json(session_id, {
                "type": "agent_response",
                "intent": agent_response.get("intent"),
                "text": agent_response.get("text_response"),
                "action_result": agent_response.get("action_result"),
                "latency_ms": 0
            })

    else:
        logger.warning(f"Unknown control message type: {msg_type}")