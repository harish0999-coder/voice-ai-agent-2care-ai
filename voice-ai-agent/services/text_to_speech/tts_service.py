"""
Text-to-Speech Service
Supports: OpenAI TTS, Google Cloud TTS, ElevenLabs, gTTS (free fallback)

Key fix: gTTS latency scales linearly with text length (~18 ms per char).
_synthesize_gtts now receives a voice-trimmed version (≤ 150 chars, sentence-
aware), while the full text is still displayed in the chat UI by the caller.
"""

import asyncio
import functools
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
for _p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break

logger = logging.getLogger(__name__)

# Maximum characters sent to the TTS engine.
# gTTS latency is roughly proportional to text length; 150 chars ≈ 1-2 spoken
# sentences and keeps gTTS under ~1 500 ms in most cases.
# Faster providers (OpenAI, Google Cloud, ElevenLabs) ignore this cap because
# they are streamed/fast enough not to need it.
_GTTS_MAX_CHARS = 150


def _trim_for_voice(text: str, max_chars: int = _GTTS_MAX_CHARS) -> str:
    """
    Return the first `max_chars` characters of *text*, preferring to cut at a
    sentence boundary (. ! ?) so the spoken output still sounds complete.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    # Find the last sentence-ending punctuation within the limit
    window = text[:max_chars]
    match = None
    for m in re.finditer(r"[.!?]", window):
        match = m
    if match:
        return window[:match.end()].strip()

    # No sentence boundary — cut at the last space to avoid mid-word truncation
    last_space = window.rfind(" ")
    return (window[:last_space] if last_space != -1 else window).strip()


class TTSService:
    GOOGLE_VOICES = {
        "en": ("en-IN-Wavenet-D", "en-IN"),
        "hi": ("hi-IN-Wavenet-A", "hi-IN"),
        "ta": ("ta-IN-Wavenet-A", "ta-IN"),
    }

    def __init__(self):
        self.provider = os.getenv("TTS_PROVIDER", "openai")
        # Auto-downgrade to gtts if OpenAI key is missing or is a Groq key
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            base = os.getenv("OPENAI_BASE_URL", "")
            if not key or key.startswith("sk-your") or "groq.com" in base:
                logger.warning("OPENAI_API_KEY not set or is a Groq key — TTS falling back to gTTS (free)")
                self.provider = "gtts"
        self._openai_client = None
        logger.info(f"TTS Provider: {self.provider}")

    def _get_openai_client(self):
        if not self._openai_client:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    async def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        if not text or not text.strip():
            return None

        start = time.time()

        # For gTTS (the free/slow provider) trim the text before synthesis so
        # the user hears the first sentence quickly while reading the full reply
        # in the chat.  All other providers are fast enough to use full text.
        voice_text = _trim_for_voice(text) if self.provider == "gtts" else text
        if self.provider == "gtts" and len(voice_text) < len(text.strip()):
            logger.info(f"TTS voice trim: {len(text)} → {len(voice_text)} chars")

        try:
            if self.provider == "openai":
                audio = await self._synthesize_openai(voice_text, language)
            elif self.provider == "google":
                audio = await self._synthesize_google(voice_text, language)
            elif self.provider == "elevenlabs":
                audio = await self._synthesize_elevenlabs(voice_text, language)
            else:
                audio = await self._synthesize_gtts(voice_text, language)

            ms = (time.time() - start) * 1000
            logger.info(f"TTS ({self.provider}) {ms:.0f}ms | {len(voice_text)} chars | lang={language}")
            return audio
        except Exception as e:
            logger.error(f"TTS error with {self.provider}: {e}")
            if self.provider != "gtts":
                logger.info("Falling back to gTTS")
                try:
                    trimmed = _trim_for_voice(text)
                    return await self._synthesize_gtts(trimmed, language)
                except Exception as e2:
                    logger.error(f"gTTS fallback also failed: {e2}")
            return None

    async def _synthesize_openai(self, text: str, language: str) -> bytes:
        client = self._get_openai_client()
        model = os.getenv("OPENAI_TTS_MODEL", "tts-1")
        response = await client.audio.speech.create(
            model=model, voice="nova", input=text,
            response_format="mp3", speed=1.0,
        )
        return response.content

    async def _synthesize_google(self, text: str, language: str) -> bytes:
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechAsyncClient()
        voice_name, lang_code = self.GOOGLE_VOICES.get(language, ("en-IN-Wavenet-D", "en-IN"))
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
        response = await client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content

    async def _synthesize_elevenlabs(self, text: str, language: str) -> bytes:
        import httpx
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": "eleven_multilingual_v2",
                      "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.content

    async def _synthesize_gtts(self, text: str, language: str) -> bytes:
        def _run(t: str, lang: str) -> bytes:
            from gtts import gTTS
            tts = gTTS(text=t, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()
        gtts_lang = {"en": "en", "hi": "hi", "ta": "ta"}.get(language, "en")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(_run, text, gtts_lang))