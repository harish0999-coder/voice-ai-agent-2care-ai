"""
Speech-to-Text Service
Supports: OpenAI Whisper API, Groq Whisper (FREE), local Whisper, Google STT
"""

import asyncio
import functools
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
for _p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self):
        self.provider = os.getenv("STT_PROVIDER", "groq_whisper").strip()

        # Auto-detect: if Groq key present and STT is demo → upgrade to groq_whisper
        groq_key = os.getenv("OPENAI_API_KEY", "")
        groq_base = os.getenv("OPENAI_BASE_URL", "")
        is_groq = "groq.com" in groq_base

        if self.provider == "demo" and is_groq and groq_key:
            logger.info("Groq detected — upgrading STT from demo to groq_whisper (free)")
            self.provider = "groq_whisper"

        if self.provider == "openai_whisper":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key or key.startswith("sk-your") or "groq" in os.getenv("OPENAI_BASE_URL",""):
                logger.warning("OpenAI Whisper needs an OpenAI key (not Groq) — switching to groq_whisper")
                self.provider = "groq_whisper"

        self._client = None
        logger.info(f"STT Provider: {self.provider}")

    def _get_client(self, base_url=None):
        """Get async OpenAI-compatible client."""
        if not self._client:
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or os.getenv("OPENAI_BASE_URL") or None
            )
        return self._client

    async def transcribe(self, audio_data: bytes, language_hint: Optional[str] = None) -> str:
        start = time.time()

        if self.provider == "demo":
            return ""   # Empty = skip pipeline gracefully

        elif self.provider == "groq_whisper":
            result = await self._transcribe_groq(audio_data, language_hint)

        elif self.provider == "openai_whisper":
            result = await self._transcribe_openai(audio_data, language_hint)

        elif self.provider == "local_whisper":
            result = await self._transcribe_local(audio_data, language_hint)

        elif self.provider == "google":
            result = await self._transcribe_google(audio_data, language_hint)
        else:
            result = ""

        ms = (time.time() - start) * 1000
        logger.info(f"STT ({self.provider}) {ms:.0f}ms: '{result[:80]}'")
        return result

    async def _transcribe_groq(self, audio_data: bytes, language_hint: Optional[str]) -> str:
        """Groq Whisper — free, fast, supports Hindi and Tamil."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"

        kwargs = {"model": "whisper-large-v3-turbo", "file": audio_file,
                  "response_format": "text"}
        # Language hint improves accuracy for Hindi/Tamil
        if language_hint and language_hint in ("hi", "ta", "en"):
            lang_map = {"en": "en", "hi": "hi", "ta": "ta"}
            kwargs["language"] = lang_map[language_hint]

        try:
            response = await client.audio.transcriptions.create(**kwargs)
            return (response if isinstance(response, str) else response.text).strip()
        except Exception as e:
            logger.error(f"Groq Whisper error: {e}")
            raise

    async def _transcribe_openai(self, audio_data: bytes, language_hint: Optional[str]) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language_hint and language_hint != "auto":
            kwargs["language"] = language_hint
        response = await client.audio.transcriptions.create(**kwargs)
        return response.text.strip()

    async def _transcribe_local(self, audio_data: bytes, language_hint: Optional[str]) -> str:
        def _run(audio_bytes, lang):
            import whisper, tempfile
            if not hasattr(self, '_local_model') or not self._local_model:
                self._local_model = whisper.load_model(os.getenv("WHISPER_MODEL_SIZE", "base"))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes); tmp = f.name
            result = self._local_model.transcribe(tmp, **({"language": lang} if lang else {}))
            os.unlink(tmp)
            return result["text"].strip()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(_run, audio_data, language_hint))

    async def _transcribe_google(self, audio_data: bytes, language_hint: Optional[str]) -> str:
        from google.cloud import speech
        client = speech.SpeechAsyncClient()
        lang_code = {"en": "en-IN", "hi": "hi-IN", "ta": "ta-IN"}.get(language_hint, "en-IN")
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=16000, language_code=lang_code)
        response = await client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript.strip() if response.results else ""
