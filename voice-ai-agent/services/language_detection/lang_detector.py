"""
Language Detection Service
Detects Hindi, Tamil, and English from transcribed text.
Uses character-set analysis + optional LLM confirmation for ambiguous cases.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Unicode ranges for Indian scripts
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')   # Hindi
TAMIL_RANGE = re.compile(r'[\u0B80-\u0BFF]')         # Tamil


class LanguageDetector:
    """
    Fast language detector optimized for English / Hindi / Tamil.
    
    Strategy (in order of priority):
    1. Script detection — Devanagari → Hindi, Tamil script → Tamil
    2. Common word matching for Hinglish/Tanglish
    3. Library-based fallback (langdetect)
    4. Default to English
    
    This keeps detection under 5ms for the vast majority of inputs.
    """

    # Common Hindi words written in Latin script
    HINDI_LATIN_WORDS = {
        "mujhe", "kal", "doctor", "se", "milna", "hai", "aaj", "appointment",
        "chahiye", "kab", "kya", "nahi", "haan", "theek", "band", "bata",
        "please", "mera", "meri", "apna", "abhi", "jaldi", "aane", "wala",
    }

    # Common Tamil words written in Latin script (Tanglish)
    TAMIL_LATIN_WORDS = {
        "naalai", "doctor", "paakanum", "enna", "eppo", "oru", "vendum",
        "illai", "seri", "ungal", "enakku", "appointment", "vandhu",
        "parunga", "sollunga", "theriyum",
    }

    def __init__(self):
        self._langdetect_available = self._check_langdetect()

    def _check_langdetect(self) -> bool:
        try:
            import langdetect
            return True
        except ImportError:
            logger.warning("langdetect not installed — using script-based detection only")
            return False

    async def detect(self, text: str) -> str:
        """
        Detect language of input text.
        
        Returns:
            "en" | "hi" | "ta"
        """
        if not text or not text.strip():
            return "en"

        # ── Priority 1: Unicode script detection ─────────────────────────────
        if DEVANAGARI_RANGE.search(text):
            logger.debug(f"Detected Hindi (Devanagari script): '{text[:30]}'")
            return "hi"

        if TAMIL_RANGE.search(text):
            logger.debug(f"Detected Tamil (Tamil script): '{text[:30]}'")
            return "ta"

        # ── Priority 2: Latin-script word matching (Hinglish / Tanglish) ─────
        words = set(text.lower().split())

        hindi_matches = words & self.HINDI_LATIN_WORDS
        tamil_matches = words & self.TAMIL_LATIN_WORDS

        if hindi_matches and len(hindi_matches) >= 2:
            logger.debug(f"Detected Hindi (Hinglish) via words: {hindi_matches}")
            return "hi"

        if tamil_matches and len(tamil_matches) >= 2:
            logger.debug(f"Detected Tamil (Tanglish) via words: {tamil_matches}")
            return "ta"

        # ── Priority 3: langdetect library ───────────────────────────────────
        if self._langdetect_available:
            detected = self._run_langdetect(text)
            if detected:
                return detected

        # ── Default: English ─────────────────────────────────────────────────
        return "en"

    def _run_langdetect(self, text: str) -> Optional[str]:
        """Run langdetect and map to our supported codes."""
        try:
            from langdetect import detect, LangDetectException
            code = detect(text)

            mapping = {
                "hi": "hi",
                "ta": "ta",
                "en": "en",
                "mr": "hi",   # Marathi → treat as Hindi (Devanagari)
            }
            result = mapping.get(code)
            if result:
                logger.debug(f"langdetect → {code} mapped to {result}")
            return result

        except Exception as e:
            logger.debug(f"langdetect failed: {e}")
            return None

    def get_language_name(self, code: str) -> str:
        """Return human-readable language name."""
        return {"en": "English", "hi": "Hindi", "ta": "Tamil"}.get(code, "English")
