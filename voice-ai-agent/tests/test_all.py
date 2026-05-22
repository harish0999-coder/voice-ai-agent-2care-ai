"""
Test Suite — Voice AI Agent
Covers: language detection, scheduling logic, agent tools, latency checks
"""

import asyncio
import json
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Language Detection Tests ─────────────────────────────────────────────────
class TestLanguageDetection:
    @pytest.fixture
    def detector(self):
        from services.language_detection.lang_detector import LanguageDetector
        return LanguageDetector()

    @pytest.mark.asyncio
    async def test_english_detection(self, detector):
        result = await detector.detect("Book an appointment with cardiologist tomorrow")
        assert result == "en"

    @pytest.mark.asyncio
    async def test_hindi_devanagari_detection(self, detector):
        result = await detector.detect("मुझे कल डॉक्टर से मिलना है")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_tamil_detection(self, detector):
        result = await detector.detect("நாளை மருத்துவரை பார்க்க வேண்டும்")
        assert result == "ta"

    @pytest.mark.asyncio
    async def test_hinglish_detection(self, detector):
        result = await detector.detect("mujhe kal doctor se milna hai")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_empty_string(self, detector):
        result = await detector.detect("")
        assert result == "en"  # Default

    @pytest.mark.asyncio
    async def test_english_medical(self, detector):
        result = await detector.detect("I need to see a dermatologist next Monday")
        assert result == "en"


# ── Date Resolution Tests ────────────────────────────────────────────────────
class TestDateResolution:
    def _resolve(self, date_str):
        from agent.tools.appointment_tools import _resolve_date
        return _resolve_date(date_str)

    def test_today(self):
        from datetime import date
        result = self._resolve("today")
        assert result == date.today().strftime("%Y-%m-%d")

    def test_tomorrow(self):
        from datetime import date, timedelta
        result = self._resolve("tomorrow")
        assert result == (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    def test_hindi_kal(self):
        from datetime import date, timedelta
        result = self._resolve("कल")
        assert result == (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    def test_already_formatted(self):
        result = self._resolve("2025-12-25")
        assert result == "2025-12-25"

    def test_day_after_tomorrow(self):
        from datetime import date, timedelta
        result = self._resolve("day after tomorrow")
        assert result == (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")


# ── Scheduler Validation Tests ───────────────────────────────────────────────
class TestSchedulerValidation:
    @pytest.fixture
    def scheduler(self):
        from scheduler.appointment_engine.scheduler import AppointmentScheduler
        return AppointmentScheduler()

    @pytest.mark.asyncio
    async def test_valid_slot(self, scheduler):
        from datetime import date, timedelta
        future_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        error = await scheduler._validate_slot("DR001", future_date, "10:00")
        assert error is None

    @pytest.mark.asyncio
    async def test_past_date_rejected(self, scheduler):
        error = await scheduler._validate_slot("DR001", "2020-01-01", "10:00")
        assert error is not None
        assert "past" in error.lower()

    @pytest.mark.asyncio
    async def test_invalid_time_slot(self, scheduler):
        from datetime import date, timedelta
        future_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        error = await scheduler._validate_slot("DR001", future_date, "13:45")
        assert error is not None  # 13:45 is not a valid slot

    @pytest.mark.asyncio
    async def test_all_slots_structure(self, scheduler):
        from scheduler.appointment_engine.scheduler import ALL_SLOTS
        assert len(ALL_SLOTS) > 0
        for slot in ALL_SLOTS:
            h, m = slot.split(":")
            assert 0 <= int(h) <= 23
            assert int(m) in (0, 30)

    @pytest.mark.asyncio
    async def test_suggest_alternatives_returns_list(self, scheduler):
        from datetime import date, timedelta
        future_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        alternatives = await scheduler._suggest_alternatives("DR001", future_date, "10:00")
        assert isinstance(alternatives, list)
        for alt in alternatives:
            assert "date" in alt
            assert "time_slot" in alt


# ── System Prompt Tests ───────────────────────────────────────────────────────
class TestSystemPrompts:
    def test_english_prompt(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="en")
        assert "English" in prompt
        assert "2Care.ai" in prompt

    def test_hindi_prompt(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="hi")
        assert "Hindi" in prompt
        assert "आप" in prompt  # Hindi "you"

    def test_tamil_prompt(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="ta")
        assert "Tamil" in prompt

    def test_patient_context_injected(self):
        from agent.prompt.system_prompts import get_system_prompt
        ctx = {
            "patient_id": "P001",
            "preferred_language": "hi",
            "last_appointment": "2024-12-01",
            "preferred_doctor": "DR001",
        }
        prompt = get_system_prompt(language="en", patient_context=ctx)
        assert "P001" in prompt
        assert "DR001" in prompt


# ── Campaign Scheduler Tests ─────────────────────────────────────────────────
class TestCampaignScheduler:
    @pytest.fixture
    def campaign_scheduler(self):
        from scheduler.appointment_engine.campaign_scheduler import CampaignScheduler
        return CampaignScheduler()

    @pytest.mark.asyncio
    async def test_create_campaign(self, campaign_scheduler):
        result = await campaign_scheduler.schedule_campaign(
            campaign_type="reminder",
            patient_ids=["P001", "P002"],
            language="en",
        )
        assert result["success"] is True
        assert result["patient_count"] == 2
        assert "campaign_id" in result

    @pytest.mark.asyncio
    async def test_campaign_status(self, campaign_scheduler):
        result = await campaign_scheduler.schedule_campaign(
            campaign_type="followup",
            patient_ids=["P003"],
            language="hi",
        )
        campaign_id = result["campaign_id"]
        status = await campaign_scheduler.get_status(campaign_id)
        assert status is not None
        assert status["type"] == "followup"

    @pytest.mark.asyncio
    async def test_list_campaigns(self, campaign_scheduler):
        campaigns = await campaign_scheduler.list_campaigns()
        assert isinstance(campaigns, list)

    def test_default_templates_all_languages(self, campaign_scheduler):
        for lang in ("en", "hi", "ta"):
            for ctype in ("reminder", "followup", "vaccination"):
                template = campaign_scheduler._default_template(ctype, lang)
                assert isinstance(template, str)
                assert len(template) > 0

    def test_personalize_message(self, campaign_scheduler):
        template = "Hello {patient_name}, your appointment with {doctor_name} is {appointment_date}."
        result = campaign_scheduler._personalize_message(template, "P001", "en")
        assert "{patient_name}" not in result
        assert "{doctor_name}" not in result


# ── Session Memory Tests ─────────────────────────────────────────────────────
class TestSessionMemory:
    @pytest.fixture
    def memory(self):
        from memory.session_memory.redis_session import SessionMemory
        return SessionMemory("test_session_001")

    @pytest.mark.asyncio
    async def test_add_and_get_history(self, memory):
        await memory.clear()
        await memory.add_message("user", "Book appointment")
        await memory.add_message("assistant", "Which doctor would you like?")
        history = await memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_session_variables(self, memory):
        await memory.set("patient_id", "P001")
        await memory.set("preferred_language", "hi")
        pid = await memory.get("patient_id")
        lang = await memory.get("preferred_language")
        assert pid == "P001"
        assert lang == "hi"

    @pytest.mark.asyncio
    async def test_clear_session(self, memory):
        await memory.add_message("user", "test")
        await memory.clear()
        history = await memory.get_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_default_value(self, memory):
        result = await memory.get("nonexistent_key", default="default_val")
        assert result == "default_val"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
