"""
Integration Tests — Voice AI Agent
Tests cover: scheduling, language detection, agent tools, and memory
Run with: pytest tests/ -v
"""

import asyncio
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ══════════════════════════════════════════════════════════════════════════════
#  Language Detection Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageDetection:

    @pytest.fixture
    def detector(self):
        from services.language_detection.lang_detector import LanguageDetector
        return LanguageDetector()

    @pytest.mark.asyncio
    async def test_detect_english(self, detector):
        result = await detector.detect("Book an appointment with cardiologist tomorrow")
        assert result == "en"

    @pytest.mark.asyncio
    async def test_detect_hindi_devanagari(self, detector):
        result = await detector.detect("मुझे कल डॉक्टर से मिलना है")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_detect_tamil_script(self, detector):
        result = await detector.detect("நாளை மருத்துவரை பார்க்க வேண்டும்")
        assert result == "ta"

    @pytest.mark.asyncio
    async def test_detect_hinglish(self, detector):
        result = await detector.detect("mujhe kal doctor se milna hai appointment chahiye")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_empty_string_defaults_english(self, detector):
        result = await detector.detect("")
        assert result == "en"

    @pytest.mark.asyncio
    async def test_none_input_defaults_english(self, detector):
        result = await detector.detect(None)
        assert result == "en"


# ══════════════════════════════════════════════════════════════════════════════
#  Scheduling Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduler:

    @pytest.fixture
    def scheduler(self):
        from scheduler.appointment_engine.scheduler import AppointmentScheduler
        return AppointmentScheduler()

    @pytest.mark.asyncio
    async def test_get_available_slots_returns_list(self, scheduler):
        from datetime import datetime, timedelta
        future_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        slots = await scheduler.get_available_slots("DR001", future_date)
        assert isinstance(slots, list)

    @pytest.mark.asyncio
    async def test_past_date_returns_no_slots(self, scheduler):
        slots = await scheduler.get_available_slots("DR001", "2020-01-01")
        assert slots == []

    @pytest.mark.asyncio
    async def test_booking_past_time_rejected(self, scheduler):
        result = await scheduler.book_appointment(
            patient_id="P001",
            doctor_id="DR001",
            date="2020-01-01",
            time_slot="09:00"
        )
        assert result["success"] is False
        assert "past" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_time_slot_rejected(self, scheduler):
        from datetime import datetime, timedelta
        future_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        result = await scheduler.book_appointment(
            patient_id="P001",
            doctor_id="DR001",
            date=future_date,
            time_slot="13:37"  # Not a valid clinic slot
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_appointment(self, scheduler):
        result = await scheduler.cancel_appointment("FAKE_ID")
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_suggest_alternatives_returns_slots(self, scheduler):
        from datetime import datetime, timedelta
        future_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        alternatives = await scheduler._suggest_alternatives("DR001", future_date, "10:00")
        assert isinstance(alternatives, list)

    def test_date_resolution_tomorrow(self, scheduler):
        from agent.tools.appointment_tools import _resolve_date
        from datetime import datetime, timedelta
        result = _resolve_date("tomorrow")
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_date_resolution_today(self, scheduler):
        from agent.tools.appointment_tools import _resolve_date
        from datetime import datetime
        result = _resolve_date("today")
        assert result == datetime.now().strftime("%Y-%m-%d")

    def test_date_resolution_hindi(self, scheduler):
        from agent.tools.appointment_tools import _resolve_date
        from datetime import datetime, timedelta
        result = _resolve_date("कल")  # "kal" = tomorrow in Hindi
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected


# ══════════════════════════════════════════════════════════════════════════════
#  Session Memory Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionMemory:

    @pytest.fixture
    def memory(self):
        from memory.session_memory.redis_session import SessionMemory
        return SessionMemory("test_session_xyz")

    @pytest.mark.asyncio
    async def test_add_and_retrieve_history(self, memory):
        await memory.clear()
        await memory.add_message("user", "Hello")
        await memory.add_message("assistant", "Hi there!")
        history = await memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_set_and_get_variable(self, memory):
        await memory.set("preferred_language", "hi")
        result = await memory.get("preferred_language")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_default_value_for_missing_key(self, memory):
        result = await memory.get("nonexistent_key", "default_val")
        assert result == "default_val"

    @pytest.mark.asyncio
    async def test_clear_removes_history(self, memory):
        await memory.add_message("user", "test message")
        await memory.clear()
        history = await memory.get_history()
        assert history == []


# ══════════════════════════════════════════════════════════════════════════════
#  System Prompt Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemPrompts:

    def test_english_prompt_generated(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="en")
        assert "appointment" in prompt.lower()
        assert "English" in prompt

    def test_hindi_prompt_contains_instructions(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="hi")
        assert "Hindi" in prompt

    def test_tamil_prompt_contains_instructions(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(language="ta")
        assert "Tamil" in prompt

    def test_patient_context_injected(self):
        from agent.prompt.system_prompts import get_system_prompt
        prompt = get_system_prompt(
            language="en",
            patient_context={
                "patient_id": "P123",
                "preferred_language": "hi",
                "last_appointment": "2024-01-15",
            }
        )
        assert "P123" in prompt
        assert "2024-01-15" in prompt


# ══════════════════════════════════════════════════════════════════════════════
#  Doctor Model Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDoctorModel:

    @pytest.mark.asyncio
    async def test_list_all_doctors_fallback(self):
        from database.models.doctor import DoctorModel
        doctors = await DoctorModel.list_doctors()
        assert len(doctors) > 0
        assert all("name" in d for d in doctors)
        assert all("specialty" in d for d in doctors)

    @pytest.mark.asyncio
    async def test_filter_by_specialty(self):
        from database.models.doctor import DoctorModel
        doctors = await DoctorModel.list_doctors(specialty="cardiologist")
        assert all("cardio" in d["specialty"].lower() for d in doctors)


# ══════════════════════════════════════════════════════════════════════════════
#  Campaign Scheduler Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCampaignScheduler:

    @pytest.fixture
    def scheduler(self):
        from scheduler.appointment_engine.campaign_scheduler import CampaignScheduler
        return CampaignScheduler()

    @pytest.mark.asyncio
    async def test_create_reminder_campaign(self, scheduler):
        result = await scheduler.schedule_campaign(
            campaign_type="reminder",
            patient_ids=["P001", "P002"],
            language="en",
        )
        assert result["success"] is True
        assert result["patient_count"] == 2
        assert result["campaign_id"]

    @pytest.mark.asyncio
    async def test_campaign_status_retrieved(self, scheduler):
        create = await scheduler.schedule_campaign(
            campaign_type="followup",
            patient_ids=["P003"],
        )
        cid = create["campaign_id"]
        status = await scheduler.get_status(cid)
        assert status is not None
        assert status["campaign_id"] == cid

    @pytest.mark.asyncio
    async def test_unknown_campaign_returns_none(self, scheduler):
        status = await scheduler.get_status("FAKE_CAMPAIGN_ID")
        assert status is None
