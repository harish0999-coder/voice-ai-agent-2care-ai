"""
Outbound Campaign Scheduler
Manages proactive appointment reminders, follow-ups, and vaccination campaigns.
Uses APScheduler for background job scheduling.
"""

import asyncio
import logging
import uuid
import json
import os
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory campaign store (replace with Redis/DB in production)
_campaigns: dict = {}


class CampaignScheduler:
    """
    Orchestrates outbound campaigns.
    
    Campaign types:
    - reminder:    "Your appointment is tomorrow at 10 AM"
    - followup:    "How are you feeling after your appointment?"
    - vaccination: "Your vaccination is due. Would you like to schedule?"
    
    Each campaign:
    1. Loads patient list + contact info
    2. Generates personalized message via LLM
    3. Initiates TTS call (Twilio / mock)
    4. Handles patient response (reschedule / confirm / decline)
    5. Logs outcome
    """

    async def schedule_campaign(
        self,
        campaign_type: str,
        patient_ids: list,
        scheduled_at: Optional[str] = None,
        message_template: Optional[str] = None,
        language: str = "en",
    ) -> dict:
        """Schedule a new outbound campaign."""
        campaign_id = str(uuid.uuid4())[:8].upper()

        campaign = {
            "campaign_id": campaign_id,
            "type": campaign_type,
            "patient_ids": patient_ids,
            "language": language,
            "status": "scheduled",
            "created_at": time.time(),
            "scheduled_at": scheduled_at or datetime.now().isoformat(),
            "message_template": message_template or self._default_template(campaign_type, language),
            "results": {},
        }

        _campaigns[campaign_id] = campaign

        # Schedule background execution
        if scheduled_at:
            logger.info(f"Campaign {campaign_id} scheduled for {scheduled_at}")
            # In production, use APScheduler or Celery:
            # scheduler.add_job(self._run_campaign, 'date', run_date=scheduled_at, args=[campaign_id])
        else:
            # Execute immediately in background
            asyncio.create_task(self._run_campaign(campaign_id))
            logger.info(f"Campaign {campaign_id} started immediately")

        return {
            "success": True,
            "campaign_id": campaign_id,
            "patient_count": len(patient_ids),
            "type": campaign_type,
            "message": f"Campaign {campaign_id} created for {len(patient_ids)} patients",
        }

    async def _run_campaign(self, campaign_id: str):
        """Execute campaign for all patients."""
        campaign = _campaigns.get(campaign_id)
        if not campaign:
            logger.error(f"Campaign {campaign_id} not found")
            return

        _campaigns[campaign_id]["status"] = "running"
        logger.info(f"Running campaign {campaign_id} for {len(campaign['patient_ids'])} patients")

        for patient_id in campaign["patient_ids"]:
            try:
                result = await self._call_patient(
                    patient_id=patient_id,
                    campaign=campaign,
                )
                _campaigns[campaign_id]["results"][patient_id] = result
                logger.info(f"Campaign {campaign_id}: patient {patient_id} → {result['outcome']}")
            except Exception as e:
                logger.error(f"Campaign {campaign_id}: failed for patient {patient_id}: {e}")
                _campaigns[campaign_id]["results"][patient_id] = {
                    "outcome": "error",
                    "error": str(e)
                }

            # Small delay between calls to avoid overwhelming the system
            await asyncio.sleep(0.5)

        _campaigns[campaign_id]["status"] = "completed"
        _campaigns[campaign_id]["completed_at"] = time.time()
        logger.info(f"✅ Campaign {campaign_id} completed")

    async def _call_patient(self, patient_id: str, campaign: dict) -> dict:
        """
        Initiate an outbound call to a patient.
        
        In production, this would use Twilio Programmable Voice or similar.
        For demo, we simulate the call and return a mock result.
        """
        # In production with Twilio:
        # from twilio.rest import Client
        # client = Client(account_sid, auth_token)
        # call = client.calls.create(
        #     to=patient_phone,
        #     from_=twilio_number,
        #     url=f"{BASE_URL}/api/campaigns/twiml/{campaign['campaign_id']}/{patient_id}"
        # )

        logger.info(
            f"[OUTBOUND] Calling patient {patient_id} | "
            f"type={campaign['type']} | lang={campaign['language']}"
        )

        # Generate personalized message
        message = self._personalize_message(
            template=campaign["message_template"],
            patient_id=patient_id,
            language=campaign["language"],
        )

        # Simulate call result
        import random
        outcomes = ["confirmed", "rescheduled", "declined", "no_answer"]
        weights = [0.5, 0.2, 0.1, 0.2]
        outcome = random.choices(outcomes, weights=weights)[0]

        return {
            "patient_id": patient_id,
            "outcome": outcome,
            "message_delivered": message,
            "timestamp": time.time(),
        }

    def _personalize_message(self, template: str, patient_id: str, language: str) -> str:
        """Replace template variables with patient data."""
        # In production, fetch patient name, doctor name, appointment time
        replacements = {
            "{patient_name}": f"Patient {patient_id}",
            "{doctor_name}": "Dr. Sharma",
            "{appointment_date}": "tomorrow",
            "{appointment_time}": "10:00 AM",
            "{clinic_name}": "Apollo Clinic",
        }
        message = template
        for key, val in replacements.items():
            message = message.replace(key, val)
        return message

    def _default_template(self, campaign_type: str, language: str) -> str:
        templates = {
            "reminder": {
                "en": "Hello {patient_name}, this is a reminder about your appointment with {doctor_name} {appointment_date} at {appointment_time}. Reply to confirm or reschedule.",
                "hi": "नमस्ते {patient_name}, यह {doctor_name} के साथ आपकी अपॉइंटमेंट की याद दिलाने के लिए है - {appointment_date} को {appointment_time} बजे।",
                "ta": "வணக்கம் {patient_name}, உங்கள் {doctor_name} உடனான சந்திப்பு {appointment_date} அன்று {appointment_time} மணிக்கு உள்ளது.",
            },
            "followup": {
                "en": "Hello {patient_name}, this is {clinic_name} following up after your recent appointment. How are you feeling?",
                "hi": "नमस्ते {patient_name}, हम आपकी हालत जानना चाहते हैं। क्या आप ठीक हैं?",
                "ta": "வணக்கம் {patient_name}, உங்கள் நலன் அறிய அழைக்கிறோம்.",
            },
            "vaccination": {
                "en": "Hello {patient_name}, your vaccination is due soon. Would you like to schedule an appointment with us?",
                "hi": "नमस्ते {patient_name}, आपके टीकाकरण का समय आ गया है। क्या हम अपॉइंटमेंट बुक करें?",
                "ta": "வணக்கம் {patient_name}, உங்கள் தடுப்பூசி நேரம் வந்துவிட்டது. சந்திப்பு பதிவு செய்யட்டுமா?",
            },
        }
        return templates.get(campaign_type, templates["reminder"]).get(language, templates.get(campaign_type, {}).get("en", ""))

    async def get_status(self, campaign_id: str) -> Optional[dict]:
        return _campaigns.get(campaign_id)

    async def list_campaigns(self) -> list:
        return [
            {
                "campaign_id": c["campaign_id"],
                "type": c["type"],
                "status": c["status"],
                "patient_count": len(c["patient_ids"]),
                "created_at": c["created_at"],
            }
            for c in _campaigns.values()
        ]
