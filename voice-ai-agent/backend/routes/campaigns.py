"""
Outbound Campaign Routes
Manage proactive reminder and follow-up campaigns
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scheduler.appointment_engine.campaign_scheduler import CampaignScheduler

router = APIRouter()
campaign_scheduler = CampaignScheduler()


class CampaignRequest(BaseModel):
    campaign_type: str          # "reminder" | "followup" | "vaccination"
    patient_ids: List[str]
    scheduled_at: Optional[str] = None   # ISO datetime, None = now
    message_template: Optional[str] = None
    language: Optional[str] = "en"


@router.post("/trigger")
async def trigger_campaign(req: CampaignRequest):
    """Trigger an outbound campaign for a list of patients."""
    result = await campaign_scheduler.schedule_campaign(
        campaign_type=req.campaign_type,
        patient_ids=req.patient_ids,
        scheduled_at=req.scheduled_at,
        message_template=req.message_template,
        language=req.language,
    )
    return result


@router.get("/status/{campaign_id}")
async def campaign_status(campaign_id: str):
    """Get status of a campaign."""
    status = await campaign_scheduler.get_status(campaign_id)
    if not status:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return status


@router.get("/")
async def list_campaigns():
    """List all campaigns."""
    return await campaign_scheduler.list_campaigns()
