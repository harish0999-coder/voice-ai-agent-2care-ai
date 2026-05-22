"""
Voice Agent - Core AI Reasoning Engine
Integrates LLM with tool orchestration for appointment management
Supports OpenAI, Claude (Anthropic), and local Llama models
"""

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from pathlib import Path
for _p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break

from agent.prompt.system_prompts import get_system_prompt
from agent.tools.appointment_tools import (
    check_availability_tool,
    book_appointment_tool,
    cancel_appointment_tool,
    reschedule_appointment_tool,
    list_doctors_tool,
)
from memory.session_memory.redis_session import SessionMemory
from memory.persistent_memory.patient_memory import PatientMemory

logger = logging.getLogger(__name__)

# ── Inline tool-call helpers ─────────────────────────────────────────────────
# llama-3.1-8b-instant on Groq sometimes returns function calls as plain text
# in message.content instead of using the structured tool_calls field, e.g.:
#   <function=list_doctors>{"specialty": "cardiology"}</function>
# We parse and execute these so users never see raw function syntax.

_INLINE_TOOL_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)


def _extract_inline_calls(text: str) -> list[tuple[str, dict]]:
    """Return list of (fn_name, args_dict) for every <function=…> tag in text."""
    calls = []
    for m in _INLINE_TOOL_RE.finditer(text):
        fn_name = m.group(1)
        try:
            args = json.loads(m.group(2).strip())
        except json.JSONDecodeError:
            args = {}
        calls.append((fn_name, args))
    return calls


def _strip_inline_calls(text: str) -> str:
    """Remove all <function=…>…</function> blocks from a string."""
    return _INLINE_TOOL_RE.sub("", text).strip()


def _sanitise_response(text: str) -> str:
    """
    Remove any leaked tool-call syntax from a final response string so it is
    never displayed raw to the user.
    """
    cleaned = _strip_inline_calls(text or "")
    # Also strip any stray JSON blobs that look like tool arguments
    cleaned = re.sub(r"\{\"[a-z_]+\":\s*\"[^\"]*\"\}", "", cleaned)
    return cleaned.strip() or "How can I help you?"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check available appointment slots for a doctor on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {"type": "string", "description": "Doctor ID or specialty name"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD or natural language like 'tomorrow'"},
                },
                "required": ["doctor_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for the patient",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {"type": "string"},
                    "date": {"type": "string"},
                    "time_slot": {"type": "string", "description": "Time in HH:MM format"},
                    "notes": {"type": "string"},
                },
                "required": ["doctor_id", "date", "time_slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an existing appointment to a new date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                    "new_date": {"type": "string"},
                    "new_time_slot": {"type": "string"},
                },
                "required": ["appointment_id", "new_date", "new_time_slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_doctors",
            "description": "List available doctors, optionally filtered by specialty",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]

TOOL_DISPATCH = {
    "check_availability": check_availability_tool,
    "book_appointment": book_appointment_tool,
    "cancel_appointment": cancel_appointment_tool,
    "reschedule_appointment": reschedule_appointment_tool,
    "list_doctors": list_doctors_tool,
}


class VoiceAgent:
    def __init__(self, session_id: str, session_memory: SessionMemory):
        self.session_id = session_id
        self.session_memory = session_memory
        self.patient_memory = PatientMemory()
        self.client = None
        self.model = None
        self.llm_provider = "none"
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client — logs a warning but never crashes if key missing."""
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key or api_key.startswith("sk-your"):
                logger.warning(
                    "OPENAI_API_KEY not set or is placeholder. "
                    "Agent will run in DEMO mode (rule-based responses). "
                    "Set OPENAI_API_KEY in your .env file for full AI."
                )
                self.llm_provider = "demo"
                return
            from openai import AsyncOpenAI
            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            provider_name = (
                "Groq" if base_url and "groq" in base_url else
                "Together" if base_url and "together" in base_url else
                "OpenAI-compatible" if base_url else "OpenAI"
            )
            logger.info(f"LLM: {provider_name} ({self.model})")

        elif self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key or api_key.startswith("sk-ant-your"):
                logger.warning("ANTHROPIC_API_KEY not set. Running in DEMO mode.")
                self.llm_provider = "demo"
                return
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
            logger.info(f"LLM: Anthropic ({self.model})")

        elif self.llm_provider == "ollama":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama"
            )
            self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
            logger.info(f"LLM: Ollama/{self.model}")

        else:
            logger.warning(f"Unknown LLM_PROVIDER '{self.llm_provider}' — using demo mode.")
            self.llm_provider = "demo"

    async def process(self, transcript: str, language: str = "en") -> dict:
        """Process a user transcript and return agent response."""
        reasoning_trace = [{"stage": "input", "transcript": transcript, "language": language}]

        raw_history = await self.session_memory.get_history()
        # Strip any extra fields — only role+content accepted by all LLM APIs
        history = [{"role": m["role"], "content": m["content"]} for m in raw_history]
        patient_id = await self.session_memory.get("patient_id", "anonymous")
        patient_context = await self.patient_memory.get_context(patient_id)
        system_prompt = get_system_prompt(language=language, patient_context=patient_context)
        messages = history + [{"role": "user", "content": transcript}]

        if self.llm_provider == "demo":
            response_data = await self._demo_response(transcript, language)
        elif self.llm_provider == "anthropic":
            response_data = await self._call_anthropic(system_prompt, messages, reasoning_trace)
        else:
            response_data = await self._call_openai_compatible(system_prompt, messages, reasoning_trace)

        text_response = response_data["text_response"]
        action_result = response_data.get("action_result")
        intent = response_data.get("intent", "conversation")

        await self.session_memory.add_message("user", transcript)
        await self.session_memory.add_message("assistant", text_response)
        await self.patient_memory.update_context(
            patient_id=patient_id,
            language=language,
            last_intent=intent,
        )

        return {
            "intent": intent,
            "text_response": text_response,
            "action_result": action_result,
            "reasoning_trace": reasoning_trace,
        }

    async def _demo_response(self, transcript: str, language: str) -> dict:
        """
        Rule-based demo responses when no LLM API key is configured.
        Still calls real tools for appointment operations.
        """
        t = transcript.lower()
        intent = "conversation"
        action_result = None

        if any(w in t for w in ["book", "appointment", "schedule", "doctor", "kal", "நாளை", "बुक"]):
            intent = "book_appointment"
            action_result = await check_availability_tool(doctor_id="DR001", date="tomorrow")
            slots = action_result.get("available_slots", [])
            if language == "hi":
                text = f"मैं आपके लिए डॉ. अनन्या शर्मा (कार्डियोलॉजिस्ट) के साथ अपॉइंटमेंट बुक कर सकता हूं। उपलब्ध स्लॉट: {', '.join(slots[:3]) if slots else 'कोई स्लॉट उपलब्ध नहीं'}। कौन सा समय सुविधाजनक है?"
            elif language == "ta":
                text = f"டாக்டர் அனன்யா சர்மா (இருதயவியல்) உடன் சந்திப்பு பதிவு செய்யலாம். கிடைக்கும் நேரங்கள்: {', '.join(slots[:3]) if slots else 'கிடைக்கவில்லை'}. எந்த நேரம் வசதியானது?"
            else:
                text = f"I can book with Dr. Ananya Sharma (Cardiologist). Available slots tomorrow: {', '.join(slots[:3]) if slots else 'none available'}. Which time works?"

        elif any(w in t for w in ["cancel", "रद्द", "ரத்து"]):
            intent = "cancel_appointment"
            if language == "hi":
                text = "अपॉइंटमेंट रद्द करने के लिए कृपया अपनी अपॉइंटमेंट ID बताएं।"
            elif language == "ta":
                text = "சந்திப்பை ரத்து செய்ய உங்கள் சந்திப்பு ID கொடுங்கள்."
            else:
                text = "To cancel, please provide your appointment ID (e.g. 'ABC12345')."

        elif any(w in t for w in ["reschedule", "change", "move", "बदल", "மாற்று"]):
            intent = "reschedule_appointment"
            if language == "hi":
                text = "अपॉइंटमेंट बदलने के लिए अपनी appointment ID और नई तारीख बताएं।"
            elif language == "ta":
                text = "சந்திப்பை மாற்ற appointment ID மற்றும் புதிய தேதி கொடுங்கள்."
            else:
                text = "To reschedule, please tell me your appointment ID and the new date."

        elif any(w in t for w in ["doctor", "डॉक्टर", "மருத்துவர", "available", "list", "who"]):
            intent = "list_doctors"
            action_result = await list_doctors_tool()
            doctors = action_result.get("doctors", [])
            doc_names = ", ".join([f"{d['name']} ({d['specialty']})" for d in doctors[:4]])
            if language == "hi":
                text = f"हमारे उपलब्ध डॉक्टर: {doc_names}। किस विशेषज्ञ से मिलना है?"
            elif language == "ta":
                text = f"கிடைக்கும் டாக்டர்கள்: {doc_names}. யாரை சந்திக்க வேண்டும்?"
            else:
                text = f"Available doctors: {doc_names}. Which specialist would you like to see?"

        else:
            if language == "hi":
                text = "नमस्ते! मैं 2Care.ai का वॉयस असिस्टेंट हूं। अपॉइंटमेंट बुक, रद्द या बदलने में मदद कर सकता हूं।"
            elif language == "ta":
                text = "வணக்கம்! நான் 2Care.ai உதவியாளர். சந்திப்பு பதிவு, ரத்து அல்லது மாற்றம் செய்ய உதவுவேன்."
            else:
                text = "Hello! I'm the 2Care.ai assistant. I can help you book, reschedule, or cancel appointments. What would you like to do?"

        return {"text_response": text, "action_result": action_result, "intent": intent}

    async def _call_openai_compatible(self, system_prompt, messages, reasoning_trace) -> dict:
        action_result = None
        intent = "conversation"
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # ── FIX: tool_use_failed recovery ────────────────────────────────────
        # llama-3.1-8b-instant occasionally emits malformed tool-call JSON
        # (error code 400, type=tool_use_failed).  We catch this and retry once
        # without tools so the user always gets a useful response.
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=512,          # Reduced from 1024 — voice replies are short
            )
        except Exception as e:
            err_str = str(e)
            logger.error(f"OpenAI API error: {e}")
            # tool_use_failed: retry without tools to get a plain text response
            if "tool_use_failed" in err_str or "failed_generation" in err_str:
                logger.warning(f"[{self.session_id}] tool_use_failed — retrying without tools")
                try:
                    fallback = await self.client.chat.completions.create(
                        model=self.model,
                        messages=full_messages,
                        temperature=0.2,
                        max_tokens=256,
                    )
                    return {
                        "text_response": fallback.choices[0].message.content or
                                         "I had trouble with that. Could you try again?",
                        "action_result": None,
                        "intent": "error",
                    }
                except Exception as e2:
                    logger.error(f"tool_use_failed fallback also failed: {e2}")
            return {
                "text_response": "I'm having trouble connecting to the AI service. Please try again.",
                "action_result": None,
                "intent": "error",
            }

        message = response.choices[0].message
        reasoning_trace.append({
            "stage": "llm_response",
            "finish_reason": response.choices[0].finish_reason,
        })

        # ── Resolve tool calls (structured OR inline text format) ─────────────
        # llama-3.1-8b-instant on Groq uses two different formats depending on
        # the request:
        #   A) Structured: message.tool_calls is populated (correct OpenAI format)
        #   B) Inline:     message.content contains <function=name>{args}</function>
        # We handle both so neither leaks raw syntax to the user.

        content_text = message.content or ""
        inline_calls = _extract_inline_calls(content_text)

        # Build a unified list of (fn_name, args, tool_call_id_or_None)
        unified_calls: list[tuple[str, dict, Any]] = []

        if message.tool_calls:
            # Path A — structured tool_calls
            seen: set = set()
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                key = (tc.function.name, frozenset(sorted(args.items())))
                if key not in seen:
                    seen.add(key)
                    unified_calls.append((tc.function.name, args, tc.id))
                else:
                    logger.warning(
                        f"[{self.session_id}] Duplicate tool call suppressed: "
                        f"{tc.function.name}({args})"
                    )
        elif inline_calls:
            # Path B — inline <function=…> tags in content
            logger.warning(
                f"[{self.session_id}] Inline tool calls detected in content "
                f"(model did not use tool_calls field): {[n for n, _ in inline_calls]}"
            )
            seen_inline: set = set()
            for fn_name, args in inline_calls:
                key = (fn_name, frozenset(sorted(args.items())))
                if key not in seen_inline:
                    seen_inline.add(key)
                    unified_calls.append((fn_name, args, None))

        if unified_calls:
            tool_results_for_api = []
            for fn_name, fn_args, tc_id in unified_calls:
                reasoning_trace.append({"stage": "tool_call", "tool": fn_name, "args": fn_args})
                logger.info(f"[{self.session_id}] Tool: {fn_name}({fn_args})")

                tool_fn = TOOL_DISPATCH.get(fn_name)
                result = await tool_fn(**fn_args) if tool_fn else {"error": f"Unknown tool: {fn_name}"}
                action_result = result
                intent = fn_name
                reasoning_trace.append({"stage": "tool_result", "tool": fn_name, "result": result})

                if tc_id:
                    # Structured path — proper tool message
                    tool_results_for_api.append({
                        "tool_call_id": tc_id,
                        "role": "tool",
                        "content": json.dumps(result),
                    })

            # Build follow-up messages depending on which path we used
            if message.tool_calls:
                follow_up_msgs = full_messages + [message] + tool_results_for_api
            else:
                # Inline path: inject results as a user-turn summary so the
                # model can formulate a clean reply without seeing raw JSON
                results_summary = "; ".join(
                    f"{fn}: {json.dumps(res)}"
                    for (fn, _, _), res in zip(unified_calls, [action_result])
                )
                follow_up_msgs = full_messages + [
                    {"role": "assistant", "content": _strip_inline_calls(content_text)},
                    {"role": "user", "content": f"Tool results: {results_summary}. Now reply to the patient in 1-2 sentences."},
                ]

            try:
                final = await self.client.chat.completions.create(
                    model=self.model,
                    messages=follow_up_msgs,
                    temperature=0.3,
                    max_tokens=256,
                )
                text_response = _sanitise_response(final.choices[0].message.content)
            except Exception as e:
                logger.error(f"OpenAI follow-up error: {e}")
                text_response = _sanitise_response(str(action_result))
        else:
            # No tool calls at all — plain conversational reply
            text_response = _sanitise_response(content_text) or "I couldn't process that request."

        return {"text_response": text_response, "action_result": action_result, "intent": intent}

    async def _call_anthropic(self, system_prompt, messages, reasoning_trace) -> dict:
        anthropic_tools = [
            {"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in TOOLS
        ]
        try:
            response = await self.client.messages.create(
                model=self.model, system=system_prompt, messages=messages,
                tools=anthropic_tools, max_tokens=512,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"text_response": "AI service error. Please try again.", "action_result": None, "intent": "error"}

        action_result = None
        intent = "conversation"
        text_response = ""

        for block in response.content:
            if block.type == "text":
                text_response = block.text
            elif block.type == "tool_use":
                fn_name, fn_args = block.name, block.input
                tool_fn = TOOL_DISPATCH.get(fn_name)
                result = await tool_fn(**fn_args) if tool_fn else {"error": f"Unknown tool: {fn_name}"}
                action_result = result
                intent = fn_name
                try:
                    follow_up = await self.client.messages.create(
                        model=self.model, system=system_prompt,
                        messages=messages + [
                            {"role": "assistant", "content": response.content},
                            {"role": "user", "content": [
                                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                            ]},
                        ],
                        max_tokens=256,
                    )
                    text_response = follow_up.content[0].text if follow_up.content else text_response
                except Exception as e:
                    logger.error(f"Anthropic follow-up error: {e}")

        return {"text_response": text_response, "action_result": action_result, "intent": intent}