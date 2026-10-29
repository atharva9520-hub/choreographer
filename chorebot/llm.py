"""LLM layer: turns a chat message into tool calls. Provider-agnostic interface + Gemini implementation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta

FREQ_HELP = (
    "Frequency is an iCal RRULE without 'RRULE:'. every day=FREQ=DAILY; Saturdays=FREQ=WEEKLY;BYDAY=SA; "
    "Wed+Sun=FREQ=WEEKLY;BYDAY=WE,SU; every other week=FREQ=WEEKLY;INTERVAL=2;BYDAY=SA; every 3 days="
    "FREQ=DAILY;INTERVAL=3; monthly on the 1st=FREQ=MONTHLY;BYMONTHDAY=1. Weekly-type with no day -> Saturday; "
    "'twice a week' with no days -> WE,SU."
)

QUERY_SCHEMA = ("Tables (TEXT columns, dates YYYY-MM-DD): members(name), chores(name,type,frequency,members,next_up,"
                "active), tasks(date,chore,assigned_to,status,done_by), busy(member,start,\"end\",active). "
                "Completed work = tasks.status='done'; done_by did it.")

_PEOPLE = {"type": "array", "items": {"type": "string"}}
_STR = {"type": "string"}

TOOLS = [
    {"name": "mark_done", "description": "Someone did chores (the sender or someone they name). One entry per person+chore.",
     "parameters": {"type": "object", "properties": {"entries": {"type": "array", "items": {
         "type": "object", "properties": {
             "member": _STR,
             "chore": {"type": "string", "description": "existing chore name, or a short description for a one-off"},
             "date": {"type": "string", "description": "YYYY-MM-DD, default today"}},
         "required": ["member", "chore"]}}}, "required": ["entries"]}},
    {"name": "set_chore", "description": "Create or change a chore (same tool for both). Fill only what the message says.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "description": "short lowercase name: mop, dishwasher, trash, bathroom"},
         "kind": {"type": "string", "enum": ["scheduled", "turn"],
                  "description": "turn = no schedule, done when needed, people take turns"},
         "frequency": {"type": "string", "description": "RRULE, scheduled chores only"},
         "people": {**_PEOPLE, "description": "exactly who does it (replaces current list)"},
         "add_people": _PEOPLE, "remove_people": _PEOPLE,
         "next_up": {"type": "string", "description": "turn chores: who is next"}}, "required": ["name"]}},
    {"name": "remove_chore", "description": "Stop tracking a chore.",
     "parameters": {"type": "object", "properties": {"name": _STR}, "required": ["name"]}},
    {"name": "set_busy", "description": "Someone is away/busy/sick for a date range.",
     "parameters": {"type": "object", "properties": {"member": _STR, "start": _STR, "end": _STR},
                    "required": ["member", "start", "end"]}},
    {"name": "set_rest", "description": "Who rests (no scheduled chores) this week.",
     "parameters": {"type": "object", "properties": {"member": _STR}, "required": ["member"]}},
    {"name": "undo_last", "description": "Take back the sender's latest logged chore/busy entry, or the latest one "
                                         "about `member`.",
     "parameters": {"type": "object", "properties": {"member": _STR}}},
    {"name": "show_schedule", "description": "Show tasks for today / this week / next week, optionally one person's.",
     "parameters": {"type": "object", "properties": {
         "when": {"type": "string", "enum": ["today", "this_week", "next_week"]}, "member": _STR},
         "required": ["when"]}},
    {"name": "whose_turn", "description": "Whose turn for a turn-based chore (omit chore for all).",
     "parameters": {"type": "object", "properties": {"chore": _STR}}},
    {"name": "history", "description": "Who did a chore / when / what someone did. Filters optional.",
     "parameters": {"type": "object", "properties": {
         "question": _STR, "chore": _STR, "member": _STR, "since": _STR, "until": _STR}, "required": ["question"]}},
    {"name": "query", "description": "Stats needing counts across people/chores. One read-only DuckDB SELECT. "
                                     + QUERY_SCHEMA,
     "parameters": {"type": "object", "properties": {"question": _STR, "sql": _STR}, "required": ["question", "sql"]}},
    {"name": "register_member", "description": "ONLY when the sender gives their OWN name.",
     "parameters": {"type": "object", "properties": {"name": _STR}, "required": ["name"]}},
    {"name": "add_member", "description": "Add other roommates by name.",
     "parameters": {"type": "object", "properties": {"names": _PEOPLE}, "required": ["names"]}},
    {"name": "reply", "description": "Say something without changing anything: one short clarifying question, or "
                                     "how-to help. Not for chit-chat.",
     "parameters": {"type": "object", "properties": {"text": _STR}, "required": ["text"]}},
]

CONFIRM_TOOLS = {"set_chore", "remove_chore", "set_rest"}

EXAMPLES = """Examples (sender Atharva):
"Yuvraj did mopping and ari vacuumed" -> mark_done [Yuvraj mop, Ari vacuum]
"dishes ho gaye" -> mark_done [Atharva dishwasher]
"took out the trash yesterday" -> mark_done [Atharva trash, date=yesterday]
"scrubbed the balcony" -> mark_done [Atharva "balcony"] (one-off, don't ask)
"we will clean the bathroom every other week" -> set_chore bathroom kind=scheduled FREQ=WEEKLY;INTERVAL=2;BYDAY=SA
"assign Yuvraj and Ari mopping every week" -> set_chore mop FREQ=WEEKLY;BYDAY=SA people=[Yuvraj, Ari]
"Yuvraj doesn't need to do dishes" -> set_chore dishwasher remove_people=[Yuvraj]
"trash is whenever it's full, take turns" -> set_chore trash kind=turn
"who did the mopping?" -> history chore=mop
"who has done the most chores this month" -> query
"what do I have this week" -> show_schedule this_week member=Atharva
"I'm out of town Thu to Sun" -> set_busy Atharva
"oops undo that" -> undo_last
"lol same" / "rent is due friday" -> no tool"""


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


def build_system_prompt(today: date, sender: str | None, members: list[str], chores: list[str],
                        examples: bool = True) -> str:
    week = ", ".join(f"{(today + timedelta(days=i)):%a}={(today + timedelta(days=i)).isoformat()}" for i in range(-1, 7))
    return f"""You are the chore bot in roommates' Telegram group. Read each message and call tools; people write
casually, with typos and Hindi/English mixed.
Today: {today:%A} {today.isoformat()}. Dates: {week}.
Sender: {sender or 'UNREGISTERED (only register_member applies)'}. Members: {', '.join(members) or 'none yet'}.
Chores: {'; '.join(chores) or 'none yet'}.
Rules:
- Not about chores -> no tool. Problems with things (a broken trash can, a noisy dishwasher) are not chores.
- I/me/my/main/maine = the sender. Map nicknames/typos to members and to existing chore names.
- Someone DID a task that isn't in the list -> mark_done as a one-off. Never ask whether to make it a chore.
- Use the recent conversation for 'it', 'he', 'same for', and answers to your own questions.
- Several tool calls or several mark_done entries per message are fine.
- A brand-new chore with no schedule and not turn-based -> reply with one short question.
- {FREQ_HELP}
- Only call tools; never plain text.
{EXAMPLES if examples else ""}"""


def format_message(sender: str | None, text: str, history: list[str] | None = None, replied_to: str = "") -> str:
    """The user turn sent to the model: recent chat for context, then the new message."""
    parts = []
    if history:
        parts.append("Recent conversation (oldest first, for context only):\n" + "\n".join(history[-10:]))
    if replied_to:
        parts.append(f'The new message is a reply to: "{replied_to[:300]}"')
    parts.append(f"NEW MESSAGE from {sender or 'unregistered user'}: {text}")
    return "\n\n".join(parts)


class LLM:
    """Interface. Swap in another provider (e.g. Ollama) by implementing these two methods."""

    def parse(self, message: str, system_prompt: str) -> list[ToolCall]:
        raise NotImplementedError

    def summarize(self, question: str, rows: list[dict]) -> str:
        raise NotImplementedError


class GeminiLLM(LLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        # Backups used when the main model is overloaded (503) or rate-limited (429).
        fallback = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-2.5-flash")
        self.models = [self.model] + [m.strip() for m in fallback.split(",") if m.strip() and m.strip() != self.model]

    def _generate(self, **kwargs):
        from google.genai import errors

        last = None
        for model in self.models:
            try:
                return self.client.models.generate_content(model=model, **kwargs)
            except errors.APIError as e:
                if e.code not in (429, 500, 503, 404):
                    raise
                last = e  # busy / unavailable: try the next model
        raise last

    def parse(self, message: str, system_prompt: str) -> list[ToolCall]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"]) for t in TOOLS])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
        )
        resp = self._generate(contents=message, config=config)
        u = getattr(resp, "usage_metadata", None)
        self.last_usage = (getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0)
        return [ToolCall(fc.name, dict(fc.args or {})) for fc in (resp.function_calls or [])]

    def summarize(self, question: str, rows: list[dict]) -> str:
        prompt = SUMMARY_PROMPT.format(question=question, rows=rows)
        resp = self._generate(contents=prompt)
        return (resp.text or "").strip()


SUMMARY_PROMPT = ("Question: {question}\nQuery result rows (JSON): {rows}\n"
                  "Answer in one or two short sentences for a group chat. If there are no rows, say so plainly.")

# OpenAI-compatible providers: name -> (base URL, API-key env var, default model)
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY", "grok-4"),
}


class OpenAICompatibleLLM(LLM):
    """Groq, xAI (Grok), or any other OpenAI-compatible API."""

    def __init__(self, provider: str = "groq", model: str | None = None):
        from openai import OpenAI

        base_url, key_env, default_model = OPENAI_COMPATIBLE[provider]
        self.client = OpenAI(api_key=os.environ[key_env], base_url=base_url)
        self.model = model or os.environ.get(f"{provider.upper()}_MODEL", default_model)
        self.tools = [{"type": "function", "function": t} for t in TOOLS]

    def parse(self, message: str, system_prompt: str) -> list[ToolCall]:
        import json

        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, tools=self.tools, tool_choice="auto",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
        )
        calls = resp.choices[0].message.tool_calls or []
        u = getattr(resp, "usage", None)
        self.last_usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        return [ToolCall(c.function.name, json.loads(c.function.arguments or "{}")) for c in calls]

    def summarize(self, question: str, rows: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(question=question, rows=rows)}],
        )
        return (resp.choices[0].message.content or "").strip()


class FallbackLLM(LLM):
    """Try each provider in order; move on if one errors (rate limit, outage, bad key)."""

    def __init__(self, providers: list[LLM]):
        self.providers = providers

    def _try(self, method: str, *args):
        import logging

        last = None
        for p in self.providers:
            try:
                return getattr(p, method)(*args)
            except Exception as e:  # noqa: BLE001
                logging.getLogger("chorebot").warning("%s failed (%s); trying next provider", type(p).__name__, e)
                last = e
        raise last

    def parse(self, message: str, system_prompt: str) -> list[ToolCall]:
        return self._try("parse", message, system_prompt)

    def summarize(self, question: str, rows: list[dict]) -> str:
        return self._try("summarize", question, rows)


def _build(name: str) -> LLM:
    name = name.strip().lower()
    if name == "gemini":
        return GeminiLLM()
    if name in OPENAI_COMPATIBLE:
        return OpenAICompatibleLLM(name)
    raise ValueError(f"Unknown LLM provider '{name}'. Use one of: gemini, {', '.join(OPENAI_COMPATIBLE)}")


def make_llm() -> LLM:
    """LLM_PROVIDER picks the main model; LLM_FALLBACK (optional, comma list) are backups."""
    names = [os.environ.get("LLM_PROVIDER", "gemini")]
    names += [n for n in os.environ.get("LLM_FALLBACK", "").split(",") if n.strip()]
    providers = [_build(n) for n in names]
    return providers[0] if len(providers) == 1 else FallbackLLM(providers)
