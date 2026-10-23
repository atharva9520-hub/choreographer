"""LLM layer: turns a chat message into tool calls. Provider-agnostic interface + Gemini implementation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

FREQ_HELP = (
    "iCal RRULE without the 'RRULE:' prefix. Examples: daily='FREQ=DAILY'; every Saturday="
    "'FREQ=WEEKLY;BYDAY=SA'; twice a week Wed+Sun='FREQ=WEEKLY;BYDAY=WE,SU'; every 2 weeks on Monday="
    "'FREQ=WEEKLY;INTERVAL=2;BYDAY=MO'; every 3 days='FREQ=DAILY;INTERVAL=3'; monthly on the 1st="
    "'FREQ=MONTHLY;BYMONTHDAY=1'. If the user only says 'twice a week', pick sensible spread-out days."
)

QUERY_SCHEMA = """DuckDB tables (all columns are TEXT; dates are 'YYYY-MM-DD'):
members(name, telegram_id, "order")
chores(name, type ['scheduled'|'turn'], frequency, members [comma list, '' = everyone], next_up, active ['yes'|'no'])
tasks(id, date, chore, assigned_to, status ['open'|'done'|'cancelled'|'undone'], done_by)
busy(id, member, start, "end", active)
log(id, timestamp, sender, message, action, details, undone)
Completed work = tasks where status='done' (done_by is who did it). Ignore cancelled/undone rows."""

TOOLS = [
    {
        "name": "register_member",
        "description": "ONLY when the SENDER states their own name, e.g. 'I'm Atharva'. Never use it for other people.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "name": "add_member",
        "description": "Add OTHER roommates by name, e.g. 'the other two are Yuvraj and Ari'.",
        "parameters": {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                       "required": ["names"]},
    },
    {
        "name": "add_chore",
        "description": "Create a chore. 'scheduled' chores repeat on a frequency. 'turn' chores have no schedule "
                       "(done when needed, e.g. trash) and rotate strictly between people.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "short lowercase name, e.g. 'mop', 'dishwasher', 'trash'"},
            "type": {"type": "string", "enum": ["scheduled", "turn"]},
            "frequency": {"type": "string", "description": "Only for scheduled. " + FREQ_HELP},
            "members": {"type": "array", "items": {"type": "string"},
                        "description": "Who shares this chore. Omit for everyone."},
            "first_up": {"type": "string", "description": "Only for turn chores: who goes first."},
        }, "required": ["name", "type"]},
    },
    {
        "name": "update_chore",
        "description": "Change an existing chore: frequency, include/exclude people, type, or whose turn is next.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "frequency": {"type": "string", "description": FREQ_HELP},
            "include": {"type": "array", "items": {"type": "string"}},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "type": {"type": "string", "enum": ["scheduled", "turn"]},
            "next_up": {"type": "string"},
        }, "required": ["name"]},
    },
    {
        "name": "remove_chore",
        "description": "Delete a chore.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "name": "set_rest",
        "description": "Choose who rests (does no scheduled chores) this week; the rotation continues from there.",
        "parameters": {"type": "object", "properties": {"member": {"type": "string"}}, "required": ["member"]},
    },
    {
        "name": "mark_done",
        "description": "Someone did one or more chores. One entry per person+chore. Works for chores done by "
                       "the sender or reported on behalf of others, and for one-off chores not in the list.",
        "parameters": {"type": "object", "properties": {"entries": {"type": "array", "items": {
            "type": "object", "properties": {
                "member": {"type": "string"},
                "chore": {"type": "string", "description": "Existing chore name if it matches, else a short description"},
                "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
            }, "required": ["member", "chore"]}}}, "required": ["entries"]},
    },
    {
        "name": "set_busy",
        "description": "Someone is busy/away and can't do chores for a date range; their tasks get reassigned.",
        "parameters": {"type": "object", "properties": {
            "member": {"type": "string"},
            "start": {"type": "string", "description": "YYYY-MM-DD"},
            "end": {"type": "string", "description": "YYYY-MM-DD (same as start for one day)"},
        }, "required": ["member", "start", "end"]},
    },
    {
        "name": "show_schedule",
        "description": "Show the plan.",
        "parameters": {"type": "object", "properties": {
            "when": {"type": "string", "enum": ["today", "this_week", "next_week"]}}, "required": ["when"]},
    },
    {
        "name": "query",
        "description": "Answer a question about chore data/history (e.g. who did most, whose turn, how many times). "
                       "Write one read-only DuckDB SQL query.\n" + QUERY_SCHEMA,
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}, "sql": {"type": "string"}}, "required": ["question", "sql"]},
    },
]

CONFIRM_TOOLS = {"add_chore", "update_chore", "remove_chore", "set_rest"}


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


def build_system_prompt(today: date, sender: str | None, members: list[str], chores: list[str]) -> str:
    return f"""You are the chore bot in a Telegram group of roommates. You read every message.
Today is {today:%A} {today.isoformat()} (America/Chicago).
Sender: {sender or 'UNREGISTERED (only register_member is useful for them)'}.
Members: {', '.join(members) or 'none yet'}.
Chores: {'; '.join(chores) or 'none yet'}.

Rules:
- If the message is ordinary conversation and not about chores, call NO tool and reply with nothing.
- 'I', 'me', 'mine' = the sender. Map nicknames/typos to the member list. Messages may mix Hindi/English.
- Resolve relative dates ('yesterday', 'this weekend', 'till Sunday') to YYYY-MM-DD.
- One message can need several tool calls or several mark_done entries.
- Use existing chore names when the user means one of them (e.g. 'vacuumed' -> 'vacuum').
- Only call tools; never answer in text."""


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
