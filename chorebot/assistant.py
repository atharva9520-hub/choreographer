"""Glue between chat messages, the LLM and the chore rules. No Telegram code here, so it's testable."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .llm import CONFIRM_TOOLS, LLM, ToolCall, build_system_prompt
from .logic import ChoreBook, monday_of, parse_frequency
from .query import run_query


@dataclass
class Result:
    text: str = ""                                     # reply to post ('' = stay silent)
    log_ids: list[str] = field(default_factory=list)   # for "undo" replies
    pending: list[ToolCall] = field(default_factory=list)  # settings changes waiting for ✅
    pending_text: str = ""


class Assistant:
    def __init__(self, book: ChoreBook, llm: LLM, tz: str = "America/Chicago"):
        self.book = book
        self.llm = llm
        self.tz = ZoneInfo(tz)

    def today(self) -> date:
        return datetime.now(self.tz).date()

    # ---------- incoming message ----------
    def handle_message(self, text: str, telegram_id, replied_to: str = "") -> Result:
        sender = self.book.member_by_telegram(telegram_id)
        today = self.today()
        chores = [self.book.describe(c) for c in self.book.chores()]
        prompt = build_system_prompt(today, sender, self.book.member_names(), chores)
        message = f"{sender or 'Unregistered user'}: {text}"
        if replied_to:
            message = f"(replying to: \"{replied_to[:300]}\")\n{message}"
        calls = self.llm.parse(message, prompt)
        if not calls:
            return Result()

        if sender is None:
            reg = [c for c in calls if c.name == "register_member"]
            if not reg:
                return Result("I don't know you yet. Say \"I'm <your name>\" first 🙂")
            calls = reg

        out, log_ids, pending = [], [], []
        for call in calls:
            try:
                if call.name in CONFIRM_TOOLS:
                    self._validate(call)
                    pending.append(call)
                    continue
                text_out, ids = self._apply_now(call, sender, text, telegram_id, today)
                out.append(text_out)
                log_ids += ids
            except ValueError as e:
                out.append(f"⚠️ {e}")
        res = Result("\n".join(o for o in out if o), log_ids, pending)
        if pending:
            res.pending_text = "Confirm these changes?\n" + "\n".join("• " + self.preview(c) for c in pending)
        return res

    def _apply_now(self, call: ToolCall, sender, text, telegram_id, today) -> tuple[str, list[str]]:
        a = call.args
        if call.name == "register_member":
            return self.book.register_member(a["name"], telegram_id), []
        if call.name == "add_member":
            return "\n".join(self.book.add_member(n) for n in a.get("names", [])), []
        if call.name == "mark_done":
            lines, ids = [], []
            for e in a.get("entries", []):
                try:
                    day = _date(e.get("date"), today)
                    reply, log_id = self.book.mark_done(e["member"], e["chore"], day, sender, text)
                    lines.append(reply)
                    ids.append(log_id)
                except ValueError as err:
                    lines.append(f"⚠️ {err}")
            return "\n".join(lines), ids
        if call.name == "set_busy":
            start, end = _date(a.get("start"), today), _date(a.get("end"), today)
            reply, log_id = self.book.set_busy(a["member"], start, end, sender, text)
            return reply, [log_id]
        if call.name == "show_schedule":
            return self.schedule_text(a.get("when", "this_week"), today), []
        if call.name == "query":
            try:
                rows = run_query(self.book.s, a["sql"])
            except Exception as err:  # noqa: BLE001  (bad SQL from the model)
                raise ValueError(f"I couldn't answer that ({err}). Try rephrasing?") from err
            return "🔎 " + self.llm.summarize(a.get("question", text), rows), []
        raise ValueError(f"Unknown action {call.name}")

    # ---------- settings changes (after ✅) ----------
    def _validate(self, call: ToolCall) -> None:
        a = call.args
        for m in a.get("members", []) + a.get("include", []) + a.get("exclude", []):
            self.book.resolve_member(m)
        for key in ("first_up", "next_up", "member"):
            if a.get(key):
                self.book.resolve_member(a[key])
        if a.get("frequency"):
            parse_frequency(a["frequency"])
        if call.name in ("update_chore", "remove_chore") and not self.book.find_chore(a["name"]):
            raise ValueError(f"No chore called '{a['name']}'")

    def preview(self, call: ToolCall) -> str:
        a = call.args
        bits = [f"{k}={', '.join(v) if isinstance(v, list) else v}" for k, v in a.items() if v and k != "name"]
        label = {"add_chore": "Add", "update_chore": "Update", "remove_chore": "Remove",
                 "set_rest": "Rest this week:"}[call.name]
        return f"{label} {a.get('name', '')} " + " · ".join(bits)

    def apply_confirmed(self, calls: list[ToolCall], sender: str) -> str:
        today = self.today()
        out = []
        for call in calls:
            a = call.args
            try:
                if call.name == "add_chore":
                    out.append(self.book.add_chore(a["name"], a["type"], a.get("frequency", ""),
                                                   a.get("members"), a.get("first_up", "")))
                elif call.name == "update_chore":
                    out.append(self.book.update_chore(a["name"], a.get("frequency", ""), a.get("include"),
                                                      a.get("exclude"), a.get("type", ""), a.get("next_up", "")))
                elif call.name == "remove_chore":
                    out.append(self.book.remove_chore(a["name"]))
                elif call.name == "set_rest":
                    out.append(self.book.set_rest(a["member"], today))
                self.book.log(sender, "", call.name, a)
            except ValueError as e:
                out.append(f"⚠️ {e}")
        self.book.make_plan(today)  # keep this week's plan in sync with the new settings
        out.append("This week's plan is updated. Ask \"what's the plan this week?\" to see it.")
        return "\n".join(out)

    # ---------- schedule ----------
    def schedule_text(self, when: str, today: date | None = None) -> str:
        today = today or self.today()
        if when == "today":
            self._ensure_plan(today)
            return self.book.day_summary(today)
        monday = monday_of(today) + (timedelta(weeks=1) if when == "next_week" else timedelta())
        self._ensure_plan(monday)
        return self.book.week_summary(monday)

    def _ensure_plan(self, day: date) -> None:
        monday = monday_of(day)
        week = (monday.isoformat(), (monday + timedelta(days=6)).isoformat())
        has = any(week[0] <= t["date"] <= week[1] and t["status"] in ("open", "done") and t["assigned_to"]
                  for t in self.book.s.rows("Tasks"))
        if not has:
            self.book.make_plan(monday)

    def undo(self, log_ids: list[str]) -> str:
        return "\n".join(self.book.undo(i) for i in log_ids) or "Nothing to undo."


def _date(value: str | None, default: date) -> date:
    if not value:
        return default
    try:
        return date.fromisoformat(value[:10])
    except ValueError as e:
        raise ValueError(f"Bad date '{value}'") from e
