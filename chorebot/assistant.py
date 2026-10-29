"""Glue between chat messages, the LLM and the chore rules. No Telegram code here, so it's testable."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import json

from .llm import CONFIRM_TOOLS, LLM, ToolCall, build_system_prompt, format_message
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
    def handle_message(self, text: str, telegram_id, replied_to: str = "",
                       history: list[str] | None = None) -> Result:
        sender = self.book.member_by_telegram(telegram_id)
        today = self.today()
        chores = [self.book.describe(c) for c in self.book.chores()]
        prompt = build_system_prompt(today, sender, self.book.member_names(), chores)
        calls = self.llm.parse(format_message(sender, text, history, replied_to), prompt)
        res = self._handle_calls(calls, sender, text, telegram_id, today)
        self._record(sender or str(telegram_id), text, calls, res)
        return res

    def _record(self, sender: str, text: str, calls: list[ToolCall], res: "Result") -> None:
        """Every message + what the model decided, for reviewing misunderstandings and building evals."""
        try:
            self.book.s.append("Messages", {
                "timestamp": datetime.now(self.tz).isoformat(timespec="seconds"), "sender": sender, "text": text,
                "tool_calls": json.dumps([{"name": c.name, "args": c.args} for c in calls], ensure_ascii=False),
                "reply": (res.text + ("\n" + res.pending_text if res.pending_text else ""))[:500],
            })
        except Exception:  # noqa: BLE001  never let logging break the bot
            pass

    def _handle_calls(self, calls: list[ToolCall], sender, text, telegram_id, today) -> "Result":
        if not calls:
            return Result()
        if sender:  # models sometimes pass "me"/"I" instead of the sender's name
            calls = [ToolCall(c.name, _replace_me(c.args, sender)) for c in calls]

        if sender is None:
            reg = [c for c in calls if c.name == "register_member"]
            if not reg:
                return Result("I don't know you yet. Say \"I'm <your name>\" first 🙂")
            calls = reg[:1]
        else:
            # Already registered: naming someone else means adding them, never renaming the sender.
            others = [c.args.get("name", "") for c in calls
                      if c.name == "register_member" and c.args.get("name", "").lower() != sender.lower()]
            calls = [c for c in calls if c.name != "register_member"]
            if others:
                calls.insert(0, ToolCall("add_member", {"names": others}))

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
        if call.name == "reply":
            return a.get("text", ""), []
        if call.name == "undo_last":
            return self.book.undo_last(sender, a.get("member", "")), []
        if call.name == "show_schedule":
            member = self.book.resolve_member(a["member"]) if a.get("member") else None
            return self.schedule_text(a.get("when", "this_week"), today, member), []
        if call.name == "whose_turn":
            turn = [c for c in self.book.chores() if c["type"] == "turn"]
            if a.get("chore"):
                c = self.book.find_chore(a["chore"])
                if c is None:
                    raise ValueError(f"There's no chore called '{a['chore']}'.")
                if c["type"] != "turn":
                    return self._scheduled_turn_text(c, today), []
                turn = [c]
            if not turn:
                return "There are no turn-based chores yet.", []
            return "\n".join(f"🔁 {c['name']}: {self.book.whose_turn(c, today)}'s turn" for c in turn), []
        if call.name == "history":
            rows = self.book.history(a.get("chore", ""), a.get("member", ""), a.get("since", ""), a.get("until", ""))
            return "🔎 " + self.llm.summarize(a.get("question", text), rows), []
        if call.name == "query":
            try:
                rows = run_query(self.book.s, a["sql"])
            except Exception as err:  # noqa: BLE001  (bad SQL from the model)
                raise ValueError(f"I couldn't answer that ({err}). Try rephrasing?") from err
            return "🔎 " + self.llm.summarize(a.get("question", text), rows), []
        raise ValueError(f"Unknown action {call.name}")

    def _scheduled_turn_text(self, chore: dict, today: date) -> str:
        self._ensure_plan(today)
        upcoming = sorted((t for t in self.book.s.rows("Tasks") if t["chore"] == chore["name"]
                           and t["status"] == "open" and t["date"] >= today.isoformat()), key=lambda t: t["date"])
        if not upcoming:
            return f"{chore['name']} has nothing open right now."
        t = upcoming[0]
        return f"🗓️ {chore['name']} is next on {date.fromisoformat(t['date']):%a %b %d}: {t['assigned_to']}"

    # ---------- settings changes (after ✅) ----------
    def _validate(self, call: ToolCall) -> None:
        a = call.args
        if call.name == "set_chore":
            self.book.check_set_chore(**_chore_args(a))
        elif call.name == "remove_chore" and not self.book.find_chore(a["name"]):
            raise ValueError(f"There's no chore called '{a['name']}'.")
        elif call.name == "set_rest":
            self.book.resolve_member(a["member"])

    def preview(self, call: ToolCall) -> str:
        a = call.args
        if call.name == "remove_chore":
            return f"Remove {a['name']}"
        if call.name == "set_rest":
            return f"{self.book.resolve_member(a['member'])} rests this week"
        exists = self.book.find_chore(a["name"]) is not None
        bits = []
        if a.get("kind") == "turn":
            bits.append("take turns, done when needed")
        if a.get("frequency"):
            bits.append(describe_frequency(a["frequency"]))
        if a.get("people"):
            bits.append("by " + ", ".join(self.book.resolve_member(m) for m in a["people"]))
        if a.get("add_people"):
            bits.append("add " + ", ".join(self.book.resolve_member(m) for m in a["add_people"]))
        if a.get("remove_people"):
            bits.append("without " + ", ".join(self.book.resolve_member(m) for m in a["remove_people"]))
        if a.get("next_up"):
            bits.append(f"{self.book.resolve_member(a['next_up'])} is next")
        return f"{'Update' if exists else 'Add'} {a['name']}: " + "; ".join(bits or ["no changes"])

    def apply_confirmed(self, calls: list[ToolCall], sender: str) -> str:
        today = self.today()
        out = []
        for call in calls:
            a = call.args
            try:
                if call.name == "set_chore":
                    out.append(self.book.set_chore(**_chore_args(a)))
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
    def schedule_text(self, when: str, today: date | None = None, member: str | None = None) -> str:
        today = today or self.today()
        if when == "today":
            self._ensure_plan(today)
            return self.book.day_summary(today, member)
        monday = monday_of(today) + (timedelta(weeks=1) if when == "next_week" else timedelta())
        self._ensure_plan(monday)
        return self.book.week_summary(monday, member)

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


def _chore_args(a: dict) -> dict:
    return {"name": a["name"], "kind": a.get("kind", ""), "frequency": a.get("frequency", ""),
            "people": a.get("people") or None, "add_people": a.get("add_people") or None,
            "remove_people": a.get("remove_people") or None, "next_up": a.get("next_up", "")}


_DAYS = {"MO": "Mon", "TU": "Tue", "WE": "Wed", "TH": "Thu", "FR": "Fri", "SA": "Sat", "SU": "Sun"}


def describe_frequency(rrule: str) -> str:
    """'FREQ=WEEKLY;INTERVAL=2;BYDAY=SA' -> 'every 2 weeks on Sat' (for confirmation messages)."""
    parts = dict(p.split("=", 1) for p in rrule.upper().removeprefix("RRULE:").split(";") if "=" in p)
    n = int(parts.get("INTERVAL", "1"))
    unit = {"DAILY": "day", "WEEKLY": "week", "MONTHLY": "month"}.get(parts.get("FREQ", ""), "")
    if not unit:
        return rrule
    every = f"every {unit}" if n == 1 else ("every other " + unit if n == 2 else f"every {n} {unit}s")
    if parts.get("BYDAY"):
        days = ", ".join(_DAYS.get(d[-2:], d) for d in parts["BYDAY"].split(","))
        every = ("every " + days) if n == 1 and unit == "week" else f"{every} on {days}"
    if parts.get("BYMONTHDAY"):
        every += f" on day {parts['BYMONTHDAY']}"
    return every


_ME = {"me", "i", "myself", "mine", "my", "main", "self", "sender"}


def _replace_me(value, sender: str):
    if isinstance(value, str):
        return sender if value.strip().lower() in _ME else value
    if isinstance(value, list):
        return [_replace_me(v, sender) for v in value]
    if isinstance(value, dict):
        return {k: (v if k in ("text", "question", "sql") else _replace_me(v, sender)) for k, v in value.items()}
    return value
