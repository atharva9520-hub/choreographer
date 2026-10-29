"""Chat with the bot in your terminal as several people, without Telegram and without touching the real sheet.

    python -m chorebot.simulate            # in-memory data, real LLM from .env
    python -m chorebot.simulate --sheet    # use the real Google Sheet (careful: changes are real)

Type as the current person, or prefix a message with a name:   Yuvraj: did the dishes
Commands start with "/" — type /help to see them.
"""
from __future__ import annotations

import sys
from collections import deque
from datetime import date, timedelta

from .assistant import Assistant
from .logic import ChoreBook, monday_of

HELP = """Commands:
  /as NAME          switch who you are typing as (e.g. /as Ari)
  NAME: message     send one message as NAME without switching
  yes / no          answer the bot's confirmation question
  undo              undo the bot's last ✅ message (like replying "undo" in Telegram)
  /date YYYY-MM-DD  pretend today is this date      /next   move one day forward
  /sunday           post next week's plan (the Sunday 7pm message)
  /morning          post today's tasks (the 8am message)
  /show TAB         print a tab: members, chores, tasks, busy, log, messages
  /help             this list           /quit   exit"""


class Simulator:
    def __init__(self, assistant: Assistant, out=print):
        self.a = assistant
        self.out = out
        self.ids: dict[str, int] = {}          # pretend Telegram ids per name
        self.current = ""
        self.pending = None
        self.last_log_ids: list[str] = []
        self.history: deque[str] = deque(maxlen=12)
        self.today = assistant.today()
        assistant.today = lambda: self.today  # let /date and /next move time

    def fake_id(self, name: str) -> int:
        key = name.strip().lower()
        if key not in self.ids:
            self.ids[key] = 1000 + len(self.ids)
        return self.ids[key]

    def bot(self, text: str) -> None:
        if text:
            self.out("🤖 " + text.replace("\n", "\n   "))
            self.history.append("Bot: " + text[:300].replace("\n", " | "))

    def handle(self, line: str) -> bool:
        """Process one line. Returns False to quit."""
        line = line.strip()
        if not line:
            return True
        low = line.lower()
        if low in ("/quit", "/exit"):
            return False
        if low == "/help":
            self.out(HELP)
            return True
        if low.startswith("/as "):
            self.current = line[4:].strip().title()
            self.out(f"(now typing as {self.current})")
            return True
        if low.startswith("/date "):
            self.today = date.fromisoformat(line[6:].strip())
            self.out(f"(today is now {self.today:%A %b %d})")
            return True
        if low == "/next":
            self.today += timedelta(days=1)
            self.out(f"(today is now {self.today:%A %b %d})")
            return True
        if low == "/sunday":
            nxt = monday_of(self.today) + timedelta(weeks=1)
            self.a.book.make_plan(nxt)
            self.bot("Next week's plan:\n" + self.a.book.week_summary(nxt))
            return True
        if low == "/morning":
            self.bot("Good morning! " + self.a.schedule_text("today"))
            return True
        if low.startswith("/show"):
            self.show(line[5:].strip() or "tasks")
            return True

        speaker = self.current
        if ":" in line and not line.startswith("http"):
            name, rest = line.split(":", 1)
            if name.strip() and " " not in name.strip():
                speaker, line, low = name.strip().title(), rest.strip(), rest.strip().lower()
        if not speaker:
            self.out("(pick who you are first, e.g. /as Atharva — or type 'Atharva: hi')")
            return True
        uid = self.fake_id(speaker)
        sender = self.a.book.member_by_telegram(uid) or speaker

        if self.pending and low in ("yes", "y", "ok", "haan", "✅", "no", "n", "nah", "❌"):
            calls, self.pending = self.pending, None
            if low in ("no", "n", "nah", "❌"):
                self.bot(f"❌ Cancelled by {sender}.")
            elif not self.a.book.member_by_telegram(uid):
                self.bot("Introduce yourself first (\"I'm <name>\").")
            else:
                self.bot(f"✅ Confirmed by {sender}\n" + self.a.apply_confirmed(calls, sender))
            return True
        if low in ("undo", "undo it", "undo that") and self.last_log_ids:
            ids, self.last_log_ids = self.last_log_ids, []
            self.bot(self.a.undo(ids))
            return True

        context = list(self.history)
        self.history.append(f"{sender}: {line[:300]}")
        try:
            res = self.a.handle_message(line, uid, "", context)
        except Exception as e:  # noqa: BLE001
            self.bot(f"⚠️ error: {type(e).__name__}: {e}")
            return True
        if not res.text and not res.pending:
            self.out("   (bot stayed silent)")
        self.bot(res.text)
        if res.log_ids:
            self.last_log_ids = res.log_ids
        if res.pending:
            self.pending = res.pending
            self.bot(res.pending_text + "\n(type yes or no)")
        return True

    def show(self, tab: str) -> None:
        name = {t.lower(): t for t in ("Members", "Chores", "Tasks", "Busy", "Log", "Messages")}.get(tab.lower())
        if not name:
            self.out("Tabs: members, chores, tasks, busy, log, messages")
            return
        rows = self.a.book.s.rows(name)
        if name == "Tasks":
            rows = [r for r in rows if r["status"] in ("open", "done")]
        if not rows:
            self.out(f"({name} is empty)")
        for r in rows[-25:]:
            self.out("   " + " | ".join(f"{k}={v}" for k, v in r.items() if v and k not in ("id", "details")))


def main():
    from dotenv import load_dotenv

    from .llm import make_llm
    from .storage import MemoryStore

    load_dotenv()
    if "--sheet" in sys.argv:
        import os

        from .storage import SheetsStore
        store = SheetsStore(os.environ["GOOGLE_SHEET_ID"], os.environ.get("GOOGLE_CREDENTIALS_FILE", "service_account.json"))
        print("⚠️  Using the REAL Google Sheet. Everything you do here is saved.")
    else:
        store = MemoryStore()
        print("Using a temporary in-memory house (nothing is saved).")
    sim = Simulator(Assistant(ChoreBook(store), make_llm()))
    print(HELP + "\n\nStart with:  /as Atharva   then   I'm Atharva\n")
    while True:
        try:
            line = input(f"{sim.current or '?'}> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sim.handle(line):
            break


if __name__ == "__main__":
    main()
