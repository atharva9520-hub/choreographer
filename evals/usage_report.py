"""Real-world numbers from your live Google Sheet (run after the bot has been used for a while).

    python -m evals.usage_report
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date

from dotenv import load_dotenv

from chorebot.storage import SheetsStore


def main():
    load_dotenv()
    s = SheetsStore(os.environ["GOOGLE_SHEET_ID"], os.environ.get("GOOGLE_CREDENTIALS_FILE", "service_account.json"))
    msgs, log, tasks = s.rows("Messages"), s.rows("Log"), s.rows("Tasks")

    acted = [m for m in msgs if m["tool_calls"] not in ("", "[]")]
    tools = Counter(c["name"] for m in acted for c in json.loads(m["tool_calls"]))
    logged = [r for r in log if r["action"] == "mark_done"]
    undone = [r for r in logged if r["undone"] == "yes"]
    done = [t for t in tasks if t["status"] == "done"]
    days = sorted({m["timestamp"][:10] for m in msgs if m["timestamp"]})
    by_person = Counter(t["done_by"] for t in done)
    covered = [t for t in done if t["assigned_to"] and t["assigned_to"] != t["done_by"]]
    past_sched = [t for t in tasks if t["assigned_to"] and t["status"] in ("open", "done")
                  and t["date"] < date.today().isoformat()]

    print(f"Period: {days[0] if days else '-'} to {days[-1] if days else '-'} ({len(days)} days with messages)")
    print(f"Messages read: {len(msgs)} · acted on: {len(acted)} ({len(acted) / max(len(msgs), 1):.0%})")
    print(f"Actions by type: {dict(tools.most_common())}")
    print(f"Chores logged: {len(logged)} · later undone (bot misunderstood or user mistake): "
          f"{len(undone)} ({len(undone) / max(len(logged), 1):.1%})")
    print(f"Completed tasks: {len(done)} · by person: {dict(by_person)}")
    print(f"Covered for someone else: {len(covered)} ({len(covered) / max(len(done), 1):.0%})")
    if past_sched:
        rate = sum(t['status'] == 'done' for t in past_sched) / len(past_sched)
        print(f"Scheduled tasks completed on/after their day: {rate:.0%} of {len(past_sched)}")


if __name__ == "__main__":
    main()
