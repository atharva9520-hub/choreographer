"""Chore rules. Plain Python, no LLM. Every public method takes explicit dates so it is testable."""
from __future__ import annotations

import difflib
import json
import uuid
from datetime import date, datetime, timedelta

from dateutil.rrule import rrulestr

RRULE_START = datetime(2026, 1, 5)  # fixed Monday so "every 2 weeks" stays stable


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_frequency(freq: str) -> str:
    """Validate an RRULE like 'FREQ=WEEKLY;BYDAY=SA'. Returns it normalised or raises ValueError."""
    freq = freq.strip().removeprefix("RRULE:").upper()
    try:
        rule = rrulestr(freq, dtstart=RRULE_START)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"I couldn't understand the frequency '{freq}'") from e
    if rule.after(datetime.now()) is None:
        raise ValueError(f"The frequency '{freq}' never happens again")
    return freq


class ChoreBook:
    COVER_WINDOW_WEEKS = 3  # covering is evened out over the next 3 plans (1/3 per week, so each cover counts once)

    def __init__(self, store, compensate: bool = True):
        self.s = store
        self.compensate = compensate  # give fewer tasks to people who recently covered for others

    # ---------- lookups ----------
    def members(self) -> list[dict]:
        return sorted(self.s.rows("Members"), key=lambda m: int(m["order"] or 0))

    def member_names(self) -> list[str]:
        return [m["name"] for m in self.members()]

    def member_by_telegram(self, telegram_id) -> str | None:
        for m in self.members():
            if m["telegram_id"] == str(telegram_id):
                return m["name"]
        return None

    def resolve_member(self, name: str) -> str:
        names = self.member_names()
        lookup = {n.lower(): n for n in names}
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
        prefix = [n for n in lookup if len(key) >= 3 and n.startswith(key)]
        if len(prefix) == 1:
            return lookup[prefix[0]]
        close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.75)
        if close:
            return lookup[close[0]]
        raise ValueError(f"I don't know anyone called '{name}'. Members: {', '.join(names) or 'none yet'}")

    def chores(self, active_only=True) -> list[dict]:
        rows = self.s.rows("Chores")
        return [c for c in rows if c["active"] == "yes"] if active_only else rows

    def find_chore(self, name: str) -> dict | None:
        chores = self.chores()
        lookup = {c["name"].lower(): c for c in chores}
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
        # 'mopping' -> 'mop', 'dishes' -> 'dishwasher', 'the trash' -> 'trash'
        key = key.removeprefix("the ").strip()
        stem = [n for n in lookup if len(key) >= 3 and len(n) >= 3 and (key.startswith(n) or n.startswith(key[:4]))]
        if len(stem) == 1:
            return lookup[stem[0]]
        close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.7)
        return lookup[close[0]] if close else None

    def pool(self, chore: dict) -> list[str]:
        """People who share this chore, in member order. Empty 'members' means everyone."""
        allowed = set(_split(chore["members"]))
        return [n for n in self.member_names() if not allowed or n in allowed]

    def setting(self, key: str, default: str = "") -> str:
        for r in self.s.rows("Settings"):
            if r["key"] == key:
                return r["value"]
        return default

    def set_setting(self, key: str, value: str) -> None:
        if not self.s.update("Settings", "key", key, {"value": value}):
            self.s.append("Settings", {"key": key, "value": value})

    def log(self, sender: str, message: str, action: str, details: dict) -> str:
        log_id = _id("l")
        self.s.append("Log", {
            "id": log_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sender": sender, "message": message, "action": action,
            "details": json.dumps(details), "undone": "",
        })
        return log_id

    # ---------- setup ----------
    def register_member(self, name: str, telegram_id) -> str:
        name = name.strip().title()
        if self.member_by_telegram(telegram_id):
            old = self.member_by_telegram(telegram_id)
            return f"You're already registered as {old}."
        if name.lower() in (n.lower() for n in self.member_names()):
            # Name exists but not linked yet (e.g. added by someone else): link it.
            for m in self.members():
                if m["name"].lower() == name.lower() and not m["telegram_id"]:
                    self.s.update("Members", "name", m["name"], {"telegram_id": str(telegram_id)})
                    return f"Linked you as {m['name']}."
            raise ValueError(f"{name} is already registered to someone else.")
        self.s.append("Members", {"name": name, "telegram_id": str(telegram_id), "order": len(self.members())})
        return f"Welcome {name}! You're member #{len(self.members())}."

    def add_member(self, name: str) -> str:
        """Add someone by name; they link their Telegram account later by saying "I'm <name>"."""
        name = name.strip().title()
        if name.lower() in (n.lower() for n in self.member_names()):
            return f"{name} is already a member."
        self.s.append("Members", {"name": name, "telegram_id": "", "order": len(self.members())})
        return f"Added {name}. {name}, say \"I'm {name}\" here once so I know it's you."

    def add_chore(self, name: str, type: str, frequency: str = "", members: list[str] | None = None,
                  first_up: str = "") -> str:
        name = name.strip().lower()
        if self.find_chore(name) and self.find_chore(name)["name"] == name:
            raise ValueError(f"'{name}' already exists")
        if type not in ("scheduled", "turn"):
            raise ValueError("type must be 'scheduled' or 'turn'")
        freq = parse_frequency(frequency) if type == "scheduled" else ""
        if type == "scheduled" and not frequency:
            raise ValueError("Scheduled chores need a frequency")
        people = [self.resolve_member(m) for m in (members or [])]
        chore = {"name": name, "type": type, "frequency": freq, "members": ",".join(people),
                 "next_up": "", "active": "yes"}
        if type == "turn":
            pool = self.pool(chore)
            chore["next_up"] = self.resolve_member(first_up) if first_up else (pool[0] if pool else "")
        existing = [c for c in self.chores(active_only=False) if c["name"] == name]
        if existing:  # re-adding a removed chore
            self.s.update("Chores", "name", name, chore)
        else:
            self.s.append("Chores", chore)
        return f"Added {self.describe(chore)}"

    def update_chore(self, name: str, frequency: str = "", include: list[str] | None = None,
                     exclude: list[str] | None = None, type: str = "", next_up: str = "") -> str:
        chore = self.find_chore(name)
        if not chore:
            raise ValueError(f"No chore called '{name}'")
        changes = {}
        if type:
            if type not in ("scheduled", "turn"):
                raise ValueError("type must be 'scheduled' or 'turn'")
            changes["type"] = type
        if frequency:
            changes["frequency"] = parse_frequency(frequency)
        if include or exclude:
            current = self.pool(chore)
            for m in include or []:
                m = self.resolve_member(m)
                if m not in current:
                    current.append(m)
            for m in exclude or []:
                m = self.resolve_member(m)
                if m in current:
                    current.remove(m)
            if not current:
                raise ValueError("A chore needs at least one person")
            everyone = set(self.member_names())
            changes["members"] = "" if set(current) == everyone else ",".join(current)
        if next_up:
            changes["next_up"] = self.resolve_member(next_up)
        new = {**chore, **changes}
        if new["type"] == "scheduled" and not new["frequency"]:
            raise ValueError("Scheduled chores need a frequency")
        if new["type"] == "turn" and new["next_up"] not in self.pool(new):
            changes["next_up"] = new["next_up"] = self.pool(new)[0]
        self.s.update("Chores", "name", chore["name"], changes)
        return f"Updated {self.describe(new)}"

    def set_chore(self, name: str, kind: str = "", frequency: str = "", people: list[str] | None = None,
                  add_people: list[str] | None = None, remove_people: list[str] | None = None,
                  next_up: str = "") -> str:
        """Create the chore if it's new, otherwise change only the fields given."""
        existing = self.find_chore(name)
        if existing is None:
            if not kind:
                kind = "scheduled" if frequency else ""
            if not kind:
                raise ValueError(f"How often should {name} happen? Or is it turn-based (done when needed)?")
            members = list(people or [])
            if add_people and members:
                members += add_people
            msg = self.add_chore(name, kind, frequency, members, next_up)
            if remove_people:
                msg = self.update_chore(name, exclude=remove_people)
            return msg
        if people:
            resolved = [self.resolve_member(m) for m in people]
            everyone = set(self.member_names())
            self.s.update("Chores", "name", existing["name"],
                          {"members": "" if set(resolved) == everyone else ",".join(resolved)})
        return self.update_chore(existing["name"], frequency=frequency, include=add_people, exclude=remove_people,
                                 type=kind, next_up=next_up)

    def check_set_chore(self, name: str, kind: str = "", frequency: str = "", people=None, add_people=None,
                        remove_people=None, next_up: str = "") -> None:
        """Validate a set_chore request before asking for confirmation."""
        for m in (people or []) + (add_people or []) + (remove_people or []):
            self.resolve_member(m)
        if next_up:
            self.resolve_member(next_up)
        if frequency:
            parse_frequency(frequency)
        if self.find_chore(name) is None and not kind and not frequency:
            raise ValueError(f"How often should {name} happen? Or is it turn-based (done when needed)?")

    def history(self, chore: str = "", member: str = "", since: str = "", until: str = "",
                limit: int = 15) -> list[dict]:
        """Completed tasks, newest first."""
        c = self.find_chore(chore) if chore else None
        chore_name = c["name"] if c else chore.strip().lower()
        who = self.resolve_member(member) if member else ""
        rows = [t for t in self.s.rows("Tasks") if t["status"] == "done"
                and (not chore_name or t["chore"] == chore_name)
                and (not who or t["done_by"] == who)
                and (not since or t["date"] >= since[:10])
                and (not until or t["date"] <= until[:10])]
        rows.sort(key=lambda t: t["date"], reverse=True)
        return [{"date": t["date"], "chore": t["chore"], "done_by": t["done_by"],
                 "was_assigned_to": t["assigned_to"]} for t in rows[:limit]]

    def undo_last(self, sender: str, member: str = "") -> str:
        """Undo the most recent chore log or busy entry sent by `sender` (or about `member`)."""
        who = self.resolve_member(member) if member else ""
        for entry in reversed(self.s.rows("Log")):
            if entry["action"] not in ("mark_done", "set_busy") or entry["undone"] == "yes":
                continue
            about = json.loads(entry["details"] or "{}").get("member", "")
            if (who and about == who) or (not who and entry["sender"] == sender):
                d = json.loads(entry["details"] or "{}")
                what = "busy entry" if entry["action"] == "set_busy" else self._task_label(d.get("task_id", ""))
                return self.undo(entry["id"]) + (f" ({what})" if what else "")
        return "I couldn't find anything recent to undo."

    def _task_label(self, task_id: str) -> str:
        t = next((t for t in self.s.rows("Tasks") if t["id"] == task_id), None)
        return f"{t['chore']} by {t['done_by'] or t['assigned_to']}" if t else ""

    def remove_chore(self, name: str) -> str:
        chore = self.find_chore(name)
        if not chore:
            raise ValueError(f"No chore called '{name}'")
        self.s.update("Chores", "name", chore["name"], {"active": "no"})
        for t in self.s.rows("Tasks"):
            if t["chore"] == chore["name"] and t["status"] == "open":
                self.s.update("Tasks", "id", t["id"], {"status": "cancelled"})
        return f"Removed {chore['name']}."

    def describe(self, chore: dict) -> str:
        who = chore["members"].replace(",", ", ") or "everyone"
        if chore["type"] == "turn":
            return f"{chore['name']} (takes turns: {who}; next: {chore['next_up']})"
        return f"{chore['name']} ({chore['frequency']}; {who})"

    # ---------- rest week ----------
    def rest_member(self, day: date) -> str | None:
        names = self.member_names()
        if len(names) < 3:
            return None
        anchor = self.setting("rest_anchor")
        if not anchor:
            anchor = monday_of(day).isoformat()
            self.set_setting("rest_anchor", anchor)
        weeks = (monday_of(day) - date.fromisoformat(anchor)).days // 7
        return names[weeks % len(names)]

    def set_rest(self, member: str, day: date) -> str:
        member = self.resolve_member(member)
        idx = self.member_names().index(member)
        self.set_setting("rest_anchor", (monday_of(day) - timedelta(weeks=idx)).isoformat())
        return f"{member} rests this week. Order after that: {', '.join(self._rest_preview(day))}"

    def _rest_preview(self, day: date, weeks=3) -> list[str]:
        return [self.rest_member(day + timedelta(weeks=i)) or "-" for i in range(1, weeks + 1)]

    # ---------- busy ----------
    def busy_on(self, member: str, day: date) -> bool:
        d = day.isoformat()
        return any(b["member"] == member and b["active"] == "yes" and b["start"] <= d <= b["end"]
                   for b in self.s.rows("Busy"))

    # ---------- weekly plan ----------
    def make_plan(self, monday: date) -> list[dict]:
        """(Re)build scheduled tasks for the week starting `monday`. Done tasks are kept."""
        monday = monday_of(monday)
        sunday = monday + timedelta(days=6)
        tasks = self.s.rows("Tasks")
        scheduled = {c["name"]: c for c in self.chores() if c["type"] == "scheduled"}
        for t in tasks:
            if t["status"] == "open" and t["chore"] in scheduled and monday.isoformat() <= t["date"] <= sunday.isoformat():
                self.s.update("Tasks", "id", t["id"], {"status": "cancelled"})
        done = {(t["date"], t["chore"]) for t in tasks if t["status"] == "done"}

        rest = self.rest_member(monday)
        names = self.member_names()
        week_idx = (monday - date(2026, 1, 5)).days // 7
        counts = {n: 0 for n in names}
        balance = self._cover_balance(monday) if self.compensate else {n: 0 for n in names}
        occurrences = []
        for c in scheduled.values():
            rule = rrulestr(c["frequency"], dtstart=RRULE_START)
            for dt in rule.between(datetime.combine(monday, datetime.min.time()),
                                   datetime.combine(sunday, datetime.max.time()), inc=True):
                occurrences.append((dt.date(), c))
        occurrences.sort(key=lambda x: (x[0], x[1]["name"]))

        new = []
        for day, c in occurrences:
            if (day.isoformat(), c["name"]) in done:
                continue
            pool = self.pool(c)
            eligible = [n for n in pool if n != rest and not self.busy_on(n, day)]
            if not eligible:  # fall back to the rest person rather than skipping the chore
                eligible = [n for n in pool if not self.busy_on(n, day)] or pool
            who = min(eligible, key=lambda n: (counts[n] + balance.get(n, 0), counts[n],
                                               (names.index(n) - week_idx) % len(names)))
            counts[who] += 1
            task = {"id": _id("t"), "date": day.isoformat(), "chore": c["name"],
                    "assigned_to": who, "status": "open", "done_by": ""}
            self.s.append("Tasks", task)
            new.append(task)
        return new

    def _cover_balance(self, monday: date) -> dict[str, float]:
        """Over the last few weeks: +1 each time someone did another person's scheduled task, -1 when someone
        else did theirs. Positive = owed a lighter week."""
        start = (monday - timedelta(weeks=self.COVER_WINDOW_WEEKS)).isoformat()
        bal = {n: 0 for n in self.member_names()}
        for t in self.s.rows("Tasks"):
            if t["status"] != "done" or not (start <= t["date"] < monday.isoformat()):
                continue
            a, d = t["assigned_to"], t["done_by"]
            if a and d and a != d:
                bal[d] = bal.get(d, 0) + 1
                bal[a] = bal.get(a, 0) - 1
        return {n: v / self.COVER_WINDOW_WEEKS for n, v in bal.items()}

    # ---------- turn-based ----------
    def whose_turn(self, chore: dict, day: date) -> str:
        """next_up, skipping anyone marked busy today."""
        pool = self.pool(chore)
        if not pool:
            return ""
        start = pool.index(chore["next_up"]) if chore["next_up"] in pool else 0
        for i in range(len(pool)):
            cand = pool[(start + i) % len(pool)]
            if not self.busy_on(cand, day):
                return cand
        return pool[start]

    # ---------- logging work ----------
    def mark_done(self, member: str, chore_name: str, day: date, sender: str, message: str) -> tuple[str, str]:
        """Returns (reply text, log id)."""
        member = self.resolve_member(member)
        chore = self.find_chore(chore_name)
        details: dict = {}
        if chore is None:  # ad hoc chore
            task = {"id": _id("t"), "date": day.isoformat(), "chore": chore_name.strip().lower(),
                    "assigned_to": "", "status": "done", "done_by": member}
            self.s.append("Tasks", task)
            details = {"task_id": task["id"], "created": True}
            reply = f"✅ {member}: {task['chore']} (one-off)"

        elif chore["type"] == "turn":
            task = {"id": _id("t"), "date": day.isoformat(), "chore": chore["name"],
                    "assigned_to": chore["next_up"], "status": "done", "done_by": member}
            self.s.append("Tasks", task)
            pool = self.pool(chore)
            nxt = pool[(pool.index(member) + 1) % len(pool)] if member in pool else chore["next_up"]
            self.s.update("Chores", "name", chore["name"], {"next_up": nxt})
            details = {"task_id": task["id"], "created": True, "chore": chore["name"],
                       "prev_next_up": chore["next_up"]}
            nxt_now = self.whose_turn({**chore, "next_up": nxt}, day)
            reply = f"✅ {member}: {chore['name']} → next: {nxt_now}"
        else:
            task = self._open_task_for(chore["name"], day)
            if task:
                self.s.update("Tasks", "id", task["id"], {"status": "done", "done_by": member})
                details = {"task_id": task["id"], "created": False}
                extra = "" if task["assigned_to"] in (member, "") else f" (covering for {task['assigned_to']})"
            else:
                task = {"id": _id("t"), "date": day.isoformat(), "chore": chore["name"],
                        "assigned_to": "", "status": "done", "done_by": member}
                self.s.append("Tasks", task)
                details = {"task_id": task["id"], "created": True}
                extra = " (extra)"
            reply = f"✅ {member}: {chore['name']}{extra}"
        details["member"] = member
        log_id = self.log(sender, message, "mark_done", details)
        return reply, log_id

    def _open_task_for(self, chore: str, day: date) -> dict | None:
        """Most recent open task on/before `day`; else the next one later this week."""
        week_end = (monday_of(day) + timedelta(days=6)).isoformat()
        d = day.isoformat()
        open_ = [t for t in self.s.rows("Tasks") if t["chore"] == chore and t["status"] == "open"]
        past = sorted([t for t in open_ if t["date"] <= d], key=lambda t: t["date"])
        if past:
            return past[-1]
        future = sorted([t for t in open_ if d < t["date"] <= week_end], key=lambda t: t["date"])
        return future[0] if future else None

    def set_busy(self, member: str, start: date, end: date, sender: str, message: str) -> tuple[str, str]:
        member = self.resolve_member(member)
        if end < start:
            start, end = end, start
        busy_id = _id("b")
        self.s.append("Busy", {"id": busy_id, "member": member, "start": start.isoformat(),
                               "end": end.isoformat(), "active": "yes"})
        moved = []
        tasks = self.s.rows("Tasks")
        for t in tasks:
            if t["assigned_to"] != member or t["status"] != "open":
                continue
            if not (start.isoformat() <= t["date"] <= end.isoformat()):
                continue
            day = date.fromisoformat(t["date"])
            chore = self.find_chore(t["chore"])
            pool = self.pool(chore) if chore else self.member_names()
            rest = self.rest_member(day)
            others = [n for n in pool if n != member and n != rest and not self.busy_on(n, day)]
            others = others or [n for n in pool if n != member and not self.busy_on(n, day)]
            if not others:
                continue
            week = (monday_of(day).isoformat(), (monday_of(day) + timedelta(days=6)).isoformat())
            load = {n: sum(1 for x in tasks if x["assigned_to"] == n and x["status"] == "open"
                           and week[0] <= x["date"] <= week[1]) for n in others}
            new = min(others, key=lambda n: load[n])
            self.s.update("Tasks", "id", t["id"], {"assigned_to": new})
            t["assigned_to"] = new
            moved.append([t["id"], member, new, t["chore"], t["date"]])
        log_id = self.log(sender, message, "set_busy", {"busy_id": busy_id, "moved": moved, "member": member})
        lines = [f"🗓️ {member} busy {start:%a %b %d} – {end:%a %b %d}."]
        lines += [f"  {chore} on {date.fromisoformat(d):%a} → {new}" for _, _, new, chore, d in moved]
        return "\n".join(lines), log_id

    def undo(self, log_id: str) -> str:
        entry = next((r for r in self.s.rows("Log") if r["id"] == log_id), None)
        if not entry:
            return "Couldn't find that entry."
        if entry["undone"] == "yes":
            return "Already undone."
        d = json.loads(entry["details"] or "{}")
        if entry["action"] == "mark_done":
            if d["created"]:
                self.s.update("Tasks", "id", d["task_id"], {"status": "undone"})
            else:
                self.s.update("Tasks", "id", d["task_id"], {"status": "open", "done_by": ""})
            if d.get("prev_next_up") is not None and d.get("chore"):
                self.s.update("Chores", "name", d["chore"], {"next_up": d["prev_next_up"]})
        elif entry["action"] == "set_busy":
            self.s.update("Busy", "id", d["busy_id"], {"active": "no"})
            for task_id, old, _new, _c, _d in d["moved"]:
                self.s.update("Tasks", "id", task_id, {"assigned_to": old})
        else:
            return "That kind of change can't be undone automatically; just tell me the fix."
        self.s.update("Log", "id", log_id, {"undone": "yes"})
        return "↩️ Undone."

    # ---------- summaries ----------
    @staticmethod
    def _who(t: dict) -> str:
        return t["done_by"] if t["status"] == "done" else t["assigned_to"]

    def day_summary(self, day: date, member: str | None = None) -> str:
        tasks = [t for t in self.s.rows("Tasks") if t["date"] == day.isoformat() and t["status"] in ("open", "done")
                 and (member is None or self._who(t) == member)]
        lines = [f"📋 {day:%A %b %d}" + (f" · {member}" if member else "")]
        for t in sorted(tasks, key=lambda t: t["chore"]):
            mark = "✅" if t["status"] == "done" else "⬜"
            who = t["done_by"] if t["status"] == "done" else t["assigned_to"]
            lines.append(f"{mark} {t['chore']}: {who}")
        for c in self.chores():
            if c["type"] == "turn" and (member is None or self.whose_turn(c, day) == member):
                lines.append(f"🔁 {c['name']}: next is {self.whose_turn(c, day)}")
        if len(lines) == 1:
            lines.append("Nothing scheduled.")
        return "\n".join(lines)

    def week_summary(self, monday: date, member: str | None = None) -> str:
        monday = monday_of(monday)
        rest = self.rest_member(monday)
        lines = [f"🗓️ Week of {monday:%b %d}" + (f" · {member}" if member else "")
                 + (f" — {rest} is resting 😴" if rest else "")]
        tasks = [t for t in self.s.rows("Tasks") if t["status"] in ("open", "done")
                 and monday.isoformat() <= t["date"] <= (monday + timedelta(days=6)).isoformat()
                 and (member is None or self._who(t) == member)]
        by_day: dict[str, list[str]] = {}
        for t in sorted(tasks, key=lambda t: (t["date"], t["chore"])):
            who = t["done_by"] if t["status"] == "done" else t["assigned_to"]
            by_day.setdefault(t["date"], []).append(f"{t['chore']}: {who}" + (" ✅" if t["status"] == "done" else ""))
        for d, items in by_day.items():
            lines.append(f"{date.fromisoformat(d):%a}: " + " · ".join(items))
        for c in self.chores():
            if c["type"] == "turn" and (member is None or self.whose_turn(c, monday) == member):
                lines.append(f"🔁 {c['name']}: next is {self.whose_turn(c, monday)}")
        if member and len(lines) == 1:
            lines.append("Nothing assigned.")
        return "\n".join(lines)
