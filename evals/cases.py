"""Benchmark: realistic group-chat messages and the tool calls they should produce.

Each case has a category, a sender, the message, the expected calls, and optional recent chat history.
`expected` is a list of (tool, args); only the args listed are checked. [] means the bot should stay silent.
Dates use helpers so the set works on any day.

These messages are held out: none of them appears in the few-shot examples inside the prompt
(tests/test_evals_harness.py checks this), so scores aren't inflated by the model copying its examples.

House used for every case: members Atharva, Yuvraj, Ari; chores dishwasher (daily), mop (Saturdays),
vacuum (Wed + Sun), trash (turn-based).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


def day(offset: int):
    return lambda today: (today + timedelta(days=offset)).isoformat()


def next_weekday(wd: int):  # 0 = Monday; the next one strictly after today
    return lambda today: (today + timedelta(days=(wd - today.weekday()) % 7 or 7)).isoformat()


@dataclass
class Case:
    category: str
    sender: str
    text: str
    expected: list
    history: list[str] = field(default_factory=list)
    also_ok: list = field(default_factory=list)  # other acceptable answers (lists of calls)


def C(category, sender, text, expected, history=None, also_ok=None):
    return Case(category, sender, text, expected, history or [], also_ok or [])


def done(*pairs, date=None):
    entries = []
    for member, chore in pairs:
        e = {"member": member, "chore": chore}
        if date:
            e["date"] = date
        entries.append(e)
    return [("mark_done", {"entries": entries})]


SAT = "FREQ=WEEKLY;BYDAY=SA"

CASES: list[Case] = [
    # ---------------- logging your own work ----------------
    C("log_self", "Atharva", "just did the dishes", done(("Atharva", "dishwasher"))),
    C("log_self", "Ari", "unloaded the dishwasher", done(("Ari", "dishwasher"))),
    C("log_self", "Yuvraj", "mopped the floor", done(("Yuvraj", "mop"))),
    C("log_self", "Ari", "vacuuming done ✅", done(("Ari", "vacuum"))),
    C("log_self", "Atharva", "garbage is out", done(("Atharva", "trash"))),
    C("log_self", "Yuvraj", "i took the bins out", done(("Yuvraj", "trash"))),
    C("log_self", "Ari", "finished mopping the kitchen and living room", done(("Ari", "mop"))),
    C("log_self", "Atharva", "emptied the dishwasher this morning", done(("Atharva", "dishwasher"))),
    C("log_self", "Yuvraj", "hoovered the whole apartment", done(("Yuvraj", "vacuum"))),
    C("log_self", "Ari", "trash taken out 👍", done(("Ari", "trash"))),
    C("log_self", "Atharva", "I did the vacuum yesterday, forgot to say",
      done(("Atharva", "vacuum"), date=day(-1))),
    C("log_self", "Yuvraj", "did dishes last night", done(("Yuvraj", "dishwasher"), date=day(-1))),
    C("log_self", "Ari", "cleaned the oven today", [("mark_done", {"entries": [{"member": "Ari"}]})]),
    C("log_self", "Atharva", "wiped down the bathroom mirror", [("mark_done", {"entries": [{"member": "Atharva"}]})]),

    # ---------------- logging for someone else ----------------
    C("log_other", "Atharva", "Ari did the dishes", done(("Ari", "dishwasher"))),
    C("log_other", "Ari", "yuvraj mopped today", done(("Yuvraj", "mop"))),
    C("log_other", "Yuvraj", "atharva vacuumed", done(("Atharva", "vacuum"))),
    C("log_other", "Atharva", "fyi ari took out the trash", done(("Ari", "trash"))),
    C("log_other", "Ari", "Atharva handled the dishwasher", done(("Atharva", "dishwasher"))),
    C("log_other", "Yuvraj", "ari cleaned up the dishes after dinner", done(("Ari", "dishwasher"))),
    C("log_other", "Atharva", "yuvraj did the mopping yesterday", done(("Yuvraj", "mop"), date=day(-1))),
    C("log_other", "Ari", "trash was taken out by Yuvraj", done(("Yuvraj", "trash"))),
    C("log_other", "Atharva", "Ari vacuumed on Sunday",
      [("mark_done", {"entries": [{"member": "Ari", "chore": "vacuum"}]})]),

    # ---------------- several updates in one message ----------------
    C("log_multi", "Atharva", "Yuvraj mopped, Ari vacuumed", done(("Yuvraj", "mop"), ("Ari", "vacuum"))),
    C("log_multi", "Ari", "I did dishes and took out the trash", done(("Ari", "dishwasher"), ("Ari", "trash"))),
    C("log_multi", "Yuvraj", "me and atharva did the mopping and vacuuming, I mopped he vacuumed",
      done(("Yuvraj", "mop"), ("Atharva", "vacuum"))),
    C("log_multi", "Atharva", "dishes done by Ari, trash by Yuvraj", done(("Ari", "dishwasher"), ("Yuvraj", "trash"))),
    C("log_multi", "Ari", "did the vacuum, mop and dishes today",
      done(("Ari", "vacuum"), ("Ari", "mop"), ("Ari", "dishwasher"))),
    C("log_multi", "Yuvraj", "Atharva took out trash and I unloaded the dishwasher",
      done(("Atharva", "trash"), ("Yuvraj", "dishwasher"))),
    C("log_multi", "Atharva", "all three of us cleaned: Ari mopped, Yuvraj vacuumed, I did dishes",
      done(("Ari", "mop"), ("Yuvraj", "vacuum"), ("Atharva", "dishwasher"))),

    # ---------------- Hindi / Hinglish ----------------
    C("hinglish", "Yuvraj", "bartan dho diye", done(("Yuvraj", "dishwasher"))),
    C("hinglish", "Atharva", "kachra bahar daal diya", done(("Atharva", "trash"))),
    C("hinglish", "Ari", "maine pocha laga diya", done(("Ari", "mop"))),
    C("hinglish", "Atharva", "Yuvraj ne vacuum kar diya", done(("Yuvraj", "vacuum"))),
    C("hinglish", "Yuvraj", "aaj dishwasher maine khali kiya", done(("Yuvraj", "dishwasher"))),
    C("hinglish", "Ari", "main kal se sunday tak bahar hu", [("set_busy", {"member": "Ari", "start": day(1),
                                                                          "end": next_weekday(6)})]),
    C("hinglish", "Atharva", "trash kiska turn hai?", [("whose_turn", {"chore": "trash"})]),
    C("hinglish", "Yuvraj", "pocha kisne lagaya tha?", [("history", {"chore": "mop"})]),

    # ---------------- typos and shorthand ----------------
    C("typos", "Atharva", "did teh dishs", done(("Atharva", "dishwasher"))),
    C("typos", "Ari", "vaccumed", done(("Ari", "vacuum"))),
    C("typos", "Yuvraj", "yuvi mopped", done(("Yuvraj", "mop"))),
    C("typos", "Atharva", "arii took out trsh", done(("Ari", "trash"))),
    C("typos", "Ari", "dishwshr done", done(("Ari", "dishwasher"))),
    C("typos", "Yuvraj", "atharv did vaccum", done(("Atharva", "vacuum"))),

    # ---------------- new chores ----------------
    C("chore_new", "Atharva", "let's add cleaning the bathroom every Sunday",
      [("set_chore", {"name": "bathroom", "frequency": "FREQ=WEEKLY;BYDAY=SU"})]),
    C("chore_new", "Ari", "we should water the plants every 3 days",
      [("set_chore", {"name": "plants", "frequency": "FREQ=DAILY;INTERVAL=3"})]),
    C("chore_new", "Yuvraj", "fridge cleanout on the 1st of every month",
      [("set_chore", {"name": "fridge", "frequency": "FREQ=MONTHLY;BYMONTHDAY=1"})]),
    C("chore_new", "Atharva", "laundry room cleaning every other saturday",
      [("set_chore", {"name": "laundry", "frequency": "FREQ=WEEKLY;INTERVAL=2;BYDAY=SA"})]),
    C("chore_new", "Ari", "groceries every monday and thursday",
      [("set_chore", {"name": "groceries", "frequency": "FREQ=WEEKLY;BYDAY=MO,TH"})]),
    C("chore_new", "Atharva", "toilet paper restock, whoever's turn when it runs out",
      [("set_chore", {"name": "toilet paper", "kind": "turn"})]),
    C("chore_new", "Yuvraj", "recycling should rotate between us, just when it fills up",
      [("set_chore", {"name": "recycling", "kind": "turn"})]),
    C("chore_new", "Atharva", "only me and Yuvraj will do the bathroom, every Sunday",
      [("set_chore", {"name": "bathroom", "frequency": "FREQ=WEEKLY;BYDAY=SU", "people": ["Atharva", "Yuvraj"]})]),
    C("chore_new", "Ari", "Yuvraj and I will dust every alternate week on friday",
      [("set_chore", {"name": "dust", "frequency": "FREQ=WEEKLY;INTERVAL=2;BYDAY=FR", "people": ["Yuvraj", "Ari"]})]),
    C("chore_new", "Atharva", "add cleaning the stove", [("reply", {})]),

    # ---------------- changing chores ----------------
    C("chore_change", "Atharva", "move mopping to Fridays", [("set_chore", {"name": "mop", "frequency": "FREQ=WEEKLY;BYDAY=FR"})]),
    C("chore_change", "Ari", "vacuum only once a week now, on sunday",
      [("set_chore", {"name": "vacuum", "frequency": "FREQ=WEEKLY;BYDAY=SU"})]),
    C("chore_change", "Yuvraj", "let's mop every other week instead",
      [("set_chore", {"name": "mop", "frequency": "FREQ=WEEKLY;INTERVAL=2;BYDAY=SA"})]),
    C("chore_change", "Atharva", "take Ari off vacuuming", [("set_chore", {"name": "vacuum", "remove_people": ["Ari"]})]),
    C("chore_change", "Ari", "I can't do dishes anymore", [("set_chore", {"name": "dishwasher", "remove_people": ["Ari"]})]),
    C("chore_change", "Atharva", "put Ari back on vacuum duty", [("set_chore", {"name": "vacuum", "add_people": ["Ari"]})]),
    C("chore_change", "Yuvraj", "from now on only Atharva and Ari do the mopping",
      [("set_chore", {"name": "mop", "people": ["Atharva", "Ari"]})]),
    C("chore_change", "Atharva", "Yuvraj will handle the dishwasher alone",
      [("set_chore", {"name": "dishwasher", "people": ["Yuvraj"]})]),
    C("chore_change", "Ari", "dishes should be shared by all three of us",
      [("set_chore", {"name": "dishwasher", "people": ["Atharva", "Yuvraj", "Ari"]})]),
    C("chore_change", "Atharva", "make vacuuming turn based", [("set_chore", {"name": "vacuum", "kind": "turn"})]),
    C("chore_change", "Yuvraj", "Ari should go next for trash", [("set_chore", {"name": "trash", "next_up": "Ari"})]),
    C("chore_change", "Atharva", "stop tracking the mop", [("remove_chore", {"name": "mop"})]),
    C("chore_change", "Ari", "delete the vacuum chore", [("remove_chore", {"name": "vacuum"})]),
    C("chore_change", "Atharva", "Yuvraj gets the week off from chores", [("set_rest", {"member": "Yuvraj"})],
      also_ok=[[("set_busy", {"member": "Yuvraj"})]]),
    C("chore_change", "Ari", "I'll take the rest week this time", [("set_rest", {"member": "Ari"})]),

    # ---------------- busy / away ----------------
    C("busy", "Yuvraj", "I'm traveling from tomorrow until Sunday",
      [("set_busy", {"member": "Yuvraj", "start": day(1), "end": next_weekday(6)})]),
    C("busy", "Atharva", "Ari is sick today, can't do chores", [("set_busy", {"member": "Ari", "start": day(0), "end": day(0)})]),
    C("busy", "Ari", "I have exams, can't do anything for the next 3 days",
      [("set_busy", {"member": "Ari", "start": day(0)})]),
    C("busy", "Yuvraj", "Atharva is away this weekend",
      [("set_busy", {"member": "Atharva"})]),
    C("busy", "Atharva", "I won't be home tomorrow", [("set_busy", {"member": "Atharva", "start": day(1), "end": day(1)})]),
    C("busy", "Ari", "Yuvraj's out of town till Wednesday", [("set_busy", {"member": "Yuvraj", "end": next_weekday(2)})]),

    # ---------------- questions ----------------
    C("question", "Atharva", "who mopped last?", [("history", {"chore": "mop"})]),
    C("question", "Ari", "when was the last time someone vacuumed", [("history", {"chore": "vacuum"})]),
    C("question", "Yuvraj", "what has Ari done this week", [("history", {"member": "Ari"})],
      also_ok=[[("query", {})]]),
    C("question", "Atharva", "did anyone do the dishes today?", [("history", {"chore": "dishwasher"})],
      also_ok=[[("show_schedule", {"when": "today"})]]),
    C("question", "Ari", "who's on trash?", [("whose_turn", {"chore": "trash"})]),
    C("question", "Yuvraj", "is it my turn for the garbage?", [("whose_turn", {"chore": "trash"})]),
    C("question", "Atharva", "what's left for me today", [("show_schedule", {"when": "today", "member": "Atharva"})]),
    C("question", "Ari", "what are my chores this week?", [("show_schedule", {"when": "this_week", "member": "Ari"})]),
    C("question", "Yuvraj", "what's the schedule for this week", [("show_schedule", {"when": "this_week"})]),
    C("question", "Atharva", "plan for next week?", [("show_schedule", {"when": "next_week"})]),
    C("question", "Ari", "what needs doing today", [("show_schedule", {"when": "today"})]),
    C("question", "Atharva", "how many times has each of us mopped this month", [("query", {})]),
    C("question", "Yuvraj", "who is slacking the most lol", [("query", {})]),
    C("question", "Ari", "rank us by chores done in October", [("query", {})]),

    # ---------------- undo ----------------
    C("undo", "Ari", "wait, undo that", [("undo_last", {})]),
    C("undo", "Yuvraj", "sorry wrong one, cancel my last message", [("undo_last", {})]),
    C("undo", "Atharva", "scratch that, I didn't actually do the dishes", [("undo_last", {})]),
    C("undo", "Ari", "remove what Yuvraj just logged, he didn't mop", [("undo_last", {"member": "Yuvraj"})]),

    # ---------------- members ----------------
    C("members", "Atharva", "we have a new roommate, Sam", [("add_member", {"names": ["Sam"]})]),
    C("members", "Ari", "please add Priya to the group", [("add_member", {"names": ["Priya"]})]),

    # ---------------- follow-ups that need the recent conversation ----------------
    C("followup", "Atharva", "and before that?", [("history", {"chore": "mop"})],
      ["Atharva: who mopped last?", "Bot: 🔎 Yuvraj mopped on Saturday."]),
    C("followup", "Ari", "same for vacuum", [("history", {"chore": "vacuum"})],
      ["Ari: who did the mopping this week?", "Bot: 🔎 Atharva mopped on Saturday."]),
    C("followup", "Atharva", "every Sunday", [("set_chore", {"name": "stove", "frequency": "FREQ=WEEKLY;BYDAY=SU"})],
      ["Atharva: add cleaning the stove", "Bot: How often should the stove be cleaned? Or is it turn-based?"]),
    C("followup", "Yuvraj", "just whenever, we take turns", [("set_chore", {"name": "bathroom", "kind": "turn"})],
      ["Yuvraj: add bathroom cleaning", "Bot: How often should bathroom happen? Or is it turn-based?"]),
    C("followup", "Ari", "he did it yesterday actually", done(("Yuvraj", "mop"), date=day(-1)),
      ["Atharva: did Yuvraj mop?", "Bot: 🔎 Nobody has logged mopping this week."]),
    C("followup", "Atharva", "her too", [("set_busy", {"member": "Ari"})],
      ["Yuvraj: I'm away this weekend", "Bot: 🗓️ Yuvraj busy Sat – Sun."]),

    # ---------------- ordinary chat: should stay silent ----------------
    C("chitchat", "Ari", "who's up for movies tonight", []),
    C("chitchat", "Yuvraj", "hahaha", []),
    C("chitchat", "Atharva", "rent is due on the 1st btw", []),
    C("chitchat", "Ari", "landlord is coming to fix the sink tomorrow", []),
    C("chitchat", "Yuvraj", "can someone grab milk on the way back", []),
    C("chitchat", "Atharva", "good morning everyone", []),
    C("chitchat", "Ari", "the dishwasher is making a weird noise", []),
    C("chitchat", "Yuvraj", "I'm so tired today", []),
    C("chitchat", "Atharva", "anyone seen my charger?", []),
    C("chitchat", "Ari", "ok", []),
    C("chitchat", "Yuvraj", "did you guys watch the game", []),
    C("chitchat", "Atharva", "the trash can in my room is broken lol", []),
    C("chitchat", "Ari", "thanks!", []),
    C("chitchat", "Yuvraj", "wifi password?", []),
    C("chitchat", "Atharva", "ordering pizza, who's in", []),
]

CATEGORIES = sorted({c.category for c in CASES})
