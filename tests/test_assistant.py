"""End-to-end conversation with a scripted fake LLM (no network)."""
from datetime import date

import pytest

from chorebot.assistant import Assistant
from chorebot.llm import LLM, TOOLS, ToolCall
from chorebot.logic import ChoreBook
from chorebot.query import run_query
from chorebot.storage import MemoryStore

TODAY = date(2026, 10, 5)


class FakeLLM(LLM):
    def __init__(self):
        self.script: dict[str, list[ToolCall]] = {}

    def parse(self, message, system_prompt):
        text = message.split(": ", 1)[1]
        return self.script.get(text, [])

    def summarize(self, question, rows):
        return f"{question} -> {rows}"


@pytest.fixture
def env(monkeypatch):
    llm = FakeLLM()
    a = Assistant(ChoreBook(MemoryStore()), llm)
    monkeypatch.setattr(a, "today", lambda: TODAY)
    return a, llm


def say(env, text, tg, calls=None):
    a, llm = env
    if calls is not None:
        llm.script[text] = calls
    return a.handle_message(text, tg)


def test_full_flow(env):
    a, _ = env
    # unregistered user doing chores is asked to introduce themselves
    r = say(env, "I mopped", 1, [ToolCall("mark_done", {"entries": [{"member": "me", "chore": "mop"}]})])
    assert "don't know you" in r.text

    for tg, name in [(1, "Atharva"), (2, "Yuvraj"), (3, "Arihant")]:
        r = say(env, f"I'm {name}", tg, [ToolCall("register_member", {"name": name})])
        assert name in r.text

    # small talk -> silence
    assert say(env, "anyone want pizza?", 1, []).text == ""

    # settings need confirmation
    r = say(env, "setup chores", 1, [
        ToolCall("add_chore", {"name": "dishwasher", "type": "scheduled", "frequency": "FREQ=DAILY"}),
        ToolCall("add_chore", {"name": "mop", "type": "scheduled", "frequency": "FREQ=WEEKLY;BYDAY=SA"}),
        ToolCall("add_chore", {"name": "trash", "type": "turn", "first_up": "Yuvraj"}),
    ])
    assert len(r.pending) == 3 and "Confirm" in r.pending_text
    assert a.book.chores() == []
    out = a.apply_confirmed(r.pending, "Atharva")
    assert "trash" in out and len(a.book.chores()) == 3

    # plan exists for this week and excludes the rest person
    week = a.schedule_text("this_week")
    rest = a.book.rest_member(TODAY)
    assert f"{rest} is resting" in week

    # logging on behalf of others, multi-entry
    r = say(env, "yuvraj took trash and arihant did dishes", 1, [ToolCall("mark_done", {"entries": [
        {"member": "Yuvraj", "chore": "trash"}, {"member": "Arihant", "chore": "dishwasher"}]})])
    assert "next: Arihant" in r.text and len(r.log_ids) == 2
    assert a.book.find_chore("trash")["next_up"] == "Arihant"

    # undo both
    a.undo(r.log_ids)
    assert a.book.find_chore("trash")["next_up"] == "Yuvraj"

    # busy
    r = say(env, "Arihant busy till sunday", 3, [ToolCall("set_busy", {"member": "Arihant", "start": "2026-10-05",
                                                                        "end": "2026-10-11"})])
    assert "busy" in r.text

    # bad member name surfaces as a warning, not a crash
    r = say(env, "bob did dishes", 1, [ToolCall("mark_done", {"entries": [{"member": "Bob", "chore": "dishwasher"}]})])
    assert "⚠️" in r.text

    # query
    say(env, "done", 2, [ToolCall("mark_done", {"entries": [{"member": "Yuvraj", "chore": "trash"}]})])
    r = say(env, "who did trash?", 1, [ToolCall("query", {
        "question": "who did trash", "sql": "select done_by, count(*) n from tasks where chore='trash' "
                                             "and status='done' group by done_by"})])
    assert "Yuvraj" in r.text


def test_query_blocks_writes():
    s = MemoryStore()
    with pytest.raises(ValueError):
        run_query(s, "delete from tasks")
    with pytest.raises(ValueError):
        run_query(s, "select 1; drop table tasks")
    assert run_query(s, "select count(*) as n from tasks") == [{"n": "0"}]


def test_invalid_settings_rejected_before_confirm(env):
    say(env, "I'm Atharva", 1, [ToolCall("register_member", {"name": "Atharva"})])
    r = say(env, "add mop", 1, [ToolCall("add_chore", {"name": "mop", "type": "scheduled", "frequency": "sometimes"})])
    assert not r.pending and "⚠️" in r.text


def test_gemini_tool_declarations_build():
    from google.genai import types
    tools = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                       parameters_json_schema=t["parameters"]) for t in TOOLS]
    assert len(tools) == len(TOOLS)


def test_add_other_members_then_they_link(env):
    a, _ = env
    say(env, "I'm Atharva", 1, [ToolCall("register_member", {"name": "Atharva"})])
    r = say(env, "the other two are yuvraj and ari", 1, [ToolCall("add_member", {"names": ["yuvraj", "ari"]})])
    assert "Added Yuvraj" in r.text and "Added Ari" in r.text
    assert a.book.member_names() == ["Atharva", "Yuvraj", "Ari"]
    r = say(env, "I'm Ari", 3, [ToolCall("register_member", {"name": "Ari"})])
    assert "Linked" in r.text and a.book.member_by_telegram(3) == "Ari"
    # sender can't accidentally take over someone else's name
    r = say(env, "I'm Yuvraj", 1, [ToolCall("register_member", {"name": "Yuvraj"})])
    assert "⚠️" in r.text and a.book.member_by_telegram(1) == "Atharva"
