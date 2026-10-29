"""The terminal simulator, driven by a scripted model."""
from datetime import date

from chorebot.assistant import Assistant
from chorebot.llm import LLM, ToolCall
from chorebot.logic import ChoreBook
from chorebot.simulate import Simulator
from chorebot.storage import MemoryStore

SCRIPT = {
    "I'm Atharva": [ToolCall("register_member", {"name": "Atharva"})],
    "I'm Yuvraj": [ToolCall("register_member", {"name": "Yuvraj"})],
    "I'm Ari": [ToolCall("register_member", {"name": "Ari"})],
    "trash is turn based": [ToolCall("set_chore", {"name": "trash", "kind": "turn"})],
    "took out trash": [ToolCall("mark_done", {"entries": [{"member": "me", "chore": "trash"}]})],
}


class Scripted(LLM):
    def parse(self, message, prompt):
        return SCRIPT.get(message.rsplit(": ", 1)[1], [])

    def summarize(self, q, rows):
        return str(rows)


def test_three_people_one_terminal():
    lines = []
    sim = Simulator(Assistant(ChoreBook(MemoryStore()), Scripted()), out=lines.append)
    sim.handle("/date 2026-11-02")
    for name in ("Atharva", "Yuvraj", "Ari"):
        sim.handle(f"{name}: I'm {name}")
    assert sim.a.book.member_names() == ["Atharva", "Yuvraj", "Ari"]
    sim.handle("/as Ari")
    sim.handle("trash is turn based")
    assert sim.pending
    sim.handle("yes")
    assert sim.a.book.find_chore("trash")
    sim.handle("Yuvraj: took out trash")
    assert "Yuvraj: trash" in "\n".join(lines) and sim.a.book.find_chore("trash")["next_up"] == "Ari"
    sim.handle("undo")
    assert sim.a.book.find_chore("trash")["next_up"] == "Atharva"
    sim.handle("anyone want pizza")
    assert "(bot stayed silent)" in lines[-1]
    sim.handle("/next")
    assert sim.a.today() == date(2026, 11, 3)
    sim.handle("/morning")
    assert "Tuesday Nov 03" in lines[-1]
    assert sim.handle("/quit") is False
