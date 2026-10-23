"""Provider wiring without network: fake OpenAI-style client and fallback chain."""
import json
from types import SimpleNamespace as NS

import pytest

from chorebot.llm import TOOLS, FallbackLLM, LLM, OpenAICompatibleLLM, ToolCall, make_llm


def fake_completion(tool_calls=None, content=None):
    msg = NS(tool_calls=tool_calls, content=content)
    return NS(choices=[NS(message=msg)])


def test_openai_compatible_parses_tool_calls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    llm = OpenAICompatibleLLM("groq")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return fake_completion([NS(function=NS(name="mark_done", arguments=json.dumps(
            {"entries": [{"member": "Yuvraj", "chore": "mop"}, {"member": "Ari", "chore": "vacuum"}]})))])

    llm.client = NS(chat=NS(completions=NS(create=create)))
    calls = llm.parse("Atharva: yuvraj mopped, ari vacuumed", "sys")
    assert calls == [ToolCall("mark_done", {"entries": [{"member": "Yuvraj", "chore": "mop"},
                                                        {"member": "Ari", "chore": "vacuum"}]})]
    assert seen["model"] == "openai/gpt-oss-120b"
    assert len(seen["tools"]) == len(TOOLS) and seen["tools"][0]["type"] == "function"


def test_openai_compatible_no_tool_calls_is_silence(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    llm = OpenAICompatibleLLM("groq")
    llm.client = NS(chat=NS(completions=NS(create=lambda **kw: fake_completion(None, "lol"))))
    assert llm.parse("Atharva: pizza?", "sys") == []


class Boom(LLM):
    def parse(self, *a):
        raise RuntimeError("rate limited")

    def summarize(self, *a):
        raise RuntimeError("down")


class Ok(LLM):
    def parse(self, *a):
        return [ToolCall("show_schedule", {"when": "today"})]

    def summarize(self, *a):
        return "ok"


def test_fallback_uses_next_provider():
    llm = FallbackLLM([Boom(), Ok()])
    assert llm.parse("m", "s")[0].name == "show_schedule"
    assert llm.summarize("q", []) == "ok"
    with pytest.raises(RuntimeError):
        FallbackLLM([Boom()]).parse("m", "s")


def test_make_llm_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_FALLBACK", "gemini")
    llm = make_llm()
    assert isinstance(llm, FallbackLLM) and len(llm.providers) == 2
    monkeypatch.setenv("LLM_FALLBACK", "")
    assert isinstance(make_llm(), OpenAICompatibleLLM)
