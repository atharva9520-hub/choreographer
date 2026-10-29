"""The benchmark checker, with scripted models (no network)."""
from datetime import date

from chorebot.llm import EXAMPLES, LLM, ToolCall
from evals.cases import CASES
from evals.run_evals import check, eval_today, make_house, report, resolve, run, score, summarize


def test_checker_matches_equivalent_answers():
    book, today = make_house(), date(2026, 10, 29)
    exp = [("set_chore", {"name": "mop", "frequency": "FREQ=WEEKLY;BYDAY=SA", "people": ["Yuvraj", "Ari"]})]
    good = [ToolCall("set_chore", {"name": "mopping", "frequency": "RRULE:FREQ=WEEKLY;BYDAY=SA",
                                   "people": ["ari", "yuvraj"]})]
    bad = [ToolCall("set_chore", {"name": "mop", "frequency": "FREQ=WEEKLY;BYDAY=SU", "people": ["Yuvraj", "Ari"]})]
    assert check(exp, good, book, today)
    assert not check(exp, bad, book, today)
    assert check([], [], book, today) and not check([], good, book, today)
    # a clarifying question is what some cases want
    assert check([("reply", {})], [ToolCall("reply", {"text": "How often?"})], book, today)
    assert not check([("reply", {})], [], book, today)
    # new chore names match loosely
    assert check([("set_chore", {"name": "plants"})], [ToolCall("set_chore", {"name": "water plants"})], book, today)


class Perfect(LLM):
    def parse(self, message, prompt):
        text = message.split("NEW MESSAGE from ", 1)[1].split(": ", 1)[1]
        for c in CASES:
            if c.text == text:
                self.last_usage = (100, 10)
                return [ToolCall(n, resolve(a, eval_today())) for n, a in c.expected]
        return []


def test_perfect_model_scores_100_and_report_renders():
    rows = run(Perfect(), verbose=False, model_name="perfect")
    assert score(rows) == 1.0
    s = summarize(rows)
    assert s["false_trigger_rate"] == 0 and s["missed_rate"] == 0 and s["avg_tokens_in"] == 100
    md = report(rows)
    assert "| perfect | full | 116/116 | 100.0%" in md and "## Accuracy by category" in md


def test_silent_model_scores_only_chitchat():
    class Silent(LLM):
        def parse(self, message, prompt):
            return []
    rows = run(Silent(), verbose=False)
    s = summarize(rows)
    assert s["missed_rate"] == 1.0 and s["false_trigger_rate"] == 0.0
    assert abs(s["accuracy"] - sum(c.category == "chitchat" for c in CASES) / len(CASES)) < 1e-9


def test_cases_are_held_out_from_prompt_examples():
    example_msgs = {line.split('"')[1].lower() for line in EXAMPLES.splitlines() if line.startswith('"')}
    leaked = [c.text for c in CASES if c.text.lower() in example_msgs]
    assert not leaked, f"benchmark messages also appear in the prompt: {leaked}"


def test_case_texts_unique():
    texts = [c.text for c in CASES]
    assert len(texts) == len(set(texts))


def test_daily_limit_skips_rest_and_errors_are_not_scored():
    class Limited(LLM):
        n = 0

        def parse(self, message, prompt):
            self.n += 1
            if self.n > 3:
                raise RuntimeError("Error code: 429 - Rate limit reached ... tokens per day (TPD)")
            return Perfect().parse(message, prompt)
    rows = run(Limited(), verbose=False)
    s = summarize(rows)
    assert s["n"] == 3 and s["accuracy"] == 1.0 and s["skipped"] == len(CASES) - 3


def test_resume_reuses_answers():
    first = run(Perfect(), verbose=False, model_name="m")
    done = {(r.model, r.variant, r.text): r for r in first}

    class Boom(LLM):
        def parse(self, message, prompt):
            raise AssertionError("should not be called")
    again = run(Boom(), verbose=False, model_name="m", done=done)
    assert score(again) == 1.0
