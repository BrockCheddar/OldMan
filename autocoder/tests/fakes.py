import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.llm import LLMClient, LLMResponse, ToolCall


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    # Distinctive opening line of _condense_batch's system prompt (see
    # agent.py) -- used below to recognize a condensation call specifically.
    _CONDENSATION_SYSTEM_MARKER = "You are building per-file notes on a codebase"

    def complete(self, system, history, tools, max_tokens):
        self.calls.append({"system": system, "history_len": len(history), "history": list(history)})
        # Condensation passes (outer/inner read-batch condensing, and the
        # post-mark_step_done dirty-file flush) are best-effort in
        # production -- agent.py already treats an LLMError from one as
        # non-fatal, not a run-ending failure -- and they fire on a
        # schedule tests don't control (batch thresholds, which files got
        # edited). Checked BEFORE popping so a condensation call can never
        # consume the response a test scripted for the next real agent
        # turn; tests don't need to account for these at all unless they
        # explicitly want to (see condensation_response below).
        if self._CONDENSATION_SYSTEM_MARKER in (system or ""):
            return text_response("")
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            # BaseException (not just Exception) so tests can script a
            # KeyboardInterrupt to simulate a real interruption mid-run,
            # exercising Agent.run's actual save-and-return path instead
            # of contriving the same effect some other way.
            raise item
        return item

    def count_tokens(self, system, history, tools):
        """Deterministic char-based approximation -- fine for a test
        double since it isn't real infrastructure. Production backends
        (OpenAICompatBackend, AnthropicBackend) always ask the real
        tokenizer; see llm.py."""
        total = len(system or "")
        if tools:
            total += len(json.dumps(tools))
        for t in history:
            if t.text:
                total += len(t.text)
            for tc in t.tool_calls:
                total += len(tc.name) + len(json.dumps(tc.input))
            for tr in t.tool_results:
                total += len(tr.content)
        return max(1, total // 4)


def text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", input_tokens=10, output_tokens=10)


def tool_response(name: str, inp: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, input=inp)],
        stop_reason="tool_use", input_tokens=10, output_tokens=10,
    )
