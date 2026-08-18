import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.llm import LLMClient, LLMResponse, ToolCall


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system, history, tools, max_tokens):
        self.calls.append({"system": system, "history_len": len(history), "history": list(history)})
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", input_tokens=10, output_tokens=10)


def tool_response(name: str, inp: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, input=inp)],
        stop_reason="tool_use", input_tokens=10, output_tokens=10,
    )
