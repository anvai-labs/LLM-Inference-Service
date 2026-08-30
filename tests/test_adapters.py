from dataclasses import dataclass

import pytest

from ollama_server.adapters import (
    HuggingFaceAdapter,
    OllamaChatAdapter,
    OllamaGenerateAdapter,
    OpenAIAdapter,
    VLLMAdapter,
)


@dataclass
class DummyModel:
    context_size: int = 8192


class DummyManager:
    def build_model_mapping(self) -> dict[str, str]:
        return {"tiny:1b": "/tiny", "large:7b": "/large"}

    def get_model_info(self, name: str) -> DummyModel | None:
        return DummyModel() if name in self.build_model_mapping() else None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4


def test_openai_round_trip_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIAdapter(DummyManager(), "0.5,0.5")  # type: ignore[arg-type]
    request = adapter.parse_request(
        {
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 42,
        }
    )
    assert request.model_name == "tiny:1b"
    assert request.max_tokens == 42
    assert request.tensor_split == "0.5,0.5"
    assert request.prompt == "Be brief\n\nUSER: Hello\nASSISTANT:"
    response = adapter.format_response("  answer  ", request)
    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["usage"]["total_tokens"] > 0
    with pytest.raises(ValueError, match="messages"):
        adapter.parse_request({})


def test_ollama_chat_and_generate_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ollama_server.adapters.base.model_inspector.get_enhanced_model_info",
        lambda _name, _info: {"context_size": 4096},
    )
    manager = DummyManager()
    chat = OllamaChatAdapter(manager)  # type: ignore[arg-type]
    chat_request = chat.parse_request(
        {
            "model": "tiny:1b",
            "messages": [{"role": "user", "content": "Hello"}],
            "options": {"num_ctx": 16_384},
        }
    )
    assert chat_request.context_size == 4096
    assert chat.format_response(" hi ", chat_request)["message"]["content"] == "hi"

    class Tracker:
        @staticmethod
        def get_request(_request_id: str) -> None:
            return None

    generate = OllamaGenerateAdapter(manager, Tracker())  # type: ignore[arg-type]
    generated_request = generate.parse_request({"model": "tiny:1b", "prompt": "Hello"})
    assert generate.format_response(" result ", generated_request)["response"] == "result"
    with pytest.raises(ValueError, match="prompt"):
        generate.parse_request({"model": "tiny:1b"})


def test_huggingface_and_vllm_protocol_shapes() -> None:
    manager = DummyManager()
    huggingface = HuggingFaceAdapter(manager)  # type: ignore[arg-type]
    request = huggingface.parse_request(
        {"model": "tiny:1b", "inputs": "hello", "parameters": {"max_new_tokens": 5}}
    )
    assert request.max_tokens == 5
    assert huggingface.format_response(" done ", request) == {"generated_text": "done"}

    vllm = VLLMAdapter(manager)  # type: ignore[arg-type]
    completion = vllm.parse_request({"model": "tiny:1b", "prompt": "hello"})
    assert vllm.format_response("done", completion)["object"] == "text_completion"
    chat = vllm.parse_request(
        {"model": "tiny:1b", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert vllm.format_response("done", chat)["object"] == "chat.completion"
    with pytest.raises(ValueError, match="messages.*prompt"):
        vllm.parse_request({"model": "tiny:1b"})
