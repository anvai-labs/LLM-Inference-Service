from ollama_server.utils.response_processing import (
    ThinkTagProcessor,
    preserve_think_tags_in_streaming,
)


def test_extracts_multiple_thinking_blocks_and_cleans_response() -> None:
    thinking, response = ThinkTagProcessor.extract_think_content(
        "<think>first</think>\n\nAnswer\n<think>second</think>"
    )
    assert thinking == "first\nsecond"
    assert response == "Answer"


def test_preservation_depends_on_protocol_and_flag() -> None:
    raw = "<think>work</think>answer"
    assert ThinkTagProcessor.process_response_with_think_tags(raw, "ollama_chat") == raw
    assert ThinkTagProcessor.process_response_with_think_tags(raw, "openai") == "answer"
    assert ThinkTagProcessor.process_response_with_think_tags(raw, "ollama_chat", False) == "answer"


def test_formats_thinking_without_mutating_unsupported_protocols() -> None:
    generate = ThinkTagProcessor.format_response_with_thinking(
        {"response": "answer"}, "<think>work</think>answer", "ollama_generate"
    )
    chat = ThinkTagProcessor.format_response_with_thinking(
        {"message": {"content": "answer"}}, "<think>work</think>answer", "ollama_chat"
    )
    openai = ThinkTagProcessor.format_response_with_thinking(
        {"content": "answer"}, "<think>work</think>answer", "openai"
    )
    assert generate["thinking"] == "work"
    assert chat["message"]["thinking"] == "work"
    assert openai == {"content": "answer"}


def test_streaming_strips_only_tags_for_non_ollama_protocols() -> None:
    chunk = "<think>work</think>answer"
    assert preserve_think_tags_in_streaming(chunk, "ollama_generate") == chunk
    assert preserve_think_tags_in_streaming(chunk, "openai") == "workanswer"
