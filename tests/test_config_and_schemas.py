import sys
from pathlib import Path

import pytest

from ollama_server.config import ServerConfig, parse_arguments
from ollama_server.core.schemas import RequestStatus


def test_server_config_paths() -> None:
    config = ServerConfig(
        Path("/models"), Path("/llama"), Path("/logs"), 8080, "127.0.0.1", False, None
    )
    assert config.models_dir == Path("/models/blobs")
    assert config.manifests_dir == Path("/models/manifests/registry.ollama.ai/library")


def test_parse_arguments_normalizes_disabled_tensor_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--model-dir",
            "/tmp/models",
            "--port",
            "9000",
            "--default-tensor-split",
            "none",
        ],
    )
    config = parse_arguments()
    assert config.model_dir == Path("/tmp/models")
    assert config.port == 9000
    assert config.default_tensor_split is None


def make_status(**overrides: object) -> RequestStatus:
    values = {
        "request_id": "r1",
        "status": "completed",
        "progress": 1,
        "total": 1,
        "output": "ok",
        "start_time": 10.0,
        "last_update": 12.0,
        "model": "test",
        "options": {},
        "actual_tokens": 20,
        "prompt_tokens": 10,
        "completion_time": 12.0,
        "eval_duration": 1_000_000_000,
    }
    values.update(overrides)
    return RequestStatus(**values)  # type: ignore[arg-type]


def test_request_status_throughput_uses_precise_and_fallback_durations() -> None:
    status = make_status()
    assert status.duration == 2.0
    assert status.total_tokens_per_second == 15.0
    assert status.generated_tokens_per_second == 20.0

    fallback = make_status(eval_duration=None, actual_tokens=4)
    assert fallback.generated_tokens_per_second == 2.0
    assert make_status(status="error").generated_tokens_per_second == 0.0
