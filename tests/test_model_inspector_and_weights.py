from types import SimpleNamespace

import pytest

from ollama_server.utils.model_inspector import ModelInspector
from ollama_server.utils.weight_distribution import WeightDistributionManager


def test_model_inspector_parses_and_enhances(monkeypatch: pytest.MonkeyPatch) -> None:
    output = """Model
architecture llama
parameters 7B
context length 128K
embedding length 4096
quantization Q4_K_M
Capabilities
completion
tools
Parameters
stop \"<eos>\"
License
Apache 2.0
"""
    inspector = ModelInspector()
    detail = inspector._parse_ollama_show_output(output)
    assert detail is not None
    assert detail.context_length == 131_072
    assert detail.stop_tokens == ["<eos>"]
    assert inspector._parse_size_string("1.5K") == 1536
    assert inspector._parse_size_string("invalid") == 4096

    monkeypatch.setattr(inspector, "get_ollama_model_info", lambda _name: detail)
    base = SimpleNamespace(
        id="m:7b",
        name="m",
        path="/m",
        size=1,
        modified_at="now",
        parameter_size="7B",
        quantization="Q4",
        context_size=4096,
    )
    enhanced = inspector.get_enhanced_model_info("m:7b", base)
    assert enhanced["context_size"] == 131_072
    assert enhanced["architecture"] == "llama"


def gpu_data() -> dict:
    return {
        "gpus": [
            {
                "gpu_id": 0,
                "memory_total": 16_000,
                "memory_used": 4_000,
                "name": "fast",
                "utilization_gpu": 20,
                "temperature": 50,
                "power_draw": 100,
            },
            {
                "gpu_id": 1,
                "memory_total": 8_000,
                "memory_used": 4_000,
                "name": "small",
                "utilization_gpu": 60,
                "temperature": 70,
                "power_draw": 150,
            },
        ]
    }


def test_weight_distribution_presets_validation_and_updates() -> None:
    manager = WeightDistributionManager()
    config = manager.create_config_from_gpu_data(gpu_data(), "test:7b")
    assert manager.validate_config(config)
    assert manager.get_tensor_split_string(config) == "0.500,0.500"

    memory = manager.apply_preset("memory-based", gpu_data(), "test:7b")
    assert memory.gpu_configs[0].weight_percentage == pytest.approx(66.67, abs=0.01)
    assert manager.validate_config(memory)
    assert manager.update_weight(0, 60)
    assert manager.current_config is not None
    assert round(sum(g.weight_percentage for g in manager.current_config.gpu_configs), 5) == 100
    summary = manager.get_config_summary()
    assert summary["gpu_count"] == 2
    assert manager.to_dict()["model_name"] == "test:7b"

    config.gpu_configs[0].weight_percentage = -1
    assert not manager.validate_config(config)
    with pytest.raises(ValueError, match="Invalid GPU data"):
        manager.create_config_from_gpu_data({})
    with pytest.raises(ValueError, match="Unknown preset"):
        manager.apply_preset("missing", gpu_data())
