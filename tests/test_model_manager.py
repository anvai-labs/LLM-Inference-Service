import json
from pathlib import Path

from ollama_server.models import ModelManager


def create_model(tmp_path: Path, model_id: str, config: dict | None = None) -> Path:
    name, tag = model_id.split(":", 1)
    digest = "a" * 64
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"sha256-{digest}"
    blob_path.write_bytes(b"GGUF")
    manifest_path = tmp_path / "manifests" / "registry.ollama.ai" / "library" / name / tag
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "config": config or {},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": f"sha256:{digest}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return blob_path


def test_discovers_model_and_normalizes_metadata(tmp_path: Path) -> None:
    blob_path = create_model(
        tmp_path,
        "qwen2:7b-q4_k_m",
        {"parameters": "num_ctx 65536\nnum_batch 512"},
    )
    manager = ModelManager(tmp_path)

    assert manager.build_model_mapping() == {"qwen2:7b-q4_k_m": str(blob_path)}
    info = manager.get_model_info("qwen2:7b-q4_k_m")

    assert info is not None
    assert info.context_size == 65_536
    assert info.parameter_size == "7B"
    assert info.quantization == "Q4_K_M"
    assert info.size == 4


def test_resolution_order_and_context_fallbacks(tmp_path: Path) -> None:
    latest = create_model(tmp_path, "llama3:latest")
    manager = ModelManager(tmp_path)

    assert manager.find_model_path("llama3") == str(latest)
    assert manager.detect_context_size(str(latest), "custom:128k") == 131_072
    assert manager.detect_context_size(str(latest), "llama3:latest") == 8_192
    assert manager.detect_context_size(str(latest), "unknown:latest") == 16_384
    assert manager.estimate_tokens("abcdefgh") == 2
    assert manager.estimate_tokens("") == 0


def test_invalid_or_missing_manifests_are_ignored(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    assert manager.build_model_mapping() == {}
    assert manager.get_model_info("missing:latest") is None

    invalid = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "bad" / "latest"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("not json", encoding="utf-8")
    (tmp_path / "blobs").mkdir()
    assert manager.build_model_mapping() == {}
