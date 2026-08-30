"""Discover Ollama model blobs and expose normalized model metadata."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Metadata required by request adapters and the inference executor."""

    id: str
    name: str
    path: str
    context_size: int
    size: int
    parameter_size: str
    quantization: str
    modified_at: str


class ModelManager:
    """Map Ollama manifests to model blobs under a configured model root."""

    DEFAULT_CONTEXT_SIZE = 16_384
    _CONTEXT_PATTERNS = (
        (re.compile(r"(\d+)[kK](?:-ctx|-context)?"), lambda value: int(value) * 1024),
        (re.compile(r"ctx(\d+)", re.IGNORECASE), int),
        (re.compile(r"(\d+)ctx", re.IGNORECASE), int),
    )
    _FAMILY_CONTEXTS = (
        ("llama-3.3-70b", 131_072),
        ("phi-3-mini-128k", 131_072),
        ("phi3-mini-128k", 131_072),
        ("mixtral-8x22b", 65_536),
        ("mistral", 32_768),
        ("mixtral", 32_768),
        ("qwen2", 32_768),
        ("codellama", 16_384),
        ("deepseek-coder", 16_384),
        ("llama3", 8_192),
        ("gemma", 8_192),
        ("starcoder", 8_192),
        ("llama2", 4_096),
        ("phi-3", 4_096),
        ("phi3", 4_096),
    )
    _PARAMETER_PATTERNS = (
        ("8x22B", re.compile(r"8x22b")),
        ("8x7B", re.compile(r"8x7b")),
        ("671B", re.compile(r"671b")),
        ("235B", re.compile(r"235b")),
        ("70B", re.compile(r"70b")),
        ("34B", re.compile(r"34b")),
        ("33B", re.compile(r"33b")),
        ("30B", re.compile(r"30b")),
        ("27B", re.compile(r"27b")),
        ("22B", re.compile(r"22b")),
        ("15B", re.compile(r"15b")),
        ("13B", re.compile(r"13b")),
        ("8B", re.compile(r"8b")),
        ("7B", re.compile(r"7b")),
        ("3.8B", re.compile(r"3\.8b")),
        ("3B", re.compile(r"3b")),
        ("2B", re.compile(r"2b")),
        ("1.5B", re.compile(r"1\.5b")),
        ("1.1B", re.compile(r"1\.1b")),
        ("1B", re.compile(r"1b")),
    )
    _QUANTIZATION_PATTERNS = (
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q5_0",
        "Q4_K_M",
        "Q4_0",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q2_K",
        "FP16",
        "F16",
        "F32",
    )

    def __init__(self, models_base_dir: Path):
        self.models_base_dir = Path(models_base_dir)
        self.models_dir = self.models_base_dir / "blobs"
        self.manifests_dir = (
            self.models_base_dir / "manifests" / "registry.ollama.ai" / "library"
        )
        self._model_cache: Dict[str, ModelInfo] = {}
        self._context_cache: Dict[str, int] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Return the deliberately conservative four-characters-per-token estimate."""
        return len(text) // 4 if text else 0

    def _manifest_path(self, model_id: str) -> Path:
        name, tag = model_id.split(":", 1) if ":" in model_id else (model_id, "latest")
        return self.manifests_dir / name / tag

    def read_manifest_metadata(self, model_id: str) -> Dict[str, Any]:
        """Read inline configuration metadata from an Ollama manifest."""
        manifest_path = self._manifest_path(model_id)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to read manifest %s: %s", manifest_path, exc)
            return {}

        config = manifest.get("config", {})
        if not isinstance(config, dict):
            return {}
        normalized = dict(config)
        parameters = config.get("parameters", "")
        if isinstance(parameters, str):
            for line in parameters.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    normalized[parts[0]] = int(parts[1])
        return normalized

    def detect_context_size(self, model_path: str, model_id: str) -> int:
        """Detect context size from manifest metadata, names, then family defaults."""
        cache_key = f"{model_path}:{model_id}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        metadata = self.read_manifest_metadata(model_id)
        for key in (
            "num_ctx",
            "n_ctx",
            "context_length",
            "max_position_embeddings",
            "n_ctx_train",
            "max_context_length",
        ):
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                self._context_cache[cache_key] = value
                return value

        for source in (model_id, Path(model_path).name):
            for pattern, converter in self._CONTEXT_PATTERNS:
                match = pattern.search(source)
                if match:
                    value = converter(match.group(1))
                    self._context_cache[cache_key] = value
                    return value

        lowered = model_id.lower()
        for family, value in self._FAMILY_CONTEXTS:
            if family in lowered:
                self._context_cache[cache_key] = value
                return value

        self._context_cache[cache_key] = self.DEFAULT_CONTEXT_SIZE
        return self.DEFAULT_CONTEXT_SIZE

    def build_model_mapping(self) -> Dict[str, str]:
        """Return model IDs mapped to existing blob paths."""
        if not self.manifests_dir.is_dir() or not self.models_dir.is_dir():
            return {}

        mapping: Dict[str, str] = {}
        for manifest_path in sorted(self.manifests_dir.glob("*/*")):
            if not manifest_path.is_file():
                continue
            model_id = f"{manifest_path.parent.name}:{manifest_path.name}"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping invalid manifest %s: %s", manifest_path, exc)
                continue
            for layer in manifest.get("layers", []):
                if not isinstance(layer, dict):
                    continue
                media_type = str(layer.get("mediaType", "")).lower()
                digest = str(layer.get("digest", ""))
                if "model" not in media_type and "gguf" not in media_type:
                    continue
                if not digest.startswith("sha256:"):
                    continue
                blob_path = self.models_dir / f"sha256-{digest.removeprefix('sha256:')}"
                if blob_path.is_file():
                    mapping[model_id] = str(blob_path)
                    break
        return mapping

    def find_model_path(self, model_id: str) -> Optional[str]:
        """Resolve an exact ID, an implicit latest tag, a prefix, or a GGUF path."""
        mapping = self.build_model_mapping()
        if model_id in mapping:
            return mapping[model_id]
        if ":" not in model_id:
            latest = mapping.get(f"{model_id}:latest")
            if latest:
                return latest
            matches = sorted(key for key in mapping if key.startswith(f"{model_id}:"))
            if matches:
                return mapping[matches[0]]
        direct_path = Path(model_id)
        if direct_path.is_file() and direct_path.suffix.lower() == ".gguf":
            return str(direct_path)
        return None

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Return normalized metadata for a discovered model."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        model_path_value = self.find_model_path(model_id)
        if not model_path_value:
            return None
        model_path = Path(model_path_value)
        lowered = model_id.lower()
        parameter_size = next(
            (label for label, pattern in self._PARAMETER_PATTERNS if pattern.search(lowered)),
            "Unknown",
        )
        quantization = next(
            (value for value in self._QUANTIZATION_PATTERNS if value.lower() in lowered),
            "Unknown",
        )
        info = ModelInfo(
            id=model_id,
            name=model_id.split(":", 1)[0],
            path=str(model_path),
            context_size=self.detect_context_size(str(model_path), model_id),
            size=model_path.stat().st_size,
            parameter_size=parameter_size,
            quantization=quantization,
            modified_at=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        )
        self._model_cache[model_id] = info
        return info
