"""comparisons — ar_model: consolidated research implementation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from comparisons.ar_data import ARExperimentConfig
    from endo_bbox_vla.utils import PolypPrediction

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ParameterReport:
    total: int
    trainable: int
    lora_trainable: int
    projector_trainable: int
    unexpected_trainable: tuple[str, ...]

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trainable_fraction": self.trainable_fraction}


@dataclass(frozen=True, slots=True)
class GradientCategoryReport:
    """Observed gradients for one allowed trainable parameter category."""

    trainable_tensors: int
    trainable_elements: int
    gradient_tensors: int
    finite_gradient_tensors: int
    nonzero_gradient_tensors: int
    nonzero_gradient_elements: int
    gradient_l2_norm: float | None
    gradient_max_abs: float | None
    missing_gradient: tuple[str, ...]
    nonfinite_gradient: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GradientFlowReport:
    """Fail-closed first-step gradient evidence for the intended SFT scope."""

    lora: GradientCategoryReport
    projector: GradientCategoryReport
    unexpected_trainable: tuple[str, ...]

    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.unexpected_trainable:
            preview = ", ".join(self.unexpected_trainable[:8])
            failures.append(f"unexpected trainable parameters: {preview}")
        for name, category in (
            ("LoRA", self.lora),
            ("visual merger/projector", self.projector),
        ):
            if category.trainable_tensors <= 0:
                failures.append(f"{name} has no trainable parameters")
            if category.missing_gradient:
                preview = ", ".join(category.missing_gradient[:8])
                failures.append(f"{name} parameters missing gradients: {preview}")
            if category.nonfinite_gradient:
                preview = ", ".join(category.nonfinite_gradient[:8])
                failures.append(
                    f"{name} parameters have non-finite gradients: {preview}"
                )
            if category.nonzero_gradient_tensors <= 0:
                failures.append(f"{name} received no finite nonzero gradient")
        return tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "lora": self.lora.to_dict(),
            "projector": self.projector.to_dict(),
            "unexpected_trainable": list(self.unexpected_trainable),
        }


@dataclass(frozen=True, slots=True)
class QwenARBundle:
    model: Any
    processor: Any
    parameters: ParameterReport
    resolved_revision: str | None


def _base_model(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def visual_projector(model: Any) -> Any:
    base = _base_model(model)
    visual = getattr(base, "visual", None)
    if visual is None:
        raise RuntimeError("Qwen2-VL model does not expose its visual module")
    merger = getattr(visual, "merger", None)
    if merger is None:
        raise RuntimeError("Qwen2-VL visual module does not expose merger/projector")
    return merger


def freeze_vision_except_projector(model: Any) -> None:
    """Freeze the ViT and unfreeze only its visual merger/projector."""
    base = _base_model(model)
    visual = getattr(base, "visual", None)
    if visual is None:
        raise RuntimeError("Qwen2-VL model does not expose its visual module")
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    for parameter in visual_projector(model).parameters():
        parameter.requires_grad_(True)


def _trainable_category(name: str) -> str | None:
    lowered = name.lower()
    if "lora_" in lowered:
        return "lora"
    if ".visual.merger." in lowered or lowered.startswith("visual.merger."):
        return "projector"
    return None


def inspect_trainable_parameters(model: Any) -> ParameterReport:
    total = 0
    trainable = 0
    lora_trainable = 0
    projector_trainable = 0
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        category = _trainable_category(name)
        if category == "lora":
            lora_trainable += count
        elif category == "projector":
            projector_trainable += count
        else:
            unexpected.append(name)
    return ParameterReport(
        total=total,
        trainable=trainable,
        lora_trainable=lora_trainable,
        projector_trainable=projector_trainable,
        unexpected_trainable=tuple(unexpected),
    )


def _gradient_category_report(entries: list[tuple[str, Any]]) -> GradientCategoryReport:
    missing: list[str] = []
    nonfinite: list[str] = []
    gradient_tensors = 0
    finite_gradient_tensors = 0
    nonzero_gradient_tensors = 0
    nonzero_gradient_elements = 0
    squared_l2_norm = 0.0
    max_abs = 0.0
    for name, parameter in entries:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        gradient_tensors += 1
        detached = gradient.detach()
        if not bool(detached.isfinite().all().item()):
            nonfinite.append(name)
            continue
        finite_gradient_tensors += 1
        nonzero = int(detached.count_nonzero().item())
        if nonzero:
            nonzero_gradient_tensors += 1
            nonzero_gradient_elements += nonzero
        if detached.numel():
            double_gradient = detached.double()
            squared_l2_norm += float(double_gradient.square().sum().item())
            max_abs = max(max_abs, float(double_gradient.abs().max().item()))
    has_nonfinite = bool(nonfinite)
    return GradientCategoryReport(
        trainable_tensors=len(entries),
        trainable_elements=sum((int(parameter.numel()) for _, parameter in entries)),
        gradient_tensors=gradient_tensors,
        finite_gradient_tensors=finite_gradient_tensors,
        nonzero_gradient_tensors=nonzero_gradient_tensors,
        nonzero_gradient_elements=nonzero_gradient_elements,
        gradient_l2_norm=None if has_nonfinite else math.sqrt(squared_l2_norm),
        gradient_max_abs=None if has_nonfinite else max_abs,
        missing_gradient=tuple(missing),
        nonfinite_gradient=tuple(nonfinite),
    )


def inspect_gradient_flow(model: Any) -> GradientFlowReport:
    """Inspect raw accumulated gradients before clipping or optimizer mutation."""
    categories: dict[str, list[tuple[str, Any]]] = {"lora": [], "projector": []}
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = _trainable_category(name)
        if category is None:
            unexpected.append(name)
            continue
        categories[category].append((name, parameter))
    return GradientFlowReport(
        lora=_gradient_category_report(categories["lora"]),
        projector=_gradient_category_report(categories["projector"]),
        unexpected_trainable=tuple(unexpected),
    )


def assert_gradient_flow(report: GradientFlowReport) -> None:
    """Reject a step unless every intended parameter is connected and finite."""
    failures = report.failures()
    if failures:
        raise RuntimeError(
            "first-step gradient-flow audit failed: " + "; ".join(failures)
        )


def assert_training_scope(report: ParameterReport) -> None:
    if report.total <= 0:
        raise RuntimeError("model contains no parameters")
    if report.lora_trainable <= 0:
        raise RuntimeError("no trainable LoRA parameters were found")
    if report.projector_trainable <= 0:
        raise RuntimeError("the visual merger/projector is not trainable")
    if report.unexpected_trainable:
        preview = ", ".join(report.unexpected_trainable[:8])
        raise RuntimeError(f"unexpected trainable base parameters: {preview}")


def _training_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen AR dependencies are missing. Sync bbox_vla/ar_qwen/pyproject.toml first."
        ) from exc
    return (
        torch,
        (LoraConfig, PeftModel, get_peft_model),
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
    )


def _processor_kwargs(config: ARExperimentConfig) -> dict[str, Any]:
    settings = config.model.get("processor", {})
    if not isinstance(settings, Mapping):
        raise TypeError("model.processor must be a mapping")
    return {
        "revision": config.model["processor_revision"],
        "use_fast": bool(settings.get("use_fast", False)),
        "min_pixels": int(settings["min_pixels"]),
        "max_pixels": int(settings["max_pixels"]),
    }


def load_processor(
    config: ARExperimentConfig,
    *,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> Any:
    _, _, AutoProcessor, _ = _training_imports()
    kwargs = _processor_kwargs(config)
    kwargs["local_files_only"] = local_files_only
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    processor = AutoProcessor.from_pretrained(config.model["repo_id"], **kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen processor does not expose a tokenizer")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return processor


def _attach_lora(model: Any, config: ARExperimentConfig) -> Any:
    _, peft_imports, _, _ = _training_imports()
    LoraConfig, _, get_peft_model = peft_imports
    settings = config.model["lora"]
    lora_config = LoraConfig(
        r=int(settings["rank"]),
        lora_alpha=int(settings["alpha"]),
        lora_dropout=float(settings["dropout"]),
        bias=str(settings.get("bias", "none")),
        target_modules=list(settings["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    freeze_vision_except_projector(model)
    report = inspect_trainable_parameters(model)
    assert_training_scope(report)
    return model


def _load_base_qwen(
    config: ARExperimentConfig,
    *,
    device: str,
    for_training: bool,
    local_files_only: bool,
    cache_dir: str | Path | None,
) -> Any:
    torch, _, _, Qwen2VLForConditionalGeneration = _training_imports()
    kwargs: dict[str, Any] = {
        "revision": config.model["revision"],
        "torch_dtype": torch.bfloat16,
        "attn_implementation": config.model.get("attention_implementation", "sdpa"),
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model["repo_id"], **kwargs
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is not None and resolved_revision != config.model["revision"]:
        raise RuntimeError(
            f"resolved model revision {resolved_revision} does not match configured revision"
        )
    model.to(device)
    model.config.use_cache = not for_training
    if for_training and config.model.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    return model


def load_training_bundle(
    config: ARExperimentConfig,
    *,
    device: str = "cuda",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> QwenARBundle:
    processor = load_processor(
        config, local_files_only=local_files_only, cache_dir=cache_dir
    )
    model = _attach_lora(
        _load_base_qwen(
            config,
            device=device,
            for_training=True,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ),
        config,
    )
    return QwenARBundle(
        model=model,
        processor=processor,
        parameters=inspect_trainable_parameters(model),
        resolved_revision=getattr(_base_model(model).config, "_commit_hash", None),
    )


CHECKPOINT_SCHEMA_VERSION = 2


def validate_checkpoint_metadata(
    config: ARExperimentConfig, metadata: Mapping[str, Any]
) -> None:
    """Reject an adapter that is not bound to this exact model/data contract."""
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "qwen2_vl_ar_polyp_sft_adapter",
        "base_model": config.model["repo_id"],
        "base_revision": config.model["revision"],
        "processor_revision": config.model["processor_revision"],
        "coordinate_convention": config.model["coordinate_convention"],
        "output_format": config.model["output_format"],
        "annotation_revision": config.data_binding_revision,
        "approved_payload_sha256": config.data_binding_sha256,
        "protocol_id": config.data["protocol_id"],
        "config_sha256": config.config_sha256,
    }
    if config.uses_grounded_action:
        cache_manifest = config.cache_manifest_path
        if cache_manifest is None or not cache_manifest.is_file():
            raise FileNotFoundError(
                f"grounded action checkpoint validation requires the cache manifest: {cache_manifest}"
            )
        expected.update(
            {
                "output_contract": config.output_contract,
                "camera_order": list(config.camera_order),
                "camera_input_sizes": {
                    key: list(value) for key, value in config.camera_input_sizes.items()
                },
                "target_camera": config.target_camera,
                "action": dict(config.model["action"]),
                "cache_manifest": str(config.data["cache_manifest"]),
                "cache_manifest_sha256": hashlib.sha256(
                    cache_manifest.read_bytes()
                ).hexdigest(),
            }
        )
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"AR checkpoint metadata does not match config: {mismatches}")


def _load_projector(model: Any, path: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to load the visual projector"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"missing trainable visual projector checkpoint: {path}"
        )
    incompatible = visual_projector(model).load_state_dict(
        load_file(str(path)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"visual projector state mismatch: {incompatible}")


PROJECTOR_FILENAME = "vision_projector.safetensors"

METADATA_FILENAME = "ar_checkpoint_metadata.json"


def load_training_bundle_from_adapter(
    config: ARExperimentConfig,
    adapter_dir: str | Path,
    *,
    device: str = "cuda",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> QwenARBundle:
    """Resume the exact SFT policy with LoRA and projector trainable.

    Loading a fresh LoRA adapter and silently dropping the trained visual
    merger would change the policy.  This entry point therefore validates the
    checkpoint identity, restores both trainable components, and re-applies the
    same frozen-vision scope used by a fresh run.
    """
    _, peft_imports, _, _ = _training_imports()
    _, PeftModel, _ = peft_imports
    adapter_path = Path(adapter_dir).resolve()
    metadata = json.loads(
        (adapter_path / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    validate_checkpoint_metadata(config, metadata)
    processor = load_processor(
        config, local_files_only=local_files_only, cache_dir=cache_dir
    )
    base = _load_base_qwen(
        config,
        device=device,
        for_training=True,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    _load_projector(model, adapter_path / PROJECTOR_FILENAME)
    freeze_vision_except_projector(model)
    report = inspect_trainable_parameters(model)
    assert_training_scope(report)
    return QwenARBundle(
        model=model,
        processor=processor,
        parameters=report,
        resolved_revision=getattr(_base_model(model).config, "_commit_hash", None),
    )


def _save_projector(model: Any, path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to save the visual projector"
        ) from exc
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in visual_projector(model).state_dict().items()
    }
    if not state:
        raise RuntimeError("visual projector state is empty")
    save_file(state, str(path))


def save_ar_adapter(
    bundle: QwenARBundle,
    output_dir: str | Path,
    *,
    config: ARExperimentConfig,
    training_state: Mapping[str, Any],
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(destination, safe_serialization=True)
    bundle.processor.save_pretrained(destination)
    _save_projector(bundle.model, destination / PROJECTOR_FILENAME)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "qwen2_vl_ar_polyp_sft_adapter",
        "base_model": config.model["repo_id"],
        "base_revision": config.model["revision"],
        "processor_revision": config.model["processor_revision"],
        "coordinate_convention": config.model["coordinate_convention"],
        "output_format": config.model["output_format"],
        "annotation_revision": config.data_binding_revision,
        "approved_payload_sha256": config.data_binding_sha256,
        "protocol_id": config.data["protocol_id"],
        "config_sha256": config.config_sha256,
        "parameters": bundle.parameters.to_dict(),
        "training_state": dict(training_state),
    }
    if config.uses_grounded_action:
        cache_manifest = config.cache_manifest_path
        if cache_manifest is None or not cache_manifest.is_file():
            raise FileNotFoundError(
                f"grounded action checkpoint requires the verified cache manifest: {cache_manifest}"
            )
        metadata.update(
            {
                "output_contract": config.output_contract,
                "camera_order": list(config.camera_order),
                "camera_input_sizes": {
                    key: list(value) for key, value in config.camera_input_sizes.items()
                },
                "target_camera": config.target_camera,
                "action": dict(config.model["action"]),
                "cache_manifest": str(config.data["cache_manifest"]),
                "cache_manifest_sha256": hashlib.sha256(
                    cache_manifest.read_bytes()
                ).hexdigest(),
            }
        )
    (destination / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def load_inference_bundle(
    config: ARExperimentConfig,
    adapter_dir: str | Path,
    *,
    device: str = "cuda",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> QwenARBundle:
    _, peft_imports, _, _ = _training_imports()
    _, PeftModel, _ = peft_imports
    adapter_path = Path(adapter_dir).resolve()
    metadata_path = adapter_path / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_checkpoint_metadata(config, metadata)
    processor = load_processor(
        config, local_files_only=local_files_only, cache_dir=cache_dir
    )
    base = _load_base_qwen(
        config,
        device=device,
        for_training=False,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    _load_projector(model, adapter_path / PROJECTOR_FILENAME)
    model.eval()
    return QwenARBundle(
        model=model,
        processor=processor,
        parameters=inspect_trainable_parameters(model),
        resolved_revision=getattr(_base_model(model).config, "_commit_hash", None),
    )


@dataclass(frozen=True, slots=True)
class PresenceTokenContract:
    y_token_id: int
    n_token_id: int

    def to_dict(self) -> dict[str, int]:
        return {"y_token_id": self.y_token_id, "n_token_id": self.n_token_id}


def validate_presence_tokens(processor: Any) -> PresenceTokenContract:
    """Require Y and N to be distinct single tokens under the pinned tokenizer."""
    from comparisons.ar_data import NEGATIVE_TARGET, POSITIVE_TARGET_PREFIX

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen processor does not expose a tokenizer")
    y_ids = tokenizer.encode(POSITIVE_TARGET_PREFIX, add_special_tokens=False)
    n_ids = tokenizer.encode(NEGATIVE_TARGET, add_special_tokens=False)
    if len(y_ids) != 1 or len(n_ids) != 1:
        raise RuntimeError(
            f"presence markers must each be one token, got Y={y_ids!r}, N={n_ids!r}"
        )
    if int(y_ids[0]) == int(n_ids[0]):
        raise RuntimeError("Y and N resolved to the same tokenizer id")
    if tokenizer.decode(y_ids, skip_special_tokens=False) != POSITIVE_TARGET_PREFIX:
        raise RuntimeError("Y presence token does not decode exactly to 'Y'")
    if tokenizer.decode(n_ids, skip_special_tokens=False) != NEGATIVE_TARGET:
        raise RuntimeError("N presence token does not decode exactly to 'N'")
    return PresenceTokenContract(int(y_ids[0]), int(n_ids[0]))


@dataclass(frozen=True, slots=True)
class ARGenerationOutput:
    """One decoded generation and its first-token presence probability."""

    text: str
    present_score: float
    generated_tokens: int | None = None
    latency_ms: float | None = None


def _as_rgb_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB") if frame.mode != "RGB" else frame
        image.load()
        return image
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for array frame inputs") from exc
    value = (
        frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    )
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("QwenARGenerator accepts exactly one frame")
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"expected HWC/CHW image, got shape {value.shape}")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"cannot infer image channels from shape {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all():
            raise ValueError("frame contains NaN or infinity")
        if value.max(initial=0.0) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).round().astype(np.uint8)
    else:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return Image.fromarray(value).convert("RGB")


def _ordered_grounded_images(observation: Any) -> tuple[Image.Image, ...]:
    """Validate the exact four-camera runtime boundary without reordering.

    Requiring the mapping's insertion order to equal the checkpoint contract
    makes missing, extra, or accidentally permuted camera inputs fail closed.
    Python mappings cannot contain duplicate keys; duplicate construction is
    therefore rejected by the mapping boundary itself.
    """
    from comparisons.ar_data import QWEN_CAMERA_ORDER

    if not isinstance(observation, Mapping):
        raise TypeError(
            "grounded AR input must be an ordered camera observation mapping"
        )
    received_order = tuple(observation.keys())
    if received_order != QWEN_CAMERA_ORDER:
        raise ValueError(
            f"grounded AR camera keys/order mismatch: expected {QWEN_CAMERA_ORDER!r}, got {received_order!r}"
        )
    return tuple((_as_rgb_pil(observation[key]) for key in QWEN_CAMERA_ORDER))


def _scalar(value: Any, *, name: str) -> float:
    """Read a one-element Python/NumPy/torch-like value without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16" and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one scalar, got shape {array.shape}."
        )
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def present_probability_from_logits(y_logit: Any, n_logit: Any) -> float:
    """Compute ``P(Y) / (P(Y) + P(N))`` with a stable two-class softmax."""
    y_value = _scalar(y_logit, name="y_logit")
    n_value = _scalar(n_logit, name="n_logit")
    difference = n_value - y_value
    if difference >= 0:
        exponent = math.exp(-difference)
        return exponent / (1.0 + exponent)
    return 1.0 / (1.0 + math.exp(difference))


class QwenARGenerator:
    """Generate one presence score and one conditional positive box.

    The unconstrained greedy pass supplies the first-position Y/N logits. If it
    emits a valid positive box, that text is retained. Otherwise a second greedy
    pass is conditioned on the literal Y token. This ensures every frame has a
    bbox candidate whose independent confidence can be calibrated like DETR.
    The two-pass case is deliberately included in latency timing.
    """

    def __init__(
        self, bundle: QwenARBundle, config: ARExperimentConfig, *, device: str = "cuda"
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.device = device
        self.grounded_action = bool(getattr(config, "uses_grounded_action", False))
        self.tokens = validate_presence_tokens(bundle.processor)
        self.bundle.model.eval()

    def _inputs(
        self, visual_input: Image.Image | tuple[Image.Image, ...]
    ) -> dict[str, Any]:
        from comparisons.ar_data import (
            BBOX_CAMERA_KEY,
            QWEN_CAMERA_ORDER,
            build_multicamera_qwen_messages,
            build_qwen_messages,
        )

        if self.grounded_action:
            if not isinstance(visual_input, tuple) or len(visual_input) != len(
                QWEN_CAMERA_ORDER
            ):
                raise TypeError(
                    "grounded AR processor input must contain four ordered images"
                )
            messages = build_multicamera_qwen_messages(
                visual_input,
                image_size=visual_input[QWEN_CAMERA_ORDER.index(BBOX_CAMERA_KEY)].size,
                camera_input_sizes=self.config.camera_input_sizes,
            )
            processor_images = list(visual_input)
        else:
            if not isinstance(visual_input, Image.Image):
                raise TypeError("legacy AR processor input must be one image")
            messages = build_qwen_messages(visual_input, image_size=visual_input.size)
            processor_images = [visual_input]
        text = self.bundle.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        values = self.bundle.processor(
            text=[text], images=processor_images, padding=True, return_tensors="pt"
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in values.items()
        }

    def _batch_inputs(
        self, visual_inputs: Sequence[Image.Image | tuple[Image.Image, ...]]
    ) -> dict[str, Any]:
        from comparisons.ar_data import (
            BBOX_CAMERA_KEY,
            QWEN_CAMERA_ORDER,
            build_multicamera_qwen_messages,
            build_qwen_messages,
        )

        if not visual_inputs:
            raise ValueError("batched AR generation requires at least one observation")
        texts: list[str] = []
        processor_images: list[Image.Image] = []
        for visual_input in visual_inputs:
            if self.grounded_action:
                if not isinstance(visual_input, tuple) or len(visual_input) != len(
                    QWEN_CAMERA_ORDER
                ):
                    raise TypeError(
                        "grounded AR processor input must contain four ordered images"
                    )
                messages = build_multicamera_qwen_messages(
                    visual_input,
                    image_size=visual_input[
                        QWEN_CAMERA_ORDER.index(BBOX_CAMERA_KEY)
                    ].size,
                    camera_input_sizes=self.config.camera_input_sizes,
                )
                processor_images.extend(visual_input)
            else:
                if not isinstance(visual_input, Image.Image):
                    raise TypeError("legacy AR processor input must be one image")
                messages = build_qwen_messages(
                    visual_input, image_size=visual_input.size
                )
                processor_images.append(visual_input)
            texts.append(
                self.bundle.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        values = self.bundle.processor(
            text=texts, images=processor_images, padding=True, return_tensors="pt"
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in values.items()
        }

    def _generation_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "use_cache": True,
            "max_new_tokens": int(self.config.generation["max_new_tokens"]),
            "return_dict_in_generate": True,
            "output_scores": True,
        }

    def _decode_new(self, sequences: Any, *, prompt_length: int) -> str:
        token_ids = sequences[:, prompt_length:]
        return self.bundle.processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _decode_new_batch(self, sequences: Any, *, prompt_length: int) -> list[str]:
        token_ids = sequences[:, prompt_length:]
        return [
            text.strip()
            for text in self.bundle.processor.batch_decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]

    def _force_positive(self, inputs: dict[str, Any]) -> str:
        import torch

        prompt_length = int(inputs["input_ids"].shape[1])
        y_token = torch.tensor(
            [[self.tokens.y_token_id]],
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        forced = dict(inputs)
        forced["input_ids"] = torch.cat((inputs["input_ids"], y_token), dim=1)
        forced["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones_like(y_token)), dim=1
        )
        result = self.bundle.model.generate(
            **forced,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            max_new_tokens=max(1, int(self.config.generation["max_new_tokens"]) - 1),
        )
        return self._decode_new(result, prompt_length=prompt_length)

    def predict_ar(self, frame: Any) -> ARGenerationOutput:
        import torch

        from comparisons.ar_data import (
            QWEN_CAMERA_ORDER,
            parse_ar_output,
            parse_grounded_action_output,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        expected_size = (
            int(self.config.data["original_width"]),
            int(self.config.data["original_height"]),
        )
        if self.grounded_action:
            images = _ordered_grounded_images(frame)
            expected_camera_sizes = self.config.camera_input_sizes
            wrong_sizes = {
                key: image.size
                for key, image in zip(QWEN_CAMERA_ORDER, images, strict=True)
                if image.size != expected_camera_sizes[key]
            }
            if wrong_sizes:
                raise ValueError(
                    f"grounded AR image resolution mismatch: expected {expected_camera_sizes}, got {wrong_sizes}"
                )
            visual_input: Image.Image | tuple[Image.Image, ...] = images
            parse_output = parse_grounded_action_output
        else:
            image = _as_rgb_pil(frame)
            if image.size != expected_size:
                raise ValueError(
                    f"AR input image must be {expected_size}, got {image.size}"
                )
            visual_input = image
            parse_output = parse_ar_output
        inputs = self._inputs(visual_input)
        prompt_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            result = self.bundle.model.generate(**inputs, **self._generation_kwargs())
            if not result.scores:
                raise RuntimeError("Qwen generation returned no first-token logits")
            first_logits = result.scores[0][0]
            score = present_probability_from_logits(
                first_logits[self.tokens.y_token_id],
                first_logits[self.tokens.n_token_id],
            )
            text = self._decode_new(result.sequences, prompt_length=prompt_length)
            parsed = parse_output(text, image_size=expected_size)
            if self.config.generation["force_positive_bbox_candidate"] and (
                not (parsed.format_valid and parsed.present is True)
            ):
                text = self._force_positive(inputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated_tokens = len(
            self.bundle.processor.tokenizer.encode(text, add_special_tokens=False)
        )
        return ARGenerationOutput(
            text=text,
            present_score=score,
            generated_tokens=generated_tokens,
            latency_ms=latency_ms,
        )

    def predict_ar_batch(self, frames: Sequence[Any]) -> tuple[ARGenerationOutput, ...]:
        """Greedily decode a fixed-resolution observation batch.

        Four-camera prompts have an identical token length, so prompt padding
        cannot leak into generation. Invalid/non-positive outputs retain the
        conditional-candidate contract by falling back to the exact
        single-sample forced-Y path only for those observations.
        """
        import torch

        from comparisons.ar_data import (
            QWEN_CAMERA_ORDER,
            parse_ar_output,
            parse_grounded_action_output,
        )

        if not frames:
            raise ValueError("batched AR generation requires at least one frame")
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        expected_size = (
            int(self.config.data["original_width"]),
            int(self.config.data["original_height"]),
        )
        visual_inputs: list[Image.Image | tuple[Image.Image, ...]] = []
        if self.grounded_action:
            expected_camera_sizes = self.config.camera_input_sizes
            parse_output = parse_grounded_action_output
            for frame in frames:
                images = _ordered_grounded_images(frame)
                wrong_sizes = {
                    key: image.size
                    for key, image in zip(QWEN_CAMERA_ORDER, images, strict=True)
                    if image.size != expected_camera_sizes[key]
                }
                if wrong_sizes:
                    raise ValueError(
                        f"grounded AR image resolution mismatch: expected {expected_camera_sizes}, got {wrong_sizes}"
                    )
                visual_inputs.append(images)
        else:
            parse_output = parse_ar_output
            for frame in frames:
                image = _as_rgb_pil(frame)
                if image.size != expected_size:
                    raise ValueError(
                        f"AR input image must be {expected_size}, got {image.size}"
                    )
                visual_inputs.append(image)
        inputs = self._batch_inputs(visual_inputs)
        prompt_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            result = self.bundle.model.generate(**inputs, **self._generation_kwargs())
            if not result.scores:
                raise RuntimeError("Qwen generation returned no first-token logits")
            first_logits = result.scores[0]
            if int(first_logits.shape[0]) != len(frames):
                raise RuntimeError("Qwen generation returned the wrong batch dimension")
            scores = [
                present_probability_from_logits(
                    row[self.tokens.y_token_id], row[self.tokens.n_token_id]
                )
                for row in first_logits
            ]
            texts = self._decode_new_batch(
                result.sequences, prompt_length=prompt_length
            )
            for index, (text, visual_input) in enumerate(
                zip(texts, visual_inputs, strict=True)
            ):
                parsed = parse_output(text, image_size=expected_size)
                if self.config.generation["force_positive_bbox_candidate"] and (
                    not (parsed.format_valid and parsed.present is True)
                ):
                    texts[index] = self._force_positive(self._inputs(visual_input))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return tuple(
            (
                ARGenerationOutput(
                    text=text,
                    present_score=score,
                    generated_tokens=len(
                        self.bundle.processor.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                    ),
                    latency_ms=latency_ms,
                )
                for text, score in zip(texts, scores, strict=True)
            )
        )


@runtime_checkable
class ARGeneratorProtocol(Protocol):
    """Small boundary implemented by the GPU-side Qwen wrapper."""

    def predict_ar(self, frame: Any) -> Any: ...


type Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ARInferenceResult:
    """Runtime prediction with generation diagnostics kept out of the public API."""

    prediction: PolypPrediction
    generation: ARGenerationOutput
    format_valid: bool
    parsed_present: bool | None
    candidate_bbox_xyxy: Box | None
    parse_error: str | None
    canonical_text: str | None


def _get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _first_present(record: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = _get(record, name)
        if value is not None:
            return value
    return None


def coerce_generation_output(output: Any) -> ARGenerationOutput:
    """Normalize an injected wrapper's output to :class:`ARGenerationOutput`.

    Accepted forms are an ``ARGenerationOutput``, ``(text, score)``, or a
    mapping/object with ``bbox_text``/``text`` (aliases: ``generated_text``,
    ``output_text``) and either ``present_score``/``score`` or explicit
    ``y_logit`` and ``n_logit``.  ``bbox_text`` takes precedence so an
    evaluation wrapper can retain a greedy ``N`` decision separately while
    providing the conditionally generated ``Y [...]`` candidate needed for
    threshold calibration.  A mapping named ``first_token_logits`` may also
    contain the explicit keys ``"Y"`` and ``"N"``.  Logit array ordering is
    intentionally never guessed.
    """
    if isinstance(output, ARGenerationOutput):
        generation = output
    elif (
        isinstance(output, Sequence)
        and (not isinstance(output, str | bytes | bytearray | np.ndarray))
        and (len(output) == 2)
    ):
        generation = ARGenerationOutput(
            text=str(output[0]), present_score=_scalar(output[1], name="score")
        )
    else:
        text = _first_present(
            output, ("bbox_text", "text", "generated_text", "output_text")
        )
        if not isinstance(text, str):
            raise TypeError(
                "AR generator output must contain generated text as a string."
            )
        score_value = _first_present(output, ("present_score", "score"))
        if score_value is not None:
            score = _scalar(score_value, name="present_score")
        else:
            y_logit = _get(output, "y_logit")
            n_logit = _get(output, "n_logit")
            token_logits = _get(output, "first_token_logits")
            if (y_logit is None or n_logit is None) and isinstance(
                token_logits, Mapping
            ):
                y_logit = token_logits.get("Y")
                n_logit = token_logits.get("N")
            if y_logit is None or n_logit is None:
                raise ValueError(
                    "AR generator output needs present_score or explicit y_logit and n_logit."
                )
            score = present_probability_from_logits(y_logit, n_logit)
        generated_tokens = _get(output, "generated_tokens")
        latency_ms = _get(output, "latency_ms")
        generation = ARGenerationOutput(
            text=text,
            present_score=score,
            generated_tokens=None
            if generated_tokens is None
            else int(generated_tokens),
            latency_ms=None if latency_ms is None else float(latency_ms),
        )
    score = float(generation.present_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"present_score must be finite and in [0, 1], got {score!r}.")
    if not isinstance(generation.text, str):
        raise TypeError("AR generation text must be a string.")
    if generation.generated_tokens is not None and generation.generated_tokens < 0:
        raise ValueError("generated_tokens cannot be negative.")
    if generation.latency_ms is not None and (
        not math.isfinite(generation.latency_ms) or generation.latency_ms < 0
    ):
        raise ValueError("latency_ms must be finite and non-negative.")
    return generation


def _frame_size(frame: Any) -> tuple[int, int]:
    """Infer ``(width, height)`` from PIL, NumPy, or torch-like image data."""
    size = getattr(frame, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        width, height = (int(size[0]), int(size[1]))
        if width > 0 and height > 0:
            return (width, height)
    shape = tuple((int(value) for value in getattr(frame, "shape", ())))
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError("ARPolypPredictor accepts exactly one frame at a time.")
        shape = shape[1:]
    if len(shape) != 3:
        raise ValueError(f"Expected HWC or CHW image data, got shape {shape!r}.")
    if shape[0] in (1, 3, 4):
        width, height = (shape[2], shape[1])
    elif shape[2] in (1, 3, 4):
        width, height = (shape[1], shape[0])
    else:
        raise ValueError(f"Cannot infer image channel dimension from shape {shape!r}.")
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    return (width, height)


type ParseCallable = Callable[..., Any]


def _grounded_parse(text: str, *, image_size: tuple[int, int]) -> Any:
    from comparisons.ar_data import (
        parse_grounded_action_output as parse_grounded_action_output,
    )

    return parse_grounded_action_output(text, image_size=image_size)


def _default_parse(text: str, *, image_size: tuple[int, int]) -> Any:
    from comparisons.ar_data import parse_ar_output as parse_ar_output

    return parse_ar_output(text, image_size=image_size)


def _grounded_target_frame(observation: Any) -> Any:
    """Return cam_low only after validating the exact ordered observation API."""
    from comparisons.ar_data import PINNED_CAMERA_ORDER as PINNED_CAMERA_ORDER
    from comparisons.ar_data import TARGET_CAMERA as TARGET_CAMERA

    if not isinstance(observation, Mapping):
        raise TypeError("grounded AR inference requires an ordered camera mapping")
    received_order = tuple(observation.keys())
    if received_order != PINNED_CAMERA_ORDER:
        raise ValueError(
            f"grounded AR camera keys/order mismatch: expected {PINNED_CAMERA_ORDER!r}, got {received_order!r}"
        )
    return observation[TARGET_CAMERA]


def _box_from_parse_result(parsed: Any, *, image_size: tuple[int, int]) -> Box | None:
    value = _get(parsed, "bbox_xyxy")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    x1, y1, x2, y2 = (float(item) for item in array)
    width, height = image_size
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return (x1, y1, x2, y2)


class ARPolypPredictor:
    """Adapt an injected Qwen AR wrapper to the bbox runtime contract.

    The injected object should implement ``predict_ar(frame)``.  For convenient
    testing, ``predict(frame)`` and a directly callable object are also accepted.
    The wrapper owns all processor/model/device logic and should return generated
    text plus the first-token presence score or explicit ``Y``/``N`` logits.
    """

    def __init__(
        self,
        generator: ARGeneratorProtocol | Callable[[Any], Any] | Any,
        *,
        confidence_threshold: float,
        parser: ParseCallable | None = None,
        input_transform: Callable[[Any], Any] | None = None,
        grounded_action: bool = False,
    ) -> None:
        if (
            not math.isfinite(confidence_threshold)
            or not 0.0 <= confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be finite and in [0, 1].")
        self.generator = generator
        self.confidence_threshold = float(confidence_threshold)
        if not isinstance(grounded_action, bool):
            raise TypeError("grounded_action must be boolean")
        self.grounded_action = grounded_action
        self.parser = parser or (_grounded_parse if grounded_action else _default_parse)
        self.input_transform = input_transform

    def _generate(self, frame: Any) -> ARGenerationOutput:
        inputs = (
            self.input_transform(frame) if self.input_transform is not None else frame
        )
        if hasattr(self.generator, "predict_ar"):
            output = self.generator.predict_ar(inputs)
        elif hasattr(self.generator, "predict"):
            output = self.generator.predict(inputs)
        elif callable(self.generator):
            output = self.generator(inputs)
        else:
            raise TypeError(
                "AR generator must implement predict_ar/predict or be callable."
            )
        return coerce_generation_output(output)

    def predict_with_diagnostics(
        self, observation: Any, frame_id: int, timestamp: float
    ) -> ARInferenceResult:
        """Run one frame and retain parser diagnostics for offline auditing."""
        from endo_bbox_vla.utils import PolypPrediction

        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite.")
        cam_low_frame = (
            _grounded_target_frame(observation) if self.grounded_action else observation
        )
        image_size = _frame_size(cam_low_frame)
        generation = self._generate(observation)
        parsed = self.parser(generation.text, image_size=image_size)
        format_valid = bool(_get(parsed, "format_valid", False))
        parsed_present_value = _get(parsed, "present")
        parsed_present = (
            parsed_present_value if isinstance(parsed_present_value, bool) else None
        )
        candidate_box = (
            _box_from_parse_result(parsed, image_size=image_size)
            if format_valid and parsed_present is True
            else None
        )
        if format_valid and parsed_present is True and (candidate_box is None):
            format_valid = False
        present = (
            format_valid
            and parsed_present is True
            and (candidate_box is not None)
            and (generation.present_score >= self.confidence_threshold)
        )
        prediction = PolypPrediction(
            present=present,
            score=float(generation.present_score),
            bbox_xyxy=candidate_box if present else None,
            frame_id=int(frame_id),
            timestamp=float(timestamp),
        )
        parse_error = _get(parsed, "error")
        canonical_text = _get(parsed, "canonical_text")
        return ARInferenceResult(
            prediction=prediction,
            generation=generation,
            format_valid=format_valid,
            parsed_present=parsed_present,
            candidate_bbox_xyxy=candidate_box,
            parse_error=None if parse_error is None else str(parse_error),
            canonical_text=None if canonical_text is None else str(canonical_text),
        )

    def predict_record(
        self, observation: Any, frame_id: int, timestamp: float
    ) -> PolypPrediction:
        return self.predict_with_diagnostics(
            observation, frame_id, timestamp
        ).prediction

    def predict_polyp(
        self, observation: Any, frame_id: int, timestamp: float
    ) -> dict[str, bool | float | int | list[float] | None]:
        """Return the exact dictionary consumed by ``LatestFrameWorker``."""
        return self.predict_record(observation, frame_id, timestamp).to_dict()


def predict_polyp(
    predictor: ARPolypPredictor, cam_low_frame: Any, frame_id: int, timestamp: float
) -> dict[str, bool | float | int | list[float] | None]:
    """Functional form matching :func:`bbox_vla.detr.infer.predict_polyp`."""
    return predictor.predict_polyp(cam_low_frame, frame_id, timestamp)
