"""comparisons — ar_train: consolidated research implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from comparisons.ar_model import METADATA_FILENAME, PROJECTOR_FILENAME

if TYPE_CHECKING:
    from comparisons.ar_data import ARExperimentConfig
    from comparisons.ar_model import QwenARBundle
from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TrainSummary:
    optimizer_steps: int
    batches: int
    examples: int
    mean_loss: float
    first_loss: float
    last_loss: float
    min_loss: float
    elapsed_seconds: float
    output_dir: str
    gradient_flow_audit: Mapping[str, Any] | None = None
    peak_cuda_memory_gib: float | None = None

    def to_dict(self) -> dict[str, Any]:
        audit = (
            None
            if self.gradient_flow_audit is None
            else json.loads(
                json.dumps(
                    dict(self.gradient_flow_audit), sort_keys=True, allow_nan=False
                )
            )
        )
        return {
            "optimizer_steps": self.optimizer_steps,
            "batches": self.batches,
            "examples": self.examples,
            "mean_loss": self.mean_loss,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "min_loss": self.min_loss,
            "elapsed_seconds": self.elapsed_seconds,
            "output_dir": self.output_dir,
            "gradient_flow_audit": audit,
            "peak_cuda_memory_gib": self.peak_cuda_memory_gib,
        }


def _decoded_token_character_spans(
    tokenizer: Any, text: str, token_ids: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Map slow-tokenizer IDs back to exact ASCII answer character spans.

    Qwen's pinned slow tokenizer can merge the space before a negative number
    with its minus sign. Independently encoding ``text[:start]`` is therefore
    not prefix-stable at every numeric boundary. Decoding cumulative token
    prefixes gives the actual token coverage without requiring the unavailable
    fast-tokenizer ``offset_mapping`` API. Canonical answers are ASCII, so a
    token may cross a structure/numeric boundary but cannot contain a partial
    multibyte character.
    """
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError(
            "processor tokenizer must expose decode() for action weighting"
        )
    spans: list[tuple[int, int]] = []
    previous = ""
    for end in range(1, len(token_ids) + 1):
        decoded = decode(
            list(token_ids[:end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise RuntimeError("processor tokenizer decode() did not return text")
        if not text.startswith(decoded) or not decoded.startswith(previous):
            raise ValueError(
                "tokenizer cannot exactly align the canonical assistant answer"
            )
        spans.append((len(previous), len(decoded)))
        previous = decoded
    if previous != text:
        raise ValueError(
            "tokenizer decode does not reproduce the canonical assistant answer"
        )
    return tuple(spans)


class QwenSFTCollator:
    """Render Qwen chats and supervise assistant tokens only.

    Prompt and full chats are processed independently, then exact token-prefix
    equality is required. This costs extra CPU preprocessing but prevents image
    or instruction tokens from silently entering the loss.
    """

    def __init__(
        self,
        processor: Any,
        *,
        max_sequence_length: int,
        action_numeric_loss_weight: float = 1.0,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 < float(action_numeric_loss_weight) <= 1.0:
            raise ValueError("action_numeric_loss_weight must be in (0, 1]")
        self.processor = processor
        self.max_sequence_length = int(max_sequence_length)
        self.action_numeric_loss_weight = float(action_numeric_loss_weight)

    def _chat_text(self, messages: Any, *, add_generation_prompt: bool) -> str:
        value = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if not isinstance(value, str) or not value:
            raise RuntimeError("processor.apply_chat_template returned no text")
        return value

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        from comparisons.ar_data import assistant_only_labels

        if not examples:
            raise ValueError("Qwen SFT batch must not be empty")
        image_groups = [
            tuple(example["images"]) if "images" in example else (example["image"],)
            for example in examples
        ]
        if any((not group for group in image_groups)):
            raise ValueError("each Qwen SFT example must contain at least one image")
        images = [image for group in image_groups for image in group]
        full_texts = [
            self._chat_text(example["messages"], add_generation_prompt=False)
            for example in examples
        ]
        prompt_texts = [
            self._chat_text(example["prompt_messages"], add_generation_prompt=True)
            for example in examples
        ]
        full = self.processor(
            text=full_texts, images=images, padding=True, return_tensors="pt"
        )
        prompt = self.processor(
            text=prompt_texts, images=images, padding=True, return_tensors="pt"
        )
        if "input_ids" not in full or "attention_mask" not in full:
            raise RuntimeError("Qwen processor omitted input_ids or attention_mask")
        if "input_ids" not in prompt or "attention_mask" not in prompt:
            raise RuntimeError("Qwen prompt processor omitted token tensors")
        import torch

        labels = torch.full_like(full["input_ids"], -100)
        loss_weights = torch.zeros_like(full["input_ids"], dtype=torch.float32)
        has_weighted_action = False
        for row, example in enumerate(examples):
            full_length = int(full["attention_mask"][row].sum().item())
            prompt_length = int(prompt["attention_mask"][row].sum().item())
            if full_length > self.max_sequence_length:
                raise ValueError(
                    f"image {example.get('image_id')} tokenized to {full_length} tokens, exceeding max_sequence_length={self.max_sequence_length}"
                )
            full_ids = full["input_ids"][row, :full_length].tolist()
            prompt_ids = prompt["input_ids"][row, :prompt_length].tolist()
            row_labels = assistant_only_labels(full_ids, prompt_ids)
            labels[row, :full_length] = torch.tensor(
                row_labels, dtype=labels.dtype, device=labels.device
            )
            supervised_positions = [
                index for index, label in enumerate(row_labels) if label != -100
            ]
            loss_weights[row, supervised_positions] = 1.0
            spans = tuple(example.get("action_numeric_spans", ()))
            if spans:
                tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError(
                        "processor tokenizer is required for action-token loss weighting"
                    )
                answer = str(example["answer"])
                answer_ids = [
                    int(value)
                    for value in tokenizer.encode(answer, add_special_tokens=False)
                ]
                supervised_ids = [row_labels[index] for index in supervised_positions]
                if supervised_ids[: len(answer_ids)] != answer_ids:
                    raise ValueError(
                        "canonical assistant answer is not an exact prefix of supervised tokens"
                    )
                token_character_spans = _decoded_token_character_spans(
                    tokenizer, answer, answer_ids
                )
                for start, end in spans:
                    numeric_start = int(start)
                    numeric_end = int(end)
                    if not 0 <= numeric_start < numeric_end <= len(answer):
                        raise ValueError(
                            "action numeric span is outside the assistant answer"
                        )
                    numeric_token_offsets = [
                        token_offset
                        for token_offset, (token_start, token_end) in enumerate(
                            token_character_spans
                        )
                        if token_start < numeric_end and token_end > numeric_start
                    ]
                    if not numeric_token_offsets:
                        raise ValueError(
                            "action numeric span has no overlapping tokenizer token"
                        )
                    for answer_offset in numeric_token_offsets:
                        loss_weights[row, supervised_positions[answer_offset]] = (
                            self.action_numeric_loss_weight
                        )
                has_weighted_action = True
        full["labels"] = labels
        if has_weighted_action:
            full["loss_weights"] = loss_weights
        return dict(full)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing") from exc
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    required = {"python", "numpy", "torch", "cuda"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"trainer RNG state is missing {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def build_train_dataset(config: ARExperimentConfig, *, verify: bool = True) -> Any:
    from comparisons.ar_data import QwenPolypARDataset, QwenScenarioStaticARDataset

    if config.data_format == "scenario_static_v1":
        return QwenScenarioStaticARDataset(
            config.scenario_contract,
            "train",
            cache_root=config.cache_root if config.uses_lossless_cache else None,
            cache_manifest=str(
                config.data.get("cache_manifest", "cache_manifest.json")
            ),
            grounded_action=config.uses_grounded_action,
            camera_input_sizes=config.camera_input_sizes
            if config.uses_grounded_action
            else None,
            image_size=(
                int(config.data["original_width"]),
                int(config.data["original_height"]),
            ),
        )
    return QwenPolypARDataset(
        config.coco_path("train"),
        config.approved_root / str(config.data["manifest"]),
        config.image_root,
        image_size=(
            int(config.data["original_width"]),
            int(config.data["original_height"]),
        ),
        verify=verify,
        enforce_manifest_split=False,
    )


def _diagnostic_subset_indices(dataset: Any, limit: int, *, seed: int) -> list[int]:
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        status = str(getattr(record.status, "value", record.status))
        group = -1 if record.group_id is None else int(record.group_id)
        buckets[status, group].append(index)
    rng = random.Random(seed)
    for indices in buckets.values():
        rng.shuffle(indices)
    selected: list[int] = []
    keys = sorted(buckets)
    offsets = {key: 0 for key in keys}
    while len(selected) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            selected.append(buckets[key][offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    if len(selected) != limit:
        raise RuntimeError(
            f"could select only {len(selected)} of {limit} diagnostic samples"
        )
    return selected


def _subset_dataset(
    dataset: Any, limit: int | None, *, seed: int = 42
) -> tuple[Any, Any]:
    import torch

    weight_builder = getattr(dataset, "balanced_sample_weights", None)
    weights = (
        weight_builder(balance_groups=True)
        if callable(weight_builder)
        else torch.ones(len(dataset), dtype=torch.double)
    )
    if limit is None:
        return (dataset, weights)
    if limit <= 0:
        raise ValueError("limit_train_samples must be positive")
    count = min(int(limit), len(dataset))
    if not hasattr(dataset, "records"):
        indices = list(range(count))
        return (torch.utils.data.Subset(dataset, indices), weights[indices])
    indices = _diagnostic_subset_indices(dataset, count, seed=seed)
    keys = [
        (
            str(
                getattr(
                    dataset.records[index].status,
                    "value",
                    dataset.records[index].status,
                )
            ),
            -1
            if dataset.records[index].group_id is None
            else int(dataset.records[index].group_id),
        )
        for index in indices
    ]
    counts = Counter(keys)
    subset_weights = torch.tensor(
        [1.0 / counts[key] for key in keys], dtype=torch.double
    )
    return (torch.utils.data.Subset(dataset, indices), subset_weights)


def _make_dataloader_generator(seed: int) -> Any:
    import torch

    return torch.Generator().manual_seed(int(seed))


def build_train_loader(
    config: ARExperimentConfig,
    dataset: Any,
    processor: Any,
    *,
    limit_train_samples: int | None = None,
) -> Any:
    import torch

    from endo_bbox_vla.data import ScenarioEpisodeBalancedSampler

    seed = int(config.training["seed"])
    worker_generator = (
        _make_dataloader_generator(seed) if config.uses_grounded_action else None
    )
    if config.data_format == "scenario_static_v1" and limit_train_samples is None:
        sampler = ScenarioEpisodeBalancedSampler(
            dataset.records,
            seed=seed,
            samples_per_epoch=config.scenario_contract.samples_per_epoch,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(config.training["batch_size"]),
            sampler=sampler,
            num_workers=int(config.training["num_workers"]),
            collate_fn=QwenSFTCollator(
                processor,
                max_sequence_length=int(config.training["max_sequence_length"]),
                action_numeric_loss_weight=float(
                    config.training.get("action_numeric_loss_weight", 1.0)
                ),
            ),
            pin_memory=True,
            persistent_workers=int(config.training["num_workers"]) > 0,
            generator=worker_generator,
        )
    selected, weights = _subset_dataset(dataset, limit_train_samples, seed=seed)
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(selected),
        replacement=True,
        generator=sampler_generator,
    )
    return torch.utils.data.DataLoader(
        selected,
        batch_size=int(config.training["batch_size"]),
        sampler=sampler,
        num_workers=int(config.training["num_workers"]),
        collate_fn=QwenSFTCollator(
            processor,
            max_sequence_length=int(config.training["max_sequence_length"]),
            action_numeric_loss_weight=float(
                config.training.get("action_numeric_loss_weight", 1.0)
            ),
        ),
        pin_memory=True,
        persistent_workers=int(config.training["num_workers"]) > 0,
        generator=worker_generator,
    )


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> Any:
    """Token-weighted next-token loss with the standard causal shift."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.shape != labels.shape:
        raise ValueError("weighted causal LM tensors have incompatible shapes")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch/sequence dimensions")
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = loss_weights[:, 1:].to(dtype=torch.float32).contiguous()
    valid = shifted_labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("weighted causal LM loss has no supervised target tokens")
    if bool((shifted_weights[valid] <= 0).any()) or not bool(
        torch.isfinite(shifted_weights[valid]).all()
    ):
        raise ValueError("supervised token weights must be finite and positive")
    per_token = functional.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    effective_weights = shifted_weights * valid.to(dtype=shifted_weights.dtype)
    denominator = effective_weights.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise ValueError("weighted causal LM loss has zero effective weight")
    return (per_token * effective_weights).sum() / denominator


SFT_GATE_BINDING_KEYS = frozenset(
    {
        "path",
        "sha256",
        "config_sha256",
        "cache_identity_sha256",
        "camera_sha256",
        "seed",
        "resolution_profile",
        "identity_sha256",
        "selection_sha256",
        "training_order_sha256",
        "checks_sha256",
        "run_sha256",
        "checkpoint_tree_sha256",
        "train_summary_sha256",
        "primary_gate_failure",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA256_PATTERN = re.compile("[0-9a-f]{64}\\Z")


def _validate_bound_artifact(
    binding: Mapping[str, Any], *, name: str, sha_fields: Sequence[str]
) -> None:
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError(f"{name} binding path must be absolute")
    source = Path(path_value)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"{name} binding artifact is not a regular file: {source}"
        )
    for field in sha_fields:
        digest = binding.get(field)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{name} binding {field} is not a SHA-256")
    if _sha256_file(source) != binding["sha256"]:
        raise ValueError(f"{name} binding artifact SHA-256 changed")


SFT_PRIMARY_FAILURE_BINDING_KEYS = frozenset(
    {
        "path",
        "sha256",
        "config_sha256",
        "resolution_profile",
        "failure_kind",
        "failure_evidence",
        "identity_sha256",
        "selection_sha256",
        "training_order_sha256",
    }
)


def _normalize_gate_binding(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("overfit_gate_binding must be a mapping")
    try:
        serialized = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("overfit_gate_binding must contain finite JSON data") from exc
    normalized = json.loads(serialized)
    if not isinstance(normalized, dict):
        raise TypeError("overfit_gate_binding must normalize to a mapping")
    if set(normalized) != SFT_GATE_BINDING_KEYS:
        raise ValueError("overfit gate binding fields changed")
    profile = normalized.get("resolution_profile")
    if profile not in {"full_resolution", "registered_76g_fallback"}:
        raise ValueError("overfit gate binding has an unregistered resolution profile")
    _validate_bound_artifact(
        normalized,
        name="overfit gate",
        sha_fields=(
            "sha256",
            "config_sha256",
            "cache_identity_sha256",
            "camera_sha256",
            "identity_sha256",
            "selection_sha256",
            "training_order_sha256",
            "checks_sha256",
            "run_sha256",
            "checkpoint_tree_sha256",
            "train_summary_sha256",
        ),
    )
    if type(normalized.get("seed")) is not int:
        raise ValueError("overfit gate binding seed must be an exact integer")
    primary = normalized.get("primary_gate_failure")
    if profile == "full_resolution":
        if primary is not None:
            raise ValueError(
                "full-resolution gate binding must not include fallback evidence"
            )
    else:
        if (
            not isinstance(primary, dict)
            or set(primary) != SFT_PRIMARY_FAILURE_BINDING_KEYS
        ):
            raise ValueError(
                "fallback gate binding lacks exact primary failure evidence"
            )
        _validate_bound_artifact(
            primary,
            name="primary gate failure",
            sha_fields=(
                "sha256",
                "config_sha256",
                "identity_sha256",
                "selection_sha256",
                "training_order_sha256",
            ),
        )
        evidence = primary.get("failure_evidence")
        if (
            primary.get("resolution_profile") != "full_resolution"
            or primary.get("failure_kind")
            not in {"cuda_out_of_memory", "peak_cuda_memory_exceeded"}
            or (not isinstance(evidence, dict))
            or (evidence.get("eligible_fallback") is not True)
            or (evidence.get("kind") != primary.get("failure_kind"))
        ):
            raise ValueError("fallback primary failure evidence is not eligible")
    return normalized


def _canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _require_gate_binding_matches_config(
    config: ARExperimentConfig, binding: Mapping[str, Any]
) -> None:
    if binding.get("config_sha256") != getattr(
        config, "config_sha256", None
    ) or binding.get("seed") != int(config.training["seed"]):
        raise ValueError("overfit gate binding config/seed differs from active SFT")
    camera = {
        "resolution_profile": binding["resolution_profile"],
        "order": list(config.camera_order),
        "input_sizes": {
            key: list(value) for key, value in config.camera_input_sizes.items()
        },
        "target": config.target_camera,
    }
    if binding.get("camera_sha256") != _canonical_sha256(camera):
        raise ValueError("overfit gate binding camera contract differs from active SFT")


SFT_CHECKPOINT_NAME_PATTERN = re.compile("checkpoint-([0-9]{6})\\Z")
TRAINER_STATE_FILENAME = "trainer_state.pt"
SFT_CONTENT_BOUND_FILENAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    PROJECTOR_FILENAME,
)
SFT_COMPLETE_CHECKPOINT_FILENAMES = (
    *SFT_CONTENT_BOUND_FILENAMES,
    METADATA_FILENAME,
    TRAINER_STATE_FILENAME,
)


def _validate_grounded_resume_output_layout(checkpoint: Path, output_dir: Path) -> int:
    output = output_dir.resolve()
    target = checkpoint.absolute()
    if target.parent.resolve() != output:
        raise ValueError(
            "resume checkpoint must be an immediate child of the SFT output"
        )
    match = SFT_CHECKPOINT_NAME_PATTERN.fullmatch(target.name)
    if match is None:
        raise ValueError("grounded resume target must be a numbered checkpoint-XXXXXX")
    if not target.is_dir() or target.is_symlink():
        raise ValueError(
            "grounded resume target must be a regular checkpoint directory"
        )
    target_step = int(match.group(1))
    if target_step <= 0:
        raise ValueError("grounded resume checkpoint step must be positive")
    if not output.is_dir() or output.is_symlink():
        raise ValueError("grounded SFT output must be a regular directory")
    numbered_steps: list[int] = []
    unexpected: list[str] = []
    incomplete: list[str] = []
    for entry in output.iterdir():
        entry_match = SFT_CHECKPOINT_NAME_PATTERN.fullmatch(entry.name)
        if entry_match is None:
            unexpected.append(entry.name)
            continue
        if not entry.is_dir() or entry.is_symlink():
            unexpected.append(entry.name)
            continue
        entry_step = int(entry_match.group(1))
        if entry_step <= 0:
            unexpected.append(entry.name)
            continue
        numbered_steps.append(entry_step)
        missing = [
            name
            for name in SFT_COMPLETE_CHECKPOINT_FILENAMES
            if not (entry / name).is_file() or (entry / name).is_symlink()
        ]
        partial = sorted(
            (
                path.name
                for path in entry.iterdir()
                if path.name.startswith(".") or path.name.endswith((".tmp", ".partial"))
            )
        )
        if missing or partial:
            incomplete.append(f"{entry.name}:missing={missing},partial={partial}")
    if unexpected:
        raise ValueError(
            f"grounded SFT output has final/summary/partial/stale forward-tail entries: {sorted(unexpected)}"
        )
    if incomplete:
        raise ValueError(
            f"grounded SFT output has incomplete checkpoints: {incomplete}"
        )
    if not numbered_steps or max(numbered_steps) != target_step:
        raise ValueError(
            "resume target is not the latest complete numbered SFT checkpoint"
        )
    return target_step


def _load_trainer_payload(checkpoint_dir: Path) -> dict[str, Any]:
    import torch

    source = checkpoint_dir / TRAINER_STATE_FILENAME
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"resume checkpoint lacks regular {TRAINER_STATE_FILENAME}: {source}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("unsupported Qwen SFT trainer-state schema")
    return dict(payload)


SFT_TRAINER_STATE_KEYS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "total_steps",
        "epoch",
        "next_batch_index",
        "optimizer_steps",
        "batches",
        "examples",
        "loss_sum",
        "first_loss",
        "last_loss",
        "min_loss",
        "elapsed_seconds",
        "gradient_flow_audit",
        "overfit_gate",
        "checkpoint_content",
        "adapter_training_state_sha256",
        "optimizer",
        "scheduler",
        "rng",
    }
)


def _checkpoint_content_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for filename in SFT_CONTENT_BOUND_FILENAMES:
        path = checkpoint_dir / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"grounded SFT checkpoint lacks regular content-bound artifact: {path}"
            )
        files[filename] = _sha256_file(path)
    return {
        "schema_version": 1,
        "files": files,
        "aggregate_sha256": _canonical_sha256(files),
    }


def _resume_progress(
    *,
    epoch: int,
    next_batch_index: int,
    optimizer_steps: int,
    batches: int,
    examples: int,
) -> dict[str, int]:
    return {
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "optimizer_steps": int(optimizer_steps),
        "batches": int(batches),
        "examples": int(examples),
    }


GROUNDED_SFT_STAGE = "grounded_action_sft_production"


def _validate_grounded_checkpoint_integrity(
    checkpoint_dir: Path,
    *,
    config: ARExperimentConfig,
    payload: Mapping[str, Any],
    expected_gate_binding: Mapping[str, Any],
    expected_checkpoint_step: int,
) -> None:
    from comparisons.ar_model import METADATA_FILENAME, validate_checkpoint_metadata

    if set(payload) != SFT_TRAINER_STATE_KEYS:
        raise ValueError("grounded SFT trainer-state fields changed")
    if payload.get("config_sha256") != getattr(config, "config_sha256", None):
        raise ValueError(
            "resume trainer-state config SHA differs from the active config"
        )
    integer_fields = (
        "total_steps",
        "epoch",
        "next_batch_index",
        "optimizer_steps",
        "batches",
        "examples",
    )
    if any((type(payload.get(field)) is not int for field in integer_fields)):
        raise ValueError("grounded SFT trainer progress must use exact integers")
    if (
        payload["total_steps"] <= 0
        or payload["epoch"] < 0
        or payload["next_batch_index"] < 0
        or (payload["optimizer_steps"] <= 0)
        or (payload["optimizer_steps"] > payload["total_steps"])
        or (payload["batches"] < payload["optimizer_steps"])
        or (payload["examples"] != payload["batches"])
    ):
        raise ValueError("grounded SFT trainer progress is inconsistent")
    finite_fields = (
        "loss_sum",
        "first_loss",
        "last_loss",
        "min_loss",
        "elapsed_seconds",
    )
    if (
        any(
            (
                isinstance(payload.get(field), bool)
                or not isinstance(payload.get(field), int | float)
                or (not math.isfinite(float(payload[field])))
                for field in finite_fields
            )
        )
        or float(payload["elapsed_seconds"]) < 0.0
    ):
        raise ValueError("grounded SFT trainer loss/time state is invalid")
    state_fields = ("optimizer", "scheduler", "rng")
    if not all((isinstance(payload.get(field), Mapping) for field in state_fields)):
        raise ValueError("grounded SFT optimizer/scheduler/RNG state is invalid")
    gradient_audit = payload.get("gradient_flow_audit")
    if (
        not isinstance(gradient_audit, Mapping)
        or gradient_audit.get("passed") is not True
    ):
        raise ValueError("grounded SFT trainer state lacks a passed gradient audit")
    recorded_binding = payload.get("overfit_gate")
    if recorded_binding != dict(expected_gate_binding):
        raise ValueError("resume trainer-state overfit gate authorization changed")
    actual_content = _checkpoint_content_manifest(checkpoint_dir)
    if payload.get("checkpoint_content") != actual_content:
        raise ValueError("resume adapter model/config/projector hashes changed")
    metadata_path = checkpoint_dir / METADATA_FILENAME
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError(
            f"resume checkpoint lacks regular metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("resume adapter metadata must be a mapping")
    validate_checkpoint_metadata(config, metadata)
    adapter_state = metadata.get("training_state")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("resume adapter metadata lacks training_state")
    if payload.get("adapter_training_state_sha256") != _canonical_sha256(adapter_state):
        raise ValueError("resume adapter metadata training state changed")
    if adapter_state.get("stage") != GROUNDED_SFT_STAGE or adapter_state.get(
        "overfit_gate"
    ) != dict(expected_gate_binding):
        raise ValueError("resume adapter metadata authorization changed")
    expected_progress = _resume_progress(
        epoch=int(payload["epoch"]),
        next_batch_index=int(payload["next_batch_index"]),
        optimizer_steps=int(payload["optimizer_steps"]),
        batches=int(payload["batches"]),
        examples=int(payload["examples"]),
    )
    if adapter_state.get("resume_progress") != expected_progress:
        raise ValueError("resume adapter metadata progress differs from trainer state")
    if adapter_state.get("optimizer_step") != payload[
        "optimizer_steps"
    ] or adapter_state.get("gradient_flow_audit") != dict(gradient_audit):
        raise ValueError(
            "resume adapter metadata training evidence differs from trainer state"
        )
    if int(payload["optimizer_steps"]) != expected_checkpoint_step:
        raise ValueError(
            "checkpoint directory step differs from trainer-state progress"
        )


def validate_grounded_resume_checkpoint(
    config: ARExperimentConfig,
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    overfit_gate_binding: Mapping[str, Any],
) -> None:
    target = Path(checkpoint).absolute()
    binding = _normalize_gate_binding(overfit_gate_binding)
    if binding is None:
        raise ValueError("grounded SFT resume requires overfit gate authorization")
    _require_gate_binding_matches_config(config, binding)
    checkpoint_step = _validate_grounded_resume_output_layout(
        target, Path(output_dir).resolve()
    )
    payload = _load_trainer_payload(target)
    _validate_grounded_checkpoint_integrity(
        target,
        config=config,
        payload=payload,
        expected_gate_binding=binding,
        expected_checkpoint_step=checkpoint_step,
    )


def _linear_scheduler(
    optimizer: Any, *, total_steps: int, warmup_fraction: float
) -> Any:
    import torch

    if total_steps <= 0:
        raise ValueError("total optimizer steps must be positive")
    warmup_steps = int(total_steps * warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _reserve_checkpoint_destination(path: Path) -> Path:
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite SFT checkpoint: {destination}"
        ) from exc
    return destination


def _save_trainer_state(
    checkpoint_dir: Path,
    *,
    config: Any,
    optimizer: Any,
    scheduler: Any,
    total_steps: int,
    epoch: int,
    next_batch_index: int,
    optimizer_steps: int,
    batches: int,
    examples: int,
    loss_sum: float,
    first_loss: float | None,
    last_loss: float | None,
    min_loss: float,
    elapsed_seconds: float,
    gradient_flow_audit: Mapping[str, Any] | None,
    overfit_gate_binding: Mapping[str, Any] | None,
    adapter_training_state: Mapping[str, Any],
) -> Path:
    import torch

    from comparisons.ar_model import METADATA_FILENAME

    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise ValueError(
            "trainer-state destination must be a reserved checkpoint directory"
        )
    grounded = bool(getattr(config, "uses_grounded_action", False))
    normalized_binding = _normalize_gate_binding(overfit_gate_binding)
    normalized_gradient_audit = (
        None
        if gradient_flow_audit is None
        else json.loads(
            json.dumps(dict(gradient_flow_audit), sort_keys=True, allow_nan=False)
        )
    )
    checkpoint_content = None
    adapter_training_state_sha256 = None
    if grounded:
        expected_adapter_state = json.loads(
            json.dumps(dict(adapter_training_state), sort_keys=True, allow_nan=False)
        )
        metadata_path = checkpoint_dir / METADATA_FILENAME
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise FileNotFoundError(
                f"grounded SFT checkpoint lacks regular metadata: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError("grounded SFT adapter metadata must be a mapping")
        if metadata.get("training_state") != expected_adapter_state:
            raise ValueError(
                "adapter metadata training state changed before trainer-state save"
            )
        checkpoint_content = _checkpoint_content_manifest(checkpoint_dir)
        adapter_training_state_sha256 = _canonical_sha256(expected_adapter_state)
    payload = {
        "schema_version": 1,
        "config_sha256": getattr(config, "config_sha256", None),
        "total_steps": int(total_steps),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "optimizer_steps": int(optimizer_steps),
        "batches": int(batches),
        "examples": int(examples),
        "loss_sum": float(loss_sum),
        "first_loss": None if first_loss is None else float(first_loss),
        "last_loss": None if last_loss is None else float(last_loss),
        "min_loss": float(min_loss),
        "elapsed_seconds": float(elapsed_seconds),
        "gradient_flow_audit": normalized_gradient_audit,
        "overfit_gate": normalized_binding,
        "checkpoint_content": checkpoint_content,
        "adapter_training_state_sha256": adapter_training_state_sha256,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
    }
    destination = checkpoint_dir / TRAINER_STATE_FILENAME
    temporary = checkpoint_dir / f".{TRAINER_STATE_FILENAME}.tmp"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite SFT trainer state: {destination}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite SFT trainer-state temp file: {temporary}"
        )
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def _load_trainer_state(
    checkpoint_dir: str | Path,
    *,
    config: Any,
    optimizer: Any,
    scheduler: Any,
    total_steps: int,
    overfit_gate_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir).absolute()
    payload = _load_trainer_payload(checkpoint)
    expected_config = getattr(config, "config_sha256", None)
    if payload.get("config_sha256") != expected_config:
        raise ValueError(
            "resume trainer-state config SHA differs from the active config"
        )
    if int(payload.get("total_steps", -1)) != int(total_steps):
        raise ValueError("resume trainer-state total step count changed")
    if bool(getattr(config, "uses_grounded_action", False)):
        binding = _normalize_gate_binding(overfit_gate_binding)
        if binding is None:
            raise ValueError("grounded SFT resume requires overfit gate authorization")
        match = SFT_CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint.name)
        if match is None:
            raise ValueError("grounded SFT resume requires a numbered checkpoint")
        _validate_grounded_checkpoint_integrity(
            checkpoint,
            config=config,
            payload=payload,
            expected_gate_binding=binding,
            expected_checkpoint_step=int(match.group(1)),
        )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return dict(payload)


def _log(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True, allow_nan=False), flush=True)


DATALOADER_EPOCH_SEED_STRIDE = 1000003


def _reset_dataloader_generator(loader: Any, *, seed: int, epoch: int) -> None:
    import torch

    generator = getattr(loader, "generator", None)
    if generator is None:
        return
    if not isinstance(generator, torch.Generator):
        raise TypeError("Qwen SFT DataLoader generator must be a torch.Generator")
    generator.manual_seed(int(seed) + DATALOADER_EPOCH_SEED_STRIDE * int(epoch))


def _authorized_training_state(
    state: Mapping[str, Any],
    *,
    overfit_gate_binding: Mapping[str, Any],
    progress: Mapping[str, int],
) -> dict[str, Any]:
    return {
        **dict(state),
        "stage": GROUNDED_SFT_STAGE,
        "overfit_gate": dict(overfit_gate_binding),
        "resume_progress": dict(progress),
    }


def _move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _peak_cuda_memory_gib() -> float:
    import torch

    return (
        max(
            float(torch.cuda.max_memory_allocated()),
            float(torch.cuda.max_memory_reserved()),
        )
        / 1024**3
    )


def fit(
    config: ARExperimentConfig,
    bundle: QwenARBundle,
    loader: Any,
    *,
    max_steps: int | None = None,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    overfit_gate_binding: Mapping[str, Any] | None = None,
    diagnostic_overfit: bool = False,
) -> TrainSummary:
    import torch

    from comparisons.ar_model import (
        assert_gradient_flow,
        inspect_gradient_flow,
        save_ar_adapter,
    )

    grounded = bool(getattr(config, "uses_grounded_action", False))
    normalized_gate_binding = _normalize_gate_binding(overfit_gate_binding)
    if grounded:
        if diagnostic_overfit and normalized_gate_binding is not None:
            raise ValueError("diagnostic overfit must not consume a production gate")
        if not diagnostic_overfit and normalized_gate_binding is None:
            raise ValueError("grounded production SFT requires a passed overfit gate")
        if normalized_gate_binding is not None:
            _require_gate_binding_matches_config(config, normalized_gate_binding)
    elif diagnostic_overfit or normalized_gate_binding is not None:
        raise ValueError(
            "overfit gate authorization applies only to grounded action SFT"
        )
    if diagnostic_overfit and resume_checkpoint is not None:
        raise ValueError("diagnostic overfit does not support resume")
    device = str(config.training["device"])
    accumulation = int(config.training["gradient_accumulation_steps"])
    planned_steps = math.ceil(len(loader) / accumulation) * int(
        config.training["epochs"]
    )
    total_steps = planned_steps if max_steps is None else int(max_steps)
    if total_steps <= 0:
        raise ValueError("max_steps must be positive")
    epochs_to_iterate = (
        int(config.training["epochs"])
        if max_steps is None
        else max(1, math.ceil(total_steps * accumulation / len(loader)))
    )
    destination = (
        Path(output_dir).resolve() if output_dir is not None else config.output_dir
    )
    resume_path = (
        None if resume_checkpoint is None else Path(resume_checkpoint).absolute()
    )
    if grounded and (not diagnostic_overfit):
        assert normalized_gate_binding is not None
        forbidden_gate_outputs = {
            Path(normalized_gate_binding["path"]).parent.resolve()
        }
        primary_failure = normalized_gate_binding.get("primary_gate_failure")
        if isinstance(primary_failure, Mapping):
            forbidden_gate_outputs.add(Path(primary_failure["path"]).parent.resolve())
        if destination in forbidden_gate_outputs:
            raise ValueError(
                "overfit gate output must differ from the production SFT output"
            )
    if resume_path is None and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Qwen AR training directory: {destination}"
        )
    if resume_path is not None and resume_path.parent.resolve() != destination:
        raise ValueError(
            "resume checkpoint must be an immediate child of the output directory"
        )
    if resume_path is not None and grounded:
        assert normalized_gate_binding is not None
        validate_grounded_resume_checkpoint(
            config,
            resume_path,
            output_dir=destination,
            overfit_gate_binding=normalized_gate_binding,
        )
    destination.mkdir(parents=True, exist_ok=True)
    parameters = [
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    scheduler = _linear_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(config.training["warmup_fraction"]),
    )
    bundle.model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    resume_epoch = 0
    resume_batch_index = 0
    optimizer_steps = 0
    batches = 0
    examples = 0
    loss_sum = 0.0
    first_loss: float | None = None
    last_loss: float | None = None
    min_loss = math.inf
    elapsed_before = 0.0
    save_steps = int(config.training["save_steps"])
    log_steps = int(config.training["log_steps"])
    gradient_flow_audit: dict[str, Any] | None = None
    if resume_path is not None:
        restored = _load_trainer_state(
            resume_path,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            overfit_gate_binding=normalized_gate_binding,
        )
        resume_epoch = int(restored["epoch"])
        resume_batch_index = int(restored["next_batch_index"])
        optimizer_steps = int(restored["optimizer_steps"])
        batches = int(restored["batches"])
        examples = int(restored["examples"])
        loss_sum = float(restored["loss_sum"])
        first_loss = (
            None if restored["first_loss"] is None else float(restored["first_loss"])
        )
        last_loss = (
            None if restored["last_loss"] is None else float(restored["last_loss"])
        )
        min_loss = float(restored["min_loss"])
        elapsed_before = float(restored["elapsed_seconds"])
        gradient_flow_audit = (
            None
            if restored["gradient_flow_audit"] is None
            else dict(restored["gradient_flow_audit"])
        )
        _log(
            {
                "event": "training_resumed",
                "checkpoint": str(resume_path),
                "epoch": resume_epoch,
                "next_batch_index": resume_batch_index,
                "optimizer_step": optimizer_steps,
            }
        )
    state_epoch = resume_epoch
    state_next_batch = resume_batch_index
    started = time.perf_counter()
    stop = False
    epoch_range = (
        range(resume_epoch, epochs_to_iterate) if optimizer_steps < total_steps else ()
    )
    for epoch in epoch_range:
        sampler = getattr(loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        _reset_dataloader_generator(
            loader, seed=int(config.training["seed"]), epoch=epoch
        )
        for batch_index, batch in enumerate(loader):
            if epoch == resume_epoch and batch_index < resume_batch_index:
                continue
            moved = _move_batch(batch, device)
            model_inputs = dict(moved)
            loss_weights = model_inputs.pop("loss_weights", None)
            weighted_labels = (
                model_inputs.pop("labels") if loss_weights is not None else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = bundle.model(**model_inputs)
                loss = (
                    output.loss
                    if loss_weights is None
                    else weighted_causal_lm_loss(
                        output.logits, weighted_labels, loss_weights
                    )
                )
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite Qwen SFT loss at batch {batch_index}: {loss}"
                )
            (loss / accumulation).backward()
            batches += 1
            examples += int(moved["input_ids"].shape[0])
            loss_value = float(loss.detach().cpu().item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            min_loss = min(min_loss, loss_value)
            loss_sum += loss_value
            is_last_batch = batch_index + 1 == len(loader)
            if batches % accumulation != 0 and (not is_last_batch):
                continue
            if optimizer_steps == 0:
                gradient_report = inspect_gradient_flow(bundle.model)
                assert_gradient_flow(gradient_report)
                gradient_flow_audit = {
                    **gradient_report.to_dict(),
                    "stage": "after_backward_before_clipping",
                    "optimizer_step": 1,
                    "accumulated_batches": batches,
                }
                _log({"event": "gradient_flow_audit", **gradient_flow_audit})
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config.training["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if device.startswith("cuda") and "peak_memory_limit_gib" in config.training:
                peak_gib = _peak_cuda_memory_gib()
                limit_gib = float(config.training["peak_memory_limit_gib"])
                if peak_gib > limit_gib:
                    raise RuntimeError(
                        f"Qwen SFT peak CUDA memory {peak_gib:.9f} GiB exceeds the {limit_gib:.9f} GiB gate; use the registered 76g fallback config"
                    )
            state_epoch = epoch + 1 if is_last_batch else epoch
            state_next_batch = 0 if is_last_batch else batch_index + 1
            if log_steps > 0 and optimizer_steps % log_steps == 0:
                _log(
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": optimizer_steps,
                        "loss": loss_value,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "examples": examples,
                    }
                )
            if save_steps > 0 and optimizer_steps % save_steps == 0:
                checkpoint_progress = _resume_progress(
                    epoch=state_epoch,
                    next_batch_index=state_next_batch,
                    optimizer_steps=optimizer_steps,
                    batches=batches,
                    examples=examples,
                )
                adapter_training_state: dict[str, Any] = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "gradient_flow_audit": gradient_flow_audit,
                }
                if grounded and (not diagnostic_overfit):
                    assert normalized_gate_binding is not None
                    adapter_training_state = _authorized_training_state(
                        adapter_training_state,
                        overfit_gate_binding=normalized_gate_binding,
                        progress=checkpoint_progress,
                    )
                checkpoint_destination = _reserve_checkpoint_destination(
                    destination / f"checkpoint-{optimizer_steps:06d}"
                )
                checkpoint_dir = save_ar_adapter(
                    bundle,
                    checkpoint_destination,
                    config=config,
                    training_state=adapter_training_state,
                )
                if Path(checkpoint_dir).absolute() != checkpoint_destination:
                    raise RuntimeError(
                        "adapter save returned an unexpected checkpoint destination"
                    )
                _save_trainer_state(
                    Path(checkpoint_dir),
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps,
                    epoch=state_epoch,
                    next_batch_index=state_next_batch,
                    optimizer_steps=optimizer_steps,
                    batches=batches,
                    examples=examples,
                    loss_sum=loss_sum,
                    first_loss=first_loss,
                    last_loss=last_loss,
                    min_loss=min_loss,
                    elapsed_seconds=elapsed_before + time.perf_counter() - started,
                    gradient_flow_audit=gradient_flow_audit,
                    overfit_gate_binding=normalized_gate_binding,
                    adapter_training_state=adapter_training_state,
                )
            if optimizer_steps >= total_steps:
                stop = True
                break
        if stop:
            break
    elapsed = elapsed_before + time.perf_counter() - started
    if optimizer_steps != total_steps:
        raise RuntimeError(
            f"training stopped at optimizer step {optimizer_steps}, expected {total_steps}"
        )
    if first_loss is None or last_loss is None:
        raise RuntimeError("training completed without observing a loss")
    if gradient_flow_audit is None:
        raise RuntimeError(
            "training completed without a first-step gradient-flow audit"
        )
    summary = TrainSummary(
        optimizer_steps=optimizer_steps,
        batches=batches,
        examples=examples,
        mean_loss=loss_sum / batches,
        first_loss=first_loss,
        last_loss=last_loss,
        min_loss=min_loss,
        elapsed_seconds=elapsed,
        output_dir=str(destination),
        gradient_flow_audit=gradient_flow_audit,
        peak_cuda_memory_gib=_peak_cuda_memory_gib()
        if device.startswith("cuda")
        else None,
    )
    final_progress = _resume_progress(
        epoch=state_epoch,
        next_batch_index=state_next_batch,
        optimizer_steps=optimizer_steps,
        batches=batches,
        examples=examples,
    )
    final_training_state = summary.to_dict()
    if grounded and (not diagnostic_overfit):
        assert normalized_gate_binding is not None
        final_training_state = _authorized_training_state(
            final_training_state,
            overfit_gate_binding=normalized_gate_binding,
            progress=final_progress,
        )
    final_destination = _reserve_checkpoint_destination(destination / "final")
    final_dir = save_ar_adapter(
        bundle, final_destination, config=config, training_state=final_training_state
    )
    if Path(final_dir).absolute() != final_destination:
        raise RuntimeError("adapter save returned an unexpected final destination")
    _save_trainer_state(
        Path(final_dir),
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=total_steps,
        epoch=state_epoch,
        next_batch_index=state_next_batch,
        optimizer_steps=optimizer_steps,
        batches=batches,
        examples=examples,
        loss_sum=loss_sum,
        first_loss=first_loss,
        last_loss=last_loss,
        min_loss=min_loss,
        elapsed_seconds=elapsed,
        gradient_flow_audit=gradient_flow_audit,
        overfit_gate_binding=normalized_gate_binding,
        adapter_training_state=final_training_state,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-data-file-verification", action="store_true")
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--overfit-gate", type=Path)
    return parser


class _GateOutputCollisionError(ValueError):
    """Raised before any gate artifact may be written into production output."""


def _require_gate_output_separation(
    config: ARExperimentConfig,
    artifact_path: Path,
    *,
    production_output: str | Path | None = None,
) -> None:
    gate_output = artifact_path.parent.resolve()
    forbidden = {config.output_dir.resolve()}
    if production_output is not None:
        forbidden.add(Path(production_output).resolve())
    if gate_output in forbidden:
        raise _GateOutputCollisionError(
            "overfit gate output must differ from the production SFT output"
        )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _read_json_mapping(path: str | Path, name: str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _mapping(payload, name)


GATE_IMAGE_SIZE = (640, 480)
GATE_FALLBACK_AUXILIARY_SIZE = (320, 240)


def _resolution_profile(config: ARExperimentConfig) -> str:
    from comparisons.ar_data import PINNED_CAMERA_ORDER, TARGET_CAMERA

    sizes = config.camera_input_sizes
    full = {key: GATE_IMAGE_SIZE for key in PINNED_CAMERA_ORDER}
    fallback = {
        key: GATE_IMAGE_SIZE if key == TARGET_CAMERA else GATE_FALLBACK_AUXILIARY_SIZE
        for key in PINNED_CAMERA_ORDER
    }
    if sizes == full:
        return "full_resolution"
    if sizes == fallback:
        return "registered_76g_fallback"
    raise ValueError(
        f"overfit32 camera sizes must match the registered full-resolution or 76g fallback profile, got {sizes}"
    )


GATE_MAX_PEAK_CUDA_MEMORY_GIB = 76.0


def _require_registered_gate_contract(config: ARExperimentConfig) -> str:
    if config.data_format != "scenario_static_v1" or not config.uses_grounded_action:
        raise ValueError("overfit32 gate requires grounded scenario-static SFT")
    if str(config.training["device"]).split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("overfit32 peak-memory gate requires a CUDA training config")
    if int(config.training["batch_size"]) != 1:
        raise ValueError("overfit32 gate requires training.batch_size=1")
    if int(config.training["gradient_accumulation_steps"]) != 2:
        raise ValueError("overfit32 gate requires gradient_accumulation_steps=2")
    configured_limit = float(config.training.get("peak_memory_limit_gib", math.nan))
    if configured_limit != GATE_MAX_PEAK_CUDA_MEMORY_GIB:
        raise ValueError("overfit32 gate requires training.peak_memory_limit_gib=76.0")
    resolution_profile = _resolution_profile(config)
    manifest_path = config.cache_manifest_path
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(
            f"overfit32 gate requires the aggregate cache manifest: {manifest_path}"
        )
    return resolution_profile


GATE_SAMPLE_COUNT = 32
GATE_SELECTION_ALGORITHM = "scenario_status_group_round_robin_v1"


def _overfit_gate_canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _selection_identity(dataset: Any, *, seed: int) -> tuple[list[int], dict[str, Any]]:
    if not hasattr(dataset, "records"):
        raise TypeError("overfit32 dataset must expose content-bound records")
    indices = _diagnostic_subset_indices(dataset, GATE_SAMPLE_COUNT, seed=seed)
    records: list[dict[str, Any]] = []
    for ordinal, dataset_position in enumerate(indices):
        record = dataset.records[dataset_position]
        identity = {
            "ordinal": ordinal,
            "dataset_position": int(dataset_position),
            "sample_id": str(record.sample_id),
            "image_id": int(record.image_id),
            "dataset_index": int(record.dataset_index),
            "episode_index": int(record.episode_index),
            "frame_index": int(record.frame_index),
            "status": str(getattr(record.status, "value", record.status)),
            "source_dataset": str(record.source_dataset),
        }
        records.append(identity)
    if len({row["sample_id"] for row in records}) != GATE_SAMPLE_COUNT:
        raise ValueError("overfit32 selection contains duplicate sample IDs")
    if len({row["dataset_position"] for row in records}) != GATE_SAMPLE_COUNT:
        raise ValueError("overfit32 selection contains duplicate dataset positions")
    selection_payload = {
        "algorithm": GATE_SELECTION_ALGORITHM,
        "seed": int(seed),
        "record_count": GATE_SAMPLE_COUNT,
        "records": records,
    }
    return (
        indices,
        {
            **selection_payload,
            "sha256": _overfit_gate_canonical_sha256(selection_payload),
        },
    )


def _identity_context(
    config: ARExperimentConfig,
    *,
    config_sha256: str,
    resolution_profile: str,
    cache_identity: Mapping[str, Any],
    selection: Mapping[str, Any],
    training_order: Mapping[str, Any],
    data_bindings: Mapping[str, Any],
    primary_gate_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "config": {"path": str(config.source_path), "sha256": config_sha256},
        "cache": dict(cache_identity),
        "camera": {
            "resolution_profile": resolution_profile,
            "order": list(config.camera_order),
            "input_sizes": {
                key: list(value) for key, value in config.camera_input_sizes.items()
            },
            "target": config.target_camera,
        },
        "seed": int(config.training["seed"]),
        "data_bindings": dict(data_bindings),
        "selection": dict(selection),
        "training_order": dict(training_order),
        "primary_gate_failure": None
        if primary_gate_failure is None
        else dict(primary_gate_failure),
    }


def _overfit_gate_sha256_file(
    path: str | Path, *, chunk_size: int = 1024 * 1024
) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_identity(config: ARExperimentConfig) -> dict[str, Any]:
    from comparisons.ar_data import CACHE_BUNDLE_KIND, CACHE_SCHEMA_VERSION

    manifest_path = config.cache_manifest_path
    if manifest_path is None:
        raise ValueError("grounded overfit32 config has no cache manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("aggregate cache manifest must be a mapping")
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("kind") != CACHE_BUNDLE_KIND
    ):
        raise ValueError("overfit32 requires the completed aggregate cache contract")
    views = manifest.get("views")
    if not isinstance(views, Mapping) or not isinstance(views.get("train"), Mapping):
        raise ValueError("aggregate cache manifest lacks a train binding")
    train = views["train"]
    return {
        "configured_root": str(config.data["cache_root"]),
        "manifest": str(config.data["cache_manifest"]),
        "aggregate_manifest_sha256": _overfit_gate_sha256_file(manifest_path),
        "train_manifest_sha256": str(train["cache_manifest_sha256"]),
        "train_completion_sha256": str(train["completion_sha256"]),
    }


GATE_TRAINING_ORDER_POLICY = "fixed_subset_sequential_no_replacement_v1"
GATE_BATCHES = 200


def _training_order_identity(indices: Sequence[int]) -> dict[str, Any]:
    selected = [int(index) for index in indices]
    if len(selected) != GATE_SAMPLE_COUNT or len(set(selected)) != GATE_SAMPLE_COUNT:
        raise ValueError("overfit32 training order requires 32 unique subset indices")
    expanded = [selected[index % GATE_SAMPLE_COUNT] for index in range(GATE_BATCHES)]
    payload = {
        "policy": GATE_TRAINING_ORDER_POLICY,
        "replacement": False,
        "shuffle": False,
        "batches": GATE_BATCHES,
        "full_subset_passes": GATE_BATCHES // GATE_SAMPLE_COUNT,
        "final_prefix_records": GATE_BATCHES % GATE_SAMPLE_COUNT,
        "expanded_dataset_positions_sha256": _overfit_gate_canonical_sha256(expanded),
    }
    return {**payload, "sha256": _overfit_gate_canonical_sha256(payload)}


def _expected_gate_identity(
    config: ARExperimentConfig,
    dataset: Any,
    *,
    primary_gate_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from comparisons.ar_data import verify_data_bindings

    resolution_profile = _require_registered_gate_contract(config)
    seed = int(config.training["seed"])
    indices, selection = _selection_identity(dataset, seed=seed)
    return _identity_context(
        config,
        config_sha256=config.config_sha256,
        resolution_profile=resolution_profile,
        cache_identity=_cache_identity(config),
        selection=selection,
        training_order=_training_order_identity(indices),
        data_bindings=verify_data_bindings(config).to_dict(),
        primary_gate_failure=primary_gate_failure,
    )


def _validate_nll_report(value: Any, name: str) -> dict[str, Any]:
    report = _mapping(value, name)
    expected_keys = {
        "metric",
        "record_count",
        "mean",
        "minimum",
        "maximum",
        "per_record",
        "per_record_sha256",
    }
    if set(report) != expected_keys:
        raise ValueError(
            f"{name} fields changed: {sorted(set(report) ^ expected_keys)}"
        )
    if (
        report["metric"] != "mean_per_record_weighted_causal_nll"
        or report["record_count"] != GATE_SAMPLE_COUNT
    ):
        raise ValueError(f"{name} metric contract changed")
    losses = report["per_record"]
    if not isinstance(losses, list) or len(losses) != GATE_SAMPLE_COUNT:
        raise ValueError(f"{name}.per_record must contain exactly 32 losses")
    converted = [float(value) for value in losses]
    if any((not math.isfinite(value) or value <= 0.0 for value in converted)):
        raise ValueError(f"{name}.per_record contains an invalid weighted NLL")
    expected = {
        "mean": math.fsum(converted) / GATE_SAMPLE_COUNT,
        "minimum": min(converted),
        "maximum": max(converted),
        "per_record_sha256": _overfit_gate_canonical_sha256(converted),
    }
    mismatches = {
        key: {"expected": expected_value, "actual": report.get(key)}
        for key, expected_value in expected.items()
        if report.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{name} aggregate/hash mismatch: {mismatches}")
    return report


_TRAINING_SUMMARY_KEYS = frozenset(
    {
        "optimizer_steps",
        "batches",
        "examples",
        "mean_loss",
        "first_loss",
        "last_loss",
        "min_loss",
        "elapsed_seconds",
        "output_dir",
        "gradient_flow_audit",
        "peak_cuda_memory_gib",
    }
)


def _validate_training_summary(value: Any, output_dir: Path) -> dict[str, Any]:
    summary = _mapping(value, "overfit gate training summary")
    if set(summary) != _TRAINING_SUMMARY_KEYS:
        raise ValueError("overfit gate training summary fields changed")
    for field in ("optimizer_steps", "batches", "examples"):
        if type(summary.get(field)) is not int or summary[field] <= 0:
            raise ValueError(f"overfit gate training summary {field} is invalid")
    for field in ("mean_loss", "first_loss", "last_loss", "min_loss"):
        value = summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or (not math.isfinite(float(value)))
            or (float(value) <= 0.0)
        ):
            raise ValueError(f"overfit gate training summary {field} is invalid")
    elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or (not math.isfinite(float(elapsed)))
        or (float(elapsed) < 0.0)
    ):
        raise ValueError("overfit gate training summary elapsed_seconds is invalid")
    if summary.get("output_dir") != str(output_dir.resolve()):
        raise ValueError("overfit gate training summary output path changed")
    return summary


_REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    METADATA_FILENAME,
    PROJECTOR_FILENAME,
    TRAINER_STATE_FILENAME,
)


def _sha256_tree(root: str | Path) -> str:
    directory = Path(root).resolve()
    files = sorted((path for path in directory.rglob("*") if path.is_file()))
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {directory}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _collect_artifact_evidence(
    output_dir: Path,
    *,
    config: ARExperimentConfig,
    summary: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from comparisons.ar_model import METADATA_FILENAME, validate_checkpoint_metadata

    summary_path = output_dir / "train_summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise FileNotFoundError(
            f"overfit32 training summary is missing: {summary_path}"
        )
    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if disk_summary != dict(summary):
        raise ValueError("in-memory and persisted overfit32 training summaries differ")
    final_dir = output_dir / "final"
    if not final_dir.is_dir() or final_dir.is_symlink():
        raise FileNotFoundError(
            f"overfit32 final checkpoint is not a regular directory: {final_dir}"
        )
    file_evidence: dict[str, Any] = {}
    for name in _REQUIRED_CHECKPOINT_FILES:
        path = final_dir / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"required overfit32 checkpoint artifact is missing: {path}"
            )
        file_evidence[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _overfit_gate_sha256_file(path),
        }
    metadata = json.loads((final_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("overfit32 checkpoint metadata must be a mapping")
    validate_checkpoint_metadata(config, metadata)
    expected_metadata = {
        "config_sha256": config.config_sha256,
        "cache_manifest_sha256": cache_identity["aggregate_manifest_sha256"],
        "camera_order": list(config.camera_order),
        "camera_input_sizes": {
            key: list(value) for key, value in config.camera_input_sizes.items()
        },
        "training_state": dict(summary),
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"overfit32 checkpoint metadata mismatch: {mismatches}")
    return {
        "train_summary": {
            "path": "train_summary.json",
            "sha256": _overfit_gate_sha256_file(summary_path),
        },
        "checkpoint": {
            "path": "final",
            "tree_sha256": _sha256_tree(final_dir),
            "files": file_evidence,
        },
    }


GATE_MAX_LOSS_RATIO = 0.5


def _loss_check(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    try:
        initial_mean = float(initial["mean"])
        final_mean = float(final["mean"])
    except (KeyError, TypeError, ValueError):
        initial_mean = math.nan
        final_mean = math.nan
    ratio = (
        final_mean / initial_mean
        if math.isfinite(initial_mean)
        and math.isfinite(final_mean)
        and (initial_mean > 0.0)
        and (final_mean > 0.0)
        else math.nan
    )
    return {
        "passed": math.isfinite(ratio) and ratio <= GATE_MAX_LOSS_RATIO,
        "metric": "mean_per_record_weighted_causal_nll",
        "initial": initial_mean if math.isfinite(initial_mean) else None,
        "final": final_mean if math.isfinite(final_mean) else None,
        "ratio": ratio if math.isfinite(ratio) else None,
        "maximum_ratio": GATE_MAX_LOSS_RATIO,
        "comparison": "ratio <= maximum_ratio",
    }


GATE_OPTIMIZER_STEPS = 100
GATE_EXAMPLES = 200


def _counter_check(summary: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "optimizer_steps": GATE_OPTIMIZER_STEPS,
        "batches": GATE_BATCHES,
        "examples": GATE_EXAMPLES,
    }
    observed = {key: summary.get(key) for key in expected}
    return {"passed": observed == expected, "expected": expected, "observed": observed}


def _gradient_check(audit: Any) -> dict[str, Any]:
    expected_stage = "after_backward_before_clipping"
    details = dict(audit) if isinstance(audit, Mapping) else None
    passed = bool(
        details is not None
        and details.get("passed") is True
        and (details.get("stage") == expected_stage)
        and (details.get("optimizer_step") == 1)
        and (details.get("accumulated_batches") == 2)
        and (details.get("unexpected_trainable") == [])
        and isinstance(details.get("lora"), Mapping)
        and (details["lora"].get("nonzero_gradient_tensors", 0) > 0)
        and (not details["lora"].get("missing_gradient"))
        and (not details["lora"].get("nonfinite_gradient"))
        and isinstance(details.get("projector"), Mapping)
        and (details["projector"].get("nonzero_gradient_tensors", 0) > 0)
        and (not details["projector"].get("missing_gradient"))
        and (not details["projector"].get("nonfinite_gradient"))
    )
    return {
        "passed": passed,
        "required_stage": expected_stage,
        "required_optimizer_step": 1,
        "required_accumulated_batches": 2,
        "observed": details,
    }


def _memory_check(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw_peak = summary.get("peak_cuda_memory_gib")
    peak = (
        float(raw_peak)
        if isinstance(raw_peak, int | float) and (not isinstance(raw_peak, bool))
        else math.nan
    )
    return {
        "passed": math.isfinite(peak) and 0.0 < peak <= GATE_MAX_PEAK_CUDA_MEMORY_GIB,
        "maximum_gib": GATE_MAX_PEAK_CUDA_MEMORY_GIB,
        "observed_gib": peak if math.isfinite(peak) else None,
    }


def assess_gate_checks(
    *,
    summary: Mapping[str, Any],
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    artifacts_passed: bool,
) -> dict[str, Any]:
    checks = {
        "loss_reduction": _loss_check(initial, final),
        "training_counters": _counter_check(summary),
        "gradient_flow": _gradient_check(summary.get("gradient_flow_audit")),
        "peak_cuda_memory": _memory_check(summary),
        "checkpoint_artifacts": {"passed": bool(artifacts_passed)},
    }
    return {
        "passed": all((check["passed"] is True for check in checks.values())),
        "checks": checks,
    }


GATE_SCHEMA_VERSION = 1
GATE_KIND = "qwen2_vl_grounded_sft_overfit32_gate"
PRIMARY_GROUNDED_CONFIG_FILENAME = "endovla_bbox_action_sft.yaml"
_PEAK_MEMORY_FAILURE_PATTERN = re.compile(
    "peak CUDA memory ([0-9]+(?:\\.[0-9]+)?) GiB exceeds the ([0-9]+(?:\\.[0-9]+)?) GiB gate",
    flags=re.IGNORECASE,
)


def _memory_failure_evidence(error: Mapping[str, Any]) -> dict[str, Any]:
    error_type = str(error.get("type", ""))
    message = str(error.get("message", ""))
    lowered = message.lower()
    if "cuda" in lowered and (
        error_type.endswith("OutOfMemoryError") or "out of memory" in lowered
    ):
        return {
            "eligible_fallback": True,
            "kind": "cuda_out_of_memory",
            "observed_peak_gib": None,
            "limit_gib": GATE_MAX_PEAK_CUDA_MEMORY_GIB,
        }
    match = _PEAK_MEMORY_FAILURE_PATTERN.search(message)
    if match is not None:
        observed = float(match.group(1))
        limit = float(match.group(2))
        eligible = (
            math.isfinite(observed)
            and math.isfinite(limit)
            and (observed > GATE_MAX_PEAK_CUDA_MEMORY_GIB)
            and (limit == GATE_MAX_PEAK_CUDA_MEMORY_GIB)
        )
        return {
            "eligible_fallback": eligible,
            "kind": "peak_cuda_memory_exceeded" if eligible else None,
            "observed_peak_gib": observed,
            "limit_gib": limit,
        }
    return {
        "eligible_fallback": False,
        "kind": None,
        "observed_peak_gib": None,
        "limit_gib": GATE_MAX_PEAK_CUDA_MEMORY_GIB,
    }


def _criteria_payload() -> dict[str, Any]:
    return {
        "sample_count": GATE_SAMPLE_COUNT,
        "optimizer_steps": GATE_OPTIMIZER_STEPS,
        "batches": GATE_BATCHES,
        "examples": GATE_EXAMPLES,
        "maximum_loss_ratio": GATE_MAX_LOSS_RATIO,
        "maximum_peak_cuda_memory_gib": GATE_MAX_PEAK_CUDA_MEMORY_GIB,
    }


def validate_primary_gate_failure_artifact(
    primary_config: ARExperimentConfig, artifact_path: str | Path, *, dataset: Any
) -> dict[str, Any]:
    """Validate one exact full-resolution memory failure authorizing fallback."""
    source = Path(artifact_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"primary gate failure artifact does not exist: {source}"
        )
    _require_gate_output_separation(primary_config, source)
    if _resolution_profile(primary_config) != "full_resolution":
        raise ValueError(
            "primary gate failure must use the registered full-resolution config"
        )
    payload = _read_json_mapping(source, "primary gate failure artifact")
    expected_keys = {
        "schema_version",
        "kind",
        "state",
        "passed",
        "identity",
        "criteria",
        "error",
        "failure_evidence",
    }
    if set(payload) != expected_keys:
        raise ValueError("primary gate failure artifact fields changed")
    if (
        payload.get("schema_version") != GATE_SCHEMA_VERSION
        or payload.get("kind") != GATE_KIND
        or payload.get("state") != "failed"
        or (payload.get("passed") is not False)
    ):
        raise ValueError("primary gate artifact is not a failed overfit32 gate")
    expected_identity = _expected_gate_identity(
        primary_config, dataset, primary_gate_failure=None
    )
    if payload.get("identity") != expected_identity:
        raise ValueError(
            "primary gate failure identity does not match the full config/data"
        )
    if payload.get("criteria") != _criteria_payload():
        raise ValueError("primary gate failure criteria changed")
    error = _mapping(payload.get("error"), "primary gate failure error")
    if set(error) != {"type", "message"}:
        raise ValueError("primary gate failure error fields changed")
    evidence = _memory_failure_evidence(error)
    if payload.get("failure_evidence") != evidence:
        raise ValueError("primary gate failure evidence does not match its exception")
    if evidence["eligible_fallback"] is not True:
        raise ValueError(
            "primary gate failure does not prove CUDA OOM or peak memory >76 GiB"
        )
    return {
        "path": str(source),
        "sha256": _overfit_gate_sha256_file(source),
        "config_sha256": primary_config.config_sha256,
        "resolution_profile": "full_resolution",
        "failure_kind": evidence["kind"],
        "failure_evidence": evidence,
        "identity_sha256": _overfit_gate_canonical_sha256(expected_identity),
        "selection_sha256": expected_identity["selection"]["sha256"],
        "training_order_sha256": expected_identity["training_order"]["sha256"],
    }


def _primary_failure_binding_for_fallback(
    config: ARExperimentConfig, value: Any, *, dataset: Any
) -> dict[str, Any]:
    from comparisons.ar_data import load_ar_config

    if not isinstance(value, Mapping):
        raise TypeError("fallback gate identity must bind a primary gate failure")
    path = value.get("path")
    primary_config_path = config.source_path.with_name(PRIMARY_GROUNDED_CONFIG_FILENAME)
    primary_config = load_ar_config(primary_config_path)
    binding = validate_primary_gate_failure_artifact(
        primary_config, str(path), dataset=dataset
    )
    if dict(value) != binding:
        raise ValueError("fallback gate primary-failure binding changed")
    return binding


def validate_overfit_gate_artifact(
    config: ARExperimentConfig,
    artifact_path: str | Path,
    *,
    dataset: Any,
    production_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and content-bind one passed gate for grounded production SFT."""
    source = Path(artifact_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"overfit gate artifact does not exist: {source}")
    _require_gate_output_separation(config, source, production_output=production_output)
    payload = _read_json_mapping(source, "overfit gate artifact")
    expected_keys = {
        "schema_version",
        "kind",
        "state",
        "passed",
        "identity",
        "criteria",
        "evaluation",
        "training_summary",
        "checks",
        "artifacts",
        "error",
    }
    if set(payload) != expected_keys:
        raise ValueError("passed overfit gate artifact fields changed")
    if (
        payload.get("schema_version") != GATE_SCHEMA_VERSION
        or payload.get("kind") != GATE_KIND
        or payload.get("state") != "passed"
        or (payload.get("passed") is not True)
        or (payload.get("error") is not None)
    ):
        raise ValueError("production SFT requires a passed overfit32 gate artifact")
    identity = _mapping(payload.get("identity"), "overfit gate identity")
    profile = _resolution_profile(config)
    primary_binding = None
    if profile == "registered_76g_fallback":
        primary_binding = _primary_failure_binding_for_fallback(
            config, identity.get("primary_gate_failure"), dataset=dataset
        )
        _require_gate_output_separation(
            config, Path(primary_binding["path"]), production_output=production_output
        )
    elif identity.get("primary_gate_failure") is not None:
        raise ValueError("full-resolution gate must not bind a primary failure")
    expected_identity = _expected_gate_identity(
        config, dataset, primary_gate_failure=primary_binding
    )
    if identity != expected_identity:
        raise ValueError(
            "overfit gate config/cache/camera/seed/selection identity changed"
        )
    if payload.get("criteria") != _criteria_payload():
        raise ValueError("overfit gate criteria changed")
    evaluation = _mapping(payload.get("evaluation"), "overfit gate evaluation")
    if set(evaluation) != {"initial", "final"}:
        raise ValueError("overfit gate evaluation fields changed")
    initial = _validate_nll_report(evaluation["initial"], "overfit gate initial NLL")
    final = _validate_nll_report(evaluation["final"], "overfit gate final NLL")
    summary = _validate_training_summary(payload.get("training_summary"), source.parent)
    expected_artifacts = _collect_artifact_evidence(
        source.parent,
        config=config,
        summary=summary,
        cache_identity=expected_identity["cache"],
    )
    if payload.get("artifacts") != expected_artifacts:
        raise ValueError("overfit gate checkpoint/summary hashes changed")
    assessment = assess_gate_checks(
        summary=summary, initial=initial, final=final, artifacts_passed=True
    )
    if (
        assessment["passed"] is not True
        or payload.get("checks") != assessment["checks"]
    ):
        raise ValueError(
            "overfit gate run/checks do not satisfy the registered criteria"
        )
    run_payload = {
        "criteria": payload["criteria"],
        "evaluation": evaluation,
        "training_summary": summary,
        "checks": assessment["checks"],
        "artifacts": expected_artifacts,
    }
    return {
        "path": str(source),
        "sha256": _overfit_gate_sha256_file(source),
        "config_sha256": expected_identity["config"]["sha256"],
        "cache_identity_sha256": _overfit_gate_canonical_sha256(
            expected_identity["cache"]
        ),
        "camera_sha256": _overfit_gate_canonical_sha256(expected_identity["camera"]),
        "seed": expected_identity["seed"],
        "resolution_profile": profile,
        "identity_sha256": _overfit_gate_canonical_sha256(expected_identity),
        "selection_sha256": identity["selection"]["sha256"],
        "training_order_sha256": identity["training_order"]["sha256"],
        "checks_sha256": _overfit_gate_canonical_sha256(assessment["checks"]),
        "run_sha256": _overfit_gate_canonical_sha256(run_payload),
        "checkpoint_tree_sha256": expected_artifacts["checkpoint"]["tree_sha256"],
        "train_summary_sha256": expected_artifacts["train_summary"]["sha256"],
        "primary_gate_failure": primary_binding,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from comparisons.ar_data import load_ar_config, verify_data_bindings
    from comparisons.ar_model import (
        load_training_bundle,
        load_training_bundle_from_adapter,
    )

    args = build_arg_parser().parse_args(argv)
    config = load_ar_config(args.config)
    if config.uses_grounded_action:
        if args.limit_train_samples is not None or args.max_steps is not None:
            raise ValueError(
                "grounded diagnostic training must use `python -m comparisons.ar_train overfit`"
            )
        if args.overfit_gate is None:
            raise ValueError("grounded production SFT requires --overfit-gate")
    elif args.overfit_gate is not None:
        raise ValueError("--overfit-gate applies only to grounded action SFT")
    bindings = verify_data_bindings(config)
    seed_everything(int(config.training["seed"]))
    dataset = build_train_dataset(config, verify=not args.skip_data_file_verification)
    _log(
        {
            "event": "data_ready",
            "bindings": bindings.to_dict(),
            "train_examples": len(dataset),
            "ignored_train_records": dataset.ignored_record_count,
        }
    )
    destination = (
        args.output_dir.resolve() if args.output_dir is not None else config.output_dir
    )
    overfit_gate_binding = None
    if config.uses_grounded_action:
        from comparisons.ar_train import (
            validate_overfit_gate_artifact as validate_overfit_gate_artifact,
        )

        assert args.overfit_gate is not None
        overfit_gate_binding = validate_overfit_gate_artifact(
            config, args.overfit_gate, dataset=dataset, production_output=destination
        )
        if args.resume_checkpoint is not None:
            validate_grounded_resume_checkpoint(
                config,
                args.resume_checkpoint,
                output_dir=destination,
                overfit_gate_binding=overfit_gate_binding,
            )
        _log({"event": "overfit_gate_authorized", **overfit_gate_binding})
    loader_kwargs = {
        "device": str(config.training["device"]),
        "local_files_only": args.local_files_only,
        "cache_dir": args.cache_dir,
    }
    bundle = (
        load_training_bundle(config, **loader_kwargs)
        if args.resume_checkpoint is None
        else load_training_bundle_from_adapter(
            config, args.resume_checkpoint, **loader_kwargs
        )
    )
    _log({"event": "model_ready", "parameters": bundle.parameters.to_dict()})
    loader = build_train_loader(
        config, dataset, bundle.processor, limit_train_samples=args.limit_train_samples
    )
    summary = fit(
        config,
        bundle,
        loader,
        max_steps=args.max_steps,
        output_dir=destination,
        resume_checkpoint=args.resume_checkpoint,
        overfit_gate_binding=overfit_gate_binding,
    )
    _log({"event": "training_complete", **summary.to_dict()})
    return 0


def evaluate_weighted_nll(
    bundle: QwenARBundle, loader: Any, *, device: str
) -> dict[str, Any]:
    """Evaluate per-record weighted causal NLL in loader order without mutation."""
    import torch

    was_training = bool(bundle.model.training)
    losses: list[float] = []
    generator_seed = getattr(loader, "_ar_qwen_generator_seed", None)
    if generator_seed is not None:
        _reset_dataloader_generator(loader, seed=int(generator_seed), epoch=0)
    bundle.model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                moved = _move_batch(batch, device)
                model_inputs = dict(moved)
                loss_weights = model_inputs.pop("loss_weights", None)
                labels = model_inputs.pop("labels", None)
                if loss_weights is None or labels is None:
                    raise RuntimeError(
                        "overfit32 requires grounded action token weights and labels"
                    )
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda")
                    else nullcontext()
                )
                with autocast:
                    output = bundle.model(**model_inputs)
                    logits = getattr(output, "logits", None)
                    if logits is None:
                        raise RuntimeError(
                            "Qwen model omitted logits during overfit32 evaluation"
                        )
                    loss = weighted_causal_lm_loss(logits, labels, loss_weights)
                value = float(loss.detach().cpu().item())
                if not math.isfinite(value) or value <= 0.0:
                    raise RuntimeError(
                        f"overfit32 evaluation produced invalid weighted NLL: {value}"
                    )
                losses.append(value)
    finally:
        bundle.model.train(was_training)
    if len(losses) != GATE_SAMPLE_COUNT:
        raise RuntimeError(
            f"overfit32 evaluated {len(losses)} records, expected {GATE_SAMPLE_COUNT}"
        )
    return {
        "metric": "mean_per_record_weighted_causal_nll",
        "record_count": GATE_SAMPLE_COUNT,
        "mean": math.fsum(losses) / GATE_SAMPLE_COUNT,
        "minimum": min(losses),
        "maximum": max(losses),
        "per_record": losses,
        "per_record_sha256": _overfit_gate_canonical_sha256(losses),
    }


GATE_FILENAME = "overfit_gate.json"


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _build_sequential_eval_loader(
    config: ARExperimentConfig, dataset: Any, processor: Any, indices: Sequence[int]
) -> Any:
    import torch

    if len(indices) != GATE_SAMPLE_COUNT:
        raise ValueError(
            f"overfit32 evaluation requires exactly {GATE_SAMPLE_COUNT} indices"
        )
    subset = torch.utils.data.Subset(dataset, [int(index) for index in indices])
    workers = int(config.training["num_workers"])
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=QwenSFTCollator(
            processor,
            max_sequence_length=int(config.training["max_sequence_length"]),
            action_numeric_loss_weight=float(
                config.training["action_numeric_loss_weight"]
            ),
        ),
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=_make_dataloader_generator(int(config.training["seed"])),
    )
    loader._ar_qwen_generator_seed = int(config.training["seed"])
    return loader


def _build_sequential_train_loader(
    config: ARExperimentConfig, dataset: Any, processor: Any, indices: Sequence[int]
) -> Any:
    """Use each fixed record once per loader epoch in the selected order."""
    return _build_sequential_eval_loader(config, dataset, processor, indices)


def _assert_training_subset(loader: Any, expected_indices: Sequence[int]) -> None:
    import torch

    if getattr(loader, "batch_size", None) != 1 or bool(
        getattr(loader, "drop_last", False)
    ):
        raise RuntimeError(
            "overfit32 training loader must use batch_size=1 without drop_last"
        )
    selected = getattr(getattr(loader, "dataset", None), "indices", None)
    if selected is None:
        raise RuntimeError(
            "overfit32 training loader does not expose its fixed subset indices"
        )
    actual = [int(index) for index in selected]
    expected = [int(index) for index in expected_indices]
    if actual != expected:
        raise RuntimeError(
            "overfit32 training and sequential evaluation subsets differ"
        )
    if not isinstance(
        getattr(loader, "sampler", None), torch.utils.data.SequentialSampler
    ):
        raise RuntimeError(
            "overfit32 training loader must use sequential no-replacement order"
        )


def _failed_report(context: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    error_payload = {"type": type(error).__name__, "message": str(error)}
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "kind": GATE_KIND,
        "state": "failed",
        "passed": False,
        "identity": dict(context),
        "criteria": _criteria_payload(),
        "error": error_payload,
        "failure_evidence": _memory_failure_evidence(error_payload),
    }


def run_overfit_gate(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    primary_gate_failure: str | Path | None = None,
) -> dict[str, Any]:
    """Run the fixed overfit32 protocol on a fresh output path."""
    from comparisons.ar_data import load_ar_config, verify_data_bindings

    destination = Path(output_dir).resolve()
    artifact_path = destination / GATE_FILENAME
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite a nonempty overfit32 output; preserve or quarantine it and choose a fresh --output-dir: {destination}"
        )
    context: dict[str, Any] = {}
    try:
        config = load_ar_config(config_path)
        _require_gate_output_separation(config, artifact_path)
        resolution_profile = _require_registered_gate_contract(config)
        initial_config_sha256 = config.config_sha256
        bindings = verify_data_bindings(config).to_dict()
        cache_identity = _cache_identity(config)
        seed = int(config.training["seed"])
        seed_everything(seed)
        dataset = build_train_dataset(config, verify=True)
        indices, selection = _selection_identity(dataset, seed=seed)
        training_order = _training_order_identity(indices)
        primary_failure_binding = None
        if resolution_profile == "registered_76g_fallback":
            if primary_gate_failure is None:
                raise ValueError(
                    "fallback overfit32 requires --primary-gate-failure proving full-res OOM"
                )
            primary_config = load_ar_config(
                config.source_path.with_name(PRIMARY_GROUNDED_CONFIG_FILENAME)
            )
            primary_failure_binding = validate_primary_gate_failure_artifact(
                primary_config, primary_gate_failure, dataset=dataset
            )
            _require_gate_output_separation(
                config, Path(primary_failure_binding["path"])
            )
        elif primary_gate_failure is not None:
            raise ValueError(
                "full-resolution overfit32 must not use --primary-gate-failure"
            )
        context = _identity_context(
            config,
            config_sha256=initial_config_sha256,
            resolution_profile=resolution_profile,
            cache_identity=cache_identity,
            selection=selection,
            training_order=training_order,
            data_bindings=bindings,
            primary_gate_failure=primary_failure_binding,
        )
        from comparisons.ar_model import load_training_bundle as load_training_bundle

        bundle = load_training_bundle(
            config,
            device=str(config.training["device"]),
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        eval_loader = _build_sequential_eval_loader(
            config, dataset, bundle.processor, indices
        )
        initial = evaluate_weighted_nll(
            bundle, eval_loader, device=str(config.training["device"])
        )
        train_loader = _build_sequential_train_loader(
            config, dataset, bundle.processor, indices
        )
        if len(train_loader) != GATE_SAMPLE_COUNT:
            raise RuntimeError(
                f"overfit32 training loader has {len(train_loader)} batches, expected {GATE_SAMPLE_COUNT}"
            )
        _assert_training_subset(train_loader, indices)
        train_summary = fit(
            config,
            bundle,
            train_loader,
            max_steps=GATE_OPTIMIZER_STEPS,
            output_dir=destination,
            diagnostic_overfit=True,
        )
        final = evaluate_weighted_nll(
            bundle, eval_loader, device=str(config.training["device"])
        )
        if _overfit_gate_sha256_file(config.source_path) != initial_config_sha256:
            raise RuntimeError("overfit32 config changed while the gate was running")
        summary = train_summary.to_dict()
        artifacts = _collect_artifact_evidence(
            destination, config=config, summary=summary, cache_identity=cache_identity
        )
        assessment = assess_gate_checks(
            summary=summary, initial=initial, final=final, artifacts_passed=True
        )
        passed = bool(assessment["passed"])
        report = {
            "schema_version": GATE_SCHEMA_VERSION,
            "kind": GATE_KIND,
            "state": "passed" if passed else "failed",
            "passed": passed,
            "identity": context,
            "criteria": _criteria_payload(),
            "evaluation": {"initial": initial, "final": final},
            "training_summary": summary,
            "checks": assessment["checks"],
            "artifacts": artifacts,
            "error": None,
        }
    except _GateOutputCollisionError:
        raise
    except Exception as error:
        report = _failed_report(context, error)
    _atomic_write_json(artifact_path, report)
    return report


def overfit_gate_build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--primary-gate-failure", type=Path)
    return parser


def overfit_gate_main(argv: Sequence[str] | None = None) -> int:
    args = overfit_gate_build_arg_parser().parse_args(argv)
    report = run_overfit_gate(
        args.config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        primary_gate_failure=args.primary_gate_failure,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["passed"] is True else 1


def cli(argv=None):
    from endo_bbox_vla.utils import run_cli

    return run_cli({"sft": main, "overfit": overfit_gate_main}, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
