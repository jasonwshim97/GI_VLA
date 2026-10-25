"""Static and temporal head training with fixed experiment contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from endo_bbox_vla.data import EXPECTED_FULL_CLIPS

if TYPE_CHECKING:
    from endo_bbox_vla.cache import (
        GrootTemporalCache,
        GrootTemporalPromptAssignment,
        LoadedConditioningFeatures,
        LoadedVisualFeatures,
    )
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.data import (
        ScenarioEpisodeBalancedSampler,
        SizeLocationBalancedSampler,
        StaticFrameRecord,
    )
    from endo_bbox_vla.loss import TemporalDetectionCriterion
    from endo_bbox_vla.network import GrootTemporalFeaturePacket

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


REPO_ROOT = PROJECT_ROOT

VARIANT_NAME = re.compile("^[a-z][a-z0-9_]*$")

TrainingVariant = str

EVAL_CACHE_KIND = "groot_n17_actual_prompt_selection_cache_complete"

TRAIN_CACHE_KIND = "groot_n17_multilevel_cache_complete"

EXPECTED_ACTION_SHAPE = [1, 40, 16]


def _scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(total_steps * warmup_fraction)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _cxcywh_to_original(box: Tensor) -> list[float] | None:
    from endo_bbox_vla.data import GROOT_N17_GEOMETRY

    value = box.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if value.numel() != 4 or not bool(torch.isfinite(value).all()):
        return None
    center, size = (value[:2], value[2:])
    xyxy = torch.cat((center - size / 2, center + size / 2)).clamp(0, 1)
    if bool((xyxy[2:] <= xyxy[:2]).any()):
        return None
    width, height = GROOT_N17_GEOMETRY.output_size
    scale = xyxy.new_tensor((width, height, width, height))
    original = GROOT_N17_GEOMETRY.inverse_xyxy((xyxy * scale).unsqueeze(0))[0]
    if bool((original[2:] <= original[:2]).any()):
        return None
    return [float(value) for value in original]


def _visible_gt(record: StaticFrameRecord) -> tuple[list[float] | None, float, bool]:
    from endo_bbox_vla.data import GROOT_N17_GEOMETRY

    if record.bbox_xyxy is None:
        return (None, 0.0, False)
    projection = GROOT_N17_GEOMETRY.project_xyxy(
        (record.bbox_xyxy,), partial_original=record.partial
    )
    if bool(projection.ignore[0]):
        return (None, 0.0, True)
    return (
        [float(value) for value in projection.visible_boxes_xyxy[0]],
        float(projection.visible_fraction[0]),
        bool(projection.partial_transform[0]),
    )


def train_static_detector(
    config: ExperimentConfig, *, variant: TrainingVariant, device: str, resume: bool
) -> dict[str, Any]:
    from endo_bbox_vla.config import _variant_config
    from endo_bbox_vla.data import (
        ScenarioEpisodeBalancedSampler,
        SharedFeatureTable,
        SizeLocationBalancedSampler,
        _evaluation_bindings,
        _load_cache_contract,
        _training_bindings,
    )
    from endo_bbox_vla.evaluate import _predict, _selection_key, _validation_report
    from endo_bbox_vla.loss import _build_criterion
    from endo_bbox_vla.network import _build_head
    from endo_bbox_vla.utils import (
        _atomic_json,
        _atomic_jsonl,
        _atomic_torch_save,
        _restore_rng_state,
        _rng_state,
        _seed_everything,
        _variant_root,
        sha256_file,
    )

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("shared detector training requires CUDA")
    training = config.section("training")
    variant_config = _variant_config(config, variant)
    seed = int(training["seed"])
    _seed_everything(seed)
    train_contract = _load_cache_contract(config, evaluation=False)
    evaluation_contract = _load_cache_contract(config, evaluation=True)
    if train_contract.model_sha256 != evaluation_contract.model_sha256:
        raise ValueError(
            "train and validation shared tokens came from different GR00T models"
        )
    train_table = SharedFeatureTable(
        train_contract,
        _training_bindings(config, train_contract),
        progress_label="train",
    )
    validation_table = SharedFeatureTable(
        evaluation_contract,
        _evaluation_bindings(config, evaluation_contract, "validation"),
        progress_label="validation",
    )
    head = _build_head(config, variant, torch_device)
    initialized_from: dict[str, Any] | None = None
    source_variant = variant_config.get("initialize_from")
    if source_variant:
        source_path = _variant_root(config, str(source_variant)) / "best.pt"
        source = torch.load(source_path, map_location=torch_device, weights_only=True)
        head.load_state_dict(source["head"], strict=True)
        initialized_from = {
            "variant": str(source_variant),
            "checkpoint": str(source_path),
            "checkpoint_sha256": sha256_file(source_path),
        }
    criterion = _build_criterion(config, variant).to(torch_device)
    sampler_kind = str(variant_config["sampler"])
    if sampler_kind == "scenario_episode_balanced":
        sampler: ScenarioEpisodeBalancedSampler | SizeLocationBalancedSampler = (
            ScenarioEpisodeBalancedSampler(
                train_table.records,
                seed=seed,
                samples_per_epoch=int(training["samples_per_epoch"]),
            )
        )
    elif sampler_kind == "size_location_balanced":
        sampler = SizeLocationBalancedSampler(
            train_table.records,
            train_table.box_weights,
            seed=seed,
            samples_per_epoch=int(training["samples_per_epoch"]),
        )
    else:
        raise ValueError(f"unsupported shared detector sampler: {sampler_kind}")
    loader = DataLoader(
        train_table,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(variant_config["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(variant_config["epochs"])
    scheduler = _scheduler(
        optimizer,
        total_steps=epochs * len(loader),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    root = _variant_root(config, variant)
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    report_path = root / "train_report.json"
    metrics_path = root / "train_metrics.jsonl"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    start_epoch = 0
    global_step = 0
    best_key: tuple[float, float, float, int] | None = None
    if last_path.is_file():
        if not resume:
            raise FileExistsError(
                f"partial shared detector training exists: {last_path}"
            )
        state = torch.load(last_path, map_location=torch_device, weights_only=True)
        head.load_state_dict(state["head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_key = None if state["best_key"] is None else tuple(state["best_key"])
        _restore_rng_state(state["rng_state"])
    elif metrics_path.exists() or best_path.exists():
        raise FileExistsError(
            "incomplete shared detector artifacts exist without last.pt"
        )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        head.train()
        totals = defaultdict(float)
        batches = 0
        for batch in loader:
            tokens = batch["tokens"].to(torch_device, non_blocking=True)
            mask = batch["padding_mask"].to(torch_device, non_blocking=True)
            targets: dict[str, Tensor] = {
                "presence": batch["presence"].to(torch_device, non_blocking=True),
                "boxes_cxcywh": batch["boxes_cxcywh"].to(
                    torch_device, non_blocking=True
                ),
            }
            if bool(variant_config.get("use_box_weights", True)):
                targets["box_weights"] = batch["box_weights"].to(
                    torch_device, non_blocking=True
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = criterion(head(tokens, mask), targets)
            loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite shared detector {variant} loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(training["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"non-finite shared detector {variant} gradient"
                )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            for name in (
                "loss",
                "presence_loss",
                "l1_loss",
                "giou_loss",
                "heatmap_loss",
            ):
                totals[name] += float(losses[name].detach())
        validation: dict[str, Any] | None = None
        if (epoch + 1) % int(
            training["validate_every_epochs"]
        ) == 0 or epoch + 1 == epochs:
            rows = _predict(
                head,
                validation_table,
                variant=variant,
                device=torch_device,
                batch_size=int(training["evaluation_batch_size"]),
            )
            validation = _validation_report(config, rows)
            validation.update(
                {"variant": variant, "epoch": epoch, "global_step": global_step}
            )
            _atomic_json(root / f"validation_epoch_{epoch + 1:03d}.json", validation)
            key = _selection_key(validation)
            if best_key is None or key > best_key:
                best_key = key
                metadata = {
                    "schema_version": 1,
                    "kind": "shared_e03_location_first_detector_head",
                    "variant": variant,
                    "epoch": epoch,
                    "global_step": global_step,
                    "config_sha256": config.sha256,
                    "base_model_sha256": train_contract.model_sha256,
                    "train_cache_manifest_sha256": train_contract.manifest_sha256,
                    "validation_cache_manifest_sha256": evaluation_contract.manifest_sha256,
                    "action_parity_sha256": train_contract.action_parity_sha256,
                    "feature_source": config.section("backbone")["feature_source"],
                    "feature_capture": "same_action_forward",
                    "visual_parameters_trainable": False,
                    "serialized_parameters": "detector_head_only",
                    "head_kind": str(variant_config.get("head_kind", "location_first")),
                    "num_queries": int(head.num_queries),
                    "focal_presence": bool(variant_config.get("focal_presence", True)),
                    "heatmap_weight": float(
                        variant_config.get(
                            "heatmap_weight",
                            config.section("loss")["heatmap"]
                            if head.kind == "location_first"
                            else 0.0,
                        )
                    ),
                    "sampler": sampler_kind,
                    "use_box_weights": bool(
                        variant_config.get("use_box_weights", True)
                    ),
                    "initialized_from": initialized_from,
                    "validation": validation,
                    "internal_test_used": False,
                }
                _atomic_torch_save(
                    best_path, {"head": head.state_dict(), "metadata": metadata}
                )
                _atomic_jsonl(root / "best_validation_predictions.jsonl", rows)
        epoch_row = {
            "variant": variant,
            "epoch": epoch,
            "global_step": global_step,
            "batches": batches,
            **{name: value / max(batches, 1) for name, value in totals.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.monotonic() - started,
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_row, sort_keys=True, allow_nan=False) + "\n")
        _atomic_torch_save(
            last_path,
            {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_key": best_key,
                "rng_state": _rng_state(),
            },
        )
        print(json.dumps(epoch_row, sort_keys=True, allow_nan=False), flush=True)
    expected_steps = epochs * len(loader)
    if global_step != expected_steps or not best_path.is_file():
        raise RuntimeError(
            f"shared detector training incomplete: {global_step} != {expected_steps}"
        )
    report = {
        "schema_version": 1,
        "kind": "shared_e03_training_report",
        "variant": variant,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "global_step": global_step,
        "termination_reason": "training_complete",
        "head_kind": head.kind,
        "num_queries": head.num_queries,
        "use_box_weights": bool(variant_config.get("use_box_weights", True)),
        "sampler": sampler.audit(),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "feature_capture": "same_action_forward",
        "visual_parameters_trainable": False,
        "internal_test_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(report_path, report)
    return report


GrootTemporalArm = Literal["mid_only", "dit_conditioned"]

FULL_SEED = 42

FULL_EPOCHS = 30

FULL_BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class GrootTemporalFullTrainingConfig:
    feature_mode: GrootTemporalArm
    seed: int = FULL_SEED
    epochs: int = FULL_EPOCHS
    batch_size: int = FULL_BATCH_SIZE
    learning_rate: float = 3e-05
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    hidden_dim: int = 256
    num_heads: int = 8
    num_points: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    conditioning_dropout: float = 0.2
    stale_probability: float = 0.5
    stale_reference_jitter: float = 0.15
    stale_confidence: float = 0.99
    teacher_hold_epochs: int = 10
    bfloat16_autocast: bool = True
    expected_clip_count: int = EXPECTED_FULL_CLIPS
    lru_rows: int = 512
    protocol: bool = True

    def __post_init__(self) -> None:
        from endo_bbox_vla.data import EXPECTED_FULL_CLIPS

        if self.feature_mode not in {"mid_only", "dit_conditioned"}:
            raise ValueError("unsupported full temporal feature mode")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.hidden_dim,
                self.num_heads,
                self.num_points,
                self.dim_feedforward,
                self.expected_clip_count,
                self.lru_rows,
            )
            <= 0
        ):
            raise ValueError("full temporal training dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.protocol and (
            self.seed != FULL_SEED
            or self.epochs != FULL_EPOCHS
            or self.batch_size != FULL_BATCH_SIZE
            or (self.expected_clip_count != EXPECTED_FULL_CLIPS)
            or (self.teacher_hold_epochs != 10)
            or (not self.bfloat16_autocast)
        ):
            raise ValueError(
                "protocol full training is fixed to seed42/30 epochs/batch16/BF16"
            )

    @property
    def experiment_id(self) -> str:
        return (
            "T1_e03_temporal_full"
            if self.feature_mode == "mid_only"
            else "T2_e03_dit_temporal_full"
        )


@dataclass(frozen=True, slots=True)
class GrootTemporalFullTrainingResult:
    head: nn.Module
    report: Mapping[str, Any]


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class GrootTemporalCachedFrame:
    assignment: GrootTemporalPromptAssignment
    record: StaticFrameRecord
    target: Mapping[str, Tensor]
    visual: LoadedVisualFeatures
    conditioning: LoadedConditioningFeatures | None


@dataclass(frozen=True, slots=True)
class GrootTemporalCachedClip:
    clip_id: str
    source_category: str
    frames: tuple[
        GrootTemporalCachedFrame,
        GrootTemporalCachedFrame,
        GrootTemporalCachedFrame,
        GrootTemporalCachedFrame,
    ]


@dataclass(frozen=True, slots=True)
class FullFrameDescriptor:
    assignment: GrootTemporalPromptAssignment
    record: StaticFrameRecord
    target: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class FullClipDescriptor:
    clip_id: str
    source_category: str
    frames: tuple[
        FullFrameDescriptor,
        FullFrameDescriptor,
        FullFrameDescriptor,
        FullFrameDescriptor,
    ]


class LazyFullCachedClipIndex:
    """Metadata-resident full dataset with a bounded verified feature LRU."""

    def __init__(
        self,
        clips: Sequence[FullClipDescriptor],
        cache: GrootTemporalCache,
        *,
        feature_mode: GrootTemporalArm,
        lru_rows: int = 512,
    ) -> None:
        if feature_mode not in {"mid_only", "dit_conditioned"}:
            raise ValueError("unsupported full temporal feature mode")
        if lru_rows <= 0:
            raise ValueError("lru_rows must be positive")
        self.clips = tuple(clips)
        self.cache = cache
        self.feature_mode = feature_mode
        self._visual = lru_cache(maxsize=lru_rows)(self._load_visual)
        self._conditioning = lru_cache(maxsize=lru_rows)(self._load_conditioning)

    def __len__(self) -> int:
        return len(self.clips)

    def _load_visual(self, sample_id: str) -> Any:
        return self.cache.load_visual(sample_id)

    def _load_conditioning(
        self, sample_id: str, prompt_id: str, generation: int, selected_prompt: str
    ) -> Any:
        return self.cache.load_conditioning(
            (sample_id, prompt_id, generation), selected_prompt=selected_prompt
        )

    def load_clip(self, index: int) -> GrootTemporalCachedClip:
        descriptor = self.clips[index]
        frames: list[GrootTemporalCachedFrame] = []
        for frame in descriptor.frames:
            assignment = frame.assignment
            conditioning = (
                None
                if self.feature_mode == "mid_only"
                else self._conditioning(
                    assignment.sample_id,
                    assignment.prompt_id,
                    assignment.context_generation,
                    assignment.selected_prompt,
                )
            )
            frames.append(
                GrootTemporalCachedFrame(
                    assignment=assignment,
                    record=frame.record,
                    target=frame.target,
                    visual=self._visual(assignment.sample_id),
                    conditioning=conditioning,
                )
            )
        return GrootTemporalCachedClip(
            clip_id=descriptor.clip_id,
            source_category=descriptor.source_category,
            frames=tuple(frames),
        )

    def load_batch(self, indices: Sequence[int]) -> list[GrootTemporalCachedClip]:
        return [self.load_clip(index) for index in indices]


CLIP_LENGTH = 4

SOURCE_CATEGORIES = (
    "task4_positive_anchor",
    "task4_pure_negative",
    "task1_hard_negative_anchor",
    "task1_random_pure_negative",
)


def _validate_assignment_record(
    assignment: GrootTemporalPromptAssignment, record: StaticFrameRecord
) -> None:
    if record.split != "train":
        raise ValueError("GR00T temporal smoke targets must come from split=train")
    expected = {
        "source_dataset": record.source_dataset,
        "source_episode_index": record.source_episode_index,
        "source_frame_index": record.source_frame_index,
        "dataset_index": record.dataset_index,
        "frame_id": record.frame_index,
        "detector_generation": record.detector_generation,
        "status": record.status,
    }
    drift = {
        name: {"plan": getattr(assignment, name), "record": value}
        for name, value in expected.items()
        if getattr(assignment, name) != value
    }
    if not math.isclose(
        assignment.timestamp, record.timestamp, rel_tol=0.0, abs_tol=1e-06
    ):
        drift["timestamp"] = {"plan": assignment.timestamp, "record": record.timestamp}
    if drift:
        raise ValueError(f"authoritative smoke-plan frame provenance drifted: {drift}")


TASK1_NEGATIVE_CATEGORIES = frozenset(
    {"task1_hard_negative_anchor", "task1_random_pure_negative"}
)


def build_full_cached_clip_index(
    *,
    full_plan: Mapping[str, Any],
    temporal_frame_manifest: str | Path | Sequence[StaticFrameRecord],
    cache: GrootTemporalCache,
    feature_mode: GrootTemporalArm,
    expected_clip_count: int = EXPECTED_FULL_CLIPS,
    lru_rows: int = 512,
) -> LazyFullCachedClipIndex:
    from endo_bbox_vla.cache import read_full_manifest_prompt_assignments
    from endo_bbox_vla.config import (
        contracts_TASK1_SOURCE_DATASET,
        contracts_TASK4_SOURCE_DATASET,
    )
    from endo_bbox_vla.data import (
        FULL_PLAN_KIND,
        build_groot_n17_target,
        read_frame_manifest,
    )

    if full_plan.get("kind") != FULL_PLAN_KIND:
        raise ValueError("full temporal trainer requires the authoritative full plan")
    summary = full_plan.get("summary")
    if not isinstance(summary, Mapping) or (
        summary.get("full_clip_count") != expected_clip_count
        or summary.get("evaluation_overlap_checked") is not True
        or summary.get("evaluation_physical_frame_overlap") != 0
        or (summary.get("evaluation_source_episode_overlap") != 0)
    ):
        raise ValueError("full plan did not prove locked-test exclusion")
    rows = full_plan.get("clips")
    if not isinstance(rows, list) or len(rows) != expected_clip_count:
        raise ValueError(
            f"full temporal plan requires {expected_clip_count} clips, got {(len(rows) if isinstance(rows, list) else 'invalid')}"
        )
    records = (
        read_frame_manifest(temporal_frame_manifest)
        if isinstance(temporal_frame_manifest, (str, Path))
        else tuple(temporal_frame_manifest)
    )
    records_by_id = {record.sample_id: record for record in records}
    assignments = read_full_manifest_prompt_assignments(full_plan)
    by_occurrence = {
        (assignment.clip_id, assignment.frame_offset): assignment
        for assignment in assignments
    }
    if len(by_occurrence) != expected_clip_count * CLIP_LENGTH:
        raise ValueError("full temporal plan repeats or omits frame occurrences")
    descriptors: list[FullClipDescriptor] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("clip"), Mapping):
            raise TypeError("full temporal clip row is invalid")
        metadata = row["clip"]
        clip_id = str(metadata.get("clip_id", ""))
        category = str(metadata.get("source_category", ""))
        sample_ids = metadata.get("sample_ids")
        if category not in SOURCE_CATEGORIES:
            raise ValueError(f"unsupported full temporal category {category!r}")
        if (
            metadata.get("split") != "train"
            or not isinstance(sample_ids, list)
            or len(sample_ids) != CLIP_LENGTH
        ):
            raise ValueError("full temporal clip must contain four train frames")
        frames: list[FullFrameDescriptor] = []
        for frame_offset, sample_id in enumerate(sample_ids):
            assignment = by_occurrence[clip_id, frame_offset]
            if assignment.sample_id != sample_id:
                raise ValueError(
                    "full plan clip sample IDs disagree with frame assignments"
                )
            record = records_by_id[assignment.sample_id]
            _validate_assignment_record(assignment, record)
            expected_source = (
                contracts_TASK1_SOURCE_DATASET
                if category in TASK1_NEGATIVE_CATEGORIES
                else contracts_TASK4_SOURCE_DATASET
            )
            if record.source_dataset != expected_source or record.split != "train":
                raise ValueError("full temporal category/source/split contract drifted")
            target = build_groot_n17_target(record)
            if bool(target["ignore"]):
                raise ValueError("full plan contains a crop-invisible positive")
            frames.append(
                FullFrameDescriptor(assignment=assignment, record=record, target=target)
            )
        descriptors.append(
            FullClipDescriptor(
                clip_id=clip_id, source_category=category, frames=tuple(frames)
            )
        )
    if len({clip.clip_id for clip in descriptors}) != expected_clip_count:
        raise ValueError("full temporal plan repeats clip IDs")
    return LazyFullCachedClipIndex(
        descriptors, cache, feature_mode=feature_mode, lru_rows=lru_rows
    )


class DeterministicPaddedBatchSampler:
    """Shuffle every clip once and repeat only enough rows to fill the last batch."""

    def __init__(self, count: int, *, batch_size: int, seed: int) -> None:
        if count <= 0 or batch_size <= 0 or seed < 0:
            raise ValueError("sampler count/batch_size/seed are invalid")
        self.count = int(count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.count / self.batch_size)

    @property
    def padded_rows_per_epoch(self) -> int:
        return self.steps_per_epoch * self.batch_size - self.count

    def batches(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch * 1000003)
        values = torch.randperm(self.count, generator=generator).tolist()
        if self.padded_rows_per_epoch:
            values.extend(values[: self.padded_rows_per_epoch])
        batches = tuple(
            (
                tuple(values[start : start + self.batch_size])
                for start in range(0, len(values), self.batch_size)
            )
        )
        if len(batches) != self.steps_per_epoch or any(
            (len(batch) != self.batch_size for batch in batches)
        ):
            raise RuntimeError("deterministic sampler emitted an incomplete epoch")
        if set(values[: self.count]) != set(range(self.count)):
            raise RuntimeError("deterministic sampler did not expose every clip")
        return batches


def _module_state_sha256(module: nn.Module) -> str:
    """Hash exact initialized detector tensors without pickle metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\x00")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(
    *,
    config: GrootTemporalFullTrainingConfig,
    plan_sha256: str,
    cache: GrootTemporalCache,
    initial_sha256: str,
    steps_per_epoch: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "groot_temporal_full_training_progress",
        "experiment_id": config.experiment_id,
        "feature_mode": config.feature_mode,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "plan_sha256": plan_sha256,
        "cache_schema_sha256": cache.schema.sha256,
        "cache_model_revision": cache.schema.model_revision,
        "cache_e03_manifest_sha256": _sha256_file(cache.visual_manifest_path),
        "cache_conditioning_manifest_sha256": _sha256_file(
            cache.conditioning_manifest_path
        ),
        "initial_shared_detector_tensors_sha256": initial_sha256,
    }


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256(
        "\x00".join((str(part) for part in parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


@dataclass(frozen=True, slots=True)
class LinearTeacherForcingSchedule:
    initial_probability: float
    final_probability: float
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        for name in ("initial_probability", "final_probability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.initial_probability < self.final_probability:
            raise ValueError("teacher forcing probability must not increase")
        if not 0 <= self.start_step < self.end_step:
            raise ValueError("teacher forcing requires 0 <= start_step < end_step")

    def probability(self, step: int) -> float:
        if step <= self.start_step:
            return float(self.initial_probability)
        if step >= self.end_step:
            return float(self.final_probability)
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return float(
            self.initial_probability
            + progress * (self.final_probability - self.initial_probability)
        )

    def mask(
        self,
        sample_keys: Sequence[str],
        *,
        step: int,
        transition_index: int,
        seed: int,
        device: torch.device | str,
    ) -> Tensor:
        probability = self.probability(step)
        values = [
            _unit_interval("teacher", seed, step, transition_index, key) < probability
            for key in sample_keys
        ]
        return torch.tensor(values, device=device, dtype=torch.bool)


@dataclass(frozen=True, slots=True)
class StaleMemoryConfig:
    probability: float
    reference_jitter: float
    confidence: float = 0.99
    frame_delta: int = 1
    time_delta_seconds: float = 1.0 / 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("stale-memory probability must be in [0,1]")
        if (
            not math.isfinite(self.reference_jitter)
            or not 0 <= self.reference_jitter <= 0.5
        ):
            raise ValueError("stale-memory jitter must be in [0,0.5]")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("stale-memory confidence must be in [0,1]")
        if self.frame_delta <= 0 or self.time_delta_seconds <= 0:
            raise ValueError("stale-memory frame/time deltas must be positive")


train_groot_temporal_GrootTemporalArm = Literal["mid_only", "dit_conditioned"]


def _gradient_groups(
    feature_mode: train_groot_temporal_GrootTemporalArm,
) -> dict[str, tuple[str, ...]]:
    groups = {
        "adapter_local": ("adapter.mid_projection",),
        "spatial_position": ("spatial_position",),
        "coarse_heatmap": ("coarse_heatmap",),
        "local_context": ("local_context_projection",),
        "query_embeddings": ("query_embeddings",),
        "temporal_memory_projection": ("memory_projection",),
        "temporal_time_projection": ("time_projection",),
        "decoder": ("decoder",),
        "presence": ("presence_head",),
        "validity": ("validity_head",),
        "bbox": ("box_head",),
        "motion": ("motion_head",),
    }
    if feature_mode == "dit_conditioned":
        groups["adapter_conditioning"] = (
            "adapter.conditioning_projection",
            "adapter.semantic_gate_mlp",
        )
    return groups


def _atomic_torch_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_targets(
    clips: Sequence[GrootTemporalCachedClip], *, frame_offset: int, device: torch.device
) -> dict[str, Tensor]:
    frames = [clip.frames[frame_offset] for clip in clips]
    presence = torch.stack([frame.target["presence"] for frame in frames])
    box_mask = torch.stack([frame.target["box_supervision"] for frame in frames]).to(
        dtype=torch.bool
    )
    boxes: list[Tensor] = []
    for frame, supervised in zip(frames, box_mask, strict=True):
        value = torch.as_tensor(frame.target["boxes_cxcywh"], dtype=torch.float32)
        if bool(supervised):
            if tuple(value.shape) == (1, 4):
                value = value[0]
            elif tuple(value.shape) != (4,):
                raise ValueError(
                    "supervised temporal target must contain exactly one box"
                )
            boxes.append(value)
        else:
            if tuple(value.shape) == (0, 4):
                value = value.new_zeros(4)
            elif tuple(value.shape) != (4,) or bool((value != 0).any()):
                raise ValueError("unsupervised temporal target must not contain a box")
            boxes.append(value)
    return {
        "presence": presence.to(device),
        "boxes": torch.stack(boxes).to(device),
        "box_mask": box_mask.to(device),
    }


def stale_memory_mask(
    sample_keys: Sequence[str],
    *,
    first_presence: Tensor,
    config: StaleMemoryConfig,
    step: int,
    seed: int,
    eligibility_mask: Tensor | None = None,
) -> Tensor:
    presence = torch.as_tensor(first_presence).reshape(len(sample_keys))
    eligibility = (
        torch.ones(len(sample_keys), device=presence.device, dtype=torch.bool)
        if eligibility_mask is None
        else torch.as_tensor(
            eligibility_mask, device=presence.device, dtype=torch.bool
        ).reshape(len(sample_keys))
    )
    selected = [
        bool(eligibility[index])
        and float(presence[index]) <= 0.5
        and (_unit_interval("stale", seed, step, key) < config.probability)
        for index, key in enumerate(sample_keys)
    ]
    return torch.tensor(selected, device=presence.device, dtype=torch.bool)


def make_stale_clip_start_state(
    head: Any,
    *,
    frame_ids: Tensor,
    timestamps: Tensor,
    detector_generation: Tensor,
    first_presence: Tensor,
    sample_keys: Sequence[str],
    config: StaleMemoryConfig,
    step: int,
    seed: int,
    eligibility_mask: Tensor | None = None,
) -> Any | None:
    """Create an explicit false-positive memory only for negative clip starts."""
    from endo_bbox_vla.network import TemporalQueryState as TemporalQueryState

    query_template = head.reset_propagated_query
    if not isinstance(query_template, Tensor) or query_template.ndim != 1:
        raise TypeError("head.reset_propagated_query must be a rank-one query tensor")
    if not query_template.is_floating_point() or not bool(
        torch.isfinite(query_template).all()
    ):
        raise ValueError(
            "head.reset_propagated_query must be finite and floating point"
        )
    batch_size = len(sample_keys)
    selected = stale_memory_mask(
        sample_keys,
        first_presence=first_presence,
        config=config,
        step=step,
        seed=seed,
        eligibility_mask=eligibility_mask,
    ).to(query_template.device)
    if not bool(selected.any()):
        return None
    references = []
    for key in sample_keys:
        x = (
            0.5
            + (2 * _unit_interval("stale-x", seed, step, key) - 1)
            * config.reference_jitter
        )
        y = (
            0.5
            + (2 * _unit_interval("stale-y", seed, step, key) - 1)
            * config.reference_jitter
        )
        references.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
    reference_points = query_template.new_tensor(references)
    reference_box = torch.cat(
        (reference_points, query_template.new_full((batch_size, 2), 0.1)), dim=1
    )
    frame_ids = frame_ids.to(query_template.device, dtype=torch.long).reshape(
        batch_size
    )
    timestamps = timestamps.to(
        query_template.device, dtype=query_template.dtype
    ).reshape(batch_size)
    generation = detector_generation.to(
        query_template.device, dtype=torch.long
    ).reshape(batch_size)
    return TemporalQueryState(
        query=query_template.expand(batch_size, -1).clone(),
        reference_points=reference_points,
        reference_box=reference_box,
        confidence=query_template.new_full((batch_size,), config.confidence),
        validity=selected,
        age=query_template.new_zeros(batch_size),
        generation=generation,
        frame_ids=frame_ids - config.frame_delta,
        timestamps=(timestamps - config.time_delta_seconds).clamp_min(0),
    )


def _audit_nonzero_gradients(
    head: nn.Module, seen: dict[str, bool], *, groups: Mapping[str, Sequence[str]]
) -> tuple[int, int, float]:
    gradient_count = 0
    nonzero_count = 0
    squared_norm = 0.0
    for name, parameter in head.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite detector gradient in {name}")
        norm = float(gradient.detach().float().norm())
        squared_norm += norm * norm
        if norm > 0:
            nonzero_count += 1
            for group, prefixes in groups.items():
                if any((name.startswith(prefix) for prefix in prefixes)):
                    seen[group] = True
    total_norm = math.sqrt(squared_norm)
    if gradient_count == 0 or nonzero_count == 0 or (not math.isfinite(total_norm)):
        raise RuntimeError("detector backward produced no finite nonzero gradients")
    return (gradient_count, nonzero_count, total_norm)


def _square_shape(number_of_tokens: int, name: str) -> tuple[int, int]:
    side = math.isqrt(number_of_tokens)
    if side * side != number_of_tokens:
        raise ValueError(f"{name} token count must form a square grid")
    return (side, side)


def build_groot_temporal_feature_packet(
    clips: Sequence[GrootTemporalCachedClip],
    *,
    frame_offset: int,
    feature_mode: train_groot_temporal_GrootTemporalArm,
    device: torch.device | str,
) -> GrootTemporalFeaturePacket:
    """Stack one frame position from a clip batch into the model contract."""
    from endo_bbox_vla.network import GrootTemporalFeaturePacket

    if not clips:
        raise ValueError("cannot build a packet from an empty clip batch")
    if not 0 <= frame_offset < CLIP_LENGTH:
        raise ValueError("frame_offset is outside the four-frame clip")
    target_device = torch.device(device)
    frames = [clip.frames[frame_offset] for clip in clips]
    mid_tokens = torch.stack([frame.visual.tokens for frame in frames])
    if target_device.type == "cpu":
        mid_tokens = mid_tokens.float()
    mid_tokens = mid_tokens.to(target_device)
    mid_masks = torch.stack([frame.visual.padding_mask for frame in frames]).to(
        target_device
    )
    if mid_tokens.requires_grad:
        raise RuntimeError("cached visual tensors must remain detached")
    frame_ids = torch.tensor(
        [frame.assignment.frame_id for frame in frames],
        device=target_device,
        dtype=torch.long,
    )
    timestamps = torch.tensor(
        [frame.assignment.timestamp for frame in frames],
        device=target_device,
        dtype=torch.float32,
    )
    detector_generations = torch.tensor(
        [frame.assignment.detector_generation for frame in frames],
        device=target_device,
        dtype=torch.long,
    )
    common = {
        "mid_tokens": mid_tokens,
        "mid_padding_mask": mid_masks,
        "mid_spatial_shape": _square_shape(mid_tokens.shape[1], "block-11 visual"),
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "prompt_id": tuple((frame.assignment.prompt_id for frame in frames)),
        "detector_generation": detector_generations,
        "mid_feature_dim": int(mid_tokens.shape[2]),
    }
    if feature_mode == "mid_only":
        return GrootTemporalFeaturePacket(**common)
    conditioning = [frame.conditioning for frame in frames]
    if any((value is None for value in conditioning)):
        raise RuntimeError(
            "VL-conditioned temporal packets require conditioning for every batch row"
        )
    available = [value for value in conditioning if value is not None]
    conditioning_tokens = torch.stack([value.tokens for value in available])
    if target_device.type == "cpu":
        conditioning_tokens = conditioning_tokens.float()
    conditioning_tokens = conditioning_tokens.to(target_device)
    conditioning_masks = torch.stack([value.padding_mask for value in available]).to(
        target_device
    )
    sequence_positions = torch.stack(
        [value.sequence_positions for value in available]
    ).to(target_device)
    if conditioning_tokens.requires_grad:
        raise RuntimeError("cached DiT conditioning must remain detached")
    context_generations = torch.tensor(
        [frame.assignment.context_generation for frame in frames],
        device=target_device,
        dtype=torch.long,
    )
    return GrootTemporalFeaturePacket(
        **common,
        dit_conditioning_tokens=conditioning_tokens,
        dit_conditioning_padding_mask=conditioning_masks,
        dit_conditioning_spatial_shape=_square_shape(
            conditioning_tokens.shape[1], "DiT conditioning"
        ),
        dit_sequence_positions=sequence_positions,
        dit_source_frame_ids=frame_ids.clone(),
        dit_source_timestamps=timestamps.clone(),
        dit_prompt_id=tuple((frame.assignment.prompt_id for frame in frames)),
        dit_context_generation=context_generations,
        dit_conditioning_dim=int(conditioning_tokens.shape[2]),
    )


def teacher_force_state(
    state: Any,
    *,
    target_presence: Tensor,
    target_boxes_cxcywh: Tensor,
    selected: Tensor,
) -> Any:
    """Replace selected positive memory geometry while preserving its query.

    The operation is label-side only and never changes current-frame detector
    outputs.  Its result is fed as the previous state for the next frame.
    """
    if not dataclasses.is_dataclass(state):
        raise TypeError("state must be a dataclass instance")
    batch_size = int(state.query.shape[0])
    presence = torch.as_tensor(
        target_presence, device=state.query.device, dtype=state.query.dtype
    ).reshape(batch_size)
    boxes = torch.as_tensor(
        target_boxes_cxcywh, device=state.query.device, dtype=state.query.dtype
    ).reshape(batch_size, 4)
    selected = torch.as_tensor(
        selected, device=state.query.device, dtype=torch.bool
    ).reshape(batch_size)
    selected = selected & (presence > 0.5)
    if not bool(selected.any()):
        return state
    center = boxes[:, :2].clamp(0, 1)
    forced_reference = torch.where(selected[:, None], center, state.reference_points)
    replacements: dict[str, Tensor] = {"reference_points": forced_reference}
    fields = {field.name for field in dataclasses.fields(state)}
    if "reference_box" in fields:
        replacements["reference_box"] = torch.where(
            selected[:, None], boxes.clamp(0, 1), state.reference_box
        )
    replacements["confidence"] = torch.where(
        selected, torch.ones_like(state.confidence), state.confidence
    )
    validity_name = "validity" if "validity" in fields else "valid"
    replacements[validity_name] = (
        state.validity | selected
        if validity_name == "validity"
        else state.valid | selected
    )
    if "age" in fields:
        replacements["age"] = torch.where(
            selected, torch.zeros_like(state.age), state.age
        )
    return dataclasses.replace(state, **replacements)


def train_groot_temporal_full(
    *,
    full_plan: Mapping[str, Any],
    temporal_frame_manifest: str | Path | Sequence[StaticFrameRecord],
    cache: GrootTemporalCache,
    config: GrootTemporalFullTrainingConfig,
    device: str | torch.device = "cuda",
    progress_checkpoint_path: str | Path,
    progress_report_path: str | Path,
    resume: bool = True,
    expected_initial_sha256: str | None = None,
) -> GrootTemporalFullTrainingResult:
    """Train one arm, committing an exact epoch-boundary resume point atomically."""
    from endo_bbox_vla.loss import FP32BoxTemporalDetectionCriterion
    from endo_bbox_vla.network import build_groot_temporal_head

    target_device = torch.device(device)
    if target_device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA full temporal training requested but unavailable")
    plan_sha256 = _canonical_sha256(full_plan)
    dataset = build_full_cached_clip_index(
        full_plan=full_plan,
        temporal_frame_manifest=temporal_frame_manifest,
        cache=cache,
        feature_mode=config.feature_mode,
        expected_clip_count=config.expected_clip_count,
        lru_rows=config.lru_rows,
    )
    sampler = DeterministicPaddedBatchSampler(
        len(dataset), batch_size=config.batch_size, seed=config.seed
    )
    total_steps = sampler.steps_per_epoch * config.epochs
    torch.manual_seed(config.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    head = build_groot_temporal_head(
        feature_mode=config.feature_mode,
        mid_dim=cache.schema.visual_shape[1],
        dit_conditioning_dim=cache.schema.conditioning_shape[1],
        hidden_dim=config.hidden_dim,
        conditioning_dropout=config.conditioning_dropout,
        num_heads=config.num_heads,
        num_points=config.num_points,
        num_layers=2,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        memory_confidence_threshold=0.5,
        memory_gate_temperature=10.0,
        memory_time_decay_seconds=1.0,
        max_time_gap_seconds=1.0,
        max_frame_gap=30,
    ).to(target_device)
    initial_sha = _module_state_sha256(head)
    if expected_initial_sha256 is not None and initial_sha != expected_initial_sha256:
        raise RuntimeError(
            "full temporal arm did not match the shared initial detector SHA"
        )
    trainable = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = FP32BoxTemporalDetectionCriterion().to(target_device)
    identity = _checkpoint_identity(
        config=config,
        plan_sha256=plan_sha256,
        cache=cache,
        initial_sha256=initial_sha,
        steps_per_epoch=sampler.steps_per_epoch,
    )
    checkpoint_path = Path(progress_checkpoint_path)
    report_path = Path(progress_report_path)
    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    category_exposures: Counter[str] = Counter()
    repeated_clip_exposures = 0
    stale_selected = 0
    stale_eligible = 0
    teacher_selected_positive = 0
    gradient_seen = dict.fromkeys(_gradient_groups(config.feature_mode), False)
    causal_gradient_seen = [False] * (CLIP_LENGTH - 1)
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"full temporal progress checkpoint exists: {checkpoint_path}"
            )
        try:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("cannot load full temporal progress checkpoint") from exc
        if not isinstance(saved, Mapping) or saved.get("identity") != identity:
            raise FileExistsError(
                "full temporal progress checkpoint belongs to another run"
            )
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["next_epoch"])
        global_step = int(saved["global_step"])
        history = list(saved["history"])
        category_exposures.update(saved["category_exposures"])
        repeated_clip_exposures = int(saved["repeated_clip_exposures"])
        stale_selected = int(saved["stale_selected"])
        stale_eligible = int(saved["stale_eligible"])
        teacher_selected_positive = int(saved["teacher_selected_positive"])
        gradient_seen.update(saved["gradient_seen"])
        causal_gradient_seen = list(saved["causal_gradient_seen"])
        torch.set_rng_state(saved["cpu_rng_state"])
        if target_device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], target_device)
        head.to(target_device)
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, Tensor):
                    state[name] = value.to(target_device)
    elif report_path.exists():
        raise FileExistsError("full temporal report exists without a resume checkpoint")
    if not 0 <= start_epoch <= config.epochs:
        raise ValueError(
            "resume checkpoint next_epoch is outside the training schedule"
        )
    if global_step != start_epoch * sampler.steps_per_epoch:
        raise ValueError("resume checkpoint epoch/optimizer-step identity drifted")
    teacher_start = min(
        config.teacher_hold_epochs * sampler.steps_per_epoch, total_steps - 2
    )
    teacher = LinearTeacherForcingSchedule(
        initial_probability=0.5,
        final_probability=0.0,
        start_step=max(0, teacher_start),
        end_step=total_steps - 1,
    )
    stale = StaleMemoryConfig(
        probability=config.stale_probability,
        reference_jitter=config.stale_reference_jitter,
        confidence=config.stale_confidence,
    )
    amp_used = bool(
        config.bfloat16_autocast
        and target_device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    head.train()
    groups = _gradient_groups(config.feature_mode)
    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        epoch_components = Counter()
        epoch_batches = sampler.batches(epoch)
        repeated_clip_exposures += sampler.padded_rows_per_epoch
        for indices in epoch_batches:
            clips = dataset.load_batch(indices)
            category_exposures.update((clip.source_category for clip in clips))
            first_targets = _batch_targets(clips, frame_offset=0, device=target_device)
            frame_ids = torch.tensor(
                [clip.frames[0].assignment.frame_id for clip in clips],
                device=target_device,
                dtype=torch.long,
            )
            timestamps = torch.tensor(
                [clip.frames[0].assignment.timestamp for clip in clips],
                device=target_device,
                dtype=torch.float32,
            )
            generations = torch.tensor(
                [clip.frames[0].assignment.detector_generation for clip in clips],
                device=target_device,
                dtype=torch.long,
            )
            eligible = torch.tensor(
                [clip.source_category in TASK1_NEGATIVE_CATEGORIES for clip in clips],
                device=target_device,
                dtype=torch.bool,
            )
            stale_eligible += int(eligible.sum())
            previous_state = make_stale_clip_start_state(
                head,
                frame_ids=frame_ids,
                timestamps=timestamps,
                detector_generation=generations,
                first_presence=first_targets["presence"],
                sample_keys=tuple((clip.clip_id for clip in clips)),
                config=stale,
                step=global_step,
                seed=config.seed,
                eligibility_mask=eligible,
            )
            if previous_state is not None:
                stale_selected += int(previous_state.validity.sum())
            optimizer.zero_grad(set_to_none=True)
            frame_losses: list[Tensor] = []
            probes: list[Tensor] = []
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_used
                else nullcontext()
            )
            with autocast:
                for frame_offset in range(CLIP_LENGTH):
                    packet = build_groot_temporal_feature_packet(
                        clips,
                        frame_offset=frame_offset,
                        feature_mode=config.feature_mode,
                        device=target_device,
                    )
                    outputs, next_state = head(
                        packet, previous_state=previous_state, detach_next_state=False
                    )
                    if frame_offset < CLIP_LENGTH - 1:
                        next_state.query.retain_grad()
                        probes.append(next_state.query)
                    targets = _batch_targets(
                        clips, frame_offset=frame_offset, device=target_device
                    )
                    if frame_offset:
                        prior = _batch_targets(
                            clips, frame_offset=frame_offset - 1, device=target_device
                        )
                        previous_boxes = prior["boxes"]
                        previous_validity = (prior["presence"] > 0.5) & prior[
                            "box_mask"
                        ]
                    else:
                        previous_boxes = torch.zeros_like(targets["boxes"])
                        previous_validity = torch.zeros(
                            len(clips), device=target_device, dtype=torch.bool
                        )
                    losses = criterion(
                        outputs,
                        {
                            "presence": targets["presence"],
                            "validity": targets["presence"],
                            "boxes_cxcywh": targets["boxes"],
                            "gt_box_mask": targets["box_mask"],
                            "previous_boxes_cxcywh": previous_boxes,
                            "previous_box_validity": previous_validity,
                        },
                    )
                    for key in ("l1_loss", "giou_loss", "motion_loss"):
                        if losses[key].dtype != torch.float32:
                            raise RuntimeError(f"{key} was not computed in FP32")
                    frame_losses.append(losses["loss"])
                    for key in (
                        "presence_loss",
                        "validity_loss",
                        "l1_loss",
                        "giou_loss",
                        "heatmap_loss",
                        "motion_loss",
                        "auxiliary_loss",
                    ):
                        epoch_components[key] += (
                            float(losses[key].detach().float()) / CLIP_LENGTH
                        )
                    if frame_offset < CLIP_LENGTH - 1:
                        selected = teacher.mask(
                            tuple((clip.clip_id for clip in clips)),
                            step=global_step,
                            transition_index=frame_offset,
                            seed=config.seed,
                            device=target_device,
                        )
                        teacher_selected_positive += int(
                            (selected & (targets["presence"] > 0.5)).sum()
                        )
                        next_state = teacher_force_state(
                            next_state,
                            target_presence=targets["presence"],
                            target_boxes_cxcywh=targets["boxes"],
                            selected=selected,
                        )
                    previous_state = next_state
                loss = torch.stack(frame_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("full temporal loss became non-finite")
            loss.backward()
            for transition, probe in enumerate(probes):
                if (
                    probe.grad is not None
                    and bool(torch.isfinite(probe.grad).all())
                    and (float(probe.grad.float().norm()) > 0)
                ):
                    causal_gradient_seen[transition] = True
            _audit_nonzero_gradients(head, gradient_seen, groups=groups)
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().float())
            global_step += 1
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": global_step,
                "mean_loss": epoch_loss / sampler.steps_per_epoch,
                "loss_path_mean": {
                    key: value / sampler.steps_per_epoch
                    for key, value in sorted(epoch_components.items())
                },
                "teacher_forcing_probability_end": teacher.probability(global_step - 1),
            }
        )
        report = {
            **identity,
            "completed_epochs": epoch + 1,
            "optimizer_steps": global_step,
            "target_optimizer_steps": total_steps,
            "effective_batch_clips": config.batch_size,
            "clip_length": CLIP_LENGTH,
            "dataset_clip_count": len(dataset),
            "padded_clip_exposures": repeated_clip_exposures,
            "category_exposures": dict(sorted(category_exposures.items())),
            "teacher_forcing": {
                "hold_epochs": config.teacher_hold_epochs,
                "initial_probability": 0.5,
                "final_probability": teacher.probability(total_steps - 1),
                "selected_positive_transitions": teacher_selected_positive,
            },
            "stale_memory": {
                "task1_negative_only": True,
                "probability": config.stale_probability,
                "eligible_clip_exposures": stale_eligible,
                "selected_clip_exposures": stale_selected,
            },
            "precision": {
                "amp": "torch.bfloat16" if amp_used else "fp32_cpu_test",
                "bbox_l1_dtype": "torch.float32",
                "bbox_giou_dtype": "torch.float32",
                "motion_dtype": "torch.float32",
                "fp32_box_autocast_disabled": True,
            },
            "gradient_groups_seen": dict(gradient_seen),
            "causal_query_gradient_seen": causal_gradient_seen,
            "locked_test_excluded": True,
            "groot_model_loaded": False,
            "history": history,
        }
        checkpoint_payload = {
            "identity": identity,
            "next_epoch": epoch + 1,
            "global_step": global_step,
            "head": {
                name: value.detach().cpu() for name, value in head.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
            "category_exposures": dict(category_exposures),
            "repeated_clip_exposures": repeated_clip_exposures,
            "stale_selected": stale_selected,
            "stale_eligible": stale_eligible,
            "teacher_selected_positive": teacher_selected_positive,
            "gradient_seen": gradient_seen,
            "causal_gradient_seen": causal_gradient_seen,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(target_device)
            if target_device.type == "cuda"
            else torch.empty(0, dtype=torch.uint8),
        }
        _atomic_torch_replace(checkpoint_path, checkpoint_payload)
        _atomic_json_replace(report_path, report)
    if global_step != total_steps:
        raise RuntimeError("full temporal optimizer-step count differs from 30 epochs")
    if not all(gradient_seen.values()):
        missing = [name for name, seen in gradient_seen.items() if not seen]
        raise RuntimeError(f"full temporal gradient groups stayed zero: {missing}")
    if not all(causal_gradient_seen):
        raise RuntimeError("full temporal BPTT query gradients were not observed")
    if not history:
        raise RuntimeError("full temporal training completed without epoch history")
    final_report = json.loads(report_path.read_text(encoding="utf-8"))
    return GrootTemporalFullTrainingResult(head=head, report=final_report)


SMOKE_CLIPS_PER_CATEGORY = 4

SMOKE_STEPS_PER_EPOCH = 4

SMOKE_BATCH_SIZE = 16


class CategoryBalancedBatchSampler(Sampler[list[int]]):
    """Four clips from each of four source categories in every batch."""

    def __init__(
        self,
        categories: Sequence[str],
        *,
        seed: int = 42,
        clips_per_category: int = SMOKE_CLIPS_PER_CATEGORY,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler seed must be a non-negative integer")
        if (
            isinstance(clips_per_category, bool)
            or not isinstance(clips_per_category, int)
            or clips_per_category != SMOKE_CLIPS_PER_CATEGORY
        ):
            raise ValueError("smoke sampler requires exactly four clips per category")
        self.seed = seed
        self.clips_per_category = clips_per_category
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, category in enumerate(categories):
            grouped[str(category)].append(index)
        if set(grouped) != set(SOURCE_CATEGORIES):
            raise ValueError(
                "category-balanced sampler requires the four registered categories"
            )
        sizes = {category: len(grouped[category]) for category in SOURCE_CATEGORIES}
        if sizes != {category: 16 for category in SOURCE_CATEGORIES}:
            raise ValueError(
                "category-balanced smoke sampler requires 16 clips per category"
            )
        self._grouped = {
            category: tuple(grouped[category]) for category in SOURCE_CATEGORIES
        }
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return SMOKE_STEPS_PER_EPOCH

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1000003 * self.epoch)
        shuffled: dict[str, list[int]] = {}
        for category in SOURCE_CATEGORIES:
            values = self._grouped[category]
            order = torch.randperm(len(values), generator=generator).tolist()
            shuffled[category] = [values[index] for index in order]
        for batch_index in range(len(self)):
            start = batch_index * self.clips_per_category
            stop = start + self.clips_per_category
            batch = [
                index
                for category in SOURCE_CATEGORIES
                for index in shuffled[category][start:stop]
            ]
            if len(batch) != SMOKE_BATCH_SIZE:
                raise RuntimeError(
                    "category-balanced sampler produced an incomplete batch"
                )
            within_batch = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in within_batch]


class GrootTemporalCachedClipDataset(Dataset[GrootTemporalCachedClip]):
    """The 64 authoritative clips with cache tensors verified once up front."""

    def __init__(self, clips: Sequence[GrootTemporalCachedClip]) -> None:
        self.clips = tuple(clips)
        if len(self.clips) != 64:
            raise ValueError(
                "GR00T temporal smoke dataset must contain exactly 64 clips"
            )
        counts = Counter((clip.source_category for clip in self.clips))
        if counts != Counter({category: 16 for category in SOURCE_CATEGORIES}):
            raise ValueError(
                "GR00T temporal smoke dataset must contain 16 clips per category"
            )
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("GR00T temporal smoke dataset repeats clip_id")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> GrootTemporalCachedClip:
        return self.clips[index]

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple((clip.source_category for clip in self.clips))


def _read_plan(source: str | Path | Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        payload = source
    elif isinstance(source, (str, Path)):
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid GR00T temporal smoke plan: {source}") from exc
    else:
        to_dict = getattr(source, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "smoke_plan must be a path, mapping, or object with to_dict()"
            )
        payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("GR00T temporal smoke plan root must be a mapping")
    return payload


def _frame_records(
    source: str | Path | Sequence[StaticFrameRecord],
) -> dict[str, StaticFrameRecord]:
    from endo_bbox_vla.data import read_frame_manifest

    records = (
        read_frame_manifest(source)
        if isinstance(source, (str, Path))
        else tuple(source)
    )
    if not records:
        raise ValueError("temporal frame records must not be empty")
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("temporal frame records repeat sample_id")
    return by_id


def build_groot_temporal_cached_dataset(
    *,
    smoke_plan: str | Path | Mapping[str, Any] | Any,
    temporal_frame_manifest: str | Path | Sequence[StaticFrameRecord],
    cache: GrootTemporalCache,
    feature_mode: train_groot_temporal_GrootTemporalArm,
) -> GrootTemporalCachedClipDataset:
    """Join assigned plan rows, crop-aware targets, and verified cache payloads."""
    from endo_bbox_vla.cache import read_full_manifest_prompt_assignments
    from endo_bbox_vla.config import (
        contracts_TASK1_SOURCE_DATASET,
        contracts_TASK4_SOURCE_DATASET,
    )
    from endo_bbox_vla.data import build_groot_n17_target

    if feature_mode not in {"mid_only", "dit_conditioned"}:
        raise ValueError("feature_mode must be mid_only or dit_conditioned")
    plan = _read_plan(smoke_plan)
    assignments = read_full_manifest_prompt_assignments(plan)
    assignment_by_occurrence = {
        (assignment.clip_id, assignment.frame_offset): assignment
        for assignment in assignments
    }
    if len(assignment_by_occurrence) != len(assignments):
        raise ValueError("smoke plan repeats a clip/frame occurrence")
    missing = cache.missing_full_manifest_keys(plan)
    if missing["e03"]:
        raise KeyError(f"smoke cache is missing visual keys: {missing['e03'][:5]}")
    if feature_mode == "dit_conditioned" and missing["conditioning"]:
        raise KeyError(
            f"smoke cache is missing conditioning keys: {missing['conditioning'][:5]}"
        )
    records_by_id = _frame_records(temporal_frame_manifest)
    visual_by_sample: dict[str, LoadedVisualFeatures] = {}
    conditioning_by_occurrence: dict[tuple[str, int], LoadedConditioningFeatures] = {}
    if feature_mode == "dit_conditioned":
        loaded = cache.load_full_manifest(plan)
        if len(loaded) != len(assignments):
            raise RuntimeError(
                "cache full-manifest loader changed assignment cardinality"
            )
        for expected_assignment, value in zip(assignments, loaded, strict=True):
            if value.assignment != expected_assignment:
                raise ValueError(
                    "cache loader reordered or changed authoritative assignments"
                )
            visual_by_sample.setdefault(expected_assignment.sample_id, value.visual)
            conditioning_by_occurrence[
                expected_assignment.clip_id, expected_assignment.frame_offset
            ] = value.conditioning
    else:
        for assignment in assignments:
            if assignment.sample_id not in visual_by_sample:
                visual_by_sample[assignment.sample_id] = cache.load_visual(
                    assignment.sample_id
                )
    clip_rows = plan.get("clips")
    if not isinstance(clip_rows, list) or len(clip_rows) != 64:
        raise ValueError("authoritative smoke plan must contain exactly 64 clips")
    clips: list[GrootTemporalCachedClip] = []
    for clip_row in clip_rows:
        if not isinstance(clip_row, Mapping) or not isinstance(
            clip_row.get("clip"), Mapping
        ):
            raise TypeError("smoke-plan clip rows must contain clip metadata")
        clip_metadata = clip_row["clip"]
        clip_id = str(clip_metadata.get("clip_id", ""))
        category = str(clip_metadata.get("source_category", ""))
        if category not in SOURCE_CATEGORIES:
            raise ValueError(f"unsupported smoke source category {category!r}")
        if clip_metadata.get("split") != "train":
            raise ValueError("GR00T temporal smoke clips must use split=train")
        sample_ids = clip_metadata.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != CLIP_LENGTH:
            raise ValueError("smoke clip metadata must contain four sample_ids")
        frame_values: list[GrootTemporalCachedFrame] = []
        for frame_offset, sample_id in enumerate(sample_ids):
            key = (clip_id, frame_offset)
            try:
                assignment = assignment_by_occurrence[key]
            except KeyError as exc:
                raise KeyError(
                    f"smoke plan is missing frame assignment {key!r}"
                ) from exc
            if assignment.sample_id != sample_id:
                raise ValueError(
                    "clip sample_ids disagree with authoritative frame assignments"
                )
            try:
                record = records_by_id[assignment.sample_id]
            except KeyError as exc:
                raise KeyError(
                    f"temporal frame manifest lacks {assignment.sample_id!r}"
                ) from exc
            _validate_assignment_record(assignment, record)
            expected_source = (
                contracts_TASK1_SOURCE_DATASET
                if category in TASK1_NEGATIVE_CATEGORIES
                else contracts_TASK4_SOURCE_DATASET
            )
            if record.source_dataset != expected_source:
                raise ValueError(
                    "smoke source_category disagrees with its source dataset"
                )
            target = build_groot_n17_target(record)
            if bool(target["ignore"]):
                raise ValueError(
                    "smoke plan contains a crop-invisible positive; it must be reselected, not relabelled"
                )
            frame_values.append(
                GrootTemporalCachedFrame(
                    assignment=assignment,
                    record=record,
                    target=target,
                    visual=visual_by_sample[assignment.sample_id],
                    conditioning=conditioning_by_occurrence.get(key),
                )
            )
        statuses = [frame.record.status for frame in frame_values]
        if category == "task4_positive_anchor":
            anchor_sample_id = clip_metadata.get("anchor_sample_id")
            if anchor_sample_id not in sample_ids:
                raise ValueError(
                    "positive-anchor smoke clip is missing its anchor frame"
                )
            anchor_index = sample_ids.index(anchor_sample_id)
            if statuses[anchor_index] != "positive" or "positive" not in statuses:
                raise ValueError("positive-anchor smoke clip has a non-positive anchor")
            if any(
                (
                    status not in {"positive", "confirmed_negative"}
                    for status in statuses
                )
            ):
                raise ValueError(
                    "positive-anchor smoke clip contains non-trainable status"
                )
        elif statuses != ["confirmed_negative"] * CLIP_LENGTH:
            raise ValueError(
                "negative smoke clips must contain four confirmed-negative frames"
            )
        clips.append(
            GrootTemporalCachedClip(
                clip_id=clip_id, source_category=category, frames=tuple(frame_values)
            )
        )
    return GrootTemporalCachedClipDataset(clips)


SMOKE_EPOCHS = 5

SMOKE_OPTIMIZER_STEPS = 20


@dataclass(frozen=True, slots=True)
class GrootTemporalSmokeTrainingConfig:
    """Fixed smoke protocol with model-size knobs for CPU unit tests."""

    feature_mode: train_groot_temporal_GrootTemporalArm
    seed: int = 42
    epochs: int = SMOKE_EPOCHS
    batch_size: int = SMOKE_BATCH_SIZE
    clips_per_category: int = SMOKE_CLIPS_PER_CATEGORY
    learning_rate: float = 3e-05
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    hidden_dim: int = 256
    num_heads: int = 8
    num_points: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    conditioning_dropout: float = 0.2
    stale_probability: float = 0.5
    stale_reference_jitter: float = 0.15
    stale_confidence: float = 0.99
    teacher_initial_probability: float = 0.5
    teacher_final_probability: float = 0.0
    bfloat16_autocast: bool = True

    def __post_init__(self) -> None:
        if self.feature_mode not in {"mid_only", "dit_conditioned"}:
            raise ValueError(
                "feature_mode must be mid_only (visual memory) or dit_conditioned (VL-conditioned)"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            self.epochs != SMOKE_EPOCHS
            or self.batch_size != SMOKE_BATCH_SIZE
            or self.clips_per_category != SMOKE_CLIPS_PER_CATEGORY
        ):
            raise ValueError(
                "GR00T temporal smoke is fixed to 5 epochs and balanced batches of 16"
            )
        for name in (
            "epochs",
            "batch_size",
            "clips_per_category",
            "hidden_dim",
            "num_heads",
            "num_points",
            "dim_feedforward",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if (
            min(self.hidden_dim, self.num_heads, self.num_points, self.dim_feedforward)
            <= 0
        ):
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.conditioning_dropout < 1:
            raise ValueError("dropout probabilities must lie in [0,1)")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.stale_probability != 0.5:
            raise ValueError("smoke stale-memory probability is fixed to 0.5")
        if not 0 <= self.stale_reference_jitter <= 0.5:
            raise ValueError("stale_reference_jitter must lie in [0,0.5]")
        if not 0 <= self.stale_confidence <= 1:
            raise ValueError("stale_confidence must lie in [0,1]")
        if (
            self.teacher_initial_probability != 0.5
            or self.teacher_final_probability != 0.0
        ):
            raise ValueError("smoke teacher forcing must linearly decay from 0.5 to 0")

    @property
    def experiment_id(self) -> str:
        return (
            "T1_e03_temporal"
            if self.feature_mode == "mid_only"
            else "T2_e03_dit_temporal"
        )

    @property
    def optimizer_steps(self) -> int:
        return SMOKE_OPTIMIZER_STEPS

    def teacher_schedule(self) -> LinearTeacherForcingSchedule:
        return LinearTeacherForcingSchedule(
            initial_probability=self.teacher_initial_probability,
            final_probability=self.teacher_final_probability,
            start_step=0,
            end_step=self.optimizer_steps - 1,
        )

    def stale_config(self) -> StaleMemoryConfig:
        return StaleMemoryConfig(
            probability=self.stale_probability,
            reference_jitter=self.stale_reference_jitter,
            confidence=self.stale_confidence,
        )


@dataclass(frozen=True, slots=True)
class GrootTemporalSmokeTrainingResult:
    head: nn.Module
    report: Mapping[str, Any]


_LOSS_KEYS = (
    "presence_loss",
    "validity_loss",
    "l1_loss",
    "giou_loss",
    "heatmap_loss",
    "motion_loss",
    "auxiliary_loss",
)


def _train_groot_temporal_canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_groot_temporal_cached_smoke(
    *,
    smoke_plan: str | Path | Mapping[str, Any] | Any,
    temporal_frame_manifest: str | Path | Sequence[StaticFrameRecord],
    cache: GrootTemporalCache,
    config: GrootTemporalSmokeTrainingConfig,
    device: torch.device | str = "cuda",
    head: nn.Module | None = None,
    criterion: TemporalDetectionCriterion | None = None,
    report_path: str | Path | None = None,
) -> GrootTemporalSmokeTrainingResult:
    """Run exactly 20 cached optimizer steps without loading GR00T parameters."""
    from endo_bbox_vla.loss import TemporalDetectionCriterion
    from endo_bbox_vla.network import build_groot_temporal_head

    target_device = torch.device(device)
    if target_device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA smoke training requested but CUDA is unavailable")
    if target_device.type == "cuda" and (not config.bfloat16_autocast):
        raise RuntimeError(
            "CUDA GR00T temporal smoke requires BF16 autocast; disabling it would mix the BF16 frozen cache with FP32 detector parameters"
        )
    if target_device.type == "cuda" and (not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "CUDA GR00T temporal smoke requires hardware-supported BF16 autocast"
        )
    torch.manual_seed(config.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    plan = _read_plan(smoke_plan)
    dataset = build_groot_temporal_cached_dataset(
        smoke_plan=plan,
        temporal_frame_manifest=temporal_frame_manifest,
        cache=cache,
        feature_mode=config.feature_mode,
    )
    sampler = CategoryBalancedBatchSampler(
        dataset.categories,
        seed=config.seed,
        clips_per_category=config.clips_per_category,
    )
    if len(sampler) * config.epochs != config.optimizer_steps:
        raise RuntimeError("smoke sampler no longer yields exactly 20 optimizer steps")
    if head is None:
        head = build_groot_temporal_head(
            feature_mode=config.feature_mode,
            mid_dim=cache.schema.visual_shape[1],
            dit_conditioning_dim=cache.schema.conditioning_shape[1],
            hidden_dim=config.hidden_dim,
            conditioning_dropout=config.conditioning_dropout,
            num_heads=config.num_heads,
            num_points=config.num_points,
            num_layers=2,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            memory_confidence_threshold=0.5,
            memory_gate_temperature=10.0,
            memory_time_decay_seconds=1.0,
            max_time_gap_seconds=1.0,
            max_frame_gap=30,
        )
    head = head.to(target_device)
    if getattr(head, "feature_mode", None) != config.feature_mode:
        raise ValueError(
            "provided head feature_mode differs from the smoke training arm"
        )
    initial_detector_state_sha256 = _module_state_sha256(head)
    criterion = TemporalDetectionCriterion() if criterion is None else criterion
    criterion = criterion.to(target_device)
    parameter_names = [name for name, _ in head.named_parameters()]
    state_names = list(head.state_dict())
    forbidden_fragments = (
        "policy",
        "backbone",
        "qwen",
        "diffusion",
        "action_head",
        "vl_self_attention",
    )
    forbidden = [
        name
        for name in set(parameter_names) | set(state_names)
        if any((fragment in name.lower() for fragment in forbidden_fragments))
    ]
    if forbidden:
        raise ValueError(
            f"detector head contains forbidden GR00T parameter names: {forbidden[:5]}"
        )
    trainable = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("GR00T temporal detector head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    head_ids = {id(parameter) for parameter in trainable}
    if optimizer_ids != head_ids:
        raise RuntimeError(
            "optimizer parameters differ from detector trainable parameters"
        )
    teacher = config.teacher_schedule()
    stale = config.stale_config()
    amp_used = bool(
        config.bfloat16_autocast
        and target_device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    loss_totals = dict.fromkeys(_LOSS_KEYS, 0.0)
    loss_nonzero_steps = dict.fromkeys(_LOSS_KEYS, 0)
    gradient_groups = _gradient_groups(config.feature_mode)
    gradient_seen = dict.fromkeys(gradient_groups, False)
    maximum_gradient_norm = 0.0
    minimum_nonzero_gradient_parameters: int | None = None
    maximum_gradient_parameters = 0
    category_exposures: Counter[str] = Counter()
    stale_eligible = 0
    stale_selected = 0
    stale_selected_outside_eligibility = 0
    stale_gate_audited_batches = 0
    stale_gate_selected_rows = 0
    stale_gate_positive_selected_rows = 0
    stale_gate_nonselected_rows = 0
    stale_gate_zero_nonselected_rows = 0
    stale_gate_minimum_selected: float | None = None
    stale_gate_maximum_nonselected_absolute = 0.0
    teacher_selected_positive = 0
    bptt_graph_transitions = 0
    causal_query_gradient_steps = [0] * (CLIP_LENGTH - 1)
    causal_query_gradient_norm_sums = [0.0] * (CLIP_LENGTH - 1)
    causal_reference_box_gradient_steps = [0] * (CLIP_LENGTH - 1)
    causal_reference_box_gradient_norm_sums = [0.0] * (CLIP_LENGTH - 1)
    clip_boundary_state_resets = 0
    global_step = 0
    metrics: list[dict[str, Any]] = []
    head.train()
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        for indices in sampler:
            clips = [dataset[index] for index in indices]
            batch_counts = Counter((clip.source_category for clip in clips))
            if batch_counts != Counter(
                {category: config.clips_per_category for category in SOURCE_CATEGORIES}
            ):
                raise RuntimeError("sampler emitted a category-imbalanced batch")
            category_exposures.update(batch_counts)
            first_targets = _batch_targets(clips, frame_offset=0, device=target_device)
            frame_ids = torch.tensor(
                [clip.frames[0].assignment.frame_id for clip in clips],
                device=target_device,
                dtype=torch.long,
            )
            timestamps = torch.tensor(
                [clip.frames[0].assignment.timestamp for clip in clips],
                device=target_device,
                dtype=torch.float32,
            )
            generations = torch.tensor(
                [clip.frames[0].assignment.detector_generation for clip in clips],
                device=target_device,
                dtype=torch.long,
            )
            stale_eligibility = torch.tensor(
                [clip.source_category in TASK1_NEGATIVE_CATEGORIES for clip in clips],
                device=target_device,
                dtype=torch.bool,
            )
            stale_eligible += int(stale_eligibility.sum())
            previous_state = make_stale_clip_start_state(
                head,
                frame_ids=frame_ids,
                timestamps=timestamps,
                detector_generation=generations,
                first_presence=first_targets["presence"],
                sample_keys=tuple((clip.clip_id for clip in clips)),
                config=stale,
                step=global_step,
                seed=config.seed,
                eligibility_mask=stale_eligibility,
            )
            initial_stale_selected = torch.zeros(
                len(clips), device=target_device, dtype=torch.bool
            )
            if previous_state is not None:
                initial_stale_selected = previous_state.validity.detach().clone()
                selected_outside = initial_stale_selected & ~stale_eligibility
                stale_selected_outside_eligibility += int(selected_outside.sum())
                if bool(selected_outside.any()):
                    raise RuntimeError(
                        "stale-memory initializer selected a row outside Task1 negatives"
                    )
                stale_selected += int(initial_stale_selected.sum())
            optimizer.zero_grad(set_to_none=True)
            frame_losses: list[Tensor] = []
            step_loss_components = dict.fromkeys(_LOSS_KEYS, 0.0)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_used
                else nullcontext()
            )
            with autocast:
                causal_state_probes: list[tuple[Tensor, Tensor]] = []
                for frame_offset in range(CLIP_LENGTH):
                    packet = build_groot_temporal_feature_packet(
                        clips,
                        frame_offset=frame_offset,
                        feature_mode=config.feature_mode,
                        device=target_device,
                    )
                    outputs, next_state = head(
                        packet, previous_state=previous_state, detach_next_state=False
                    )
                    if frame_offset == 0:
                        gate = outputs.memory_gate.detach().float()
                        if tuple(gate.shape) != (len(clips),) or not bool(
                            torch.isfinite(gate).all()
                        ):
                            raise RuntimeError("frame-0 stale memory gate is invalid")
                        selected_gate = gate[initial_stale_selected]
                        nonselected_gate = gate[~initial_stale_selected]
                        if selected_gate.numel() and (
                            not bool((selected_gate > 0).all())
                        ):
                            raise RuntimeError(
                                "frame-0 stale-selected rows did not receive a positive memory gate"
                            )
                        if nonselected_gate.numel() and (
                            not bool((nonselected_gate == 0).all())
                        ):
                            raise RuntimeError(
                                "frame-0 nonselected rows received a nonzero stale memory gate"
                            )
                        stale_gate_audited_batches += 1
                        stale_gate_selected_rows += int(selected_gate.numel())
                        stale_gate_positive_selected_rows += int(
                            (selected_gate > 0).sum()
                        )
                        stale_gate_nonselected_rows += int(nonselected_gate.numel())
                        stale_gate_zero_nonselected_rows += int(
                            (nonselected_gate == 0).sum()
                        )
                        if selected_gate.numel():
                            minimum = float(selected_gate.min())
                            stale_gate_minimum_selected = (
                                minimum
                                if stale_gate_minimum_selected is None
                                else min(stale_gate_minimum_selected, minimum)
                            )
                        if nonselected_gate.numel():
                            stale_gate_maximum_nonselected_absolute = max(
                                stale_gate_maximum_nonselected_absolute,
                                float(nonselected_gate.abs().max()),
                            )
                    targets = _batch_targets(
                        clips, frame_offset=frame_offset, device=target_device
                    )
                    if frame_offset == 0:
                        previous_boxes = torch.zeros_like(targets["boxes"])
                        previous_validity = torch.zeros(
                            len(clips), device=target_device, dtype=torch.bool
                        )
                    else:
                        prior = _batch_targets(
                            clips, frame_offset=frame_offset - 1, device=target_device
                        )
                        previous_boxes = prior["boxes"]
                        previous_validity = (prior["presence"] > 0.5) & prior[
                            "box_mask"
                        ]
                    loss_dict = criterion(
                        outputs,
                        {
                            "presence": targets["presence"],
                            "validity": targets["presence"],
                            "boxes_cxcywh": targets["boxes"],
                            "gt_box_mask": targets["box_mask"],
                            "previous_boxes_cxcywh": previous_boxes,
                            "previous_box_validity": previous_validity,
                        },
                    )
                    frame_losses.append(loss_dict["loss"])
                    for key in _LOSS_KEYS:
                        value = float(loss_dict[key].detach().float())
                        if not math.isfinite(value) or value < 0:
                            raise FloatingPointError(
                                f"invalid {key} at step {global_step}"
                            )
                        step_loss_components[key] += value / CLIP_LENGTH
                    if frame_offset < CLIP_LENGTH - 1:
                        selected = teacher.mask(
                            tuple((clip.clip_id for clip in clips)),
                            step=global_step,
                            transition_index=frame_offset,
                            seed=config.seed,
                            device=target_device,
                        )
                        teacher_selected_positive += int(
                            (selected & (targets["presence"] > 0.5)).sum()
                        )
                        next_state = teacher_force_state(
                            next_state,
                            target_presence=targets["presence"],
                            target_boxes_cxcywh=targets["boxes"],
                            selected=selected,
                        )
                        if (
                            next_state.query.grad_fn is None
                            or next_state.reference_box.grad_fn is None
                        ):
                            raise RuntimeError(
                                "temporal state was detached inside a clip"
                            )
                        next_state.query.retain_grad()
                        next_state.reference_box.retain_grad()
                        causal_state_probes.append(
                            (next_state.query, next_state.reference_box)
                        )
                        bptt_graph_transitions += 1
                    previous_state = next_state
                loss = torch.stack(frame_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite smoke loss at step {global_step}")
            loss.backward()
            for transition_index, (query, reference_box) in enumerate(
                causal_state_probes
            ):
                query_gradient = query.grad
                if query_gradient is None or not bool(
                    torch.isfinite(query_gradient).all()
                ):
                    raise RuntimeError(
                        f"later-frame loss did not produce a finite gradient for temporal query transition {transition_index}"
                    )
                query_norm = float(query_gradient.detach().float().norm())
                causal_query_gradient_norm_sums[transition_index] += query_norm
                if query_norm > 0:
                    causal_query_gradient_steps[transition_index] += 1
                reference_gradient = reference_box.grad
                if reference_gradient is None or not bool(
                    torch.isfinite(reference_gradient).all()
                ):
                    raise RuntimeError(
                        f"later-frame loss did not produce a finite gradient for temporal reference-box transition {transition_index}"
                    )
                reference_norm = float(reference_gradient.detach().float().norm())
                causal_reference_box_gradient_norm_sums[transition_index] += (
                    reference_norm
                )
                if reference_norm > 0:
                    causal_reference_box_gradient_steps[transition_index] += 1
            gradient_count, nonzero_count, gradient_norm = _audit_nonzero_gradients(
                head, gradient_seen, groups=gradient_groups
            )
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            maximum_gradient_parameters = max(
                maximum_gradient_parameters, gradient_count
            )
            minimum_nonzero_gradient_parameters = (
                nonzero_count
                if minimum_nonzero_gradient_parameters is None
                else min(minimum_nonzero_gradient_parameters, nonzero_count)
            )
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()
            for key, value in step_loss_components.items():
                loss_totals[key] += value
                if value > 0:
                    loss_nonzero_steps[key] += 1
            metrics.append(
                {
                    "epoch": epoch,
                    "step": global_step + 1,
                    "loss": float(loss.detach().float()),
                    "teacher_forcing_probability": teacher.probability(global_step),
                    **step_loss_components,
                }
            )
            previous_state = None
            clip_boundary_state_resets += 1
            global_step += 1
    if global_step != config.optimizer_steps:
        raise RuntimeError(
            f"smoke training expected 20 optimizer steps, got {global_step}"
        )
    if any((count == 0 for count in loss_nonzero_steps.values())):
        missing = [name for name, count in loss_nonzero_steps.items() if count == 0]
        raise RuntimeError(
            f"TemporalDetectionCriterion loss paths stayed zero: {missing}"
        )
    if not all(gradient_seen.values()):
        missing = [name for name, seen in gradient_seen.items() if not seen]
        raise RuntimeError(f"detector gradient groups stayed zero: {missing}")
    if any((count == 0 for count in causal_query_gradient_steps)):
        missing = [
            f"frame{index}_to_frame{index + 1}"
            for index, count in enumerate(causal_query_gradient_steps)
            if count == 0
        ]
        raise RuntimeError(f"causal query BPTT paths stayed zero: {missing}")
    if any((count == 0 for count in causal_reference_box_gradient_steps)):
        missing = [
            f"frame{index}_to_frame{index + 1}"
            for index, count in enumerate(causal_reference_box_gradient_steps)
            if count == 0
        ]
        raise RuntimeError(f"causal reference-box BPTT paths stayed zero: {missing}")
    if stale_selected == 0 or stale_selected > stale_eligible:
        raise RuntimeError(
            "50% Task1 stale initializer did not exercise a valid selection"
        )
    if teacher_selected_positive == 0:
        raise RuntimeError(
            "teacher forcing never selected a positive temporal transition"
        )
    expected_frame0_rows = global_step * config.batch_size
    if (
        stale_selected_outside_eligibility != 0
        or stale_gate_audited_batches != global_step
        or stale_gate_selected_rows != stale_selected
        or (stale_gate_positive_selected_rows != stale_selected)
        or (stale_gate_nonselected_rows != expected_frame0_rows - stale_selected)
        or (stale_gate_zero_nonselected_rows != stale_gate_nonselected_rows)
        or (stale_gate_minimum_selected is None)
        or (stale_gate_minimum_selected <= 0)
        or (stale_gate_maximum_nonselected_absolute != 0)
    ):
        raise RuntimeError(
            "frame-0 stale-memory gate audit is incomplete or inconsistent"
        )
    expected_exposures = Counter({category: 80 for category in SOURCE_CATEGORIES})
    if category_exposures != expected_exposures:
        raise RuntimeError("category-balanced exposure count changed")
    trainable_parameter_count = sum((parameter.numel() for parameter in trainable))
    report = {
        "schema_version": 1,
        "kind": "groot_temporal_cached_smoke_training",
        "experiment_id": config.experiment_id,
        "feature_mode": config.feature_mode,
        "seed": config.seed,
        "plan_sha256": _train_groot_temporal_canonical_sha256(plan),
        "initial_shared_detector_tensors_sha256": initial_detector_state_sha256,
        "cache_schema_sha256": cache.schema.sha256,
        "cache_model_revision": cache.schema.model_revision,
        "e03_source_hook": cache.schema.visual_source_hook,
        "conditioning_source_hook": cache.schema.conditioning_source_hook,
        "epochs": config.epochs,
        "batches_per_epoch": len(sampler),
        "optimizer_steps": global_step,
        "batch_size_clips": config.batch_size,
        "clip_length": CLIP_LENGTH,
        "frame_exposures": global_step * config.batch_size * CLIP_LENGTH,
        "category_exposures": dict(sorted(category_exposures.items())),
        "all_batches_category_balanced": True,
        "clip_boundary_state_resets": clip_boundary_state_resets,
        "state_detach_calls_inside_clip": 0,
        "bptt_graph_transitions": bptt_graph_transitions,
        "causal_query_gradient_audit": {
            "method": "retain_grad_on_each_nonterminal_next_state_query",
            "nonzero_optimizer_steps_by_transition": {
                f"frame{index}_to_frame{index + 1}": count
                for index, count in enumerate(causal_query_gradient_steps)
            },
            "gradient_norm_sum_by_transition": {
                f"frame{index}_to_frame{index + 1}": value
                for index, value in enumerate(causal_query_gradient_norm_sums)
            },
        },
        "causal_reference_box_gradient_audit": {
            "method": "retain_grad_on_each_nonterminal_state_reference_box",
            "nonzero_optimizer_steps_by_transition": {
                f"frame{index}_to_frame{index + 1}": count
                for index, count in enumerate(causal_reference_box_gradient_steps)
            },
            "gradient_norm_sum_by_transition": {
                f"frame{index}_to_frame{index + 1}": value
                for index, value in enumerate(causal_reference_box_gradient_norm_sums)
            },
        },
        "stale_memory": {
            "negative_only": stale_selected_outside_eligibility == 0,
            "task1_probability": config.stale_probability,
            "eligible_clip_exposures": stale_eligible,
            "selected_clip_exposures": stale_selected,
            "frame0_gate_audit": {
                "audited_batches": stale_gate_audited_batches,
                "selected_rows": stale_gate_selected_rows,
                "selected_rows_with_positive_gate": stale_gate_positive_selected_rows,
                "selected_rows_outside_task1_eligibility": stale_selected_outside_eligibility,
                "minimum_selected_gate": stale_gate_minimum_selected,
                "nonselected_rows": stale_gate_nonselected_rows,
                "nonselected_rows_with_zero_gate": stale_gate_zero_nonselected_rows,
                "maximum_nonselected_absolute_gate": stale_gate_maximum_nonselected_absolute,
            },
        },
        "teacher_forcing": {
            "initial_probability": teacher.probability(0),
            "final_probability": teacher.probability(global_step - 1),
            "selected_positive_transitions": teacher_selected_positive,
        },
        "loss_path_mean": {key: loss_totals[key] / global_step for key in _LOSS_KEYS},
        "loss_path_nonzero_steps": dict(loss_nonzero_steps),
        "gradient_audit": {
            "groups_with_nonzero_gradient": dict(gradient_seen),
            "minimum_nonzero_parameter_tensors_per_step": minimum_nonzero_gradient_parameters,
            "maximum_parameter_tensors_with_gradient": maximum_gradient_parameters,
            "maximum_unclipped_gradient_norm": maximum_gradient_norm,
        },
        "amp": {
            "requested_bfloat16": config.bfloat16_autocast,
            "used_cuda_bfloat16_autocast": amp_used,
            "cpu_test_path_is_fp32": target_device.type == "cpu",
            "box_giou_loss_precision": "inherits_detector_autocast",
            "full_training_todo": "evaluate FP32 L1/GIoU/motion loss math before full temporal training",
        },
        "parameter_audit": {
            "trainable_detector_parameters": trainable_parameter_count,
            "optimizer_parameter_tensors": len(optimizer_ids),
            "optimizer_exactly_matches_head": optimizer_ids == head_ids,
            "forbidden_groot_parameter_names": forbidden,
            "groot_policy_parameter_count": 0,
            "groot_model_loaded": False,
            "cached_features_require_grad": False,
        },
        "metrics": metrics,
    }
    if report_path is not None:
        _atomic_json(Path(report_path), report)
    return GrootTemporalSmokeTrainingResult(head=head, report=report)


def main(argv=None) -> int:
    """Train a static or temporal head from externally supplied frozen features."""
    import argparse

    from endo_bbox_vla.cache import GrootTemporalCache
    from endo_bbox_vla.config import TRAINING_STAGES, paper_model
    from endo_bbox_vla.data import load_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    static = commands.add_parser(
        "static", help="train Ours or a controlled static ablation"
    )
    static.add_argument("--config", type=Path, required=True)
    static.add_argument(
        "--stage",
        choices=tuple(TRAINING_STAGES),
        default="hard_positive",
        help="Ours: spatial_queries first, then hard_positive continuation",
    )
    static.add_argument("--device", default="cuda")
    static.add_argument("--resume", action="store_true")
    temporal = commands.add_parser(
        "temporal", help="train a completed temporal comparison protocol"
    )
    temporal.add_argument("--plan", type=Path, required=True)
    temporal.add_argument("--frame-manifest", type=Path, required=True)
    temporal.add_argument("--cache", type=Path, required=True)
    temporal.add_argument("--output", type=Path, required=True)
    temporal.add_argument(
        "--model",
        choices=("temporal_visual_memory", "temporal_vl_conditioned"),
        required=True,
    )
    temporal.add_argument("--device", default="cuda")
    temporal.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "static":
        train_static_detector(
            load_config(args.config),
            variant=TRAINING_STAGES[args.stage],
            device=args.device,
            resume=args.resume,
        )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        plan = json.loads(args.plan.read_text())
        train_groot_temporal_full(
            full_plan=plan,
            temporal_frame_manifest=args.frame_manifest,
            cache=GrootTemporalCache(args.cache),
            config=GrootTemporalFullTrainingConfig(
                feature_mode=paper_model(args.model).feature_mode
            ),
            device=args.device,
            progress_checkpoint_path=args.output / "progress.pt",
            progress_report_path=args.output / "progress.json",
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
