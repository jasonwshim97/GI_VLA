"""Reviewed-frame datasets, target geometry, sampling and clip plans."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable as _data_Callable
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol
from typing import Iterable as _coco_dataset_Iterable
from typing import Iterable as _transforms_Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from annotation.prepare import AnnotationStatus
from endo_bbox_vla.config import (
    CANONICAL_TASK_PROMPTS,
    NEUTRAL_PROMPT,
    PROMPT_FRACTIONS,
    contracts_TASK1_SOURCE_DATASET,
    contracts_TASK4_SOURCE_DATASET,
)

if TYPE_CHECKING:
    from annotation.prepare import AnnotationStatus, DatasetSplit, FrameRecord
    from endo_bbox_vla.cache import CacheRow
    from endo_bbox_vla.config import ExperimentConfig

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class GrootN17BoxProjection:
    """Visible and transformed forms of a set of original-coordinate boxes."""

    original_boxes_xyxy: Tensor
    visible_boxes_xyxy: Tensor
    output_boxes_xyxy: Tensor
    visible_fraction: Tensor
    partial_original: Tensor
    partial_transform: Tensor
    ignore: Tensor

    @property
    def box_supervision(self) -> Tensor:
        """Mask of boxes that remain observable after the checkpoint crop."""
        return ~self.ignore


def _boxes_tensor(boxes: Tensor | Iterable[Iterable[float]]) -> Tensor:
    result = torch.as_tensor(boxes, dtype=torch.float32)
    if result.numel() == 0:
        return result.reshape(0, 4).clone()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("boxes contain NaN or infinity")
    if bool(
        (result[:, 2] <= result[:, 0]).any() or (result[:, 3] <= result[:, 1]).any()
    ):
        raise ValueError("boxes must have positive width and height")
    return result.clone()


def _partial_flags(
    value: bool | Tensor | Iterable[bool], count: int, device: torch.device
) -> Tensor:
    if isinstance(value, bool):
        return torch.full((count,), value, dtype=torch.bool, device=device)
    result = torch.as_tensor(value, dtype=torch.bool, device=device).flatten()
    if tuple(result.shape) != (count,):
        raise ValueError(f"partial_original must have shape ({count},)")
    return result.clone()


GridLevel = Literal["raw", "merged"]


def _size(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain (width, height)")
    if any(
        (
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value


@dataclass(frozen=True, slots=True)
class GrootN17Geometry:
    """Exact saved-checkpoint image and token-grid geometry.

    All image sizes use ``(width, height)``. Coordinates are continuous image
    edge coordinates, matching COCO/xyxy bbox transforms.
    """

    original_size: tuple[int, int] = (640, 480)
    resize_size: tuple[int, int] = (256, 256)
    crop_size: tuple[int, int] = (230, 230)
    output_size: tuple[int, int] = (256, 256)
    raw_grid_size: tuple[int, int] = (16, 16)
    merged_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        for name in (
            "original_size",
            "resize_size",
            "crop_size",
            "output_size",
            "raw_grid_size",
            "merged_grid_size",
        ):
            _size(getattr(self, name), name)
        if any(
            (
                crop > resize
                for crop, resize in zip(self.crop_size, self.resize_size, strict=True)
            )
        ):
            raise ValueError("crop_size cannot exceed resize_size")
        if any(
            (
                raw % merged
                for raw, merged in zip(
                    self.raw_grid_size, self.merged_grid_size, strict=True
                )
            )
        ):
            raise ValueError("raw_grid_size must be divisible by merged_grid_size")

    @property
    def crop_offset(self) -> tuple[int, int]:
        """Integer center-crop offset used by the GR00T torch preprocessor."""
        return (
            (self.resize_size[0] - self.crop_size[0]) // 2,
            (self.resize_size[1] - self.crop_size[1]) // 2,
        )

    @property
    def scale(self) -> tuple[float, float]:
        """Direct original-to-output affine scale after resize and crop."""
        return (
            self.resize_size[0]
            / self.original_size[0]
            * self.output_size[0]
            / self.crop_size[0],
            self.resize_size[1]
            / self.original_size[1]
            * self.output_size[1]
            / self.crop_size[1],
        )

    @property
    def offset(self) -> tuple[float, float]:
        """Direct original-to-output affine offset after resize and crop."""
        crop_left, crop_top = self.crop_offset
        return (
            -crop_left * self.output_size[0] / self.crop_size[0],
            -crop_top * self.output_size[1] / self.crop_size[1],
        )

    @property
    def crop_window_original(self) -> tuple[float, float, float, float]:
        """Original-image rectangle that survives the deterministic crop."""
        crop_left, crop_top = self.crop_offset
        resize_scale_x = self.resize_size[0] / self.original_size[0]
        resize_scale_y = self.resize_size[1] / self.original_size[1]
        return (
            crop_left / resize_scale_x,
            crop_top / resize_scale_y,
            (crop_left + self.crop_size[0]) / resize_scale_x,
            (crop_top + self.crop_size[1]) / resize_scale_y,
        )

    def _forward_unchecked(self, boxes: Tensor) -> Tensor:
        scale_x, scale_y = self.scale
        offset_x, offset_y = self.offset
        result = boxes.clone()
        result[:, (0, 2)] = result[:, (0, 2)] * scale_x + offset_x
        result[:, (1, 3)] = result[:, (1, 3)] * scale_y + offset_y
        return result

    def forward_xyxy(
        self, boxes: Tensor | Iterable[Iterable[float]], *, clip: bool = True
    ) -> Tensor:
        """Map original 640x480 xyxy boxes into the 256x256 crop output."""
        result = self._forward_unchecked(_boxes_tensor(boxes))
        if clip:
            result[:, (0, 2)] = result[:, (0, 2)].clamp(0, self.output_size[0])
            result[:, (1, 3)] = result[:, (1, 3)].clamp(0, self.output_size[1])
        return result

    def inverse_xyxy(
        self, boxes: Tensor | Iterable[Iterable[float]], *, clip: bool = True
    ) -> Tensor:
        """Map output xyxy boxes back to the retained original-image region."""
        result = _boxes_tensor(boxes)
        scale_x, scale_y = self.scale
        offset_x, offset_y = self.offset
        result[:, (0, 2)] = (result[:, (0, 2)] - offset_x) / scale_x
        result[:, (1, 3)] = (result[:, (1, 3)] - offset_y) / scale_y
        if clip:
            result[:, (0, 2)] = result[:, (0, 2)].clamp(0, self.original_size[0])
            result[:, (1, 3)] = result[:, (1, 3)].clamp(0, self.original_size[1])
        return result

    def project_xyxy(
        self,
        boxes: Tensor | Iterable[Iterable[float]],
        *,
        partial_original: bool | Tensor | Iterable[bool] = False,
        epsilon: float = 1e-06,
    ) -> GrootN17BoxProjection:
        """Intersect with the visible crop and produce fail-closed metadata.

        A fully crop-invisible box is returned with ``ignore=True`` and a zero
        output box. It must not be relabelled as a confirmed negative.
        """
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        original = _boxes_tensor(boxes)
        count = len(original)
        original_partial = _partial_flags(partial_original, count, original.device)
        if count == 0:
            empty_float = original.new_empty((0,))
            empty_bool = torch.empty((0,), dtype=torch.bool, device=original.device)
            return GrootN17BoxProjection(
                original,
                original.clone(),
                original.clone(),
                empty_float,
                original_partial,
                empty_bool,
                empty_bool.clone(),
            )
        left, top, right, bottom = self.crop_window_original
        visible = original.clone()
        visible[:, 0] = visible[:, 0].clamp_min(left)
        visible[:, 1] = visible[:, 1].clamp_min(top)
        visible[:, 2] = visible[:, 2].clamp_max(right)
        visible[:, 3] = visible[:, 3].clamp_max(bottom)
        widths = (visible[:, 2] - visible[:, 0]).clamp_min(0)
        heights = (visible[:, 3] - visible[:, 1]).clamp_min(0)
        visible_area = widths * heights
        original_area = (original[:, 2] - original[:, 0]) * (
            original[:, 3] - original[:, 1]
        )
        visible_fraction = (visible_area / original_area).clamp(0, 1)
        ignore = (widths <= epsilon) | (heights <= epsilon)
        changed = ~torch.isclose(visible, original, rtol=0, atol=epsilon).all(dim=1)
        partial_transform = changed & ~ignore
        visible = torch.where(ignore[:, None], torch.zeros_like(visible), visible)
        output = self._forward_unchecked(visible)
        output[:, (0, 2)] = output[:, (0, 2)].clamp(0, self.output_size[0])
        output[:, (1, 3)] = output[:, (1, 3)].clamp(0, self.output_size[1])
        output = torch.where(ignore[:, None], torch.zeros_like(output), output)
        return GrootN17BoxProjection(
            original_boxes_xyxy=original,
            visible_boxes_xyxy=visible,
            output_boxes_xyxy=output,
            visible_fraction=visible_fraction,
            partial_original=original_partial,
            partial_transform=partial_transform,
            ignore=ignore,
        )

    def padding_mask(
        self,
        level: GridLevel = "merged",
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return the all-valid flattened spatial mask for one or more cameras.

        ``True`` would denote padding. The saved checkpoint does no letterbox
        padding, so every raw 16x16 or merged 8x8 spatial position is valid.
        """
        if level not in ("raw", "merged"):
            raise ValueError("level must be 'raw' or 'merged'")
        width, height = self.raw_grid_size if level == "raw" else self.merged_grid_size
        shape = (height * width,)
        if batch_size is not None:
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size <= 0
            ):
                raise ValueError("batch_size must be a positive integer")
            shape = (batch_size, *shape)
        return torch.zeros(shape, dtype=torch.bool, device=device)


GROOT_N17_GEOMETRY = GrootN17Geometry()


def load_config(path: str | Path) -> ExperimentConfig:
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.train import VARIANT_NAME
    from endo_bbox_vla.utils import _resolve_repo_path, sha256_file

    source = _resolve_repo_path(path)
    payload = source.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("shared detector config must be a schema-v1 mapping")
    backbone = raw["backbone"]
    data = raw["data"]
    if backbone.get("feature_capture") != "same_action_forward":
        raise ValueError("shared detector requires same_action_forward feature capture")
    if backbone.get("feature_source") != "qwen3_vl_visual_block_11_output_pre_merger":
        raise ValueError(
            "shared detector requires the original block-11 block-11 visual tap"
        )
    if backbone.get("visual_parameters_trainable") is not False:
        raise ValueError("shared detector must keep every visual parameter frozen")
    if (
        tuple(backbone.get("grid_size", ())) != (16, 16)
        or int(backbone.get("feature_dim", -1)) != 1024
    ):
        raise ValueError("shared detector requires a 16x16x1024 token grid")
    if tuple(data.get("original_size", ())) != GROOT_N17_GEOMETRY.original_size:
        raise ValueError("shared detector original image geometry changed")
    if tuple(data.get("detector_size", ())) != GROOT_N17_GEOMETRY.output_size:
        raise ValueError("shared detector detector crop geometry changed")
    if data.get("target_geometry") != "groot_n17_crop":
        raise ValueError("shared detector must use the GR00T N1.7 crop target")
    variants = raw.get("training", {}).get("variants", {})
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("shared detector requires at least one training variant")
    for name, settings in variants.items():
        if not isinstance(name, str) or VARIANT_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid shared detector variant name: {name!r}")
        if not isinstance(settings, Mapping):
            raise TypeError(f"shared detector variant {name!r} must be a mapping")
        kind = str(settings.get("head_kind", "location_first"))
        if kind not in {"one_query", "learned_queries", "location_first"}:
            raise ValueError(f"unsupported shared detector head kind: {kind!r}")
        queries = int(
            settings.get(
                "num_queries",
                1
                if kind == "one_query"
                else raw.get("head", {}).get("location_queries", 4),
            )
        )
        if kind == "one_query" and queries != 1 or queries <= 0:
            raise ValueError(
                f"invalid query count for shared detector variant {name!r}"
            )
    for split in ("train", "validation", "test"):
        manifest = _resolve_repo_path(data[f"{split}_manifest"])
        if sha256_file(manifest) != data[f"{split}_manifest_sha256"]:
            raise ValueError(f"{split} manifest SHA changed")
    return ExperimentConfig(
        raw=raw, path=source, sha256=hashlib.sha256(payload).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class CacheContract:
    root: Path
    rows: tuple[CacheRow, ...]
    manifest_sha256: str
    action_parity: dict[str, Any]
    action_parity_sha256: str
    model_sha256: str


def _validate_action_parity(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    from endo_bbox_vla.train import EXPECTED_ACTION_SHAPE

    if not path.is_file():
        raise FileNotFoundError(
            f"same-forward action parity artifact is missing: {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    required_true = (
        "torch_equal_action",
        "torch_equal_rng_after",
        "torch_equal_mid_action_vs_backbone_only",
        "torch_equal_projected_action_vs_backbone_only",
    )
    if any((report.get(key) is not True for key in required_true)):
        raise ValueError("same-forward action/RNG/feature parity did not pass")
    metadata = report.get("feature_shape_metadata")
    expected = {
        "base_checkpoint": config.section("backbone")["checkpoint"],
        "base_revision": config.section("backbone")["revision"],
        "camera_key": config.section("backbone")["camera_key"],
        "policy_camera_order": config.section("backbone")["camera_order"],
        "mid_shapes": [[256, 1024]],
        "mid_spatial_shapes": [[16, 16]],
        "mid_layer_indices": [11],
    }
    if report.get("action_shape") != EXPECTED_ACTION_SHAPE or not isinstance(
        metadata, Mapping
    ):
        raise ValueError("same-forward parity shape metadata is invalid")
    mismatches = {
        key: (wanted, metadata.get(key))
        for key, wanted in expected.items()
        if metadata.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"same-forward block-11 visual contract changed: {mismatches}")
    return dict(report)


def _load_cache_contract(
    config: ExperimentConfig, *, evaluation: bool
) -> CacheContract:
    from endo_bbox_vla.cache import _read_cache_manifest
    from endo_bbox_vla.train import EVAL_CACHE_KIND, TRAIN_CACHE_KIND
    from endo_bbox_vla.utils import _resolve_repo_path, sha256_file

    section = config.section("feature_cache")
    root = _resolve_repo_path(
        section["evaluation_root"] if evaluation else section["train_root"]
    )
    complete_path = root / "cache.complete.json"
    manifest_path = root / "manifest.jsonl"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_kind = EVAL_CACHE_KIND if evaluation else TRAIN_CACHE_KIND
    expected_sha = section[
        "evaluation_manifest_sha256" if evaluation else "train_manifest_sha256"
    ]
    actual_sha = sha256_file(manifest_path)
    if complete.get("kind") != expected_kind:
        raise ValueError(f"unexpected shared detector cache kind at {root}")
    if complete.get("manifest_sha256") != actual_sha or actual_sha != expected_sha:
        raise ValueError(f"shared detector cache manifest SHA changed at {root}")
    model_sha = str(complete.get("model_sha256"))
    if model_sha != config.section("backbone")["expected_model_sha256"]:
        raise ValueError("shared detector cache came from another GR00T checkpoint")
    rows = _read_cache_manifest(manifest_path)
    expected_count = (
        int(config.section("data")["expected_counts"]["validation"])
        + int(config.section("data")["expected_counts"]["test"])
        if evaluation
        else int(config.section("data")["expected_counts"]["train"])
    )
    if (
        len(rows) != expected_count
        or int(complete.get("entries", -1)) != expected_count
    ):
        raise ValueError(f"shared detector cache count changed at {root}")
    parity_path = root / "action_parity.json"
    parity = _validate_action_parity(parity_path, config)
    return CacheContract(
        root=root,
        rows=rows,
        manifest_sha256=actual_sha,
        action_parity=parity,
        action_parity_sha256=sha256_file(parity_path),
        model_sha256=model_sha,
    )


FRAME_SCHEMA_VERSION = 1


def _safe_relative_posix(value: Any, name: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or (not path.parts):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return path.as_posix()


ALLOWED_SPLITS = frozenset(
    {"train", "calibration", "validation", "locked_test", "test", "negative_control"}
)


def _bbox(
    value: Any, status: str, width: int, height: int
) -> tuple[float, float, float, float] | None:
    if value is None:
        if status == "positive":
            raise ValueError("positive static frame requires bbox_xyxy")
        return None
    if status != "positive":
        raise ValueError(f"{status} static frame must not contain bbox_xyxy")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 4
    ):
        raise TypeError("bbox_xyxy must contain four numbers")
    result = tuple((float(item) for item in value))
    if not all((math.isfinite(item) for item in result)):
        raise ValueError("bbox_xyxy contains NaN or infinity")
    x1, y1, x2, y2 = result
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox_xyxy {result} is outside {(width, height)}")
    return result


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


TRAINABLE_STATUSES = frozenset({"positive", "confirmed_negative"})

UNCERTAIN_STATUS = "uncertain_ignore"


@dataclass(frozen=True, slots=True)
class StaticFrameRecord:
    sample_id: str
    image_id: int
    source_dataset: str
    source_episode_index: int
    source_frame_index: int
    dataset_index: int
    episode_index: int
    frame_index: int
    timestamp: float
    source_video: str
    split: str
    status: str
    bbox_xyxy: tuple[float, float, float, float] | None
    detector_generation: int
    context_generation: int
    partial: bool = False
    occluded: bool = False
    blurred: bool = False
    group_id: int | None = None
    width: int = 640
    height: int = 480
    schema_version: int = FRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id or self.sample_id.strip() != self.sample_id:
            raise ValueError("sample_id must be a non-empty, trimmed string")
        if (
            not self.source_dataset
            or self.source_dataset.strip() != self.source_dataset
        ):
            raise ValueError("source_dataset must be a non-empty, trimmed string")
        for name in (
            "image_id",
            "source_episode_index",
            "source_frame_index",
            "dataset_index",
            "episode_index",
            "frame_index",
            "detector_generation",
            "context_generation",
        ):
            _strict_int(getattr(self, name), name)
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        _safe_relative_posix(self.source_video, "source_video")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported static-frame split: {self.split!r}")
        if self.status not in TRAINABLE_STATUSES | {UNCERTAIN_STATUS}:
            raise ValueError(f"unsupported static-frame status: {self.status!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        for name in ("partial", "occluded", "blurred"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.group_id is not None:
            _strict_int(self.group_id, "group_id")
        _bbox(self.bbox_xyxy, self.status, self.width, self.height)
        if self.schema_version != FRAME_SCHEMA_VERSION:
            raise ValueError(f"unsupported static-frame schema {self.schema_version}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StaticFrameRecord:
        fields = {
            "sample_id",
            "image_id",
            "source_dataset",
            "source_episode_index",
            "source_frame_index",
            "dataset_index",
            "episode_index",
            "frame_index",
            "timestamp",
            "source_video",
            "split",
            "status",
            "bbox_xyxy",
            "detector_generation",
            "context_generation",
            "partial",
            "occluded",
            "blurred",
            "group_id",
            "width",
            "height",
            "schema_version",
        }
        unknown = set(payload) - fields
        optional = {"width", "height", "partial", "occluded", "blurred", "group_id"}
        missing = fields - optional - set(payload)
        if unknown:
            raise ValueError(f"unknown static-frame fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing static-frame fields: {sorted(missing)}")
        status = str(payload["status"])
        width = _strict_int(payload.get("width", 640), "width", minimum=1)
        height = _strict_int(payload.get("height", 480), "height", minimum=1)
        return cls(
            sample_id=str(payload["sample_id"]),
            image_id=_strict_int(payload["image_id"], "image_id"),
            source_dataset=str(payload["source_dataset"]),
            source_episode_index=_strict_int(
                payload["source_episode_index"], "source_episode_index"
            ),
            source_frame_index=_strict_int(
                payload["source_frame_index"], "source_frame_index"
            ),
            dataset_index=_strict_int(payload["dataset_index"], "dataset_index"),
            episode_index=_strict_int(payload["episode_index"], "episode_index"),
            frame_index=_strict_int(payload["frame_index"], "frame_index"),
            timestamp=float(payload["timestamp"]),
            source_video=_safe_relative_posix(payload["source_video"], "source_video"),
            split=str(payload["split"]),
            status=status,
            bbox_xyxy=_bbox(payload["bbox_xyxy"], status, width, height),
            detector_generation=_strict_int(
                payload["detector_generation"], "detector_generation"
            ),
            context_generation=_strict_int(
                payload["context_generation"], "context_generation"
            ),
            partial=bool(payload.get("partial", False)),
            occluded=bool(payload.get("occluded", False)),
            blurred=bool(payload.get("blurred", False)),
            group_id=None
            if payload.get("group_id") is None
            else _strict_int(payload["group_id"], "group_id"),
            width=width,
            height=height,
            schema_version=_strict_int(
                payload["schema_version"], "schema_version", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox_xyxy"] = (
            list(self.bbox_xyxy) if self.bbox_xyxy is not None else None
        )
        return result


@dataclass(frozen=True, slots=True)
class BoundFeature:
    record: StaticFrameRecord
    row: CacheRow
    presence: Tensor
    box: Tensor
    box_weight: float


class SharedFeatureTable(Dataset[dict[str, Tensor]]):
    """In-memory, SHA-verified block-11 visual tokens with crop-coordinate targets."""

    def __init__(
        self,
        contract: CacheContract,
        bindings: Sequence[BoundFeature],
        *,
        progress_label: str,
    ) -> None:
        from endo_bbox_vla.cache import _validated_tensor_path

        count = len(bindings)
        self.features = torch.empty((count, 256, 1024), dtype=torch.bfloat16)
        self.padding_masks = torch.empty((count, 256), dtype=torch.bool)
        self.presence = torch.empty(count, dtype=torch.float32)
        self.boxes = torch.empty((count, 4), dtype=torch.float32)
        self.box_weights = torch.empty(count, dtype=torch.float32)
        self.records = tuple((binding.record for binding in bindings))
        started = time.monotonic()
        for index, binding in enumerate(bindings):
            path = _validated_tensor_path(contract.root, binding.row)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise TypeError(f"shared detector tensor must be a mapping: {path}")
            tokens = torch.as_tensor(payload.get("mid_tokens"))
            mask = torch.as_tensor(payload.get("mid_padding_mask"), dtype=torch.bool)
            metadata = payload.get("shape_metadata")
            if tuple(tokens.shape) != (256, 1024) or tokens.dtype != torch.bfloat16:
                raise ValueError(f"invalid shared detector tokens at {path}")
            if tuple(mask.shape) != (256,) or bool(mask.any()):
                raise ValueError(f"invalid shared detector padding mask at {path}")
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError(f"non-finite shared detector tokens at {path}")
            if not isinstance(metadata, Mapping) or (
                metadata.get("mid_shapes") != [[256, 1024]]
                or metadata.get("mid_spatial_shapes") != [[16, 16]]
                or metadata.get("mid_layer_indices") != [11]
            ):
                raise ValueError(f"invalid shared detector shape metadata at {path}")
            self.features[index].copy_(tokens)
            self.padding_masks[index].copy_(mask)
            self.presence[index] = binding.presence
            self.boxes[index].copy_(binding.box)
            self.box_weights[index] = binding.box_weight
            if (index + 1) % 1000 == 0 or index + 1 == count:
                print(
                    json.dumps(
                        {
                            "kind": "shared_e03_feature_load_progress",
                            "table": progress_label,
                            "loaded": index + 1,
                            "total": count,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.features[index],
            "padding_mask": self.padding_masks[index],
            "presence": self.presence[index],
            "boxes_cxcywh": self.boxes[index],
            "box_weights": self.box_weights[index],
        }


@dataclass(frozen=True, slots=True)
class PromptSelection:
    mode: str
    actual_prompt: str
    selected_prompt: str
    prompt_id: str
    context_generation: int

    def with_context_generation(self, value: int) -> PromptSelection:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("context_generation must be a non-negative integer")
        return PromptSelection(
            mode=self.mode,
            actual_prompt=self.actual_prompt,
            selected_prompt=self.selected_prompt,
            prompt_id=self.prompt_id,
            context_generation=value,
        )


@dataclass(frozen=True, slots=True)
class GrootStaticExample:
    sample_key: str
    contexts: tuple[dict[str, Any], ...]
    targets: tuple[dict[str, Any], ...]
    records: tuple[StaticFrameRecord, ...]
    prompts: tuple[PromptSelection, ...]
    context_sha256: tuple[str, ...]


PROMPT_MODES = ("actual", "counterfactual", "neutral")


def _apportioned_counts(length: int, fractions: Mapping[str, float]) -> dict[str, int]:
    if length < 0:
        raise ValueError("length must be non-negative")
    raw = {mode: length * float(fractions[mode]) for mode in PROMPT_MODES}
    counts = {mode: math.floor(raw[mode]) for mode in PROMPT_MODES}
    remaining = length - sum(counts.values())
    priority = {mode: index for index, mode in enumerate(PROMPT_MODES)}
    order = sorted(
        PROMPT_MODES, key=lambda mode: (-(raw[mode] - counts[mode]), priority[mode])
    )
    for mode in order[:remaining]:
        counts[mode] += 1
    return counts


def _digest(*parts: object) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def deterministic_mode_assignments(
    sample_keys: Sequence[str] | Iterable[str],
    *,
    seed: int,
    fractions: Mapping[str, float] = PROMPT_FRACTIONS,
) -> dict[str, str]:
    """Assign modes exactly (up to integer apportionment), independent of workers.

    Assignment is based only on the immutable sample key and seed.  DataLoader
    order, number of workers, and global RNG state cannot change it.
    """
    keys = tuple((str(key) for key in sample_keys))
    if len(set(keys)) != len(keys):
        raise ValueError("prompt assignment sample keys must be unique")
    if set(fractions) != set(PROMPT_MODES):
        raise ValueError(f"fractions must contain {list(PROMPT_MODES)} exactly")
    parsed = {mode: float(fractions[mode]) for mode in PROMPT_MODES}
    if any((not math.isfinite(value) or value < 0 for value in parsed.values())):
        raise ValueError("prompt fractions must be finite and non-negative")
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("prompt fractions must sum to one")
    ordered = sorted(keys, key=lambda key: (_digest(seed, key), key))
    counts = _apportioned_counts(len(ordered), parsed)
    result: dict[str, str] = {}
    cursor = 0
    for mode in PROMPT_MODES:
        for key in ordered[cursor : cursor + counts[mode]]:
            result[key] = mode
        cursor += counts[mode]
    if Counter(result.values()) != Counter(counts):
        raise RuntimeError("internal prompt apportionment mismatch")
    return result


class PromptPlanner:
    """Resolve assigned modes using the preregistered Task1/Task4 contrast.

    Counterfactual prompts are deliberately not sampled from all five tasks:
    Task4 observations receive the Task1 instruction and Task1 observations
    receive the Task4 instruction.  Detection labels and boxes are never
    changed by prompt assignment.
    """

    def __init__(
        self,
        assignments: Mapping[str, str],
        *,
        seed: int,
        canonical_tasks: Sequence[str] = CANONICAL_TASK_PROMPTS,
        neutral_prompt: str = NEUTRAL_PROMPT,
    ) -> None:
        self.assignments = {str(key): str(value) for key, value in assignments.items()}
        self.seed = int(seed)
        self.canonical_tasks = tuple((str(prompt) for prompt in canonical_tasks))
        self.neutral_prompt = str(neutral_prompt)
        if len(self.canonical_tasks) != 5 or len(set(self.canonical_tasks)) != 5:
            raise ValueError("canonical_tasks must contain five distinct prompts")
        if any((not prompt.strip() for prompt in self.canonical_tasks)):
            raise ValueError("canonical task prompts must not be empty")
        if (
            self.neutral_prompt in self.canonical_tasks
            or not self.neutral_prompt.strip()
        ):
            raise ValueError(
                "neutral prompt must be non-empty and distinct from task prompts"
            )
        unknown = set(self.assignments.values()) - set(PROMPT_MODES)
        if unknown:
            raise ValueError(f"unknown prompt modes: {sorted(unknown)}")

    def select_mode(
        self,
        sample_key: str,
        actual_prompt: str,
        mode: str,
        *,
        context_generation: int = 0,
    ) -> PromptSelection:
        sample_key = str(sample_key)
        actual_prompt = str(actual_prompt)
        if actual_prompt not in self.canonical_tasks:
            raise ValueError(
                f"actual task is not one of the five pinned instructions: {actual_prompt!r}"
            )
        if mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode {mode!r}")
        if isinstance(context_generation, bool) or not isinstance(
            context_generation, int
        ):
            raise TypeError("context_generation must be an integer")
        if context_generation < 0:
            raise ValueError("context_generation must be non-negative")
        if mode == "actual":
            selected = actual_prompt
        elif mode == "neutral":
            selected = self.neutral_prompt
        else:
            task1_prompt = self.canonical_tasks[0]
            task4_prompt = self.canonical_tasks[3]
            if actual_prompt == task4_prompt:
                selected = task1_prompt
            elif actual_prompt == task1_prompt:
                selected = task4_prompt
            else:
                raise ValueError(
                    "Counterfactual assignment is defined only for Task1 negatives and Task4 polyp frames"
                )
        prompt_payload = {
            "actual_prompt": actual_prompt,
            "mode": mode,
            "selected_prompt": selected,
        }
        prompt_id = hashlib.sha256(
            json.dumps(prompt_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return PromptSelection(
            mode=mode,
            actual_prompt=actual_prompt,
            selected_prompt=selected,
            prompt_id=prompt_id,
            context_generation=context_generation,
        )

    def select(self, sample_key: str, actual_prompt: str) -> PromptSelection:
        sample_key = str(sample_key)
        if sample_key not in self.assignments:
            raise KeyError(
                f"sample has no deterministic prompt assignment: {sample_key!r}"
            )
        return self.select_mode(sample_key, actual_prompt, self.assignments[sample_key])

    def all_modes(
        self, sample_key: str, actual_prompt: str
    ) -> tuple[PromptSelection, ...]:
        """Return same-frame actual/counterfactual/neutral evaluation arms."""
        return tuple(
            (
                self.select_mode(sample_key, actual_prompt, mode, context_generation=0)
                for mode in PROMPT_MODES
            )
        )


def _actual_prompt(record: StaticFrameRecord) -> str:
    from endo_bbox_vla.config import (
        CANONICAL_TASK_PROMPTS,
        TASK1_SOURCE_DATASET,
        TASK4_SOURCE_DATASET,
    )

    if record.source_dataset == TASK1_SOURCE_DATASET:
        return CANONICAL_TASK_PROMPTS[0]
    if record.source_dataset == TASK4_SOURCE_DATASET:
        return CANONICAL_TASK_PROMPTS[3]
    raise ValueError(f"unsupported static source {record.source_dataset!r}")


GROOT_N17_CROP_TARGET_GEOMETRY = "groot_n17_crop"


def _raw_target(record: StaticFrameRecord) -> dict[str, Any]:
    boxes = (
        torch.tensor([record.bbox_xyxy], dtype=torch.float32)
        if record.bbox_xyxy is not None
        else torch.empty((0, 4), dtype=torch.float32)
    )
    return {
        "boxes": boxes,
        "presence": torch.tensor(
            float(record.status == "positive"), dtype=torch.float32
        ),
        "image_id": torch.tensor(record.image_id, dtype=torch.int64),
        "original_size": torch.tensor([record.height, record.width], dtype=torch.int64),
        "size": torch.tensor([record.height, record.width], dtype=torch.int64),
    }


def _transforms_boxes_tensor(
    boxes: Tensor | _transforms_Iterable[_transforms_Iterable[float]],
) -> Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    if tensor.ndim != 2 or tensor.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("boxes contain NaN or infinity")
    return tensor


def xyxy_to_cxcywh(
    boxes: Tensor | _transforms_Iterable[_transforms_Iterable[float]],
) -> Tensor:
    boxes = _transforms_boxes_tensor(boxes)
    result = torch.empty_like(boxes)
    result[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
    result[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return result


def _normalized_target_from_xyxy(target: Mapping[str, Tensor]) -> Tensor:
    boxes = target["boxes"]
    if len(boxes) == 0:
        return boxes.new_zeros((4,))
    if len(boxes) != 1:
        raise ValueError("single-polyp target must contain zero or one box")
    height, width = target["size"].tolist()
    return xyxy_to_cxcywh(boxes)[0] / boxes.new_tensor((width, height, width, height))


def build_groot_target(record: StaticFrameRecord) -> dict[str, Any]:
    geometry = GROOT_N17_GEOMETRY
    if (record.width, record.height) != geometry.original_size:
        raise ValueError(
            f"GR00T target requires {geometry.original_size}, got {(record.width, record.height)}"
        )
    target = _raw_target(record)
    projection = geometry.project_xyxy(target["boxes"], partial_original=record.partial)
    if record.status == "positive":
        if len(projection.ignore) != 1:
            raise RuntimeError("positive static target must contain exactly one box")
        ignore = projection.ignore[0]
        partial_original = projection.partial_original[0]
        partial_transform = projection.partial_transform[0]
        visible_fraction = projection.visible_fraction[0]
        boxes = projection.output_boxes_xyxy[projection.box_supervision]
    else:
        ignore = torch.tensor(False, dtype=torch.bool)
        partial_original = torch.tensor(record.partial, dtype=torch.bool)
        partial_transform = torch.tensor(False, dtype=torch.bool)
        visible_fraction = torch.tensor(0.0, dtype=torch.float32)
        boxes = target["boxes"]
    target["boxes"] = boxes
    target["area"] = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    target["size"] = torch.tensor(
        [geometry.output_size[1], geometry.output_size[0]], dtype=torch.int64
    )
    target["boxes_cxcywh"] = _normalized_target_from_xyxy(target)
    target["visible_fraction"] = visible_fraction
    target["partial_original"] = partial_original
    target["partial_transform"] = partial_transform
    target["ignore"] = ignore
    target["box_supervision"] = torch.tensor(
        record.status == "positive" and (not bool(ignore)), dtype=torch.bool
    )
    target["spatial_padding_mask"] = geometry.padding_mask("merged")
    target["target_geometry"] = GROOT_N17_CROP_TARGET_GEOMETRY
    return target


class GrootCachedStaticDataset(Dataset[GrootStaticExample]):
    def __init__(
        self,
        records: Sequence[StaticFrameRecord],
        *,
        split: str,
        prompt_seed: int,
        output_size: tuple[int, int],
        cache_records: Mapping[tuple[str, str, int], Any],
    ) -> None:
        from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS

        if tuple(output_size) != GROOT_N17_GEOMETRY.output_size:
            raise ValueError(
                f"GR00T cached targets require {GROOT_N17_GEOMETRY.output_size}, got {output_size}"
            )
        self.records = tuple(
            (
                record
                for record in records
                if record.split == split and record.status in TRAINABLE_STATUSES
            )
        )
        if not self.records:
            raise ValueError(f"cached static split {split!r} is empty")
        assignments = deterministic_mode_assignments(
            [record.sample_id for record in self.records], seed=prompt_seed
        )
        self.prompt_planner = PromptPlanner(
            assignments, seed=prompt_seed, canonical_tasks=CANONICAL_TASK_PROMPTS
        )
        self.cache_records = cache_records
        self.required_cache_keys: set[tuple[str, str, int]] = set()
        for index in range(len(self.records)):
            self[index]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GrootStaticExample:
        record = self.records[index]
        prompt = self.prompt_planner.select(record.sample_id, _actual_prompt(record))
        key = (record.sample_id, prompt.prompt_id, prompt.context_generation)
        self.required_cache_keys.add(key)
        try:
            cached = self.cache_records[key]
        except KeyError as exc:
            raise KeyError(
                f"frozen-feature cache is missing {record.sample_id}/{prompt.mode}"
            ) from exc
        expected = {
            "source_dataset": record.source_dataset,
            "source_episode_index": record.source_episode_index,
            "source_frame_index": record.source_frame_index,
            "dataset_index": record.dataset_index,
            "frame_id": record.frame_index,
            "timestamp": record.timestamp,
            "prompt_mode": prompt.mode,
            "prompt_id": prompt.prompt_id,
            "detector_generation": record.detector_generation,
            "context_generation": prompt.context_generation,
        }
        mismatches = {
            name: (value, getattr(cached, name))
            for name, value in expected.items()
            if getattr(cached, name) != value
        }
        selected_hash = hashlib.sha256(prompt.selected_prompt.encode()).hexdigest()
        if cached.selected_prompt_sha256 != selected_hash:
            mismatches["selected_prompt_sha256"] = (
                selected_hash,
                cached.selected_prompt_sha256,
            )
        if mismatches:
            raise ValueError(
                f"frozen-feature metadata mismatch for {record.sample_id}: {mismatches}"
            )
        return GrootStaticExample(
            sample_key=record.sample_id,
            contexts=({},),
            targets=(build_groot_target(record),),
            records=(record,),
            prompts=(prompt,),
            context_sha256=(str(cached.context_sha256),),
        )


def validate_frame_records(records: Sequence[StaticFrameRecord]) -> None:
    if not records:
        raise ValueError("static-frame manifest is empty")
    sample_ids: set[str] = set()
    image_ids: set[int] = set()
    dataset_indices: set[int] = set()
    physical_frames: set[tuple[str, int, int]] = set()
    video_splits: dict[tuple[str, str], str] = {}
    for record in records:
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate static-frame sample_id: {record.sample_id}")
        if record.image_id in image_ids:
            raise ValueError(f"duplicate static-frame image_id: {record.image_id}")
        if record.dataset_index in dataset_indices:
            raise ValueError(f"duplicate merged dataset_index: {record.dataset_index}")
        physical = (
            record.source_dataset,
            record.source_episode_index,
            record.source_frame_index,
        )
        if physical in physical_frames:
            raise ValueError(f"duplicate physical source frame: {physical}")
        video_key = (record.source_dataset, record.source_video)
        prior_split = video_splits.setdefault(video_key, record.split)
        if prior_split != record.split and "negative_control" not in {
            prior_split,
            record.split,
        }:
            raise ValueError(
                f"source MP4 {record.source_video!r} crosses splits: {prior_split!r} and {record.split!r}"
            )
        sample_ids.add(record.sample_id)
        image_ids.add(record.image_id)
        dataset_indices.add(record.dataset_index)
        physical_frames.add(physical)


def read_frame_manifest(path: str | Path) -> tuple[StaticFrameRecord, ...]:
    source = Path(path)
    records: list[StaticFrameRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"static-frame row {line_number} must be a mapping")
            try:
                records.append(StaticFrameRecord.from_dict(payload))
            except Exception as exc:
                raise ValueError(
                    f"invalid static-frame row {line_number}: {exc}"
                ) from exc
    validate_frame_records(records)
    return tuple(records)


def _box_weight(record: StaticFrameRecord) -> float:
    if record.bbox_xyxy is None:
        return 1.0
    x1, y1, x2, y2 = record.bbox_xyxy
    short_side = min(x2 - x1, y2 - y1)
    weight = 2.5 if short_side < 35 else 1.5 if short_side <= 70 else 1.0
    center_x, center_y = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if not (
        0.25 * record.width <= center_x <= 0.75 * record.width
        and 0.25 * record.height <= center_y <= 0.75 * record.height
    ):
        weight *= 1.5
    if record.partial:
        weight *= 2.0
    left, top, right, bottom = GROOT_N17_GEOMETRY.crop_window_original
    if x1 < left or x2 > right or y1 < top or (y2 > bottom):
        weight *= 1.5
    return min(weight, 4.0)


def _target_binding(record: StaticFrameRecord, row: CacheRow) -> BoundFeature:
    target = build_groot_target(record)
    if bool(torch.as_tensor(target["ignore"])):
        raise ValueError(
            f"crop-invisible positive cannot train the shared detector: {record.sample_id}"
        )
    return BoundFeature(
        record=record,
        row=row,
        presence=torch.as_tensor(target["presence"], dtype=torch.float32),
        box=torch.as_tensor(target["boxes_cxcywh"], dtype=torch.float32),
        box_weight=_box_weight(record),
    )


def _training_bindings(
    config: ExperimentConfig, contract: CacheContract
) -> tuple[BoundFeature, ...]:
    from endo_bbox_vla.utils import _resolve_repo_path

    data = config.section("data")
    records = tuple(read_frame_manifest(_resolve_repo_path(data["train_manifest"])))
    rows = {row.identity: row for row in contract.rows}
    base = GrootCachedStaticDataset(
        records,
        split="train",
        prompt_seed=int(data["prompt_seed"]),
        output_size=GROOT_N17_GEOMETRY.output_size,
        cache_records=rows,
    )
    output: list[BoundFeature] = []
    for index in range(len(base)):
        example = base[index]
        prompt = example.prompts[0]
        record = example.records[0]
        row = rows[record.sample_id, prompt.prompt_id, prompt.context_generation]
        output.append(_target_binding(record, row))
    if len(output) != int(data["expected_counts"]["train"]):
        raise ValueError("training binding count changed")
    return tuple(output)


def _actual_training_prompt(record: StaticFrameRecord) -> str:
    from endo_bbox_vla.config import (
        CANONICAL_TASK_PROMPTS,
        TASK1_SOURCE_DATASET,
        TASK4_SOURCE_DATASET,
    )

    if record.source_dataset == TASK1_SOURCE_DATASET:
        return CANONICAL_TASK_PROMPTS[0]
    if record.source_dataset == TASK4_SOURCE_DATASET:
        return CANONICAL_TASK_PROMPTS[3]
    raise ValueError(f"unsupported block-11 visual source: {record.source_dataset}")


def _evaluation_bindings(
    config: ExperimentConfig,
    contract: CacheContract,
    split: Literal["validation", "test"],
) -> tuple[BoundFeature, ...]:
    from endo_bbox_vla.utils import _resolve_repo_path

    data = config.section("data")
    records = tuple(
        (
            record
            for record in read_frame_manifest(
                _resolve_repo_path(data[f"{split}_manifest"])
            )
            if record.status in {"positive", "confirmed_negative"}
        )
    )
    by_sample = {row.sample_id: row for row in contract.rows}
    output: list[BoundFeature] = []
    for record in records:
        row = by_sample.get(record.sample_id)
        if row is None:
            raise KeyError(f"evaluation cache misses {record.sample_id}")
        prompt = _actual_training_prompt(record)
        expected_prompt_id = "actual:" + hashlib.sha256(prompt.encode()).hexdigest()
        expected = {
            "source_dataset": record.source_dataset,
            "source_episode_index": record.source_episode_index,
            "source_frame_index": record.source_frame_index,
            "dataset_index": record.dataset_index,
            "frame_id": record.frame_index,
            "timestamp": record.timestamp,
            "prompt_mode": "actual",
            "prompt_id": expected_prompt_id,
            "detector_generation": record.detector_generation,
            "context_generation": 0,
        }
        mismatches = {
            key: (wanted, getattr(row, key))
            for key, wanted in expected.items()
            if getattr(row, key) != wanted
        }
        if row.selected_prompt_sha256 != hashlib.sha256(prompt.encode()).hexdigest():
            mismatches["selected_prompt_sha256"] = "changed"
        if mismatches:
            raise ValueError(
                f"evaluation cache metadata changed for {record.sample_id}: {mismatches}"
            )
        output.append(_target_binding(record, row))
    if len(output) != int(data["expected_counts"][split]):
        raise ValueError(f"{split} binding count changed")
    return tuple(output)


def apportion(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total <= 0 or not weights:
        raise ValueError(
            "apportionment requires a positive total and non-empty weights"
        )
    parsed = {name: float(value) for name, value in weights.items()}
    if any((not math.isfinite(value) or value < 0 for value in parsed.values())):
        raise ValueError("sampler weights must be finite and non-negative")
    denominator = sum(parsed.values())
    if denominator <= 0:
        raise ValueError("sampler weights must have positive mass")
    raw = {name: total * value / denominator for name, value in parsed.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(parsed, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


class ScenarioRecord(Protocol):
    status: str
    source_dataset: str
    source_episode_index: int


TASK4_DATASET = "Task4_Polyp_1to8"


def static_train_scenario(source_episode_index: int) -> str:
    episode = int(source_episode_index)
    if 0 <= episode < 30:
        return "S1"
    if 50 <= episode < 100:
        return "S2"
    if 100 <= episode < 150:
        return "S3"
    if 150 <= episode < 200:
        return "S4"
    if 200 <= episode < 210:
        return "S5"
    raise ValueError(
        f"Task4 positive episode is outside scenario-static-v1 train: {episode}"
    )


TASK1_DATASET = "ENDO_task1_Withdraw_400episode"


class ScenarioEpisodeBalancedSampler(Sampler[int]):
    """Balance presence, scenario, negative source, and source episodes."""

    def __init__(
        self, records: Sequence[ScenarioRecord], *, seed: int, samples_per_epoch: int
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.records = tuple(records)
        self.seed = int(seed)
        self.samples_per_epoch = int(samples_per_epoch)
        self.epoch = 0
        self.groups: dict[str, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record in enumerate(self.records):
            if record.status == "positive":
                if record.source_dataset != TASK4_DATASET:
                    raise ValueError(
                        "scenario-static-v1 positives must come from Task4"
                    )
                branch = (
                    f"positive_{static_train_scenario(record.source_episode_index)}"
                )
            elif record.status == "confirmed_negative":
                if record.source_dataset == TASK4_DATASET:
                    branch = "negative_Task4"
                elif record.source_dataset == TASK1_DATASET:
                    branch = "negative_Task1"
                else:
                    raise ValueError(
                        f"unsupported negative source {record.source_dataset!r}"
                    )
            else:
                raise ValueError(f"untrainable sampler status {record.status!r}")
            self.groups[branch][record.source_episode_index].append(index)
        expected = {
            *(f"positive_S{number}" for number in range(1, 6)),
            "negative_Task4",
            "negative_Task1",
        }
        if set(self.groups) != expected:
            raise ValueError(
                f"scenario-balanced sampler branch mismatch: {sorted(self.groups)} != {sorted(expected)}"
            )
        presence = apportion(self.samples_per_epoch, {"positive": 0.5, "negative": 0.5})
        self.branch_counts = {
            **{
                f"positive_{name}": count
                for name, count in apportion(
                    presence["positive"], {f"S{number}": 0.2 for number in range(1, 6)}
                ).items()
            },
            **{
                f"negative_{name}": count
                for name, count in apportion(
                    presence["negative"], {"Task4": 0.5, "Task1": 0.5}
                ).items()
            },
        }
        if sum(self.branch_counts.values()) != self.samples_per_epoch:
            raise RuntimeError("scenario sampler apportionment drift")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _draw_episode_balanced(
        groups: Mapping[int, Sequence[int]], count: int, *, generator: torch.Generator
    ) -> list[int]:
        episodes = sorted(groups)
        if not episodes:
            raise ValueError("cannot draw from an empty episode branch")
        frame_orders: dict[int, list[int]] = {}
        frame_cursors: dict[int, int] = defaultdict(int)
        output: list[int] = []
        while len(output) < count:
            episode_order = torch.randperm(len(episodes), generator=generator).tolist()
            for episode_offset in episode_order:
                episode = episodes[episode_offset]
                frames = groups[episode]
                cursor = frame_cursors[episode]
                order = frame_orders.get(episode)
                if order is None or cursor >= len(order):
                    order = torch.randperm(len(frames), generator=generator).tolist()
                    frame_orders[episode] = order
                    cursor = 0
                output.append(int(frames[order[cursor]]))
                frame_cursors[episode] = cursor + 1
                if len(output) == count:
                    break
        return output

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        indices: list[int] = []
        for branch in sorted(self.branch_counts):
            indices.extend(
                self._draw_episode_balanced(
                    self.groups[branch], self.branch_counts[branch], generator=generator
                )
            )
        if len(indices) != self.samples_per_epoch:
            raise RuntimeError("scenario sampler emitted the wrong epoch length")
        shuffled = torch.randperm(len(indices), generator=generator).tolist()
        return iter([indices[index] for index in shuffled])

    def audit(self) -> dict[str, object]:
        return {
            "kind": "scenario_episode_balanced",
            "samples_per_epoch": self.samples_per_epoch,
            "branch_counts": dict(sorted(self.branch_counts.items())),
            "episode_counts": {
                branch: len(episodes)
                for branch, episodes in sorted(self.groups.items())
            },
            "sampling_with_replacement_across_an_epoch": True,
        }


class SizeLocationBalancedSampler(Sampler[int]):
    """Keep scenario/episode balance while oversampling difficult positives."""

    def __init__(
        self,
        records: Sequence[StaticFrameRecord],
        weights: Tensor,
        *,
        seed: int,
        samples_per_epoch: int,
    ) -> None:
        self.base = ScenarioEpisodeBalancedSampler(
            records, seed=seed, samples_per_epoch=samples_per_epoch
        )
        if tuple(weights.shape) != (len(records),):
            raise ValueError("size/location weights do not match records")
        self.weights = weights.to(dtype=torch.float64, device="cpu")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        output: list[int] = []
        for branch in sorted(self.base.branch_counts):
            groups = self.base.groups[branch]
            episodes = sorted(groups)
            count = self.base.branch_counts[branch]
            branch_output: list[int] = []
            while len(branch_output) < count:
                for offset in torch.randperm(
                    len(episodes), generator=generator
                ).tolist():
                    members = list(groups[episodes[offset]])
                    selected = torch.multinomial(
                        self.weights[members], 1, replacement=True, generator=generator
                    ).item()
                    branch_output.append(members[selected])
                    if len(branch_output) == count:
                        break
            output.extend(branch_output)
        shuffled = torch.randperm(len(output), generator=generator).tolist()
        return iter([output[index] for index in shuffled])

    def audit(self) -> dict[str, Any]:
        return {
            **self.base.audit(),
            "kind": "scenario_episode_size_location_balanced",
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
        }


EXPECTED_FULL_CLIPS = 9871

FULL_PLAN_KIND = "groot_n17_temporal_full_train_plan"


data_GROOT_N17_CROP_TARGET_GEOMETRY = "groot_n17_crop"


def _target(record: StaticFrameRecord) -> dict[str, Any]:
    boxes = (
        torch.tensor([record.bbox_xyxy], dtype=torch.float32)
        if record.bbox_xyxy is not None
        else torch.empty((0, 4), dtype=torch.float32)
    )
    return {
        "boxes": boxes,
        "presence": torch.tensor(
            float(record.status == "positive"), dtype=torch.float32
        ),
        "image_id": torch.tensor(record.image_id, dtype=torch.int64),
        "original_size": torch.tensor([record.height, record.width], dtype=torch.int64),
        "size": torch.tensor([record.height, record.width], dtype=torch.int64),
    }


def normalized_target_from_xyxy(target: Mapping[str, Any]) -> Tensor:
    """Return one normalized cxcywh box or zeros for a confirmed negative."""
    boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return torch.zeros(4, dtype=torch.float32)
    if len(boxes) != 1:
        raise ValueError("endo_bbox_vla supports at most one polyp box per frame")
    size = torch.as_tensor(target["size"], dtype=torch.float32).reshape(2)
    height, width = size
    if width <= 0 or height <= 0:
        raise ValueError("transformed target size must be positive")
    return xyxy_to_cxcywh(boxes)[0] / boxes.new_tensor((width, height, width, height))


def build_groot_n17_target(
    record: StaticFrameRecord, *, geometry: GrootN17Geometry = GROOT_N17_GEOMETRY
) -> dict[str, Any]:
    """Build a crop-aware target for the saved five-task GR00T checkpoint.

    Fully crop-invisible positives retain ``presence=1`` but carry
    ``ignore=True`` and no bbox. Callers must filter/ignore the whole sample;
    they must never reinterpret it as a confirmed negative.
    """
    if (record.width, record.height) != geometry.original_size:
        raise ValueError(
            f"GR00T N1.7 target geometry requires source size {geometry.original_size}, got {(record.width, record.height)}"
        )
    target = _target(record)
    projection = geometry.project_xyxy(target["boxes"], partial_original=record.partial)
    if record.status == "positive":
        if len(projection.ignore) != 1:
            raise RuntimeError(
                "positive endo_bbox_vla target must contain exactly one box"
            )
        ignore = projection.ignore[0]
        partial_original = projection.partial_original[0]
        partial_transform = projection.partial_transform[0]
        visible_fraction = projection.visible_fraction[0]
        boxes = projection.output_boxes_xyxy[projection.box_supervision]
    else:
        ignore = torch.tensor(False, dtype=torch.bool)
        partial_original = torch.tensor(record.partial, dtype=torch.bool)
        partial_transform = torch.tensor(False, dtype=torch.bool)
        visible_fraction = torch.tensor(0.0, dtype=torch.float32)
        boxes = target["boxes"]
    target["boxes"] = boxes
    target["area"] = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    target["size"] = torch.tensor(
        [geometry.output_size[1], geometry.output_size[0]], dtype=torch.int64
    )
    target["boxes_cxcywh"] = normalized_target_from_xyxy(target)
    target["visible_fraction"] = visible_fraction
    target["partial_original"] = partial_original
    target["partial_transform"] = partial_transform
    target["ignore"] = ignore
    target["box_supervision"] = torch.tensor(
        record.status == "positive" and (not bool(ignore)), dtype=torch.bool
    )
    target["spatial_padding_mask"] = geometry.padding_mask("merged")
    target["target_geometry"] = data_GROOT_N17_CROP_TARGET_GEOMETRY
    return target


CLIP_SCHEMA_VERSION = 2


def _data_strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _data_safe_relative_posix(value: Any, name: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or (not path.parts):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class EndoBBoxVLAClipRecord:
    clip_id: str
    sample_ids: tuple[str, str, str, str]
    episode_index: int
    source_video: str
    split: str
    detector_generation: int
    context_generations: tuple[int, int, int, int]
    source_category: str
    anchor_sample_id: str | None
    prompt_switch_index: int | None
    schema_version: int = CLIP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        from endo_bbox_vla.config import CLIP_LENGTH

        if not self.clip_id or self.clip_id.strip() != self.clip_id:
            raise ValueError("clip_id must be a non-empty, trimmed string")
        if (
            len(self.sample_ids) != CLIP_LENGTH
            or len(set(self.sample_ids)) != CLIP_LENGTH
        ):
            raise ValueError(
                f"endo_bbox_vla clips must contain {CLIP_LENGTH} distinct frames"
            )
        if any((not sample_id for sample_id in self.sample_ids)):
            raise ValueError("clip sample IDs must not be empty")
        _data_strict_int(self.episode_index, "episode_index")
        _data_strict_int(self.detector_generation, "detector_generation")
        if len(self.context_generations) != CLIP_LENGTH:
            raise ValueError(
                "clip context_generations must contain one value per frame"
            )
        for value in self.context_generations:
            _data_strict_int(value, "context_generation")
        if any(
            (
                second < first
                for first, second in zip(
                    self.context_generations, self.context_generations[1:], strict=False
                )
            )
        ):
            raise ValueError("clip context_generations must be non-decreasing")
        if self.source_category not in {
            "task4_positive_anchor",
            "task4_pure_negative",
            "task1_hard_negative_anchor",
            "task1_random_pure_negative",
        }:
            raise ValueError(
                f"unsupported temporal source_category {self.source_category!r}"
            )
        anchored = self.source_category in {
            "task4_positive_anchor",
            "task1_hard_negative_anchor",
        }
        if anchored and (not self.anchor_sample_id):
            raise ValueError(
                "anchored temporal clips require anchor_sample_id provenance"
            )
        if not anchored and self.anchor_sample_id is not None:
            raise ValueError(
                "random pure-negative temporal clips must not declare an anchor"
            )
        if self.prompt_switch_index is not None:
            if self.source_category != "task4_positive_anchor":
                raise ValueError(
                    "prompt switch is allowed only for positive Task4 clips"
                )
            if self.prompt_switch_index != CLIP_LENGTH // 2:
                raise ValueError(
                    "endo_bbox_vla prompt switch must occur at the middle frame"
                )
            expected = (0, 0, 1, 1)
            if self.context_generations != expected:
                raise ValueError(
                    f"switched clips must rotate semantic generation as {expected}"
                )
        elif len(set(self.context_generations)) != 1:
            raise ValueError(
                "non-switched clips must keep one semantic context generation"
            )
        _data_safe_relative_posix(self.source_video, "source_video")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported endo_bbox_vla split: {self.split!r}")
        if self.schema_version != CLIP_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported endo_bbox_vla clip schema {self.schema_version}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EndoBBoxVLAClipRecord:
        fields = {
            "clip_id",
            "sample_ids",
            "episode_index",
            "source_video",
            "split",
            "detector_generation",
            "context_generations",
            "source_category",
            "anchor_sample_id",
            "prompt_switch_index",
            "schema_version",
        }
        if set(payload) != fields:
            raise ValueError(
                f"endo_bbox_vla clip fields must be exactly {sorted(fields)}, got {sorted(payload)}"
            )
        sample_ids = payload["sample_ids"]
        if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, str | bytes):
            raise TypeError("clip sample_ids must be a sequence")
        return cls(
            clip_id=str(payload["clip_id"]),
            sample_ids=tuple((str(item) for item in sample_ids)),
            episode_index=_data_strict_int(payload["episode_index"], "episode_index"),
            source_video=_data_safe_relative_posix(
                payload["source_video"], "source_video"
            ),
            split=str(payload["split"]),
            detector_generation=_data_strict_int(
                payload["detector_generation"], "detector_generation"
            ),
            context_generations=tuple(
                (
                    _data_strict_int(value, "context_generation")
                    for value in payload["context_generations"]
                )
            ),
            source_category=str(payload["source_category"]),
            anchor_sample_id=None
            if payload["anchor_sample_id"] is None
            else str(payload["anchor_sample_id"]),
            prompt_switch_index=None
            if payload["prompt_switch_index"] is None
            else _data_strict_int(
                payload["prompt_switch_index"], "prompt_switch_index"
            ),
            schema_version=_data_strict_int(
                payload["schema_version"], "schema_version", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "sample_ids": list(self.sample_ids),
            "context_generations": list(self.context_generations),
        }


@dataclass(frozen=True, slots=True)
class GrootTemporalSmokeFramePlan:
    frame_offset: int
    sample_id: str
    source_dataset: str
    source_episode_index: int
    source_frame_index: int
    dataset_index: int
    frame_id: int
    timestamp: float
    detector_generation: int
    status: str
    prompt_mode: str
    actual_prompt: str
    selected_prompt: str
    prompt_id: str
    context_generation: int

    @property
    def conditioning_key(self) -> tuple[str, str, int]:
        return (self.sample_id, self.prompt_id, self.context_generation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_offset": self.frame_offset,
            "sample_id": self.sample_id,
            "source_dataset": self.source_dataset,
            "source_episode_index": self.source_episode_index,
            "source_frame_index": self.source_frame_index,
            "dataset_index": self.dataset_index,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detector_generation": self.detector_generation,
            "status": self.status,
            "prompt_mode": self.prompt_mode,
            "actual_prompt": self.actual_prompt,
            "selected_prompt": self.selected_prompt,
            "prompt_id": self.prompt_id,
            "context_generation": self.context_generation,
            "local_cache_key": self.sample_id,
            "conditioning_key": {
                "sample_id": self.sample_id,
                "prompt_id": self.prompt_id,
                "context_generation": self.context_generation,
            },
        }


@dataclass(frozen=True, slots=True)
class GrootTemporalFullClipPlan:
    record: EndoBBoxVLAClipRecord
    base_prompt_mode: str
    frames: tuple[
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
    ]

    @property
    def clip_id(self) -> str:
        return self.record.clip_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.record.to_dict(),
            "base_prompt_mode": self.base_prompt_mode,
            "frames": [frame.to_dict() for frame in self.frames],
        }


FULL_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GrootTemporalFullPlan:
    prompt_seed: int
    canonical_full_clip_set_sha256: str
    canonical_full_prompt_assignment_sha256: str
    clips: tuple[GrootTemporalFullClipPlan, ...]
    _summary_json: str

    @property
    def summary(self) -> dict[str, Any]:
        value = json.loads(self._summary_json)
        if not isinstance(value, dict):
            raise RuntimeError("internal GR00T temporal full summary is not an object")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FULL_PLAN_SCHEMA_VERSION,
            "kind": FULL_PLAN_KIND,
            "prompt_seed": self.prompt_seed,
            "canonical_full_clip_set_sha256": self.canonical_full_clip_set_sha256,
            "canonical_full_prompt_assignment_sha256": self.canonical_full_prompt_assignment_sha256,
            "summary": self.summary,
            "clips": [clip.to_dict() for clip in self.clips],
        }


DEFAULT_PROMPT_SEED = 42


def validate_clip_records(
    clips: Sequence[EndoBBoxVLAClipRecord],
    frames_by_id: Mapping[str, StaticFrameRecord],
) -> None:
    from endo_bbox_vla.config import CLIP_LENGTH

    if not clips:
        raise ValueError("endo_bbox_vla clip manifest is empty")
    clip_ids: set[str] = set()
    for clip in clips:
        if clip.clip_id in clip_ids:
            raise ValueError(f"duplicate endo_bbox_vla clip_id: {clip.clip_id}")
        clip_ids.add(clip.clip_id)
        try:
            frames = tuple((frames_by_id[sample_id] for sample_id in clip.sample_ids))
        except KeyError as exc:
            raise ValueError(
                f"clip {clip.clip_id} references an unknown frame: {exc.args[0]}"
            ) from exc
        if any((frame.status == UNCERTAIN_STATUS for frame in frames)):
            raise ValueError(f"clip {clip.clip_id} contains an uncertain frame")
        if any((frame.status not in TRAINABLE_STATUSES for frame in frames)):
            raise ValueError(f"clip {clip.clip_id} contains a non-trainable frame")
        positives = {frame.sample_id for frame in frames if frame.status == "positive"}
        if clip.source_category == "task4_positive_anchor":
            if clip.anchor_sample_id not in positives:
                raise ValueError(
                    f"clip {clip.clip_id} positive anchor is absent or not positive"
                )
        elif clip.source_category == "task1_hard_negative_anchor":
            if positives:
                raise ValueError(
                    f"Task1 hard-negative clip {clip.clip_id} contains a positive"
                )
            if clip.anchor_sample_id not in clip.sample_ids:
                raise ValueError(
                    f"Task1 hard-negative anchor is absent from {clip.clip_id}"
                )
            anchor_frame = frames[clip.sample_ids.index(str(clip.anchor_sample_id))]
            if (
                anchor_frame.source_dataset != "ENDO_task1_Withdraw_400episode"
                or anchor_frame.status != "confirmed_negative"
            ):
                raise ValueError(
                    f"invalid Task1 hard-negative anchor in {clip.clip_id}"
                )
        elif positives:
            raise ValueError(
                f"pure-negative clip {clip.clip_id} contains a positive frame"
            )
        identities = {
            (
                frame.episode_index,
                frame.source_video,
                frame.split,
                frame.detector_generation,
            )
            for frame in frames
        }
        expected = {
            (
                clip.episode_index,
                clip.source_video,
                clip.split,
                clip.detector_generation,
            )
        }
        if identities != expected:
            raise ValueError(
                f"clip {clip.clip_id} crosses an episode, MP4, split, or detector generation"
            )
        frame_indices = [frame.frame_index for frame in frames]
        dataset_indices = [frame.dataset_index for frame in frames]
        timestamps = [frame.timestamp for frame in frames]
        if (
            frame_indices != sorted(frame_indices)
            or len(set(frame_indices)) != CLIP_LENGTH
        ):
            raise ValueError(
                f"clip {clip.clip_id} frame indices are not strictly increasing"
            )
        if (
            dataset_indices != sorted(dataset_indices)
            or len(set(dataset_indices)) != CLIP_LENGTH
        ):
            raise ValueError(
                f"clip {clip.clip_id} dataset indices are not strictly increasing"
            )
        if any(
            (
                second <= first
                for first, second in zip(timestamps, timestamps[1:], strict=False)
            )
        ):
            raise ValueError(
                f"clip {clip.clip_id} timestamps are not strictly increasing"
            )
        source_indices = [frame.source_frame_index for frame in frames]
        if any(
            (
                second - first != 6
                for first, second in zip(
                    source_indices, source_indices[1:], strict=False
                )
            )
        ):
            raise ValueError(
                f"clip {clip.clip_id} is not exact 5 Hz (+6 source frames)"
            )
        if any(
            (
                not math.isclose(second - first, 0.2, rel_tol=0.0, abs_tol=0.001)
                for first, second in zip(timestamps, timestamps[1:], strict=False)
            )
        ):
            raise ValueError(f"clip {clip.clip_id} is not exact 5 Hz (0.2 seconds)")


def read_clip_manifest(
    path: str | Path, frames_by_id: Mapping[str, StaticFrameRecord]
) -> tuple[EndoBBoxVLAClipRecord, ...]:
    clips: list[EndoBBoxVLAClipRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"endo_bbox_vla clip row {line_number} must be a mapping"
                )
            try:
                clips.append(EndoBBoxVLAClipRecord.from_dict(payload))
            except Exception as exc:
                raise ValueError(
                    f"invalid endo_bbox_vla clip row {line_number}: {exc}"
                ) from exc
    validate_clip_records(clips, frames_by_id)
    return tuple(clips)


EXPECTED_FULL_PROMPT_SWITCHES = 824

EXPECTED_FULL_CLIP_COUNTS = MappingProxyType(
    {
        "task4_positive_anchor": 3297,
        "task1_hard_negative_anchor": 2191,
        "task1_random_pure_negative": 4125,
        "task4_pure_negative": 258,
    }
)

EXPECTED_FULL_PROMPT_COUNTS = MappingProxyType(
    {"actual": 4935, "counterfactual": 2468, "neutral": 2468}
)


def _validate_full_manifest_and_assign_prompts(
    clips: Sequence[EndoBBoxVLAClipRecord], *, prompt_seed: int
) -> dict[str, str]:
    if isinstance(prompt_seed, bool) or not isinstance(prompt_seed, int):
        raise TypeError("prompt_seed must be an integer")
    clip_ids = [clip.clip_id for clip in clips]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("full temporal manifest contains duplicate clip IDs")
    category_counts = Counter((clip.source_category for clip in clips))
    if category_counts != Counter(EXPECTED_FULL_CLIP_COUNTS):
        raise ValueError(
            f"GR00T smoke selection requires the complete frozen temporal manifest: expected categories {dict(EXPECTED_FULL_CLIP_COUNTS)}, got {dict(category_counts)}"
        )
    if any((clip.split != "train" for clip in clips)):
        raise ValueError("GR00T temporal smoke source must contain train clips only")
    switch_count = sum((clip.prompt_switch_index is not None for clip in clips))
    if switch_count != EXPECTED_FULL_PROMPT_SWITCHES:
        raise ValueError(
            f"GR00T smoke selection requires the frozen prompt-switch population: expected {EXPECTED_FULL_PROMPT_SWITCHES}, got {switch_count}"
        )
    assignments = deterministic_mode_assignments(clip_ids, seed=prompt_seed)
    assignment_counts = Counter(assignments.values())
    if assignment_counts != Counter(EXPECTED_FULL_PROMPT_COUNTS):
        raise ValueError(
            f"full-manifest prompt assignment drifted: expected {dict(EXPECTED_FULL_PROMPT_COUNTS)}, got {dict(assignment_counts)}"
        )
    return assignments


def _overlap_summary(
    selected_frames: Sequence[StaticFrameRecord],
    evaluation_records: Sequence[StaticFrameRecord] | None,
) -> dict[str, Any]:
    if evaluation_records is None:
        return {
            "evaluation_overlap_checked": False,
            "evaluation_physical_frame_overlap": None,
            "evaluation_source_episode_overlap": None,
        }
    selected_physical = {
        (value.source_dataset, value.source_episode_index, value.source_frame_index)
        for value in selected_frames
    }
    evaluation_physical = {
        (value.source_dataset, value.source_episode_index, value.source_frame_index)
        for value in evaluation_records
    }
    selected_episodes = {
        (value.source_dataset, value.source_episode_index) for value in selected_frames
    }
    evaluation_episodes = {
        (value.source_dataset, value.source_episode_index)
        for value in evaluation_records
    }
    physical_overlap = selected_physical & evaluation_physical
    episode_overlap = selected_episodes & evaluation_episodes
    if physical_overlap or episode_overlap:
        raise ValueError(
            f"GR00T temporal smoke selection leaks evaluation identity: physical_frames={len(physical_overlap)}, source_episodes={len(episode_overlap)}"
        )
    return {
        "evaluation_overlap_checked": True,
        "evaluation_physical_frame_overlap": 0,
        "evaluation_source_episode_overlap": 0,
    }


EXPECTED_FULL_FRAME_OCCURRENCES = 39484

EXPECTED_FULL_UNIQUE_VISUAL_FRAMES = 32528

EXPECTED_FULL_LOGICAL_CONDITIONING = 37094

EXPECTED_FULL_PHYSICAL_CONDITIONING = 36709

EXPECTED_FULL_GENERATION_ALIASES = 385

EXPECTED_FULL_FRAME_PROMPT_COUNTS = {
    "actual": 19660,
    "counterfactual": 10744,
    "neutral": 9080,
}

EXPECTED_FULL_CONTEXT_GENERATION_COUNTS = {0: 37836, 1: 1648}

EXPECTED_FULL_FRAME_STATUS_COUNTS = {"confirmed_negative": 26296, "positive": 13188}

EXPECTED_FULL_SOURCE_FRAME_COUNTS = {
    contracts_TASK1_SOURCE_DATASET: 25264,
    contracts_TASK4_SOURCE_DATASET: 14220,
}

EXPECTED_FULL_MOTION_TRANSITIONS = 9891


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _full_summary(
    plans: Sequence[GrootTemporalFullClipPlan],
    *,
    assignments: Mapping[str, str],
    frames_by_id: Mapping[str, StaticFrameRecord],
    evaluation_records: Sequence[StaticFrameRecord] | None,
) -> dict[str, Any]:
    frames = [frame for clip in plans for frame in clip.frames]
    unique_sample_ids = {frame.sample_id for frame in frames}
    logical_conditioning = {frame.conditioning_key for frame in frames}
    physical_conditioning = {(frame.sample_id, frame.prompt_id) for frame in frames}
    prompt_counts = Counter((frame.prompt_mode for frame in frames))
    generation_counts = Counter((frame.context_generation for frame in frames))
    status_counts = Counter((frame.status for frame in frames))
    source_counts = Counter((frame.source_dataset for frame in frames))
    category_counts = Counter((clip.record.source_category for clip in plans))
    switch_count = sum((clip.record.prompt_switch_index is not None for clip in plans))
    motion_transitions = sum(
        (
            first.status == "positive" and second.status == "positive"
            for clip in plans
            for first, second in zip(clip.frames, clip.frames[1:], strict=False)
        )
    )
    selected_records = [frames_by_id[frame.sample_id] for frame in frames]
    checks = {
        "clip_count": len(plans) == EXPECTED_FULL_CLIPS,
        "clip_categories": dict(category_counts) == dict(EXPECTED_FULL_CLIP_COUNTS),
        "base_prompt_counts": Counter(assignments.values())
        == Counter(EXPECTED_FULL_PROMPT_COUNTS),
        "prompt_switch_count": switch_count == EXPECTED_FULL_PROMPT_SWITCHES,
        "frame_occurrences": len(frames) == EXPECTED_FULL_FRAME_OCCURRENCES,
        "unique_e03": len(unique_sample_ids) == EXPECTED_FULL_UNIQUE_VISUAL_FRAMES,
        "logical_conditioning": len(logical_conditioning)
        == EXPECTED_FULL_LOGICAL_CONDITIONING,
        "physical_conditioning": len(physical_conditioning)
        == EXPECTED_FULL_PHYSICAL_CONDITIONING,
        "generation_aliases": len(logical_conditioning) - len(physical_conditioning)
        == EXPECTED_FULL_GENERATION_ALIASES,
        "frame_prompt_counts": dict(prompt_counts) == EXPECTED_FULL_FRAME_PROMPT_COUNTS,
        "context_generation_counts": dict(generation_counts)
        == EXPECTED_FULL_CONTEXT_GENERATION_COUNTS,
        "frame_status_counts": dict(status_counts) == EXPECTED_FULL_FRAME_STATUS_COUNTS,
        "source_frame_counts": dict(source_counts) == EXPECTED_FULL_SOURCE_FRAME_COUNTS,
        "motion_transitions": motion_transitions == EXPECTED_FULL_MOTION_TRANSITIONS,
    }
    failed = sorted((name for name, passed in checks.items() if not passed))
    if failed:
        raise ValueError(f"full GR00T temporal plan contract failed: {failed}")
    overlap = _overlap_summary(selected_records, evaluation_records)
    return {
        "full_clip_count": len(plans),
        "full_prompt_assignment_counts": {
            mode: Counter(assignments.values())[mode] for mode in PROMPT_MODES
        },
        "frame_occurrence_count": len(frames),
        "unique_frame_count": len(unique_sample_ids),
        "e03_logical_entry_count": len(unique_sample_ids),
        "logical_conditioning_key_count": len(logical_conditioning),
        "physical_conditioning_tensor_count": len(physical_conditioning),
        "generation_alias_count": len(logical_conditioning)
        - len(physical_conditioning),
        "minimum_feature_forward_count": len(physical_conditioning),
        "clip_counts_by_source_category": dict(sorted(category_counts.items())),
        "prompt_switch_clip_count": switch_count,
        "frame_prompt_mode_counts": {
            mode: prompt_counts[mode] for mode in PROMPT_MODES
        },
        "context_generation_counts": {
            str(key): value for key, value in sorted(generation_counts.items())
        },
        "frame_status_counts": dict(sorted(status_counts.items())),
        "source_frame_counts": dict(sorted(source_counts.items())),
        "motion_supervised_transition_count": motion_transitions,
        "full_rows_sha256": _canonical_sha256([clip.to_dict() for clip in plans]),
        "all_exact_contract_checks_passed": True,
        **overlap,
    }


def _groot_temporal_smoke_actual_prompt(record: StaticFrameRecord) -> str:
    from endo_bbox_vla.config import (
        contracts_CANONICAL_TASK_PROMPTS,
        contracts_TASK1_SOURCE_DATASET,
        contracts_TASK4_SOURCE_DATASET,
    )

    if record.source_dataset == contracts_TASK1_SOURCE_DATASET:
        return contracts_CANONICAL_TASK_PROMPTS[0]
    if record.source_dataset == contracts_TASK4_SOURCE_DATASET:
        return contracts_CANONICAL_TASK_PROMPTS[3]
    raise ValueError(
        f"unsupported GR00T smoke source dataset: {record.source_dataset!r}"
    )


def _frame_plans(
    clip: EndoBBoxVLAClipRecord,
    *,
    base_prompt_mode: str,
    planner: PromptPlanner,
    frames_by_id: Mapping[str, StaticFrameRecord],
) -> tuple[
    GrootTemporalSmokeFramePlan,
    GrootTemporalSmokeFramePlan,
    GrootTemporalSmokeFramePlan,
    GrootTemporalSmokeFramePlan,
]:
    rows: list[GrootTemporalSmokeFramePlan] = []
    for frame_offset, sample_id in enumerate(clip.sample_ids):
        record = frames_by_id[sample_id]
        actual_prompt = _groot_temporal_smoke_actual_prompt(record)
        mode = (
            "counterfactual"
            if clip.prompt_switch_index is not None
            and frame_offset < clip.prompt_switch_index
            else "actual"
            if clip.prompt_switch_index is not None
            else base_prompt_mode
        )
        prompt = planner.select_mode(
            clip.clip_id,
            actual_prompt,
            mode,
            context_generation=clip.context_generations[frame_offset],
        )
        rows.append(
            GrootTemporalSmokeFramePlan(
                frame_offset=frame_offset,
                sample_id=record.sample_id,
                source_dataset=record.source_dataset,
                source_episode_index=record.source_episode_index,
                source_frame_index=record.source_frame_index,
                dataset_index=record.dataset_index,
                frame_id=record.frame_index,
                timestamp=record.timestamp,
                detector_generation=record.detector_generation,
                status=record.status,
                prompt_mode=prompt.mode,
                actual_prompt=prompt.actual_prompt,
                selected_prompt=prompt.selected_prompt,
                prompt_id=prompt.prompt_id,
                context_generation=prompt.context_generation,
            )
        )
    if len(rows) != 4:
        raise RuntimeError("GR00T smoke clips must contain exactly four frames")
    return tuple(rows)


def build_groot_temporal_full_plan(
    clips: Sequence[EndoBBoxVLAClipRecord],
    frames_by_id: Mapping[str, StaticFrameRecord],
    *,
    prompt_seed: int = DEFAULT_PROMPT_SEED,
    evaluation_records: Sequence[StaticFrameRecord] | None = None,
) -> GrootTemporalFullPlan:
    """Build the immutable full-train plan without selecting a smoke subset."""
    full_clips = tuple(clips)
    if not full_clips:
        raise ValueError("full temporal clip manifest is empty")
    validate_clip_records(full_clips, frames_by_id)
    assignments = _validate_full_manifest_and_assign_prompts(
        full_clips, prompt_seed=prompt_seed
    )
    planner = PromptPlanner(assignments, seed=prompt_seed)
    plans = tuple(
        (
            GrootTemporalFullClipPlan(
                record=clip,
                base_prompt_mode=assignments[clip.clip_id],
                frames=_frame_plans(
                    clip,
                    base_prompt_mode=assignments[clip.clip_id],
                    planner=planner,
                    frames_by_id=frames_by_id,
                ),
            )
            for clip in sorted(full_clips, key=lambda value: value.clip_id)
        )
    )
    summary = _full_summary(
        plans,
        assignments=assignments,
        frames_by_id=frames_by_id,
        evaluation_records=evaluation_records,
    )
    canonical_clips = [
        clip.to_dict() for clip in sorted(full_clips, key=lambda x: x.clip_id)
    ]
    canonical_assignments = [
        {"clip_id": clip_id, "prompt_mode": assignments[clip_id]}
        for clip_id in sorted(assignments)
    ]
    return GrootTemporalFullPlan(
        prompt_seed=prompt_seed,
        canonical_full_clip_set_sha256=_canonical_sha256(canonical_clips),
        canonical_full_prompt_assignment_sha256=_canonical_sha256(
            canonical_assignments
        ),
        clips=plans,
        _summary_json=json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


def build_groot_temporal_full_plan_from_manifests(
    *,
    temporal_frame_manifest: str | Path,
    clip_manifest: str | Path,
    evaluation_frame_manifest: str | Path,
    prompt_seed: int = DEFAULT_PROMPT_SEED,
) -> GrootTemporalFullPlan:
    frames = read_frame_manifest(temporal_frame_manifest)
    frames_by_id = {record.sample_id: record for record in frames}
    clips = read_clip_manifest(clip_manifest, frames_by_id)
    evaluation = read_frame_manifest(evaluation_frame_manifest)
    return build_groot_temporal_full_plan(
        clips, frames_by_id, prompt_seed=prompt_seed, evaluation_records=evaluation
    )


SCENARIO_STATIC_PROTOCOL_ID = "scenario-static-v1"


@dataclass(frozen=True, slots=True)
class ScenarioStaticDataContract:
    source_config: Path
    source_config_sha256: str
    dataset_root: Path
    dataset_repo: str
    dataset_revision: str
    video_backend: str
    camera_key: str
    manifests: dict[str, Path]
    manifest_sha256: dict[str, str]
    counts: dict[str, dict[str, int]]
    samples_per_epoch: int
    protocol_id: str = SCENARIO_STATIC_PROTOCOL_ID

    def records(self, view: str) -> tuple[StaticFrameRecord, ...]:
        from endo_bbox_vla.utils import sha256_file

        try:
            path = self.manifests[view]
        except KeyError as exc:
            raise KeyError(f"unknown scenario-static view: {view!r}") from exc
        digest = sha256_file(path)
        if digest != self.manifest_sha256[view]:
            raise ValueError(
                f"scenario-static {view} manifest changed after contract load"
            )
        return read_frame_manifest(path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _resolve_inside_bbox(repo_root: Path, value: str | Path, *, name: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )
    bbox_root = (repo_root / "bbox_vla").resolve()
    try:
        resolved.relative_to(bbox_root)
    except ValueError as exc:
        raise ValueError(f"{name} must resolve inside bbox_vla: {resolved}") from exc
    return resolved


SCENARIO_STATIC_STUDY_ID = "groot_n17_e02_e03_scenario_static_v1"


def _declared_hash(payload: Mapping[str, Any], path: Path) -> str:
    hashes = _mapping(payload.get("output_sha256"), "provenance.output_sha256")
    direct = hashes.get(path.name)
    if isinstance(direct, str):
        return direct
    matches = [
        str(digest)
        for declared_path, digest in hashes.items()
        if Path(str(declared_path)).name == path.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"provenance does not identify exactly one hash for {path.name}"
        )
    return matches[0]


def _status_counts(records: tuple[StaticFrameRecord, ...]) -> dict[str, int]:
    statuses = Counter((record.status for record in records))
    sources = Counter(
        (
            record.source_dataset
            for record in records
            if record.status == "confirmed_negative"
        )
    )
    return {
        "task4_positive": statuses["positive"],
        "task4_confirmed_negative": sources["Task4_Polyp_1to8"],
        "task1_confirmed_negative": sources["ENDO_task1_Withdraw_400episode"],
        "total": len(records),
    }


def load_scenario_static_contract(path: str | Path) -> ScenarioStaticDataContract:
    """Load and re-verify the exact manifests consumed by the paper block-11 visual run."""
    from endo_bbox_vla.utils import sha256_file

    source = Path(path).resolve()
    repo_root = PROJECT_ROOT
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("block-11 visual source config must be a mapping")
    if (
        raw.get("schema_version") != 1
        or raw.get("study_id") != SCENARIO_STATIC_STUDY_ID
    ):
        raise ValueError(
            "source config is not the paper scenario-static block-11 visual study"
        )
    data = _mapping(raw.get("data"), "data")
    evaluation = _mapping(raw.get("evaluation"), "evaluation")
    training = _mapping(raw.get("training"), "training")
    backbone = _mapping(raw.get("backbone"), "backbone")
    sampler = _mapping(training.get("sampler"), "training.sampler")
    if sampler.get("kind") != "scenario_episode_balanced":
        raise ValueError(
            "paper block-11 visual data must use the scenario/episode-balanced sampler"
        )
    manifests = {
        "train": _resolve_inside_bbox(
            repo_root, str(data["frame_manifest"]), name="data.frame_manifest"
        ),
        "validation": _resolve_inside_bbox(
            repo_root,
            str(evaluation["validation_manifest"]),
            name="evaluation.validation_manifest",
        ),
        "test": _resolve_inside_bbox(
            repo_root, str(evaluation["test_manifest"]), name="evaluation.test_manifest"
        ),
    }
    train_provenance_path = _resolve_inside_bbox(
        repo_root, str(data["provenance_audit"]), name="data.provenance_audit"
    )
    train_provenance = json.loads(train_provenance_path.read_text(encoding="utf-8"))
    final_root = manifests["validation"].parent
    final_protocol = json.loads(
        (final_root / "protocol.json").read_text(encoding="utf-8")
    )
    final_provenance = json.loads(
        (final_root / "provenance_audit.json").read_text(encoding="utf-8")
    )
    if (
        final_protocol.get("state") != "complete"
        or final_provenance.get("state") != "complete"
    ):
        raise ValueError("scenario-static validation/test review is not complete")
    if final_protocol.get("protocol_id") != SCENARIO_STATIC_PROTOCOL_ID:
        raise ValueError("scenario-static protocol ID mismatch")
    manifest_sha256 = {
        "train": _declared_hash(train_provenance, manifests["train"]),
        "validation": _declared_hash(final_provenance, manifests["validation"]),
        "test": _declared_hash(final_provenance, manifests["test"]),
    }
    records = {
        name: read_frame_manifest(manifest) for name, manifest in manifests.items()
    }
    for name, manifest in manifests.items():
        if sha256_file(manifest) != manifest_sha256[name]:
            raise ValueError(f"scenario-static {name} manifest hash mismatch")
        if {record.split for record in records[name]} != {name}:
            raise ValueError(f"scenario-static {name} manifest has a wrong split label")
        if any((record.status == "uncertain_ignore" for record in records[name])):
            raise ValueError(f"scenario-static {name} contains uncertain supervision")
    counts = {name: _status_counts(rows) for name, rows in records.items()}
    expected_train = {
        key: int(value)
        for key, value in _mapping(
            data["expected_counts"], "data.expected_counts"
        ).items()
    }
    if counts["train"] != expected_train:
        raise ValueError(
            f"scenario-static train count drift: {counts['train']} != {expected_train}"
        )
    for name in ("validation", "test"):
        expected = {
            key: int(value)
            for key, value in _mapping(
                final_protocol[f"{name}_counts"], f"protocol.{name}_counts"
            ).items()
        }
        if counts[name] != expected:
            raise ValueError(
                f"scenario-static {name} count drift: {counts[name]} != {expected}"
            )
    return ScenarioStaticDataContract(
        source_config=source,
        source_config_sha256=sha256_file(source),
        dataset_root=_resolve_inside_bbox(
            repo_root, str(data["dataset_root"]), name="dataset_root"
        ),
        dataset_repo=str(data["dataset_repo"]),
        dataset_revision=str(data["dataset_revision"]),
        video_backend=str(data["video_backend"]),
        camera_key=str(backbone["camera_key"]),
        manifests=manifests,
        manifest_sha256=manifest_sha256,
        counts=counts,
        samples_per_epoch=int(sampler["samples_per_epoch"]),
    )


def image_to_float_tensor(image: Image.Image | Tensor | np.ndarray) -> Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
    elif isinstance(image, Tensor):
        tensor = image
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}")
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError(
            f"image must have CHW shape with 1 or 3 channels, got {tuple(tensor.shape)}"
        )
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255)
    result = tensor.to(torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("image contains NaN or infinity")
    return result


class ToFloatTensor:
    def __call__(
        self, image: Any, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Any]]:
        return (image_to_float_tensor(image), target)


def xywh_to_xyxy(
    boxes: Tensor | _transforms_Iterable[_transforms_Iterable[float]],
) -> Tensor:
    boxes = _transforms_boxes_tensor(boxes)
    result = boxes.clone()
    result[:, 2:] += result[:, :2]
    return result


class COCOPolypDataset(Dataset):
    """Return ``(image, target, FrameRecord)`` for reviewed trainable frames.

    ``target['boxes']`` is always float32 xyxy with shape ``[N, 4]``. A
    confirmed negative has ``N=0`` and ``target['presence']==0``.
    """

    def __init__(
        self,
        coco_path: str | Path,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        split: DatasetSplit | str | None = None,
        statuses: _coco_dataset_Iterable[AnnotationStatus | str] = (
            AnnotationStatus.POSITIVE,
            AnnotationStatus.CONFIRMED_NEGATIVE,
        ),
        transform: Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]]
        | None = None,
        verify: bool = True,
        enforce_manifest_split: bool = True,
    ):
        from annotation.prepare import (
            AnnotationStatus,
            DatasetSplit,
            read_coco_json,
            read_manifest_jsonl,
            validate_coco_manifest,
        )

        self.coco_path = Path(coco_path)
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.coco = read_coco_json(self.coco_path)
        all_records = read_manifest_jsonl(self.manifest_path)
        requested_split = (
            DatasetSplit(split)
            if split is not None and enforce_manifest_split
            else None
        )
        coco_image_ids = {int(image["id"]) for image in self.coco.get("images", ())}
        split_records = tuple(
            (
                record
                for record in all_records
                if record.image_id in coco_image_ids
                and (requested_split is None or record.split == requested_split)
            )
        )
        if verify:
            validate_coco_manifest(
                self.coco,
                split_records,
                image_root=self.image_root,
                require_images=True,
            )
        selected_statuses = frozenset((AnnotationStatus(status) for status in statuses))
        if not selected_statuses:
            raise ValueError("statuses must not be empty")
        self.records = tuple(
            (record for record in split_records if record.status in selected_statuses)
        )
        self.transform = transform if transform is not None else ToFloatTensor()
        self._annotations = {
            int(annotation["image_id"]): annotation
            for annotation in self.coco["annotations"]
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any], FrameRecord]:
        from annotation.prepare import AnnotationStatus

        record = self.records[index]
        image_path = self.image_root / record.image_path
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.load()
        except Exception as exc:
            raise RuntimeError(
                f"could not load image {record.image_id}: {image_path}"
            ) from exc
        if image.size != (record.width, record.height):
            raise ValueError(
                f"image {record.image_id} has size {image.size}, expected {(record.width, record.height)}"
            )
        annotation = self._annotations.get(record.image_id)
        if annotation is None:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            area = torch.empty((0,), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            iscrowd = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = xywh_to_xyxy([annotation["bbox"]])
            area = torch.tensor([annotation["area"]], dtype=torch.float32)
            labels = torch.tensor([annotation["category_id"]], dtype=torch.int64)
            iscrowd = torch.tensor([annotation.get("iscrowd", 0)], dtype=torch.int64)
        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(record.image_id, dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "presence": torch.tensor(
                float(record.status == AnnotationStatus.POSITIVE), dtype=torch.float32
            ),
            "original_size": torch.tensor(
                [record.height, record.width], dtype=torch.int64
            ),
            "size": torch.tensor([record.height, record.width], dtype=torch.int64),
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return (image, target, record)

    def balanced_sample_weights(self, *, balance_groups: bool = True) -> Tensor:
        """Inverse-frequency weights over status and optionally acquisition group."""
        keys = [
            (record.status, record.group_id) if balance_groups else (record.status,)
            for record in self.records
        ]
        counts = Counter(keys)
        return torch.tensor([1.0 / counts[key] for key in keys], dtype=torch.double)


def scalar_int(value: Any, name: str) -> int:
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one scalar")
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or int(value) != value
    ):
        raise TypeError(f"{name} must be an integer-like scalar")
    return int(value)


def load_record_sample(source: Any, record: StaticFrameRecord) -> Mapping[str, Any]:
    if not 0 <= record.dataset_index < len(source):
        raise IndexError(
            f"dataset_index is outside the pinned dataset: {record.dataset_index}"
        )
    sample = source[record.dataset_index]
    if not isinstance(sample, Mapping):
        raise TypeError("LeRobot source sample must be a mapping")
    for key, expected in (
        ("episode_index", record.episode_index),
        ("frame_index", record.frame_index),
    ):
        if key not in sample:
            raise KeyError(f"source sample {record.dataset_index} is missing {key!r}")
        actual = scalar_int(sample[key], key)
        if actual != expected:
            raise ValueError(
                f"source {key} mismatch for {record.sample_id}: expected {expected}, got {actual}"
            )
    return sample


class LeRobotFrameSource:
    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str,
        revision: str,
        video_backend: str = "pyav",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = str(repo_id)
        self.revision = str(revision)
        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=Path(root),
            revision=self.revision,
            video_backend=video_backend,
        )
        self.features = self.dataset.features

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.dataset[index]


PROTOCOL_SCHEMA_VERSION = 1

VIEW_FILENAMES = {
    "train": "train.coco.json",
    "calibration": "calibration.coco.json",
    "negative_control": "negative_control.coco.json",
    "validation": "validation.coco.json",
    "locked_test": "test.coco.json",
}


def select_primary_v2_views(
    records: Sequence[FrameRecord],
) -> dict[str, tuple[FrameRecord, ...]]:
    from annotation.prepare import AnnotationStatus

    views: dict[str, list[FrameRecord]] = {name: [] for name in VIEW_FILENAMES}
    for record in records:
        episode = record.episode_index
        if record.group_id == 0:
            if episode <= 29:
                views["train"].append(record)
            elif episode <= 39:
                views["calibration"].append(record)
            else:
                views["negative_control"].append(record)
        elif 1 <= record.group_id <= 4:
            views["train"].append(record)
        elif record.group_id == 5:
            if record.frame_index % 6 == 0:
                views["train"].append(record)
        elif record.group_id == 6:
            views["validation"].append(record)
        elif record.group_id == 7:
            views["locked_test"].append(record)
    result = {name: tuple(rows) for name, rows in views.items()}
    all_ids: list[int] = [
        record.image_id for rows in result.values() for record in rows
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("experiment views overlap by image_id")
    for name in ("train", "calibration", "negative_control"):
        statuses = {record.status for record in result[name]}
        required = {AnnotationStatus.POSITIVE, AnnotationStatus.CONFIRMED_NEGATIVE}
        if not required.issubset(statuses):
            raise ValueError(
                f"{name} must contain both positive and confirmed-negative frames"
            )
    for name in ("validation", "locked_test"):
        if not any(
            (record.status == AnnotationStatus.POSITIVE for record in result[name])
        ):
            raise ValueError(f"{name} must contain positive frames")
    return result


def _subset_coco(
    source: Mapping[str, Any],
    records: Sequence[FrameRecord],
    *,
    protocol_id: str,
    view: str,
    annotation_revision: str,
) -> dict[str, Any]:
    from annotation.prepare import validate_coco_manifest

    image_ids = {record.image_id for record in records}
    payload = dict(source)
    payload["info"] = dict(
        source.get("info", {}),
        split=view,
        experiment_protocol=protocol_id,
        annotation_revision=annotation_revision,
    )
    payload["images"] = [
        image for image in source["images"] if int(image["id"]) in image_ids
    ]
    payload["annotations"] = [
        annotation
        for annotation in source["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    validate_coco_manifest(payload, records)
    return payload


def verify_primary_v2_views(
    protocol_root: str | Path, approved_root: str | Path
) -> dict[str, Any]:
    from annotation.prepare import (
        read_coco_json,
        read_manifest_jsonl,
        sha256_file,
        verify_locked_revision,
    )

    root = Path(protocol_root).resolve()
    approved_path = Path(approved_root).resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported experiment protocol schema")
    if protocol.get("protocol_id") != root.name:
        raise ValueError("protocol_id does not match protocol directory")
    approval = verify_locked_revision(approved_path, verify_source=False)
    if protocol.get("annotation_revision") != approval.revision:
        raise ValueError("experiment protocol annotation revision mismatch")
    if protocol.get("approved_payload_sha256") != approval.approved_payload_sha256:
        raise ValueError("experiment protocol approved payload hash mismatch")
    records = read_manifest_jsonl(approved_path / "manifest.jsonl")
    source_coco = read_coco_json(approved_path / "instances.json")
    views = select_primary_v2_views(records)
    for view, filename in VIEW_FILENAMES.items():
        path = root / filename
        expected = _subset_coco(
            source_coco,
            views[view],
            protocol_id=root.name,
            view=view,
            annotation_revision=approval.revision,
        )
        actual = read_coco_json(path)
        if actual != expected:
            raise ValueError(f"experiment view drift: {view}")
        metadata = protocol.get("views", {}).get(view, {})
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"experiment view hash mismatch: {view}")
    return protocol


def write_frame_manifest(
    path: str | Path, records: Sequence[StaticFrameRecord]
) -> Path:
    validate_frame_records(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            (json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records)
        ),
        encoding="utf-8",
    )
    return destination


PI05_LETTERBOX_TARGET_GEOMETRY = "pi05_letterbox"

CachedTargetGeometry = Literal["pi05_letterbox", "groot_n17_crop"]


@dataclass(frozen=True, slots=True)
class LetterboxSpec:
    """Exact geometry produced by LeRobot's ``resize_with_pad_torch``."""

    original_size: tuple[int, int]
    target_size: tuple[int, int]
    resized_size: tuple[int, int]
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def scale_x(self) -> float:
        return self.resized_size[0] / self.original_size[0]

    @property
    def scale_y(self) -> float:
        return self.resized_size[1] / self.original_size[1]


def compute_letterbox_spec(
    original_size: tuple[int, int], target_size: tuple[int, int] = (224, 224)
) -> LetterboxSpec:
    """Mimic LeRobot's integer truncation and centered remainder padding."""
    original_width, original_height = original_size
    target_width, target_height = target_size
    if min(original_width, original_height, target_width, target_height) <= 0:
        raise ValueError("original_size and target_size must be positive")
    ratio = max(original_width / target_width, original_height / target_height)
    resized_width = int(original_width / ratio)
    resized_height = int(original_height / ratio)
    pad_left, remainder_width = divmod(target_width - resized_width, 2)
    pad_top, remainder_height = divmod(target_height - resized_height, 2)
    pad_right = pad_left + remainder_width
    pad_bottom = pad_top + remainder_height
    return LetterboxSpec(
        original_size=original_size,
        target_size=target_size,
        resized_size=(resized_width, resized_height),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def apply_letterbox_xyxy(
    boxes: Tensor | _transforms_Iterable[_transforms_Iterable[float]],
    spec: LetterboxSpec,
    *,
    clip: bool = True,
) -> Tensor:
    boxes = _transforms_boxes_tensor(boxes).clone()
    boxes[:, (0, 2)] = boxes[:, (0, 2)] * spec.scale_x + spec.pad_left
    boxes[:, (1, 3)] = boxes[:, (1, 3)] * spec.scale_y + spec.pad_top
    if clip:
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, spec.target_size[0])
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, spec.target_size[1])
    return boxes


def _cached_target(
    record: StaticFrameRecord,
    output_size: tuple[int, int],
    target_geometry: CachedTargetGeometry = PI05_LETTERBOX_TARGET_GEOMETRY,
) -> dict[str, Any]:
    """Build a selectable frozen-feature target without decoding an image."""
    if target_geometry == data_GROOT_N17_CROP_TARGET_GEOMETRY:
        if tuple(output_size) != GROOT_N17_GEOMETRY.output_size:
            raise ValueError(
                f"GR00T N1.7 cached targets require output_size {GROOT_N17_GEOMETRY.output_size}, got {tuple(output_size)}"
            )
        return build_groot_n17_target(record)
    if target_geometry != PI05_LETTERBOX_TARGET_GEOMETRY:
        raise ValueError(f"unsupported cached target geometry: {target_geometry!r}")
    target = _target(record)
    spec = compute_letterbox_spec((record.width, record.height), output_size)
    boxes = apply_letterbox_xyxy(target["boxes"], spec)
    target["boxes"] = boxes
    target["area"] = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    target["size"] = torch.tensor([output_size[1], output_size[0]], dtype=torch.int64)
    target["boxes_cxcywh"] = normalized_target_from_xyxy(target)
    return target


@dataclass(frozen=True, slots=True)
class EndoBBoxVLAExample:
    sample_key: str
    contexts: tuple[dict[str, Any], ...]
    targets: tuple[dict[str, Any], ...]
    records: tuple[StaticFrameRecord, ...]
    prompts: tuple[PromptSelection, ...]
    context_sha256: tuple[str, ...]

    @property
    def prompt(self) -> PromptSelection:
        """Compatibility accessor for one-frame examples only."""
        if len(self.prompts) != 1:
            raise AttributeError(
                "temporal examples have frame-specific prompts; use prompts"
            )
        return self.prompts[0]


def _actual_prompt_for_record(record: StaticFrameRecord) -> str:
    from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS

    if record.source_dataset == "ENDO_task1_Withdraw_400episode":
        return CANONICAL_TASK_PROMPTS[0]
    if record.source_dataset == "Task4_Polyp_1to8":
        return CANONICAL_TASK_PROMPTS[3]
    raise ValueError(
        f"unsupported cached endo_bbox_vla source {record.source_dataset!r}"
    )


class _BaseCachedEndoBBoxVLADataset(Dataset[EndoBBoxVLAExample]):
    """Metadata/target-only view for self-contained frozen-feature training."""

    def __init__(
        self,
        *,
        sample_keys: Sequence[str],
        prompt_seed: int,
        output_size: tuple[int, int],
        cache_records: Mapping[tuple[str, str, int], Any],
        target_geometry: CachedTargetGeometry = PI05_LETTERBOX_TARGET_GEOMETRY,
    ) -> None:
        from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS

        self.output_size = tuple((int(value) for value in output_size))
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError(
                "cached endo_bbox_vla output_size must contain two positive values"
            )
        if target_geometry not in (
            PI05_LETTERBOX_TARGET_GEOMETRY,
            data_GROOT_N17_CROP_TARGET_GEOMETRY,
        ):
            raise ValueError(f"unsupported cached target geometry: {target_geometry!r}")
        if (
            target_geometry == data_GROOT_N17_CROP_TARGET_GEOMETRY
            and self.output_size != GROOT_N17_GEOMETRY.output_size
        ):
            raise ValueError(
                f"GR00T N1.7 cached targets require output_size {GROOT_N17_GEOMETRY.output_size}, got {self.output_size}"
            )
        self.target_geometry = target_geometry
        assignments = deterministic_mode_assignments(sample_keys, seed=prompt_seed)
        self.prompt_planner = PromptPlanner(
            assignments, seed=prompt_seed, canonical_tasks=CANONICAL_TASK_PROMPTS
        )
        self.cache_records = cache_records
        self.required_cache_keys: set[tuple[str, str, int]] = set()

    def _prompts(
        self,
        sample_key: str,
        records: Sequence[StaticFrameRecord],
        clip: EndoBBoxVLAClipRecord | None,
    ) -> tuple[PromptSelection, ...]:
        actual_prompts = {_actual_prompt_for_record(record) for record in records}
        if len(actual_prompts) != 1:
            raise ValueError(f"cached clip {sample_key} crosses task prompts")
        actual = actual_prompts.pop()
        if clip is not None and clip.prompt_switch_index is not None:
            return tuple(
                (
                    self.prompt_planner.select_mode(
                        sample_key,
                        actual,
                        "counterfactual"
                        if index < clip.prompt_switch_index
                        else "actual",
                        context_generation=clip.context_generations[index],
                    )
                    for index in range(len(records))
                )
            )
        base = self.prompt_planner.select(sample_key, actual)
        generations = clip.context_generations if clip is not None else (0,)
        return tuple((base.with_context_generation(value) for value in generations))

    def _cache_hashes(
        self, records: Sequence[StaticFrameRecord], prompts: Sequence[PromptSelection]
    ) -> tuple[str, ...]:
        hashes: list[str] = []
        for record, prompt in zip(records, prompts, strict=True):
            key = (record.sample_id, prompt.prompt_id, prompt.context_generation)
            self.required_cache_keys.add(key)
            try:
                cached = self.cache_records[key]
            except KeyError as exc:
                raise KeyError(
                    f"frozen-feature cache is incomplete for metadata-only training: {record.sample_id}/{prompt.mode}/generation-{prompt.context_generation}"
                ) from exc
            expected = {
                "source_dataset": record.source_dataset,
                "source_episode_index": record.source_episode_index,
                "source_frame_index": record.source_frame_index,
                "dataset_index": record.dataset_index,
                "frame_id": record.frame_index,
                "timestamp": record.timestamp,
                "prompt_mode": prompt.mode,
                "prompt_id": prompt.prompt_id,
                "detector_generation": record.detector_generation,
                "context_generation": prompt.context_generation,
            }
            mismatches = {
                name: (value, getattr(cached, name))
                for name, value in expected.items()
                if getattr(cached, name) != value
            }
            selected_hash = hashlib.sha256(prompt.selected_prompt.encode()).hexdigest()
            if cached.selected_prompt_sha256 != selected_hash:
                mismatches["selected_prompt_sha256"] = (
                    selected_hash,
                    cached.selected_prompt_sha256,
                )
            if mismatches:
                raise ValueError(
                    f"frozen-feature metadata mismatch for {record.sample_id}: {mismatches}"
                )
            hashes.append(str(cached.context_sha256))
        return tuple(hashes)

    def _example(
        self,
        sample_key: str,
        records: Sequence[StaticFrameRecord],
        clip: EndoBBoxVLAClipRecord | None = None,
    ) -> EndoBBoxVLAExample:
        prompts = self._prompts(sample_key, records, clip)
        return EndoBBoxVLAExample(
            sample_key=sample_key,
            contexts=tuple(({} for _ in records)),
            targets=tuple(
                (
                    _cached_target(record, self.output_size, self.target_geometry)
                    for record in records
                )
            ),
            records=tuple(records),
            prompts=prompts,
            context_sha256=self._cache_hashes(records, prompts),
        )


class EndoBBoxVLACachedStaticDataset(_BaseCachedEndoBBoxVLADataset):
    def __init__(
        self,
        records: Sequence[StaticFrameRecord],
        *,
        split: str,
        prompt_seed: int,
        output_size: tuple[int, int],
        cache_records: Mapping[tuple[str, str, int], Any],
        target_geometry: CachedTargetGeometry = PI05_LETTERBOX_TARGET_GEOMETRY,
    ) -> None:
        self.records = tuple(
            (
                record
                for record in records
                if record.split == split and record.status in TRAINABLE_STATUSES
            )
        )
        if not self.records:
            raise ValueError(f"cached endo_bbox_vla static split {split!r} is empty")
        super().__init__(
            sample_keys=[record.sample_id for record in self.records],
            prompt_seed=prompt_seed,
            output_size=output_size,
            cache_records=cache_records,
            target_geometry=target_geometry,
        )
        for index in range(len(self.records)):
            self[index]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EndoBBoxVLAExample:
        record = self.records[index]
        return self._example(record.sample_id, (record,))


@dataclass(frozen=True)
class EndoBBoxVLABatch:
    contexts: dict[str, Any]
    targets: dict[str, Tensor]
    records: tuple[tuple[StaticFrameRecord, ...], ...]
    prompts: tuple[tuple[PromptSelection, ...], ...]
    sample_keys: tuple[str, ...]
    context_sha256: tuple[tuple[str, ...], ...]

    @property
    def batch_size(self) -> int:
        return int(self.targets["presence"].shape[0])

    @property
    def clip_length(self) -> int:
        return int(self.targets["presence"].shape[1])


def endo_bbox_vla_collate(batch: Sequence[EndoBBoxVLAExample]) -> EndoBBoxVLABatch:
    from endo_bbox_vla.config import CAMERA_KEYS, STATE_KEY, TASK_KEY

    if not batch:
        raise ValueError("cannot collate an empty endo_bbox_vla batch")
    clip_lengths = {len(example.records) for example in batch}
    if len(clip_lengths) != 1:
        raise ValueError(
            f"endo_bbox_vla batch mixes clip lengths: {sorted(clip_lengths)}"
        )
    ignored = [
        (example.sample_key, frame_offset)
        for example in batch
        for frame_offset, target in enumerate(example.targets)
        if bool(torch.as_tensor(target.get("ignore", False), dtype=torch.bool).item())
    ]
    if ignored:
        raise ValueError(
            f"endo_bbox_vla batch contains crop-invisible positive targets marked ignore; filter them before training instead of relabelling them negative: {ignored[:5]}"
        )
    context_presence = [
        bool(context) for example in batch for context in example.contexts
    ]
    if any(context_presence) and (not all(context_presence)):
        raise ValueError("endo_bbox_vla batch mixes decoded and cache-only contexts")
    contexts: dict[str, Any] = {}
    if all(context_presence):
        for key in (*CAMERA_KEYS, STATE_KEY):
            contexts[key] = torch.stack(
                [
                    torch.stack(
                        [torch.as_tensor(context[key]) for context in example.contexts]
                    )
                    for example in batch
                ]
            )
        contexts[TASK_KEY] = [
            [str(context[TASK_KEY]) for context in example.contexts]
            for example in batch
        ]
    targets = {
        key: torch.stack(
            [
                torch.stack(
                    [torch.as_tensor(target[key]) for target in example.targets]
                )
                for example in batch
            ]
        )
        for key in ("presence", "boxes_cxcywh", "image_id", "size", "original_size")
    }
    optional_target_keys = (
        "box_supervision",
        "visible_fraction",
        "partial_original",
        "partial_transform",
        "ignore",
        "spatial_padding_mask",
    )
    flat_targets = [target for example in batch for target in example.targets]
    for key in optional_target_keys:
        availability = [key in target for target in flat_targets]
        if any(availability) and (not all(availability)):
            raise ValueError(
                f"endo_bbox_vla batch mixes targets with and without {key!r}"
            )
        if all(availability):
            targets[key] = torch.stack(
                [
                    torch.stack(
                        [torch.as_tensor(target[key]) for target in example.targets]
                    )
                    for example in batch
                ]
            )
    targets["frame_ids"] = torch.tensor(
        [[record.frame_index for record in example.records] for example in batch],
        dtype=torch.long,
    )
    targets["timestamps"] = torch.tensor(
        [[record.timestamp for record in example.records] for example in batch],
        dtype=torch.float32,
    )
    targets["context_generations"] = torch.tensor(
        [
            [prompt.context_generation for prompt in example.prompts]
            for example in batch
        ],
        dtype=torch.long,
    )
    targets["detector_generations"] = torch.tensor(
        [
            [record.detector_generation for record in example.records]
            for example in batch
        ],
        dtype=torch.long,
    )
    return EndoBBoxVLABatch(
        contexts=contexts,
        targets=targets,
        records=tuple((example.records for example in batch)),
        prompts=tuple((example.prompts for example in batch)),
        sample_keys=tuple((example.sample_key for example in batch)),
        context_sha256=tuple((example.context_sha256 for example in batch)),
    )


class FullContextSource(Protocol):
    repo_id: str
    revision: str
    features: Mapping[str, Any]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


Transform = _data_Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], str],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
]


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, Image.Image):
        return (3, value.height, value.width)
    if isinstance(value, Tensor | np.ndarray):
        shape = tuple((int(item) for item in value.shape))
        if len(shape) == 3 and shape[-1] in {1, 3, 4} and (shape[0] not in {1, 3}):
            shape = (min(shape[-1], 3), shape[0], shape[1])
        return shape
    raise TypeError(
        f"full-context image/state must be a tensor, ndarray, or PIL image, got {type(value)!r}"
    )


def _finite(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    array = np.asarray(value)
    return (
        bool(np.isfinite(array).all())
        if np.issubdtype(array.dtype, np.number)
        else False
    )


def validate_full_context_sample(
    sample: Mapping[str, Any],
    *,
    expected_camera_shape: tuple[int, int, int] = (3, 480, 640),
    canonical_tasks: Sequence[str] = CANONICAL_TASK_PROMPTS,
) -> None:
    """Validate every required modality; missing context is never substituted."""
    from endo_bbox_vla.config import CAMERA_KEYS, STATE_DIM, STATE_KEY, TASK_KEY

    missing = [key for key in (*CAMERA_KEYS, STATE_KEY, TASK_KEY) if key not in sample]
    if missing:
        raise KeyError(f"full-context sample is missing required fields: {missing}")
    for key in CAMERA_KEYS:
        actual_shape = _shape(sample[key])
        if actual_shape != expected_camera_shape:
            raise ValueError(
                f"{key} must have CHW shape {expected_camera_shape}, got {actual_shape}"
            )
        if not _finite(sample[key]):
            raise ValueError(f"{key} contains NaN or infinity")
    state_shape = _shape(sample[STATE_KEY])
    if state_shape != (STATE_DIM,):
        raise ValueError(
            f"{STATE_KEY} must have shape ({STATE_DIM},), got {state_shape}"
        )
    if not _finite(sample[STATE_KEY]):
        raise ValueError(f"{STATE_KEY} contains NaN or infinity")
    task = sample[TASK_KEY]
    if not isinstance(task, str) or not task.strip():
        raise TypeError(f"{TASK_KEY} must be a non-empty string")
    if task not in canonical_tasks:
        raise ValueError(
            f"{TASK_KEY} is not one of the five canonical instructions: {task!r}"
        )


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be an integer-like scalar")
    if not math.isfinite(float(value)) or int(value) != float(value):
        raise ValueError(f"{name} must be a finite integer-like scalar")
    return int(value)


def _validate_source_identity(
    sample: Mapping[str, Any], record: StaticFrameRecord
) -> None:
    for key, expected in (
        ("episode_index", record.episode_index),
        ("frame_index", record.frame_index),
    ):
        if key not in sample:
            raise KeyError(
                f"five-task sample {record.dataset_index} is missing identity field {key!r}"
            )
        actual = _scalar_int(sample[key], key)
        if actual != expected:
            raise ValueError(
                f"five-task {key} mismatch for {record.sample_id}: expected {expected}, got {actual}"
            )


def _load_context(
    source: FullContextSource, record: StaticFrameRecord
) -> tuple[dict[str, Any], str]:
    from endo_bbox_vla.config import CAMERA_KEYS, STATE_KEY, TASK_KEY

    if not 0 <= record.dataset_index < len(source):
        raise IndexError(
            f"dataset_index is outside the five-task dataset: {record.dataset_index}"
        )
    sample = source[record.dataset_index]
    validate_full_context_sample(
        sample, expected_camera_shape=(3, record.height, record.width)
    )
    _validate_source_identity(sample, record)
    actual_task = str(sample[TASK_KEY])
    context = {key: sample[key] for key in (*CAMERA_KEYS, STATE_KEY, TASK_KEY)}
    return (context, actual_task)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def full_context_sha256(context: Mapping[str, Any]) -> str:
    from endo_bbox_vla.config import CAMERA_KEYS, STATE_KEY, TASK_KEY

    missing = [key for key in (*CAMERA_KEYS, STATE_KEY, TASK_KEY) if key not in context]
    if missing:
        raise KeyError(f"cannot hash incomplete VLA context: {missing}")
    digest = hashlib.sha256()
    for key in (*CAMERA_KEYS, STATE_KEY):
        tensor = torch.as_tensor(context[key]).detach().cpu().contiguous()
        if tensor.is_floating_point() and (not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"context {key} contains NaN or infinity")
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    task = context[TASK_KEY]
    if not isinstance(task, str) or not task:
        raise TypeError("context task must be a non-empty string")
    digest.update(TASK_KEY.encode("utf-8"))
    digest.update(task.encode("utf-8"))
    return digest.hexdigest()


class _BaseEndoBBoxVLADataset(Dataset[EndoBBoxVLAExample]):
    def __init__(
        self,
        source: FullContextSource,
        *,
        sample_keys: Sequence[str],
        prompt_seed: int,
        transform: Transform,
    ) -> None:
        from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS

        self.source = source
        self.transform = transform
        assignments = deterministic_mode_assignments(sample_keys, seed=prompt_seed)
        self.prompt_planner = PromptPlanner(
            assignments, seed=prompt_seed, canonical_tasks=CANONICAL_TASK_PROMPTS
        )

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.transform, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def _prompted(
        self,
        sample_key: str,
        records: Sequence[StaticFrameRecord],
        *,
        clip: EndoBBoxVLAClipRecord | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[PromptSelection, ...]]:
        from endo_bbox_vla.config import TASK_KEY

        loaded = [_load_context(self.source, record) for record in records]
        actual_tasks = {actual_task for _, actual_task in loaded}
        if len(actual_tasks) != 1:
            raise ValueError(
                f"endo_bbox_vla clip {sample_key} crosses task prompts: {sorted(actual_tasks)}"
            )
        actual_prompt = actual_tasks.pop()
        if clip is not None and clip.prompt_switch_index is not None:
            prompts = tuple(
                (
                    self.prompt_planner.select_mode(
                        sample_key,
                        actual_prompt,
                        "counterfactual"
                        if frame_offset < clip.prompt_switch_index
                        else "actual",
                        context_generation=clip.context_generations[frame_offset],
                    )
                    for frame_offset in range(len(records))
                )
            )
        else:
            base = self.prompt_planner.select(sample_key, actual_prompt)
            generations = (
                clip.context_generations
                if clip is not None
                else tuple((0 for _ in records))
            )
            prompts = tuple(
                (base.with_context_generation(generation) for generation in generations)
            )
        contexts = []
        targets = []
        for (context, _), record, prompt in zip(loaded, records, prompts, strict=True):
            context = dict(context)
            context[TASK_KEY] = prompt.selected_prompt
            contexts.append(context)
            targets.append(_target(record))
        return (contexts, targets, prompts)

    def _example(
        self,
        sample_key: str,
        records: Sequence[StaticFrameRecord],
        *,
        clip: EndoBBoxVLAClipRecord | None = None,
    ) -> EndoBBoxVLAExample:
        contexts, targets, prompts = self._prompted(sample_key, records, clip=clip)
        transformed_contexts, transformed_targets = self.transform(
            contexts, targets, sample_key
        )
        if len(transformed_contexts) != len(records) or len(transformed_targets) != len(
            records
        ):
            raise RuntimeError("endo_bbox_vla transform changed clip length")
        return EndoBBoxVLAExample(
            sample_key=sample_key,
            contexts=tuple(transformed_contexts),
            targets=tuple(transformed_targets),
            records=tuple(records),
            prompts=prompts,
            context_sha256=tuple(
                (full_context_sha256(context) for context in contexts)
            ),
        )

    def _raw_example(
        self,
        sample_key: str,
        records: Sequence[StaticFrameRecord],
        *,
        clip: EndoBBoxVLAClipRecord | None = None,
    ) -> EndoBBoxVLAExample:
        """Return exact decoded 480x640 full context for action-call capture."""
        contexts, targets, prompts = self._prompted(sample_key, records, clip=clip)
        return EndoBBoxVLAExample(
            sample_key=sample_key,
            contexts=tuple(contexts),
            targets=tuple(targets),
            records=tuple(records),
            prompts=prompts,
            context_sha256=tuple(
                (full_context_sha256(context) for context in contexts)
            ),
        )


class EndoBBoxVLAStaticDataset(_BaseEndoBBoxVLADataset):
    def __init__(
        self,
        source: FullContextSource,
        records: Sequence[StaticFrameRecord],
        *,
        split: str,
        prompt_seed: int,
        transform: Transform,
    ) -> None:
        self.records = tuple(
            (
                record
                for record in records
                if record.split == split and record.status in TRAINABLE_STATUSES
            )
        )
        if not self.records:
            raise ValueError(
                f"endo_bbox_vla static split {split!r} is empty after excluding uncertain frames"
            )
        super().__init__(
            source,
            sample_keys=[record.sample_id for record in self.records],
            prompt_seed=prompt_seed,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EndoBBoxVLAExample:
        record = self.records[index]
        return self._example(record.sample_id, (record,))

    def raw_example(self, index: int) -> EndoBBoxVLAExample:
        record = self.records[index]
        return self._raw_example(record.sample_id, (record,))


data_Transform = _data_Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], str],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
]


class GrootStaticDataset(Dataset[GrootStaticExample]):
    def __init__(
        self,
        source: Any,
        records: Sequence[StaticFrameRecord],
        *,
        split: str,
        prompt_seed: int,
        transform: data_Transform,
    ) -> None:
        from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS

        self.source = source
        self.records = tuple(
            (
                record
                for record in records
                if record.split == split and record.status in TRAINABLE_STATUSES
            )
        )
        if not self.records:
            raise ValueError(
                f"static split {split!r} is empty after excluding uncertain frames"
            )
        self.transform = transform
        assignments = deterministic_mode_assignments(
            [record.sample_id for record in self.records], seed=prompt_seed
        )
        self.prompt_planner = PromptPlanner(
            assignments, seed=prompt_seed, canonical_tasks=CANONICAL_TASK_PROMPTS
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.transform, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def _raw_example(self, record: StaticFrameRecord) -> GrootStaticExample:
        from endo_bbox_vla.config import CAMERA_KEYS, STATE_KEY, TASK_KEY

        sample = load_record_sample(self.source, record)
        validate_full_context_sample(
            sample, expected_camera_shape=(3, record.height, record.width)
        )
        actual_prompt = str(sample[TASK_KEY])
        expected_prompt = _actual_prompt(record)
        if actual_prompt != expected_prompt:
            raise ValueError(
                f"source task mismatch for {record.sample_id}: {actual_prompt!r} != {expected_prompt!r}"
            )
        prompt = self.prompt_planner.select(record.sample_id, actual_prompt)
        context = {key: sample[key] for key in (*CAMERA_KEYS, STATE_KEY, TASK_KEY)}
        context[TASK_KEY] = prompt.selected_prompt
        return GrootStaticExample(
            sample_key=record.sample_id,
            contexts=(context,),
            targets=(_raw_target(record),),
            records=(record,),
            prompts=(prompt,),
            context_sha256=(full_context_sha256(context),),
        )

    def raw_example(self, index: int) -> GrootStaticExample:
        return self._raw_example(self.records[index])

    def __getitem__(self, index: int) -> GrootStaticExample:
        example = self._raw_example(self.records[index])
        contexts, targets = self.transform(
            example.contexts, example.targets, example.sample_key
        )
        if len(contexts) != 1 or len(targets) != 1:
            raise RuntimeError("static transform changed example length")
        return GrootStaticExample(
            sample_key=example.sample_key,
            contexts=tuple(contexts),
            targets=tuple(targets),
            records=example.records,
            prompts=example.prompts,
            context_sha256=example.context_sha256,
        )


@dataclass(frozen=True, slots=True)
class GrootTemporalSmokeClipPlan:
    record: EndoBBoxVLAClipRecord
    base_prompt_mode: str
    selection_stratum: str
    rank_sha256: str
    frames: tuple[
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
        GrootTemporalSmokeFramePlan,
    ]

    @property
    def clip_id(self) -> str:
        return self.record.clip_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.record.to_dict(),
            "base_prompt_mode": self.base_prompt_mode,
            "selection_stratum": self.selection_stratum,
            "rank_sha256": self.rank_sha256,
            "frames": [frame.to_dict() for frame in self.frames],
        }


SMOKE_PLAN_SCHEMA_VERSION = 1

SMOKE_PLAN_KIND = "groot_n17_temporal_balanced_smoke_plan"


@dataclass(frozen=True, slots=True)
class GrootTemporalSmokePlan:
    prompt_seed: int
    ranking_salt: str
    canonical_full_clip_set_sha256: str
    canonical_full_prompt_assignment_sha256: str
    selected_clips: tuple[GrootTemporalSmokeClipPlan, ...]
    _summary_json: str

    @property
    def summary(self) -> dict[str, Any]:
        """Return an independent JSON-compatible copy of the audited summary."""
        value = json.loads(self._summary_json)
        if not isinstance(value, dict):
            raise RuntimeError(
                "internal GR00T temporal smoke summary is not a JSON object"
            )
        return value

    @property
    def selected_clip_records(self) -> tuple[EndoBBoxVLAClipRecord, ...]:
        return tuple((value.record for value in self.selected_clips))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SMOKE_PLAN_SCHEMA_VERSION,
            "kind": SMOKE_PLAN_KIND,
            "prompt_seed": self.prompt_seed,
            "ranking_salt": self.ranking_salt,
            "canonical_full_clip_set_sha256": self.canonical_full_clip_set_sha256,
            "canonical_full_prompt_assignment_sha256": self.canonical_full_prompt_assignment_sha256,
            "summary": self.summary,
            "clips": [clip.to_dict() for clip in self.selected_clips],
        }


DEFAULT_RANKING_SALT = "groot-temporal-smoke-v1"

SMOKE_STRATUM_QUOTAS = MappingProxyType(
    {
        "task4_positive_anchor/prompt_switch/any": 4,
        "task4_positive_anchor/non_switch/actual": 6,
        "task4_positive_anchor/non_switch/counterfactual": 2,
        "task4_positive_anchor/non_switch/neutral": 4,
        "task1_hard_negative_anchor/non_switch/actual": 8,
        "task1_hard_negative_anchor/non_switch/counterfactual": 4,
        "task1_hard_negative_anchor/non_switch/neutral": 4,
        "task1_random_pure_negative/non_switch/actual": 8,
        "task1_random_pure_negative/non_switch/counterfactual": 4,
        "task1_random_pure_negative/non_switch/neutral": 4,
        "task4_pure_negative/non_switch/actual": 8,
        "task4_pure_negative/non_switch/counterfactual": 4,
        "task4_pure_negative/non_switch/neutral": 4,
    }
)


def _selection_stratum(clip: EndoBBoxVLAClipRecord, base_prompt_mode: str) -> str:
    if clip.prompt_switch_index is not None:
        return f"{clip.source_category}/prompt_switch/any"
    return f"{clip.source_category}/non_switch/{base_prompt_mode}"


def rank_groot_temporal_smoke_clip(
    clip_id: str, *, ranking_salt: str = DEFAULT_RANKING_SALT
) -> str:
    """Return the stable SHA-256 rank used inside one selection stratum."""
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("clip_id must be a non-empty string")
    if not isinstance(ranking_salt, str) or not ranking_salt:
        raise ValueError("ranking_salt must be a non-empty string")
    return hashlib.sha256(f"{ranking_salt}\x00{clip_id}".encode("utf-8")).hexdigest()


def _select_clips(
    clips: Sequence[EndoBBoxVLAClipRecord],
    assignments: Mapping[str, str],
    *,
    ranking_salt: str,
) -> tuple[tuple[EndoBBoxVLAClipRecord, str, str, str], ...]:
    strata: dict[str, list[tuple[str, str, EndoBBoxVLAClipRecord]]] = {
        name: [] for name in SMOKE_STRATUM_QUOTAS
    }
    for clip in clips:
        base_mode = assignments[clip.clip_id]
        stratum = _selection_stratum(clip, base_mode)
        if stratum not in strata:
            raise ValueError(
                f"full temporal clip resolved to an unregistered smoke stratum: {stratum}"
            )
        rank = rank_groot_temporal_smoke_clip(clip.clip_id, ranking_salt=ranking_salt)
        strata[stratum].append((rank, clip.clip_id, clip))
    selected: list[tuple[EndoBBoxVLAClipRecord, str, str, str]] = []
    for stratum, quota in SMOKE_STRATUM_QUOTAS.items():
        ranked = sorted(strata[stratum], key=lambda value: (value[0], value[1]))
        if len(ranked) < quota:
            raise ValueError(
                f"smoke stratum {stratum!r} has {len(ranked)} clips but requires {quota}"
            )
        for rank, _clip_id, clip in ranked[:quota]:
            selected.append((clip, assignments[clip.clip_id], stratum, rank))
    selected_ids = [value[0].clip_id for value in selected]
    if len(selected) != 64 or len(set(selected_ids)) != 64:
        raise RuntimeError(
            "balanced GR00T smoke selection must contain 64 distinct clips"
        )
    return tuple(selected)


def _groot_temporal_smoke_canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_and_validate(
    selected: Sequence[GrootTemporalSmokeClipPlan],
    *,
    assignments: Mapping[str, str],
    frames_by_id: Mapping[str, StaticFrameRecord],
    evaluation_records: Sequence[StaticFrameRecord] | None,
) -> dict[str, Any]:
    from endo_bbox_vla.config import (
        contracts_TASK1_SOURCE_DATASET,
        contracts_TASK4_SOURCE_DATASET,
    )

    clip_category_counts = Counter((value.record.source_category for value in selected))
    stratum_counts = Counter((value.selection_stratum for value in selected))
    switch_count = sum(
        (value.record.prompt_switch_index is not None for value in selected)
    )
    frames = [frame for clip in selected for frame in clip.frames]
    frame_prompt_counts = Counter((frame.prompt_mode for frame in frames))
    generation_counts = Counter((frame.context_generation for frame in frames))
    frame_status_counts = Counter((frame.status for frame in frames))
    source_frame_counts = Counter((frame.source_dataset for frame in frames))
    unique_sample_ids = {frame.sample_id for frame in frames}
    conditioning_keys = {frame.conditioning_key for frame in frames}
    physical_conditioning_keys = {
        (frame.sample_id, frame.prompt_id) for frame in frames
    }
    source_episodes = {
        (frame.source_dataset, frame.source_episode_index) for frame in frames
    }
    selected_frame_records = [frames_by_id[frame.sample_id] for frame in frames]
    motion_transitions = sum(
        (
            first.status == "positive" and second.status == "positive"
            for clip in selected
            for first, second in zip(clip.frames, clip.frames[1:], strict=False)
        )
    )
    expected_categories = {name: 16 for name in EXPECTED_FULL_CLIP_COUNTS}
    checks = {
        "clip_count": len(selected) == 64,
        "clip_categories": dict(clip_category_counts) == expected_categories,
        "stratum_quotas": dict(stratum_counts) == dict(SMOKE_STRATUM_QUOTAS),
        "prompt_switches": switch_count == 4,
        "frame_count": len(frames) == 256,
        "unique_frames": len(unique_sample_ids) == 256,
        "frame_prompt_arms": dict(frame_prompt_counts)
        == {"actual": 128, "counterfactual": 64, "neutral": 64},
        "context_generations": dict(generation_counts) == {0: 248, 1: 8},
        "conditioning_keys": len(conditioning_keys) == 256,
        "physical_conditioning_keys": len(physical_conditioning_keys) == 256,
        "source_frames": dict(source_frame_counts)
        == {contracts_TASK1_SOURCE_DATASET: 128, contracts_TASK4_SOURCE_DATASET: 128},
        "frame_statuses": dict(frame_status_counts)
        == {"confirmed_negative": 192, "positive": 64},
        "motion_transitions": motion_transitions == 48,
    }
    failed = sorted((name for name, passed in checks.items() if not passed))
    if failed:
        raise ValueError(f"balanced GR00T temporal smoke contract failed: {failed}")
    overlap = _overlap_summary(selected_frame_records, evaluation_records)
    selected_rows = [value.to_dict() for value in selected]
    return {
        "full_clip_count": len(assignments),
        "full_prompt_assignment_counts": dict(Counter(assignments.values())),
        "selected_clip_count": len(selected),
        "selected_frame_occurrences": len(frames),
        "unique_frame_count": len(unique_sample_ids),
        "logical_conditioning_key_count": len(conditioning_keys),
        "physical_conditioning_tensor_count": len(physical_conditioning_keys),
        "clip_counts_by_source_category": dict(sorted(clip_category_counts.items())),
        "selection_counts_by_stratum": {
            name: stratum_counts[name] for name in SMOKE_STRATUM_QUOTAS
        },
        "prompt_switch_clip_count": switch_count,
        "frame_prompt_mode_counts": {
            mode: frame_prompt_counts[mode] for mode in PROMPT_MODES
        },
        "context_generation_counts": {
            str(generation): count
            for generation, count in sorted(generation_counts.items())
        },
        "frame_status_counts": dict(sorted(frame_status_counts.items())),
        "source_frame_counts": dict(sorted(source_frame_counts.items())),
        "source_episode_counts": {
            source: sum((value[0] == source for value in source_episodes))
            for source in (
                contracts_TASK1_SOURCE_DATASET,
                contracts_TASK4_SOURCE_DATASET,
            )
        },
        "source_episode_count": len(source_episodes),
        "motion_supervised_transition_count": motion_transitions,
        "selected_rows_sha256": _groot_temporal_smoke_canonical_sha256(selected_rows),
        "all_exact_contract_checks_passed": True,
        **overlap,
    }


def build_groot_temporal_smoke_plan(
    clips: Sequence[EndoBBoxVLAClipRecord],
    frames_by_id: Mapping[str, StaticFrameRecord],
    *,
    prompt_seed: int = DEFAULT_PROMPT_SEED,
    ranking_salt: str = DEFAULT_RANKING_SALT,
    evaluation_records: Sequence[StaticFrameRecord] | None = None,
) -> GrootTemporalSmokePlan:
    """Build and fully audit the fixed 64-clip GR00T temporal smoke plan."""
    full_clips = tuple(clips)
    if not full_clips:
        raise ValueError("full temporal clip manifest is empty")
    validate_clip_records(full_clips, frames_by_id)
    assignments = _validate_full_manifest_and_assign_prompts(
        full_clips, prompt_seed=prompt_seed
    )
    selected = _select_clips(full_clips, assignments, ranking_salt=ranking_salt)
    planner = PromptPlanner(assignments, seed=prompt_seed)
    selected_plans = tuple(
        (
            GrootTemporalSmokeClipPlan(
                record=clip,
                base_prompt_mode=base_mode,
                selection_stratum=stratum,
                rank_sha256=rank,
                frames=_frame_plans(
                    clip,
                    base_prompt_mode=base_mode,
                    planner=planner,
                    frames_by_id=frames_by_id,
                ),
            )
            for clip, base_mode, stratum, rank in selected
        )
    )
    summary = _summarize_and_validate(
        selected_plans,
        assignments=assignments,
        frames_by_id=frames_by_id,
        evaluation_records=evaluation_records,
    )
    canonical_clips = [
        clip.to_dict() for clip in sorted(full_clips, key=lambda item: item.clip_id)
    ]
    canonical_assignments = [
        {"clip_id": clip_id, "prompt_mode": assignments[clip_id]}
        for clip_id in sorted(assignments)
    ]
    return GrootTemporalSmokePlan(
        prompt_seed=prompt_seed,
        ranking_salt=ranking_salt,
        canonical_full_clip_set_sha256=_groot_temporal_smoke_canonical_sha256(
            canonical_clips
        ),
        canonical_full_prompt_assignment_sha256=_groot_temporal_smoke_canonical_sha256(
            canonical_assignments
        ),
        selected_clips=selected_plans,
        _summary_json=json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    )
