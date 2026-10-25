"""annotation — prepare: consolidated research implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable as _build_scenario_static_v1_Iterable
from collections.abc import Mapping as _build_scenario_static_v1_Mapping
from collections.abc import Mapping as _finalize_scenario_static_v1_Mapping
from collections.abc import Sequence as _build_scenario_static_v1_Sequence
from collections.abc import Sequence as _finalize_scenario_static_v1_Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from pathlib import Path as _ProjectPath
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


def read_coco_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object: {path}")
    return payload


class DatasetSplit(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    LOCKED_TEST = "locked_test"


class AnnotationStatus(StrEnum):
    POSITIVE = "positive"
    CONFIRMED_NEGATIVE = "confirmed_negative"
    UNCERTAIN_IGNORE = "uncertain_ignore"
    UNREVIEWED = "unreviewed"

    @property
    def is_trainable(self) -> bool:
        return self in {self.POSITIVE, self.CONFIRMED_NEGATIVE}


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PRIMARY_SPLIT_PROTOCOL_VERSION = 2
LEGACY_SPLIT_PROTOCOL_VERSION = 1
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {LEGACY_SPLIT_PROTOCOL_VERSION, PRIMARY_SPLIT_PROTOCOL_VERSION}
)
_IMAGE_ID_FRAME_FACTOR = 1000000


def stable_image_id(episode_index: int, frame_index: int) -> int:
    """Produce a stable COCO id independent of extraction subset and ordering."""
    if episode_index < 0 or frame_index < 0 or frame_index >= _IMAGE_ID_FRAME_FACTOR:
        raise ValueError(
            "episode_index and frame_index must be non-negative; frame_index must be <1,000,000"
        )
    return 1 + episode_index * _IMAGE_ID_FRAME_FACTOR + frame_index


@dataclass(frozen=True, slots=True)
class GroupDefinition:
    """A physical acquisition group backed by one consolidated video file."""

    group_id: int
    episode_start: int
    episode_end: int
    video_file_index: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.episode_start < 0 or self.episode_end < self.episode_start:
            raise ValueError("invalid inclusive episode range")
        if self.video_file_index < 0:
            raise ValueError("video_file_index must be non-negative")

    @property
    def episode_indices(self) -> range:
        return range(self.episode_start, self.episode_end + 1)

    def contains(self, episode_index: int) -> bool:
        return self.episode_start <= episode_index <= self.episode_end


GROUPS: tuple[GroupDefinition, ...] = (
    GroupDefinition(0, 0, 49, 0),
    GroupDefinition(1, 50, 99, 1),
    GroupDefinition(2, 100, 149, 2),
    GroupDefinition(3, 150, 199, 3),
    GroupDefinition(4, 200, 229, 4),
    GroupDefinition(5, 230, 259, 5),
    GroupDefinition(6, 260, 289, 6),
    GroupDefinition(7, 290, 319, 7),
)


def group_for_episode(episode_index: int) -> GroupDefinition:
    for group in GROUPS:
        if group.contains(episode_index):
            return group
    raise ValueError(f"episode {episode_index} is outside the fixed Task4 range 0..319")


def sampling_stride(source_fps: int, target_hz: int) -> int:
    """Return an exact integer stride; reject phase-drifting resampling."""
    if source_fps <= 0 or target_hz <= 0:
        raise ValueError("source_fps and target_hz must be positive")
    quotient, remainder = divmod(source_fps, target_hz)
    if remainder or quotient < 1:
        raise ValueError(
            f"target_hz={target_hz} must be an integer divisor of source_fps={source_fps}"
        )
    return quotient


SOURCE_FPS = 30


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    """Immutable group split with an episode-local annotation frequency."""

    name: DatasetSplit
    group_ids: tuple[int, ...]
    annotation_hz: int
    expected_frames: int

    def __post_init__(self) -> None:
        if not self.group_ids or len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("group_ids must be non-empty and unique")
        if any(
            (
                group_id not in {group.group_id for group in GROUPS}
                for group_id in self.group_ids
            )
        ):
            raise ValueError(f"unknown group in {self.group_ids}")
        sampling_stride(SOURCE_FPS, self.annotation_hz)
        if self.expected_frames <= 0:
            raise ValueError("expected_frames must be positive")

    @property
    def groups(self) -> tuple[GroupDefinition, ...]:
        selected = set(self.group_ids)
        return tuple((group for group in GROUPS if group.group_id in selected))

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(
            (index for group in self.groups for index in group.episode_indices)
        )

    @property
    def stride(self) -> int:
        return sampling_stride(SOURCE_FPS, self.annotation_hz)

    def contains_episode(self, episode_index: int) -> bool:
        return any((group.contains(episode_index) for group in self.groups))


LEGACY_TRAIN_SPLIT = SplitDefinition(DatasetSplit.TRAIN, (0, 1, 2, 3, 4, 5), 5, 16209)
VALIDATION_SPLIT = SplitDefinition(DatasetSplit.VALIDATION, (6,), 15, 4907)
LOCKED_TEST_SPLIT = SplitDefinition(DatasetSplit.LOCKED_TEST, (7,), 15, 4938)
LEGACY_PRIMARY_SPLITS: Mapping[DatasetSplit, SplitDefinition] = {
    split.name: split
    for split in (LEGACY_TRAIN_SPLIT, VALIDATION_SPLIT, LOCKED_TEST_SPLIT)
}
TRAIN_SPLIT = SplitDefinition(DatasetSplit.TRAIN, (0, 1, 2, 3, 4), 5, 14715)
CALIBRATION_SPLIT = SplitDefinition(DatasetSplit.CALIBRATION, (5,), 15, 4455)
PRIMARY_SPLITS: Mapping[DatasetSplit, SplitDefinition] = {
    split.name: split
    for split in (TRAIN_SPLIT, CALIBRATION_SPLIT, VALIDATION_SPLIT, LOCKED_TEST_SPLIT)
}
SPLIT_PROTOCOLS: Mapping[int, Mapping[DatasetSplit, SplitDefinition]] = {
    LEGACY_SPLIT_PROTOCOL_VERSION: LEGACY_PRIMARY_SPLITS,
    PRIMARY_SPLIT_PROTOCOL_VERSION: PRIMARY_SPLITS,
}


def split_for_episode(
    episode_index: int, *, protocol_version: int = PRIMARY_SPLIT_PROTOCOL_VERSION
) -> SplitDefinition:
    group_id = group_for_episode(episode_index).group_id
    try:
        protocol = SPLIT_PROTOCOLS[protocol_version]
    except KeyError as exc:
        raise ValueError(
            f"unsupported split protocol version {protocol_version}; expected one of {sorted(SPLIT_PROTOCOLS)}"
        ) from exc
    for split in protocol.values():
        if group_id in split.group_ids:
            return split
    raise AssertionError(
        f"group {group_id} is not assigned in split protocol version {protocol_version}"
    )


@dataclass(frozen=True, slots=True)
class FrameRecord:
    image_id: int
    episode_index: int
    frame_index: int
    timestamp: float
    video_timestamp: float
    group_id: int
    split: DatasetSplit
    source_video: str
    image_path: str
    status: AnnotationStatus = AnnotationStatus.UNREVIEWED
    partial: bool = False
    occluded: bool = False
    blurred: bool = False
    review_version: str = "draft"
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    schema_version: int = PRIMARY_SPLIT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(self, "status", AnnotationStatus(self.status))
        if self.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported manifest schema_version={self.schema_version}; expected one of {sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}"
            )
        if self.image_id != stable_image_id(self.episode_index, self.frame_index):
            raise ValueError(
                "image_id must equal stable_image_id(episode_index, frame_index)"
            )
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if not isfinite(self.video_timestamp) or self.video_timestamp < 0:
            raise ValueError("video_timestamp must be finite and non-negative")
        expected_group = group_for_episode(self.episode_index).group_id
        if self.group_id != expected_group:
            raise ValueError(
                f"episode {self.episode_index} belongs to group {expected_group}, got {self.group_id}"
            )
        expected_split = split_for_episode(
            self.episode_index, protocol_version=self.schema_version
        ).name
        if self.split != expected_split:
            raise ValueError(
                f"episode {self.episode_index} belongs to split {expected_split.value} under manifest schema/split protocol v{self.schema_version}, got {self.split.value}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        for field_name in ("source_video", "image_path", "review_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        image_path = PurePosixPath(self.image_path)
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError(
                "image_path must be a safe path relative to the annotation root"
            )
        source_path = PurePosixPath(self.source_video)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(
                "source_video must be a safe path relative to the dataset root"
            )

    @property
    def key(self) -> tuple[int, int]:
        return (self.episode_index, self.frame_index)

    @property
    def is_trainable(self) -> bool:
        return self.status.is_trainable

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = self.split.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrameRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown FrameRecord fields: {sorted(unknown)}")
        required = {
            "image_id",
            "episode_index",
            "frame_index",
            "timestamp",
            "video_timestamp",
            "group_id",
            "split",
            "source_video",
            "image_path",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing FrameRecord fields: {sorted(missing)}")
        return cls(**dict(payload))


def resolve_split(split: DatasetSplit | SplitDefinition | str) -> SplitDefinition:
    if isinstance(split, SplitDefinition):
        return split
    try:
        return PRIMARY_SPLITS[DatasetSplit(split)]
    except (KeyError, ValueError) as exc:
        choices = ", ".join((item.value for item in DatasetSplit))
        raise ValueError(f"unknown split {split!r}; expected one of {choices}") from exc


class AnnotationValidationError(ValueError):
    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        preview = "\n".join((f"- {issue}" for issue in self.issues[:30]))
        suffix = (
            "" if len(self.issues) <= 30 else f"\n- ... {len(self.issues) - 30} more"
        )
        super().__init__(
            f"annotation validation failed with {len(self.issues)} issue(s):\n{preview}{suffix}"
        )


def validate_manifest_records(
    records: Iterable[FrameRecord],
) -> tuple[FrameRecord, ...]:
    materialized = tuple(records)
    issues: list[str] = []
    seen_ids: set[int] = set()
    seen_keys: set[tuple[int, int]] = set()
    seen_paths: set[str] = set()
    schema_versions: set[int] = set()
    for record in materialized:
        if record.image_id in seen_ids:
            issues.append(f"duplicate image_id {record.image_id}")
        if record.key in seen_keys:
            issues.append(f"duplicate episode/frame key {record.key}")
        if record.image_path in seen_paths:
            issues.append(f"duplicate image_path {record.image_path!r}")
        seen_ids.add(record.image_id)
        seen_keys.add(record.key)
        seen_paths.add(record.image_path)
        schema_versions.add(record.schema_version)
        split = resolve_split(record.split)
        if record.frame_index % split.stride != 0:
            issues.append(
                f"image {record.image_id}: frame {record.frame_index} violates episode-local {split.annotation_hz} Hz stride {split.stride}"
            )
        expected_timestamp = record.frame_index / SOURCE_FPS
        if not math.isclose(record.timestamp, expected_timestamp, abs_tol=1e-07):
            issues.append(
                f"image {record.image_id}: timestamp {record.timestamp} != frame/source_fps {expected_timestamp}"
            )
    if len(schema_versions) > 1:
        issues.append(
            f"manifest must not mix schema/split protocols: {sorted(schema_versions)}"
        )
    if issues:
        raise AnnotationValidationError(issues)
    return tuple(sorted(materialized, key=lambda record: record.image_id))


def read_manifest_jsonl(path: str | Path) -> tuple[FrameRecord, ...]:
    path = Path(path)
    records: list[FrameRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be a JSON object")
                records.append(FrameRecord.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid manifest record at {path}:{line_number}: {exc}"
                ) from exc
    return validate_manifest_records(records)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    images: int
    annotations: int
    positive: int
    confirmed_negative: int
    ignored: int
    unreviewed: int


def _load_coco(coco: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    return read_coco_json(coco) if isinstance(coco, (str, Path)) else dict(coco)


def _load_records(
    records: Sequence[FrameRecord] | str | Path,
) -> tuple[FrameRecord, ...]:
    if isinstance(records, (str, Path)):
        return read_manifest_jsonl(records)
    return validate_manifest_records(records)


POLYP_CATEGORY_ID = 1


def validate_coco_manifest(
    coco: Mapping[str, Any] | str | Path,
    records: Sequence[FrameRecord] | str | Path,
    *,
    image_root: str | Path | None = None,
    require_images: bool = False,
    require_complete_primary: bool = False,
) -> ValidationSummary:
    """Validate IDs, statuses, boxes, split provenance, and optional files."""
    payload = _load_coco(coco)
    materialized = _load_records(records)
    issues: list[str] = []
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            issues.append(f"COCO {key!r} must be a list")
    if issues:
        raise AnnotationValidationError(issues)
    categories = payload["categories"]
    if not any(
        (
            item.get("id") == POLYP_CATEGORY_ID and item.get("name") == "polyp"
            for item in categories
        )
    ):
        issues.append("COCO categories must contain {id: 1, name: 'polyp'}")
    record_by_id = {record.image_id: record for record in materialized}
    image_by_id: dict[int, Mapping[str, Any]] = {}
    for image in payload["images"]:
        if not isinstance(image, dict) or "id" not in image:
            issues.append("every COCO image must be an object with an id")
            continue
        image_id = image["id"]
        if not isinstance(image_id, int):
            issues.append(f"COCO image id must be int, got {image_id!r}")
            continue
        if image_id in image_by_id:
            issues.append(f"duplicate COCO image id {image_id}")
        image_by_id[image_id] = image
    missing_images = set(record_by_id) - set(image_by_id)
    extra_images = set(image_by_id) - set(record_by_id)
    if missing_images:
        issues.append(
            f"COCO is missing manifest image ids {sorted(missing_images)[:20]}"
        )
    if extra_images:
        issues.append(
            f"COCO contains ids absent from manifest {sorted(extra_images)[:20]}"
        )
    for image_id in set(record_by_id) & set(image_by_id):
        record = record_by_id[image_id]
        image = image_by_id[image_id]
        expected = {
            "file_name": record.image_path,
            "width": record.width,
            "height": record.height,
        }
        for key, value in expected.items():
            if image.get(key) != value:
                issues.append(
                    f"image {image_id}: COCO {key}={image.get(key)!r}, manifest={value!r}"
                )
        if require_images:
            if image_root is None:
                issues.append("require_images=True requires image_root")
                require_images = False
            elif not (Path(image_root) / record.image_path).is_file():
                issues.append(
                    f"image {image_id}: missing file {Path(image_root) / record.image_path}"
                )
    annotation_ids: set[int] = set()
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in record_by_id
    }
    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            issues.append("every COCO annotation must be an object")
            continue
        annotation_id = annotation.get("id")
        if not isinstance(annotation_id, int) or annotation_id <= 0:
            issues.append(
                f"annotation id must be a positive int, got {annotation_id!r}"
            )
        elif annotation_id in annotation_ids:
            issues.append(f"duplicate annotation id {annotation_id}")
        else:
            annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if image_id not in record_by_id:
            issues.append(
                f"annotation {annotation_id} references unknown image {image_id}"
            )
            continue
        annotations_by_image[image_id].append(annotation)
        if annotation.get("category_id") != POLYP_CATEGORY_ID:
            issues.append(
                f"annotation {annotation_id}: category_id must be {POLYP_CATEGORY_ID}"
            )
        if annotation.get("iscrowd", 0) != 0:
            issues.append(f"annotation {annotation_id}: iscrowd must be 0")
        bbox = annotation.get("bbox")
        record = record_by_id[image_id]
        if not isinstance(bbox, list) or len(bbox) != 4:
            issues.append(
                f"annotation {annotation_id}: bbox must be [x, y, width, height]"
            )
            continue
        if any(
            (
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in bbox
            )
        ):
            issues.append(
                f"annotation {annotation_id}: bbox contains non-finite/non-numeric values"
            )
            continue
        x, y, width, height = map(float, bbox)
        if x < 0 or y < 0 or width <= 0 or (height <= 0):
            issues.append(f"annotation {annotation_id}: invalid bbox {bbox}")
        if x + width > record.width + 1e-06 or y + height > record.height + 1e-06:
            issues.append(
                f"annotation {annotation_id}: bbox {bbox} exceeds {record.width}x{record.height}"
            )
        area = annotation.get("area")
        if (
            not isinstance(area, (int, float))
            or not math.isfinite(area)
            or (
                not math.isclose(
                    float(area), width * height, rel_tol=1e-06, abs_tol=1e-06
                )
            )
        ):
            issues.append(
                f"annotation {annotation_id}: area must equal bbox width*height"
            )
    counts = {status: 0 for status in AnnotationStatus}
    for record in materialized:
        counts[record.status] += 1
        annotation_count = len(annotations_by_image.get(record.image_id, []))
        if record.status == AnnotationStatus.POSITIVE and annotation_count != 1:
            issues.append(
                f"positive image {record.image_id} must have exactly one box, got {annotation_count}"
            )
        elif record.status != AnnotationStatus.POSITIVE and annotation_count != 0:
            issues.append(
                f"{record.status.value} image {record.image_id} must not have a box"
            )
    if require_complete_primary:
        schema_versions = {record.schema_version for record in materialized}
        if schema_versions != {PRIMARY_SPLIT_PROTOCOL_VERSION}:
            issues.append(
                f"complete primary manifest must use current schema/split protocol v{PRIMARY_SPLIT_PROTOCOL_VERSION}, got {sorted(schema_versions)}"
            )
        for split in PRIMARY_SPLITS.values():
            split_records = tuple(
                (record for record in materialized if record.split == split.name)
            )
            if len(split_records) != split.expected_frames:
                issues.append(
                    f"complete manifest split {split.name.value} has {len(split_records)} frames, expected {split.expected_frames}"
                )
            actual_groups = {record.group_id for record in split_records}
            if actual_groups != set(split.group_ids):
                issues.append(
                    f"complete manifest split {split.name.value} has groups {sorted(actual_groups)}, expected {sorted(split.group_ids)}"
                )
            actual_episodes = {record.episode_index for record in split_records}
            if actual_episodes != set(split.episode_indices):
                issues.append(
                    f"complete manifest split {split.name.value} has {len(actual_episodes)} episodes, expected {len(split.episode_indices)}"
                )
            for episode_index in split.episode_indices:
                frame_indices = sorted(
                    (
                        record.frame_index
                        for record in split_records
                        if record.episode_index == episode_index
                    )
                )
                if not frame_indices:
                    continue
                if frame_indices[0] != 0:
                    issues.append(
                        f"complete manifest episode {episode_index} must reset sampling at frame 0"
                    )
                if any(
                    (
                        current - previous != split.stride
                        for previous, current in zip(frame_indices, frame_indices[1:])
                    )
                ):
                    issues.append(
                        f"complete manifest episode {episode_index} is not contiguous at {split.annotation_hz} Hz stride {split.stride}"
                    )
    if issues:
        raise AnnotationValidationError(issues)
    return ValidationSummary(
        images=len(materialized),
        annotations=len(payload["annotations"]),
        positive=counts[AnnotationStatus.POSITIVE],
        confirmed_negative=counts[AnnotationStatus.CONFIRMED_NEGATIVE],
        ignored=counts[AnnotationStatus.UNCERTAIN_IGNORE],
        unreviewed=counts[AnnotationStatus.UNREVIEWED],
    )


@dataclass(frozen=True, slots=True)
class RevisionVerificationSummary:
    revision: str
    source_revision: str
    frames: int
    annotations: int
    positive: int
    confirmed_negative: int
    ignored: int
    unreviewed: int
    source_revision_sha256: str
    approved_payload_sha256: str
    qa_artifacts_sha256: str
    training_allowed: bool


APPROVAL_FILENAME = "approval.json"
APPROVAL_CHECKSUM_FILENAME = "approval.sha256"
ARTIFACT_LEDGER_FILENAME = "artifact_hashes.json"
MANIFEST_FILENAME = "manifest.jsonl"
GLOBAL_COCO_FILENAME = "instances.json"
PROPOSALS_FILENAME = "proposals.jsonl"
CORRECTION_HISTORY_FILENAME = "correction_history.jsonl"
SUMMARY_FILENAME = "annotation_summary.json"


class RevisionLockError(ValueError):
    """Raised when a revision cannot be safely approved or verified."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionLockError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RevisionLockError(f"JSON root must be an object: {path}")
    return payload


LOCK_SCHEMA_VERSION = 1
_REVISION_PATTERN = re.compile("[A-Za-z0-9][A-Za-z0-9._-]*\\Z")


def _validate_revision_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise RevisionLockError(
            f"{field} must match {_REVISION_PATTERN.pattern!r}, got {value!r}"
        )
    return value


def _relative_posix(path: Path, root: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RevisionLockError(f"{label} must be inside {root}: {resolved}") from exc
    if not relative.parts:
        raise RevisionLockError(f"{label} must name a file below {root}")
    return relative.as_posix()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_entry(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RevisionLockError(f"missing {role} artifact: {path}")
    return {
        "path": _relative_posix(path, root, label=role),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_entries(
    entries: Sequence[Mapping[str, Any]], root: Path, *, label: str
) -> None:
    actual_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RevisionLockError(f"{label} artifact entries must be objects")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RevisionLockError(
                f"{label} artifact entry has invalid path {path_value!r}"
            )
        path = (root / Path(path_value)).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RevisionLockError(
                f"{label} artifact escapes its root: {path_value}"
            ) from exc
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise RevisionLockError(f"{label} artifact {path_value} has invalid role")
        actual_entries.append(_artifact_entry(path, root, role=role))
    if list(entries) != actual_entries:
        for expected, actual in zip(entries, actual_entries, strict=False):
            if expected != actual:
                raise RevisionLockError(
                    f"{label} artifact hash mismatch for {expected.get('path')}: expected {expected}, actual {actual}"
                )
        raise RevisionLockError(f"{label} artifact inventory mismatch")


def _artifact_set_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    """Hash the sorted ``path/NUL/size/NUL/digest`` artifact inventory."""
    digest = hashlib.sha256()
    ordered = sorted(entries, key=lambda item: str(item["path"]))
    seen: set[str] = set()
    for entry in ordered:
        path = str(entry["path"])
        if path in seen:
            raise RevisionLockError(
                f"duplicate artifact path in hash inventory: {path}"
            )
        seen.add(path)
        digest.update(path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(int(entry["bytes"])).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(entry["sha256"]).lower().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def partition_coco_by_split(
    coco: Mapping[str, Any], records: Sequence[FrameRecord]
) -> dict[str, dict[str, Any]]:
    """Create exact split COCO payloads while keeping a single global manifest."""
    validate_coco_manifest(coco, records)
    result: dict[str, dict[str, Any]] = {}
    for split in PRIMARY_SPLITS.values():
        split_records = tuple(
            (record for record in records if record.split == split.name)
        )
        if not split_records:
            continue
        image_ids = {record.image_id for record in split_records}
        payload = dict(coco)
        payload["info"] = dict(coco.get("info", {}), split=split.name.value)
        payload["images"] = [
            image for image in coco["images"] if image["id"] in image_ids
        ]
        payload["annotations"] = [
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in image_ids
        ]
        validate_coco_manifest(payload, split_records)
        result[split.name.value] = payload
    return result


_SPLIT_FILENAMES = {
    "train": "train.coco.json",
    "calibration": "calibration.coco.json",
    "validation": "validation.coco.json",
    "locked_test": "test.coco.json",
}


def _validate_split_coco_files(
    revision_root: Path, records: Sequence[FrameRecord], global_coco: Mapping[str, Any]
) -> tuple[str, ...]:
    expected_by_split = partition_coco_by_split(global_coco, records)
    filenames: list[str] = []
    for split, expected in expected_by_split.items():
        filename = _SPLIT_FILENAMES[split]
        path = revision_root / filename
        if not path.is_file():
            raise RevisionLockError(
                f"missing COCO file for populated split {split}: {path}"
            )
        actual = read_coco_json(path)
        split_records = tuple(
            (record for record in records if record.split.value == split)
        )
        validate_coco_manifest(actual, split_records)
        if actual != expected:
            raise RevisionLockError(
                f"{filename} is not the exact deterministic partition of {GLOBAL_COCO_FILENAME}"
            )
        filenames.append(filename)
    absent = set(_SPLIT_FILENAMES) - set(expected_by_split)
    for split in absent:
        path = revision_root / _SPLIT_FILENAMES[split]
        if path.exists():
            raise RevisionLockError(
                f"unexpected COCO file for empty split {split}: {path}"
            )
    return tuple(sorted(filenames))


def _status_counts(records: Sequence[FrameRecord]) -> dict[str, int]:
    counts = Counter((record.status.value for record in records))
    return {status.value: counts[status.value] for status in AnnotationStatus}


def _validate_annotation_summary(
    payload: Mapping[str, Any],
    records: Sequence[FrameRecord],
    *,
    expected_revision: str,
    expected_state: str,
    training_allowed: bool,
) -> None:
    issues: list[str] = []
    if payload.get("revision") != expected_revision:
        issues.append(
            f"annotation_summary revision={payload.get('revision')!r}, expected {expected_revision!r}"
        )
    if payload.get("revision_state") != expected_state:
        issues.append(
            f"annotation_summary revision_state={payload.get('revision_state')!r}, expected {expected_state!r}"
        )
    if payload.get("training_allowed") is not training_allowed:
        issues.append(
            f"annotation_summary training_allowed={payload.get('training_allowed')!r}, expected {training_allowed}"
        )
    actual_counts = _status_counts(records)
    reported_counts = payload.get("status_counts")
    if not isinstance(reported_counts, dict):
        issues.append("annotation_summary status_counts must be an object")
    else:
        for status, count in actual_counts.items():
            if reported_counts.get(status) != count:
                issues.append(
                    f"annotation_summary status_counts.{status}={reported_counts.get(status)!r}, expected {count}"
                )
    scope = payload.get("scope")
    if not isinstance(scope, dict) or scope.get("frames") != len(records):
        issues.append(f"annotation_summary scope.frames must equal {len(records)}")
    if issues:
        raise RevisionLockError("; ".join(issues))


def _validate_revision_data(
    revision_root: Path,
    *,
    expected_revision: str,
    expected_state: str,
    training_allowed: bool,
) -> tuple[tuple[FrameRecord, ...], dict[str, Any], ValidationSummary, tuple[str, ...]]:
    records = read_manifest_jsonl(revision_root / MANIFEST_FILENAME)
    if not records:
        raise RevisionLockError("annotation revision manifest is empty")
    wrong_versions = [
        record.image_id
        for record in records
        if record.review_version != expected_revision
    ]
    if wrong_versions:
        raise RevisionLockError(
            f"manifest review_version must be {expected_revision!r}; first mismatches={wrong_versions[:20]}"
        )
    coco = read_coco_json(revision_root / GLOBAL_COCO_FILENAME)
    if coco.get("info", {}).get("version") != expected_revision:
        raise RevisionLockError(
            f"COCO info.version must be {expected_revision!r}, got {coco.get('info', {}).get('version')!r}"
        )
    summary = validate_coco_manifest(coco, records)
    if summary.unreviewed != 0:
        raise RevisionLockError(
            f"approved source must have unreviewed=0, got {summary.unreviewed}"
        )
    split_filenames = _validate_split_coco_files(revision_root, records, coco)
    annotation_summary = _read_json_object(revision_root / SUMMARY_FILENAME)
    _validate_annotation_summary(
        annotation_summary,
        records,
        expected_revision=expected_revision,
        expected_state=expected_state,
        training_allowed=training_allowed,
    )
    return (records, coco, summary, split_filenames)


def _proposal_bbox_by_image(
    coco: Mapping[str, Any],
) -> dict[int, tuple[float, float, float, float]]:
    return {
        int(annotation["image_id"]): tuple(
            (float(value) for value in annotation["bbox"])
        )
        for annotation in coco["annotations"]
    }


@dataclass(frozen=True, slots=True)
class LabelProposal:
    image_id: int
    status: AnnotationStatus
    bbox: tuple[float, float, float, float] | None = None
    partial: bool = False
    occluded: bool = False
    blurred: bool = False
    review_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AnnotationStatus(self.status))
        if not isinstance(self.image_id, int) or self.image_id <= 0:
            raise ValueError("image_id must be a positive integer")
        if self.status == AnnotationStatus.POSITIVE:
            if self.bbox is None or len(self.bbox) != 4:
                raise ValueError("positive proposal requires bbox=[x,y,width,height]")
            bbox = tuple((float(value) for value in self.bbox))
            if any((not math.isfinite(value) for value in bbox)):
                raise ValueError("bbox contains NaN or infinity")
            x, y, width, height = bbox
            if x < 0 or y < 0 or width <= 0 or (height <= 0):
                raise ValueError(f"invalid bbox {bbox}")
            object.__setattr__(self, "bbox", bbox)
        elif self.bbox is not None:
            raise ValueError(f"{self.status.value} proposal must not contain bbox")
        if self.review_version is not None and (not self.review_version.strip()):
            raise ValueError("review_version must be non-empty when provided")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelProposal":
        known = {
            "image_id",
            "status",
            "bbox",
            "partial",
            "occluded",
            "blurred",
            "review_version",
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown proposal fields: {sorted(unknown)}")
        if "image_id" not in payload or "status" not in payload:
            raise ValueError("proposal requires image_id and status")
        return cls(**dict(payload))


def _validate_proposals_against_revision(
    proposals: Sequence[LabelProposal],
    records: Sequence[FrameRecord],
    coco: Mapping[str, Any],
) -> None:
    proposal_by_id = {proposal.image_id: proposal for proposal in proposals}
    record_by_id = {record.image_id: record for record in records}
    if set(proposal_by_id) != set(record_by_id):
        missing = sorted(set(record_by_id) - set(proposal_by_id))[:20]
        extra = sorted(set(proposal_by_id) - set(record_by_id))[:20]
        raise RevisionLockError(
            f"proposals must exactly cover the manifest; missing={missing}, extra={extra}"
        )
    bbox_by_id = _proposal_bbox_by_image(coco)
    issues: list[str] = []
    for image_id, record in record_by_id.items():
        proposal = proposal_by_id[image_id]
        if proposal.status == AnnotationStatus.UNREVIEWED:
            issues.append(f"proposal {image_id} remains unreviewed")
        for field in ("status", "partial", "occluded", "blurred"):
            if getattr(proposal, field) != getattr(record, field):
                issues.append(
                    f"proposal {image_id} {field}={getattr(proposal, field)!r}, manifest={getattr(record, field)!r}"
                )
        expected_bbox = bbox_by_id.get(image_id)
        if proposal.bbox != expected_bbox:
            issues.append(
                f"proposal {image_id} bbox={proposal.bbox!r}, COCO={expected_bbox!r}"
            )
        if len(issues) >= 30:
            break
    if issues:
        raise RevisionLockError("proposal/revision mismatch: " + "; ".join(issues))


def _read_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RevisionLockError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(payload)
    except json.JSONDecodeError as exc:
        raise RevisionLockError(
            f"invalid JSONL at {path}:{exc.lineno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise RevisionLockError(f"cannot read JSONL {path}: {exc}") from exc
    return tuple(rows)


def _validate_correction_history(path: Path) -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl_objects(path)
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        correction_id = row.get("correction_id")
        if not isinstance(correction_id, str) or not correction_id.strip():
            raise RevisionLockError(
                f"correction history row {index} requires non-empty correction_id"
            )
        if correction_id in seen:
            raise RevisionLockError(f"duplicate correction_id {correction_id!r}")
        seen.add(correction_id)
    return rows


HASH_ALGORITHM = "sha256"


def read_proposals_jsonl(path: str | Path) -> tuple[LabelProposal, ...]:
    proposals: list[LabelProposal] = []
    seen: set[int] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("proposal must be a JSON object")
                proposal = LabelProposal.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid proposal at {path}:{line_number}: {exc}"
                ) from exc
            if proposal.image_id in seen:
                raise ValueError(f"duplicate proposal for image_id {proposal.image_id}")
            seen.add(proposal.image_id)
            proposals.append(proposal)
    return tuple(proposals)


def verify_locked_revision(
    revision_root: str | Path, *, verify_source: bool = True
) -> RevisionVerificationSummary:
    """Fail closed if any approved payload, metadata, or bound source artifact changed."""
    revision_root = Path(revision_root).resolve()
    if not revision_root.is_dir():
        raise RevisionLockError(
            f"locked revision directory does not exist: {revision_root}"
        )
    expected_files = {
        APPROVAL_FILENAME,
        APPROVAL_CHECKSUM_FILENAME,
        ARTIFACT_LEDGER_FILENAME,
        MANIFEST_FILENAME,
        GLOBAL_COCO_FILENAME,
        PROPOSALS_FILENAME,
        CORRECTION_HISTORY_FILENAME,
        SUMMARY_FILENAME,
    }
    checksum_line = (
        (revision_root / APPROVAL_CHECKSUM_FILENAME).read_text(encoding="ascii").strip()
    )
    expected_checksum_line = (
        f"{sha256_file(revision_root / APPROVAL_FILENAME)}  {APPROVAL_FILENAME}"
    )
    if checksum_line != expected_checksum_line:
        raise RevisionLockError("approval.sha256 does not match approval.json")
    approval = _read_json_object(revision_root / APPROVAL_FILENAME)
    if approval.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise RevisionLockError("unsupported approval lock schema_version")
    revision = _validate_revision_name(
        str(approval.get("revision", "")), field="revision"
    )
    if revision_root.name != revision:
        raise RevisionLockError(
            f"locked directory name {revision_root.name!r} != approval revision {revision!r}"
        )
    if approval.get("revision_state") != "approved_locked":
        raise RevisionLockError("revision_state must be approved_locked")
    if approval.get("training_allowed") is not True:
        raise RevisionLockError("training_allowed must be true")
    if approval.get("user_approved") is not True:
        raise RevisionLockError("user_approved must be true")
    integrity = approval.get("integrity")
    if not isinstance(integrity, dict):
        raise RevisionLockError("approval integrity metadata must be an object")
    ledger_path = revision_root / str(integrity.get("artifact_ledger", ""))
    if (
        ledger_path.name != ARTIFACT_LEDGER_FILENAME
        or ledger_path.parent != revision_root
    ):
        raise RevisionLockError("approval artifact_ledger path is invalid")
    if sha256_file(ledger_path) != integrity.get("artifact_ledger_sha256"):
        raise RevisionLockError("artifact_hashes.json does not match approval metadata")
    ledger = _read_json_object(ledger_path)
    if (
        ledger.get("schema_version") != LOCK_SCHEMA_VERSION
        or ledger.get("algorithm") != HASH_ALGORITHM
    ):
        raise RevisionLockError("unsupported artifact ledger schema or hash algorithm")
    approved_section = ledger.get("approved_payload")
    if not isinstance(approved_section, dict) or approved_section.get("root") != ".":
        raise RevisionLockError("invalid approved_payload ledger section")
    approved_entries = approved_section.get("artifacts")
    if not isinstance(approved_entries, list):
        raise RevisionLockError("approved_payload artifacts must be a list")
    _verify_entries(approved_entries, revision_root, label="approved payload")
    approved_payload_sha256 = _artifact_set_digest(approved_entries)
    if approved_payload_sha256 != approved_section.get("payload_sha256"):
        raise RevisionLockError("approved payload inventory digest mismatch")
    if approved_payload_sha256 != integrity.get("approved_payload_sha256"):
        raise RevisionLockError(
            "approved payload digest does not match approval metadata"
        )
    payload_names = {str(entry["path"]) for entry in approved_entries}
    expected_files.update(payload_names)
    actual_files = {
        path.relative_to(revision_root).as_posix()
        for path in revision_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RevisionLockError(
            f"locked revision has missing or unrecorded files; missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}"
        )
    source_section = ledger.get("source")
    qa_section = ledger.get("qa")
    if not isinstance(source_section, dict) or not isinstance(qa_section, dict):
        raise RevisionLockError("source and qa ledger sections must be objects")
    source_entries = source_section.get("artifacts")
    qa_entries = qa_section.get("artifacts")
    if not isinstance(source_entries, list) or not isinstance(qa_entries, list):
        raise RevisionLockError("source and qa artifacts must be lists")
    source_revision_sha256 = _artifact_set_digest(source_entries)
    qa_artifacts_sha256 = _artifact_set_digest(qa_entries)
    if source_revision_sha256 != source_section.get("revision_sha256"):
        raise RevisionLockError("source revision inventory digest mismatch")
    if source_revision_sha256 != approval.get("source", {}).get("revision_sha256"):
        raise RevisionLockError(
            "source revision digest does not match approval metadata"
        )
    if qa_artifacts_sha256 != qa_section.get("artifacts_sha256"):
        raise RevisionLockError("QA artifact inventory digest mismatch")
    if qa_artifacts_sha256 != integrity.get("qa_artifacts_sha256"):
        raise RevisionLockError("QA artifact digest does not match approval metadata")
    if qa_section.get("artifact_count") != len(qa_entries):
        raise RevisionLockError("QA artifact_count does not match inventory")
    if qa_section.get("total_bytes") != sum(
        (int(entry["bytes"]) for entry in qa_entries)
    ):
        raise RevisionLockError("QA total_bytes does not match inventory")
    if verify_source:
        scope_value = source_section.get("scope_root")
        if scope_value != "..":
            raise RevisionLockError("source scope_root must be '..'")
        scope_root = revision_root.parent.resolve()
        _verify_entries(source_entries, scope_root, label="source")
        _verify_entries(qa_entries, scope_root, label="QA")
    records, coco, validation, _ = _validate_revision_data(
        revision_root,
        expected_revision=revision,
        expected_state="approved_locked",
        training_allowed=True,
    )
    _validate_proposals_against_revision(
        read_proposals_jsonl(revision_root / PROPOSALS_FILENAME), records, coco
    )
    _validate_correction_history(revision_root / CORRECTION_HISTORY_FILENAME)
    source_revision = _validate_revision_name(
        str(approval.get("source", {}).get("revision", "")), field="source_revision"
    )
    return RevisionVerificationSummary(
        revision=revision,
        source_revision=source_revision,
        frames=validation.images,
        annotations=validation.annotations,
        positive=validation.positive,
        confirmed_negative=validation.confirmed_negative,
        ignored=validation.ignored,
        unreviewed=validation.unreviewed,
        source_revision_sha256=source_revision_sha256,
        approved_payload_sha256=approved_payload_sha256,
        qa_artifacts_sha256=qa_artifacts_sha256,
        training_allowed=True,
    )


@dataclass(frozen=True, slots=True)
class LocalLeRobotVideoMetadata:
    """Read-only projection of the standard LeRobot v3 video metadata."""

    root: Path
    info: dict
    episodes: tuple[dict, ...]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict:
        return self.info["features"]

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(
            (
                key
                for key, feature in self.features.items()
                if feature.get("dtype") == "video"
            )
        )

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        row = self.episodes[episode_index]
        return Path(
            self.info["video_path"].format(
                video_key=video_key,
                chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                file_index=int(row[f"videos/{video_key}/file_index"]),
            )
        )


DEFAULT_DATASET_REPO_ID = "Jason-Doldol-Research/task4_polyp"
CAMERA_KEY = "observation.images.cam_low"


def _load_metadata(
    dataset_root: Path, repo_id: str = DEFAULT_DATASET_REPO_ID
) -> LocalLeRobotVideoMetadata:
    """Load local standard metadata without creating Hugging Face cache files."""
    del repo_id
    import pyarrow.parquet as pq

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(dataset_root / "meta" / "episodes")
    columns = [
        "episode_index",
        "length",
        f"videos/{CAMERA_KEY}/chunk_index",
        f"videos/{CAMERA_KEY}/file_index",
        f"videos/{CAMERA_KEY}/from_timestamp",
        f"videos/{CAMERA_KEY}/to_timestamp",
    ]
    rows = [
        row
        for path in episode_paths
        for row in pq.read_table(path, columns=columns).to_pylist()
    ]
    rows.sort(key=lambda row: int(row["episode_index"]))
    return LocalLeRobotVideoMetadata(root=dataset_root, info=info, episodes=tuple(rows))


def _episode_rows(metadata) -> dict[int, dict]:
    rows = {int(row["episode_index"]): dict(row) for row in metadata.episodes}
    if set(rows) != set(range(320)):
        raise ValueError(
            f"Task4 metadata episode indices must be exactly 0..319, got {min(rows, default='none')}..{max(rows, default='none')} ({len(rows)} rows)"
        )
    return rows


def sample_episode_frame_indices(
    episode_length: int, target_hz: int, *, source_fps: int = SOURCE_FPS
) -> tuple[int, ...]:
    """Sample from local frame zero, resetting phase at every episode."""
    if episode_length < 0:
        raise ValueError("episode_length must be non-negative")
    stride = sampling_stride(source_fps, target_hz)
    return tuple(range(0, episode_length, stride))


def expected_sample_count(
    episode_lengths: Mapping[int, int] | Sequence[int],
    split: DatasetSplit | SplitDefinition | str,
) -> int:
    definition = resolve_split(split)
    return sum(
        (
            len(
                sample_episode_frame_indices(
                    int(episode_lengths[index]), definition.annotation_hz
                )
            )
            for index in definition.episode_indices
        )
    )


def validate_expected_counts(
    episode_lengths: Mapping[int, int] | Sequence[int],
) -> None:
    if len(episode_lengths) != 320:
        raise ValueError(
            f"Task4 metadata must contain 320 episodes, got {len(episode_lengths)}"
        )
    if isinstance(episode_lengths, Mapping) and set(episode_lengths) != set(range(320)):
        raise ValueError("Task4 metadata episode indices must be exactly 0..319")
    mismatches = []
    for split in PRIMARY_SPLITS.values():
        actual = expected_sample_count(episode_lengths, split)
        if actual != split.expected_frames:
            mismatches.append(
                f"{split.name.value}: expected {split.expected_frames}, got {actual}"
            )
    if mismatches:
        raise ValueError(
            "Task4 episode lengths do not match the locked dataset: "
            + "; ".join(mismatches)
        )


def evenly_spaced_subset(
    items: Sequence[tuple[int, int]], count: int
) -> tuple[tuple[int, int], ...]:
    """Select a deterministic sequence-wide pilot subset without duplicate keys."""
    if count < 0 or count > len(items):
        raise ValueError(f"count must be in [0, {len(items)}], got {count}")
    if count == 0:
        return ()
    if count == 1:
        return (items[len(items) // 2],)
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("evenly-spaced selection produced duplicate indices")
    return tuple((items[index] for index in indices))


PILOT_EXPECTED_FRAMES_BY_GROUP: Mapping[int, int] = {
    0: 200,
    1: 200,
    2: 200,
    3: 200,
    4: 200,
    5: 200,
    6: 500,
}
PILOT_SELECTION_HZ_BY_GROUP: Mapping[int, int] = {
    0: 5,
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5: 5,
    6: 15,
}


def pilot_sample_keys(
    episode_lengths: Mapping[int, int] | Sequence[int],
) -> frozenset[tuple[int, int]]:
    """Return the exact user-approved 1,700 pilot keys.

    The pilot predates the dedicated calibration split.  Its 200 group-5 keys
    remain on the historical 5 Hz grid so reviewed image IDs do not change.
    They are a strict subset of the final 15 Hz calibration sampling grid.
    """
    selected: set[tuple[int, int]] = set()
    for group in GROUPS[:6]:
        candidates = [
            (episode_index, frame_index)
            for episode_index in group.episode_indices
            for frame_index in sample_episode_frame_indices(
                int(episode_lengths[episode_index]),
                PILOT_SELECTION_HZ_BY_GROUP[group.group_id],
            )
        ]
        selected.update(
            evenly_spaced_subset(
                candidates, PILOT_EXPECTED_FRAMES_BY_GROUP[group.group_id]
            )
        )
    validation_candidates = [
        (episode_index, frame_index)
        for episode_index in GROUPS[6].episode_indices
        for frame_index in sample_episode_frame_indices(
            int(episode_lengths[episode_index]), PILOT_SELECTION_HZ_BY_GROUP[6]
        )
    ]
    selected.update(
        evenly_spaced_subset(validation_candidates, PILOT_EXPECTED_FRAMES_BY_GROUP[6])
    )
    if len(selected) != 1700:
        raise AssertionError(
            f"pilot must contain 1,700 unique frames, got {len(selected)}"
        )
    return frozenset(selected)


def build_frame_records(
    dataset_root: str | Path,
    *,
    splits: Sequence[DatasetSplit | str] | None = None,
    mode: str = "full",
    review_version: str = "draft",
    repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> tuple[FrameRecord, ...]:
    """Build records from official video offsets without decoding image data."""
    dataset_root = Path(dataset_root).resolve()
    metadata = _load_metadata(dataset_root, repo_id=repo_id)
    if int(metadata.fps) != SOURCE_FPS:
        raise ValueError(f"source dataset must be {SOURCE_FPS} Hz, got {metadata.fps}")
    if CAMERA_KEY not in metadata.video_keys:
        raise ValueError(f"source dataset is missing video key {CAMERA_KEY!r}")
    camera_feature = metadata.features[CAMERA_KEY]
    if tuple(camera_feature.get("shape", ())) != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError(
            f"{CAMERA_KEY} must have shape {(FRAME_HEIGHT, FRAME_WIDTH, 3)}, got {camera_feature.get('shape')}"
        )
    if mode not in {"full", "pilot"}:
        raise ValueError("mode must be 'full' or 'pilot'")
    if splits is None:
        splits = (
            (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION, DatasetSplit.VALIDATION)
            if mode == "pilot"
            else tuple(PRIMARY_SPLITS)
        )
    rows = _episode_rows(metadata)
    lengths = {episode_index: int(row["length"]) for episode_index, row in rows.items()}
    validate_expected_counts(lengths)
    requested_splits = tuple((resolve_split(split) for split in splits))
    if not requested_splits:
        raise ValueError("splits must not be empty")
    if len({split.name for split in requested_splits}) != len(requested_splits):
        raise ValueError("splits must not contain duplicates")
    if mode == "pilot" and any(
        (split.name == DatasetSplit.LOCKED_TEST for split in requested_splits)
    ):
        raise ValueError(
            "the approved pilot is limited to train+calibration+validation and must not include locked_test"
        )
    pilot_keys = pilot_sample_keys(lengths) if mode == "pilot" else None
    records: list[FrameRecord] = []
    for split in requested_splits:
        for episode_index in split.episode_indices:
            row = rows[episode_index]
            group = group_for_episode(episode_index)
            video_path = metadata.get_video_file_path(episode_index, CAMERA_KEY)
            actual_file_index = int(row[f"videos/{CAMERA_KEY}/file_index"])
            if actual_file_index != group.video_file_index:
                raise ValueError(
                    f"episode {episode_index} metadata uses video file {actual_file_index}, fixed group {group.group_id} expects {group.video_file_index}"
                )
            source_video = video_path.as_posix()
            if not (dataset_root / video_path).is_file():
                raise FileNotFoundError(dataset_root / video_path)
            video_start = float(row[f"videos/{CAMERA_KEY}/from_timestamp"])
            for frame_index in sample_episode_frame_indices(
                int(row["length"]), split.annotation_hz
            ):
                key = (episode_index, frame_index)
                if pilot_keys is not None and key not in pilot_keys:
                    continue
                relative_image_path = (
                    Path("images")
                    / split.name.value
                    / f"group-{group.group_id:02d}"
                    / f"episode-{episode_index:06d}"
                    / f"frame-{frame_index:06d}.jpg"
                ).as_posix()
                timestamp = frame_index / SOURCE_FPS
                records.append(
                    FrameRecord(
                        image_id=stable_image_id(episode_index, frame_index),
                        episode_index=episode_index,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        video_timestamp=video_start + timestamp,
                        group_id=group.group_id,
                        split=split.name,
                        source_video=source_video,
                        image_path=relative_image_path,
                        status=AnnotationStatus.UNREVIEWED,
                        review_version=review_version,
                    )
                )
    ordered = tuple(
        sorted(records, key=lambda record: (record.episode_index, record.frame_index))
    )
    if mode == "full":
        for split in requested_splits:
            actual = sum((record.split == split.name for record in ordered))
            if actual != split.expected_frames:
                raise AssertionError(
                    f"{split.name.value} extraction planned {actual}, expected {split.expected_frames}"
                )
    else:
        for split in requested_splits:
            for group in split.groups:
                expected = PILOT_EXPECTED_FRAMES_BY_GROUP[group.group_id]
                actual = sum((record.group_id == group.group_id for record in ordered))
                if actual != expected:
                    raise AssertionError(
                        f"pilot group {group.group_id} ({split.name.value}) has {actual} frames, expected {expected}"
                    )
    return ordered


def _save_jpeg(frame, output_path: Path, *, quality: int, overwrite: bool) -> bool:
    if output_path.is_file() and (not overwrite):
        with Image.open(output_path) as existing:
            if existing.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ValueError(
                    f"existing frame has unexpected size: {output_path} ({existing.size})"
                )
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        image = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy(), mode="RGB")
        image.save(temporary, format="JPEG", quality=quality, subsampling=0)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def extract_images(
    records: Iterable[FrameRecord],
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    backend: str = "pyav",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> int:
    """Decode records episode-wise using LeRobot's timestamp-aware decoder."""
    from lerobot.datasets.video_utils import decode_video_frames

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    grouped: dict[tuple[int, str], list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_index, record.source_video].append(record)
    written = 0
    for (episode_index, source_video), episode_records in sorted(grouped.items()):
        episode_records.sort(key=lambda record: record.frame_index)
        frames = decode_video_frames(
            dataset_root / source_video,
            [record.video_timestamp for record in episode_records],
            tolerance_s=0.5 / SOURCE_FPS + 1e-06,
            backend=backend,
            return_uint8=True,
        )
        if len(frames) != len(episode_records):
            raise RuntimeError(
                f"episode {episode_index}: decoder returned {len(frames)} frames for {len(episode_records)} timestamps"
            )
        for frame, record in zip(frames, episode_records, strict=True):
            if tuple(frame.shape) != (3, FRAME_HEIGHT, FRAME_WIDTH):
                raise ValueError(
                    f"image {record.image_id}: decoded shape is {tuple(frame.shape)}"
                )
            written += int(
                _save_jpeg(
                    frame,
                    output_root / record.image_path,
                    quality=jpeg_quality,
                    overwrite=overwrite,
                )
            )
    return written


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest_jsonl(path: str | Path, records: Iterable[FrameRecord]) -> None:
    ordered = validate_manifest_records(records)
    text = "".join(
        (json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in ordered)
    )
    _atomic_text(Path(path), text)


POLYP_CATEGORY = {"id": POLYP_CATEGORY_ID, "name": "polyp", "supercategory": "lesion"}


def build_empty_coco(
    records: Iterable[FrameRecord],
    *,
    dataset_name: str = "task4_polyp_bbox",
    annotation_revision: str = "draft",
) -> dict[str, Any]:
    ordered = validate_manifest_records(records)
    split_values = {record.split.value for record in ordered}
    split_name = next(iter(split_values)) if len(split_values) == 1 else "mixed"
    schema_versions = {record.schema_version for record in ordered}
    schema_version = next(iter(schema_versions), PRIMARY_SPLIT_PROTOCOL_VERSION)
    return {
        "info": {
            "description": dataset_name,
            "version": annotation_revision,
            "manifest_schema_version": schema_version,
            "split": split_name,
        },
        "licenses": [],
        "images": [
            {
                "id": record.image_id,
                "file_name": record.image_path,
                "width": record.width,
                "height": record.height,
                "episode_index": record.episode_index,
                "frame_index": record.frame_index,
                "group_id": record.group_id,
                "split": record.split.value,
            }
            for record in ordered
        ],
        "annotations": [],
        "categories": [dict(POLYP_CATEGORY)],
    }


def write_coco_json(path: str | Path, coco: Mapping[str, Any]) -> None:
    _atomic_text(Path(path), json.dumps(dict(coco), indent=2, sort_keys=True) + "\n")


def write_split_coco_files(
    output_dir: str | Path, coco: Mapping[str, Any], records: Sequence[FrameRecord]
) -> dict[str, Path]:
    filenames = {
        "train": "train.coco.json",
        "calibration": "calibration.coco.json",
        "validation": "validation.coco.json",
        "locked_test": "test.coco.json",
    }
    outputs: dict[str, Path] = {}
    for split, payload in partition_coco_by_split(coco, records).items():
        path = Path(output_dir) / filenames[split]
        write_coco_json(path, payload)
        outputs[split] = path
    return outputs


LOGGER = logging.getLogger(__name__)


def create_annotation_workspace(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    splits: Sequence[DatasetSplit | str] | None = None,
    mode: str,
    review_version: str,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    metadata_only: bool = False,
    backend: str = "pyav",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> tuple[FrameRecord, ...]:
    records = build_frame_records(
        dataset_root,
        splits=splits,
        mode=mode,
        review_version=review_version,
        repo_id=repo_id,
    )
    output_root = Path(output_root)
    if not metadata_only:
        written = extract_images(
            records,
            dataset_root=dataset_root,
            output_root=output_root,
            backend=backend,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )
        LOGGER.info("wrote %d new image(s); %d total record(s)", written, len(records))
    write_manifest_jsonl(output_root / "manifest.jsonl", records)
    coco = build_empty_coco(records, annotation_revision=review_version)
    write_coco_json(output_root / "instances.json", coco)
    write_split_coco_files(output_root, coco, records)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple((split.value for split in DatasetSplit)),
        default=None,
        help="defaults to train+calibration+validation in pilot mode and all four primary splits in full mode",
    )
    parser.add_argument("--review-version", default="draft")
    parser.add_argument("--backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = create_annotation_workspace(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        splits=args.splits,
        mode=args.mode,
        review_version=args.review_version,
        repo_id=args.repo_id,
        metadata_only=args.metadata_only,
        backend=args.backend,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "annotation workspace ready: %s (%d records)", args.output_root, len(records)
    )
    return 0


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_PROMPT = "a smooth orange sphere."
DEFAULT_SHORTEST_EDGE = 384
DEFAULT_LONGEST_EDGE = 640


def _resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device without silently falling back from CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(
            f"CUDA device {requested!r} was requested but CUDA is unavailable"
        )
    return device


def _load_resume_prefix(
    path: Path, records: Sequence[FrameRecord]
) -> list[dict[str, Any]]:
    """Read an atomic partial shard and require it to be the exact record prefix."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid resume JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"resume row at {path}:{line_number} must be an object"
                )
            rows.append(row)
    if len(rows) > len(records):
        raise ValueError(
            f"resume output has {len(rows)} rows but deterministic shard has {len(records)}"
        )
    for row, record in zip(rows, records, strict=False):
        expected = {
            "image_id": record.image_id,
            "episode_index": record.episode_index,
            "frame_index": record.frame_index,
            "group_id": record.group_id,
            "split": record.split.value,
            "image_path": record.image_path,
        }
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"resume output is not the exact deterministic shard prefix at image_id={record.image_id}: {mismatches}"
            )
        if not isinstance(row.get("candidates"), list):
            raise ValueError(
                f"resume row image_id={record.image_id} requires candidates list"
            )
    return rows


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _candidate_from_result(
    score: torch.Tensor,
    text_label: str,
    box: torch.Tensor,
    *,
    width: int,
    height: int,
    maximum_area_ratio: float,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width, box_height = (x2 - x1, y2 - y1)
    if box_width <= 1.0 or box_height <= 1.0:
        return None
    area_ratio = box_width * box_height / (width * height)
    aspect_ratio = box_width / box_height
    if area_ratio > maximum_area_ratio or not 0.35 <= aspect_ratio <= 3.0:
        return None
    return {
        "score": float(score),
        "text_label": str(text_label),
        "bbox_xyxy": [x1, y1, x2, y2],
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
    }


def generate_candidates(
    *,
    manifest_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    model_id: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    batch_size: int = 8,
    box_threshold: float = 0.1,
    text_threshold: float = 0.1,
    maximum_area_ratio: float = 0.2,
    shortest_edge: int = DEFAULT_SHORTEST_EDGE,
    longest_edge: int = DEFAULT_LONGEST_EDGE,
    local_files_only: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    torch_threads: int | None = None,
    device: str = "auto",
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Run a deterministic manifest shard and atomically persist raw candidates."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shortest_edge <= 0 or longest_edge < shortest_edge:
        raise ValueError("Require 0 < shortest_edge <= longest_edge")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    records = read_manifest_jsonl(manifest_path)
    records = tuple(
        (
            record
            for index, record in enumerate(records)
            if index % num_shards == shard_index
        )
    )
    output_path = Path(output_path)
    if output_path.exists() and (not resume) and (not overwrite):
        raise FileExistsError(
            f"candidate output already exists: {output_path}; pass --resume or --overwrite"
        )
    results = _load_resume_prefix(output_path, records) if resume else []
    runtime_device = _resolve_device(device)
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        size={"shortest_edge": shortest_edge, "longest_edge": longest_edge},
    )
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        .to(runtime_device)
        .eval()
    )
    image_root = Path(image_root)
    print(
        f"shard={shard_index}/{num_shards} device={runtime_device} resume_rows={len(results)} total={len(records)}",
        flush=True,
    )
    started = time.perf_counter()
    for offset in range(len(results), len(records), batch_size):
        batch_records = records[offset : offset + batch_size]
        images: list[Image.Image] = []
        for record in batch_records:
            with Image.open(image_root / record.image_path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(runtime_device)
        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1] for image in images],
        )
        for record, detection in zip(batch_records, processed, strict=True):
            candidates = []
            for score, label, box in zip(
                detection["scores"],
                detection["text_labels"],
                detection["boxes"],
                strict=True,
            ):
                candidate = _candidate_from_result(
                    score,
                    label,
                    box,
                    width=record.width,
                    height=record.height,
                    maximum_area_ratio=maximum_area_ratio,
                )
                if candidate is not None:
                    candidates.append(candidate)
            candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
            results.append(
                {
                    "image_id": record.image_id,
                    "episode_index": record.episode_index,
                    "frame_index": record.frame_index,
                    "group_id": record.group_id,
                    "split": record.split.value,
                    "image_path": record.image_path,
                    "candidates": candidates,
                }
            )
        elapsed = time.perf_counter() - started
        print(
            f"shard={shard_index}/{num_shards} completed={len(results)}/{len(records)} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        _atomic_write_jsonl(output_path, results)
    return tuple(results)


def merge_candidate_shards(
    paths: Sequence[str | Path], output_path: str | Path
) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = int(row["image_id"])
                if image_id in seen:
                    raise ValueError(f"duplicate candidate image_id {image_id}")
                seen.add(image_id)
                rows.append(row)
    rows.sort(key=lambda row: int(row["image_id"]))
    _atomic_write_jsonl(Path(output_path), rows)


def render_candidate_sheets(
    candidate_path: str | Path,
    *,
    image_root: str | Path,
    output_root: str | Path,
    columns: int = 4,
    rows: int = 4,
    thumbnail_size: tuple[int, int] = (320, 240),
) -> tuple[Path, ...]:
    entries = [
        json.loads(line)
        for line in Path(candidate_path).read_text(encoding="utf-8").splitlines()
    ]
    per_page = columns * rows
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for page_index, offset in enumerate(range(0, len(entries), per_page)):
        page = entries[offset : offset + per_page]
        canvas = Image.new(
            "RGB", (columns * thumbnail_size[0], rows * thumbnail_size[1]), "black"
        )
        for index, entry in enumerate(page):
            with Image.open(Path(image_root) / entry["image_path"]) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            candidate = entry["candidates"][0] if entry["candidates"] else None
            color = (
                (0, 255, 0)
                if candidate and candidate["score"] >= 0.3
                else (255, 190, 0)
            )
            if candidate is not None:
                draw.rectangle(candidate["bbox_xyxy"], outline=color, width=3)
                score_text = f"{candidate['score']:.2f}"
            else:
                score_text = "none"
            draw.text(
                (8, 8),
                f"id={entry['image_id']} e{entry['episode_index']} f{entry['frame_index']} {score_text}",
                fill=color,
                font=ImageFont.load_default(),
                stroke_width=2,
                stroke_fill="black",
            )
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            cell = Image.new("RGB", thumbnail_size, "black")
            cell.paste(
                image,
                (
                    (thumbnail_size[0] - image.width) // 2,
                    (thumbnail_size[1] - image.height) // 2,
                ),
            )
            column, row = (index % columns, index // columns)
            canvas.paste(cell, (column * thumbnail_size[0], row * thumbnail_size[1]))
        target = output_root / f"candidate_sheet_{page_index:04d}.jpg"
        canvas.save(target, quality=95, subsampling=0)
        outputs.append(target)
    return tuple(outputs)


def grounding_dino_candidates_build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--image-root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model-id", default=DEFAULT_MODEL)
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--box-threshold", type=float, default=0.1)
    generate.add_argument("--text-threshold", type=float, default=0.1)
    generate.add_argument("--maximum-area-ratio", type=float, default=0.2)
    generate.add_argument("--shortest-edge", type=int, default=DEFAULT_SHORTEST_EDGE)
    generate.add_argument("--longest-edge", type=int, default=DEFAULT_LONGEST_EDGE)
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--torch-threads", type=int)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--overwrite", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--candidates", type=Path, required=True)
    render.add_argument("--image-root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    return parser


def grounding_dino_candidates_main(argv: Sequence[str] | None = None) -> int:
    args = grounding_dino_candidates_build_arg_parser().parse_args(argv)
    if args.command == "generate":
        generate_candidates(
            manifest_path=args.manifest,
            image_root=args.image_root,
            output_path=args.output,
            model_id=args.model_id,
            prompt=args.prompt,
            batch_size=args.batch_size,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            maximum_area_ratio=args.maximum_area_ratio,
            shortest_edge=args.shortest_edge,
            longest_edge=args.longest_edge,
            local_files_only=args.local_files_only,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            torch_threads=args.torch_threads,
            device=args.device,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    elif args.command == "merge":
        merge_candidate_shards(args.inputs, args.output)
    else:
        render_candidate_sheets(
            args.candidates, image_root=args.image_root, output_root=args.output_root
        )
    return 0


SCENARIOS: dict[str, dict[str, Any]] = {
    "S1": {
        "description": "late appearance",
        "acquisition_groups": [0],
        "train": range(0, 30),
        "validation": range(30, 40),
        "test": range(40, 50),
        "guard": range(0, 0),
    },
    "S2": {
        "description": "bottom/lower appearance",
        "acquisition_groups": [1, 6],
        "train": range(50, 100),
        "validation": range(260, 270),
        "test": range(280, 290),
        "guard": range(270, 280),
    },
    "S3": {
        "description": "left appearance",
        "acquisition_groups": [2, 7],
        "train": range(100, 150),
        "validation": range(290, 300),
        "test": range(310, 320),
        "guard": range(300, 310),
    },
    "S4": {
        "description": "upper appearance",
        "acquisition_groups": [3, 5],
        "train": range(150, 200),
        "validation": range(230, 240),
        "test": range(250, 260),
        "guard": range(240, 250),
    },
    "S5": {
        "description": "right appearance",
        "acquisition_groups": [4],
        "train": range(200, 210),
        "validation": range(210, 220),
        "test": range(220, 230),
        "guard": range(0, 0),
    },
}


def _episode_sets() -> dict[str, dict[str, set[int]]]:
    result: dict[str, dict[str, set[int]]] = {}
    for split in ("train", "validation", "test", "guard"):
        result[split] = {
            scenario: {int(value) for value in definition[split]}
            for scenario, definition in SCENARIOS.items()
        }
    return result


def _validate_episode_protocol(
    episode_sets: _build_scenario_static_v1_Mapping[
        str, _build_scenario_static_v1_Mapping[str, set[int]]
    ],
) -> None:
    by_split = {
        split: set().union(*scenario_sets.values())
        for split, scenario_sets in episode_sets.items()
    }
    names = tuple(by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                raise ValueError(
                    f"Task4 episode leakage between {left}/{right}: {sorted(overlap)}"
                )
    expected_episode_counts = {"train": 190, "validation": 50, "test": 50, "guard": 30}
    actual = {split: len(episodes) for split, episodes in by_split.items()}
    if actual != expected_episode_counts:
        raise ValueError(f"Task4 episode-count drift: {actual}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _validate_bbox(
    value: _build_scenario_static_v1_Sequence[Any], *, image_id: int
) -> list[float]:
    bbox = [float(item) for item in value]
    if len(bbox) != 4 or not all((math.isfinite(item) for item in bbox)):
        raise ValueError(f"invalid bbox for image_id={image_id}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
        raise ValueError(f"out-of-bounds bbox for image_id={image_id}: {bbox}")
    return bbox


def _approved_indexes(
    approved_manifest: Path, approved_instances: Path
) -> tuple[
    dict[int, dict[str, Any]], dict[int, list[float]], dict[int, dict[str, Any]]
]:
    rows = _read_jsonl(approved_manifest)
    by_image = {int(row["image_id"]): row for row in rows}
    if len(by_image) != len(rows):
        raise ValueError("approved Task4 manifest contains duplicate image IDs")
    instances = _read_json(approved_instances)
    annotations: dict[int, list[float]] = {}
    for annotation in instances["annotations"]:
        image_id = int(annotation["image_id"])
        x, y, width, height = (float(item) for item in annotation["bbox"])
        annotations[image_id] = _validate_bbox(
            [x, y, x + width, y + height], image_id=image_id
        )
    image_metadata = {int(row["id"]): row for row in instances["images"]}
    if set(by_image) != set(image_metadata):
        raise ValueError("approved Task4 COCO images and manifest differ")
    for image_id, row in by_image.items():
        has_box = image_id in annotations
        if (row["status"] == "positive") != has_box:
            raise ValueError(
                f"approved status/annotation mismatch for image_id={image_id}"
            )
    return (by_image, annotations, image_metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_completed_review(review_root: Path) -> dict[int, dict[str, Any]]:
    manifest_path = review_root / "review_manifest.csv"
    decisions_path = review_root / "review_decisions.json"
    completion_path = review_root / "REVIEW_COMPLETE.json"
    completion = _read_json(completion_path)
    decisions_payload = _read_json(decisions_path)
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("completed QC manifest hash mismatch")
    if completion.get("decisions_sha256") != _sha256(decisions_path):
        raise ValueError("completed QC decision hash mismatch")
    if decisions_payload.get("manifest_sha256") != completion["manifest_sha256"]:
        raise ValueError("QC decisions belong to a different manifest")
    if not bool(completion.get("summary", {}).get("complete")):
        raise ValueError("the source QC review is not complete")
    qc_to_image: dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qc_to_image[str(int(row["qc_index"]))] = int(row["image_id"])
    decisions = decisions_payload.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(qc_to_image):
        raise ValueError("QC decision coverage does not exactly match its manifest")
    by_image: dict[int, dict[str, Any]] = {}
    for qc_index, decision in decisions.items():
        image_id = qc_to_image[qc_index]
        if image_id in by_image:
            raise ValueError(f"duplicate reviewed image_id={image_id}")
        name = str(decision.get("decision"))
        if name not in {"KEEP", "FIX", "REMOVE"}:
            raise ValueError(f"unsupported QC decision {name!r}")
        bbox = decision.get("bbox_xyxy")
        if name == "FIX":
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"FIX image_id={image_id} lacks a box")
            _validate_bbox(bbox, image_id=image_id)
        elif bbox is not None:
            raise ValueError(f"{name} image_id={image_id} unexpectedly has a box")
        by_image[image_id] = dict(decision)
    return by_image


TASK4_DATASET = "Task4_Polyp_1to8"


def _candidate_identity_index(
    paths: _build_scenario_static_v1_Sequence[Path],
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    immutable = (
        "sample_id",
        "dataset_index",
        "episode_index",
        "frame_index",
        "image_id",
    )
    for path in paths:
        for row in _read_jsonl(path):
            if row.get("source_dataset") != TASK4_DATASET:
                continue
            key = (int(row["source_episode_index"]), int(row["source_frame_index"]))
            prior = result.get(key)
            if prior is not None:
                for field in immutable:
                    if prior[field] != row[field]:
                        raise ValueError(
                            f"Task4 identity disagreement for {key}, field={field}"
                        )
            result[key] = row
    return result


build_scenario_static_v1_SOURCE_FPS = 30
STATIC_FPS = 5
STATIC_STRIDE = build_scenario_static_v1_SOURCE_FPS // STATIC_FPS


def _validate_manifest_rows(
    rows: _build_scenario_static_v1_Sequence[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
    *,
    split: str,
) -> None:
    if not rows:
        raise ValueError(f"empty {split} manifest")
    unique_fields = {
        "sample_id": [str(row["sample_id"]) for row in rows],
        "image_id": [int(row["image_id"]) for row in rows],
        "dataset_index": [int(row["dataset_index"]) for row in rows],
    }
    for name, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name} in {split} manifest")
    physical = [
        (
            str(row["source_dataset"]),
            int(row["source_episode_index"]),
            int(row["source_frame_index"]),
        )
        for row in rows
    ]
    if len(physical) != len(set(physical)):
        raise ValueError(f"duplicate physical frames in {split} manifest")
    if any((str(row["split"]) != split for row in rows)):
        raise ValueError(f"wrong split label inside {split} manifest")
    if any((str(row["status"]) == "uncertain_ignore" for row in rows)):
        raise ValueError(f"uncertain frame leaked into {split} manifest")
    for row in rows:
        frame = int(row["source_frame_index"])
        if frame % STATIC_STRIDE:
            raise ValueError(f"non-5 Hz frame in {split}: {row['sample_id']}")


EXPECTED_TRAIN = {
    "task4_positive": 2390,
    "task4_confirmed_negative": 1106,
    "task1_confirmed_negative": 2191,
    "total": 5687,
}
EXPECTED_REVIEW_GAP = {
    "validation": {"S1": 324, "S2": 0, "S3": 622, "S4": 391, "S5": 373},
    "test": {"S1": 320, "S2": 0, "S3": 500, "S4": 365, "S5": 358},
}


def _build_scenario_static_v1_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_jsonl(
    path: Path,
    rows: _build_scenario_static_v1_Iterable[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
) -> None:
    _build_scenario_static_v1_atomic_text(
        path, "".join((_canonical_json(dict(row)) + "\n" for row in rows))
    )


def _write_review_set(
    *,
    review_root: Path,
    pending: _build_scenario_static_v1_Sequence[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
    annotation_root: Path,
    bbox_root: Path,
    provenance_index: _build_scenario_static_v1_Mapping[
        int, _build_scenario_static_v1_Mapping[str, Any]
    ],
    batch_size: int,
) -> dict[str, Any]:
    if review_root.exists():
        raise FileExistsError(f"review root already exists: {review_root}")
    review_root.mkdir(parents=True)
    ordered = sorted(
        pending,
        key=lambda row: (
            str(row["protocol_split"]),
            str(row["scenario_id"]),
            int(row["episode_index"]),
            int(row["frame_index"]),
        ),
    )
    split_indices: Counter[str] = Counter()
    split_totals = Counter((str(row["protocol_split"]) for row in ordered))
    review_rows: list[dict[str, Any]] = []
    for qc_index, row in enumerate(ordered, start=1):
        model_split = str(row["protocol_split"])
        split_indices[model_split] += 1
        split_index = split_indices[model_split]
        source_path = annotation_root / str(row["image_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        try:
            source_relative = (
                source_path.resolve().relative_to(bbox_root.resolve()).as_posix()
            )
        except ValueError:
            source_relative = source_path.relative_to(bbox_root).as_posix()
        diagnostics = (
            provenance_index.get(int(row["image_id"]), {}).get("diagnostics") or {}
        )
        center_jump_raw = diagnostics.get("normalized_center_jump")
        bbox = [float(value) for value in row["bbox_xyxy"]]
        review_rows.append(
            {
                "qc_index": qc_index,
                "model_split": model_split,
                "split_index": split_index,
                "batch": math.ceil(split_index / batch_size),
                "sample_id": row["sample_id"],
                "image_id": int(row["image_id"]),
                "source_episode_index": int(row["episode_index"]),
                "source_frame_index": int(row["frame_index"]),
                "source_image": source_relative,
                "overlay_image": "",
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "partial": bool(row.get("partial", False)),
                "occluded": bool(row.get("occluded", False)),
                "blurred": bool(row.get("blurred", False)),
                "interpolated_hint": bool(
                    diagnostics.get("interpolated_empty_source", False)
                ),
                "normalized_center_jump": float(center_jump_raw)
                if center_jump_raw is not None
                else "",
                "review_decision": "",
                "corrected_x1": "",
                "corrected_y1": "",
                "corrected_x2": "",
                "corrected_y2": "",
                "review_notes": "",
            }
        )
    if not review_rows:
        raise ValueError("expected a non-empty evaluation review gap")
    fieldnames = list(review_rows[0])
    csv_path = review_root / "review_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    _write_jsonl(review_root / "review_manifest.jsonl", review_rows)
    _build_scenario_static_v1_atomic_text(
        review_root / "README.md",
        "# scenario-static-v1 evaluation GT review\n\nThis set contains only the positive 5 Hz validation/test boxes that were not covered by the completed projected/block-11 model-used review. Source data is read-only.\n\n- KEEP: the current box is correct.\n- FIX: redraw the box.\n- REMOVE: discard this ambiguous/incorrect frame entirely; it will not become a negative.\n- Finish all items, then press `Mark entire review complete`.\n",
    )
    audit = {
        "schema_version": 1,
        "purpose": "complete exact-5 Hz validation/test positive GT coverage",
        "total_images": len(review_rows),
        "split_counts": dict(split_totals),
        "batch_size": batch_size,
        "source_images_modified": False,
        "remove_semantics": "exclude frame entirely",
        "manifest_sha256": _sha256(csv_path),
    }
    _build_scenario_static_v1_atomic_text(
        review_root / "audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def _scenario_for(
    episode: int,
    split: str,
    episode_sets: _build_scenario_static_v1_Mapping[
        str, _build_scenario_static_v1_Mapping[str, set[int]]
    ],
) -> str | None:
    matches = [
        scenario
        for scenario, episodes in episode_sets[split].items()
        if episode in episodes
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Task4 episode {episode} belongs to multiple {split} scenarios"
        )
    return matches[0] if matches else None


TASK1_DATASET = "ENDO_task1_Withdraw_400episode"


def _counts(
    rows: _build_scenario_static_v1_Sequence[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
) -> dict[str, int]:
    return {
        "task4_positive": sum(
            (
                row["source_dataset"] == TASK4_DATASET and row["status"] == "positive"
                for row in rows
            )
        ),
        "task4_confirmed_negative": sum(
            (
                row["source_dataset"] == TASK4_DATASET
                and row["status"] == "confirmed_negative"
                for row in rows
            )
        ),
        "task1_confirmed_negative": sum(
            (
                row["source_dataset"] == TASK1_DATASET
                and row["status"] == "confirmed_negative"
                for row in rows
            )
        ),
        "total": len(rows),
    }


def _candidate_from_approved(
    source: _build_scenario_static_v1_Mapping[str, Any],
    identity_index: _build_scenario_static_v1_Mapping[
        tuple[int, int], _build_scenario_static_v1_Mapping[str, Any]
    ],
    *,
    split: str,
    bbox: _build_scenario_static_v1_Sequence[float] | None,
) -> dict[str, Any]:
    key = (int(source["episode_index"]), int(source["frame_index"]))
    identity = identity_index.get(key)
    if identity is None:
        raise KeyError(f"no five-task endo_bbox_vla identity for Task4 frame {key}")
    result = dict(identity)
    result["split"] = split
    result["status"] = str(source["status"])
    result["bbox_xyxy"] = list(bbox) if bbox is not None else None
    for name in ("partial", "occluded", "blurred", "group_id", "width", "height"):
        if name in source:
            result[name] = source[name]
    return result


def _apply_positive_qc(
    record: _build_scenario_static_v1_Mapping[str, Any],
    approved_box: _build_scenario_static_v1_Sequence[float],
    decision: _build_scenario_static_v1_Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any] | None:
    name = str(decision["decision"])
    if name == "REMOVE":
        return None
    box = decision.get("bbox_xyxy") if name == "FIX" else approved_box
    result = dict(record)
    result["split"] = split
    result["status"] = "positive"
    result["bbox_xyxy"] = _validate_bbox(box, image_id=int(result["image_id"]))
    return result


def _make_coco(
    rows: _build_scenario_static_v1_Sequence[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
    approved_by_image: _build_scenario_static_v1_Mapping[
        int, _build_scenario_static_v1_Mapping[str, Any]
    ],
    *,
    split: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for row in rows:
        if row["source_dataset"] != TASK4_DATASET:
            continue
        image_id = int(row["image_id"])
        source = approved_by_image[image_id]
        images.append(
            {
                "id": image_id,
                "file_name": f"annotation/task4_polyp_v1/{source['image_path']}",
                "width": int(source["width"]),
                "height": int(source["height"]),
                "episode_index": int(source["episode_index"]),
                "frame_index": int(source["frame_index"]),
                "group_id": source.get("group_id"),
                "split": split,
            }
        )
        if row["status"] == "positive":
            x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"])
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    return {
        "info": {
            "description": f"scenario-static-v1 {split} Task4 view",
            "fps": STATIC_FPS,
            "review_state": "prefill_only" if split != "train" else "complete",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "polyp", "supercategory": "lesion"}],
        "images": images,
        "annotations": annotations,
    }


def _scenario_counts(
    rows: _build_scenario_static_v1_Sequence[
        _build_scenario_static_v1_Mapping[str, Any]
    ],
    *,
    split: str,
    episode_sets: _build_scenario_static_v1_Mapping[
        str, _build_scenario_static_v1_Mapping[str, set[int]]
    ],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {scenario: Counter() for scenario in SCENARIOS}
    for row in rows:
        if row["source_dataset"] != TASK4_DATASET:
            continue
        scenario = _scenario_for(int(row["source_episode_index"]), split, episode_sets)
        if scenario is None:
            raise ValueError(
                f"Task4 row is outside {split} episode protocol: {row['sample_id']}"
            )
        counts[scenario][str(row["status"])] += 1
    return {scenario: dict(counter) for scenario, counter in counts.items()}


EXPECTED_TASK1_EVAL = {"validation": 3380, "test": 4394}


def build(
    *,
    approved_root: Path,
    annotation_root: Path,
    legacy_train_manifest: Path,
    full_identity_manifest: Path,
    evaluation_identity_manifest: Path,
    task1_train_manifest: Path,
    prior_review_root: Path,
    output_root: Path,
    review_root: Path,
    bbox_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    episode_sets = _episode_sets()
    _validate_episode_protocol(episode_sets)
    approved_manifest = approved_root / "manifest.jsonl"
    approved_instances = approved_root / "instances.json"
    approved_by_image, approved_boxes, _ = _approved_indexes(
        approved_manifest, approved_instances
    )
    decisions = _load_completed_review(prior_review_root)
    identity_index = _candidate_identity_index(
        [legacy_train_manifest, full_identity_manifest, evaluation_identity_manifest]
    )
    legacy_rows = _read_jsonl(legacy_train_manifest)
    train_task4: list[dict[str, Any]] = []
    removed_train: list[dict[str, Any]] = []
    for row in legacy_rows:
        if row["source_dataset"] != TASK4_DATASET:
            continue
        episode = int(row["source_episode_index"])
        scenario = _scenario_for(episode, "train", episode_sets)
        if scenario is None:
            continue
        if int(row["source_frame_index"]) % STATIC_STRIDE:
            raise ValueError(
                f"legacy Task4 training row is not 5 Hz: {row['sample_id']}"
            )
        if row["status"] == "uncertain_ignore":
            removed_train.append({"reason": "uncertain_ignore", "row": row})
            continue
        candidate = dict(row)
        candidate["split"] = "train"
        if row["status"] == "positive":
            image_id = int(row["image_id"])
            decision = decisions.get(image_id)
            if decision is None:
                raise KeyError(
                    f"training positive image_id={image_id} lacks completed QC"
                )
            applied = _apply_positive_qc(
                candidate, approved_boxes[image_id], decision, split="train"
            )
            if applied is None:
                removed_train.append({"reason": "review_remove", "row": row})
                continue
            candidate = applied
        train_task4.append(candidate)
    task1_train = _read_jsonl(task1_train_manifest)
    for row in task1_train:
        if (
            row["source_dataset"] != TASK1_DATASET
            or row["status"] != "confirmed_negative"
        ):
            raise ValueError(
                "Task1 train manifest contains a non-negative/non-Task1 row"
            )
        row["split"] = "train"
    train = sorted(
        [*train_task4, *task1_train],
        key=lambda row: (
            str(row["source_dataset"]),
            int(row["source_episode_index"]),
            int(row["source_frame_index"]),
        ),
    )
    _validate_manifest_rows(train, split="train")
    if _counts(train) != EXPECTED_TRAIN:
        raise ValueError(
            f"scenario-static train count drift: {_counts(train)} != {EXPECTED_TRAIN}"
        )
    approved_rows = list(approved_by_image.values())
    provenance_path = (
        annotation_root / "candidates/temporal-dino-only-nontest-review-only-v1.jsonl"
    )
    provenance_index = {
        int(row["image_id"]): row for row in _read_jsonl(provenance_path)
    }
    task1_evaluation = _read_jsonl(evaluation_identity_manifest)
    eval_prefill: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    pending: list[dict[str, Any]] = []
    excluded_eval: list[dict[str, Any]] = []
    source_eval_counts: dict[str, Counter[str]] = {
        "validation": Counter(),
        "test": Counter(),
        "guard": Counter(),
    }
    guard_rows: list[dict[str, Any]] = []
    for source in approved_rows:
        episode = int(source["episode_index"])
        frame = int(source["frame_index"])
        if frame % STATIC_STRIDE:
            continue
        target_split = next(
            (
                split
                for split in ("validation", "test", "guard")
                if _scenario_for(episode, split, episode_sets) is not None
            ),
            None,
        )
        if target_split is None:
            continue
        scenario = _scenario_for(episode, target_split, episode_sets)
        assert scenario is not None
        status = str(source["status"])
        source_eval_counts[target_split][status] += 1
        if status == "uncertain_ignore":
            excluded_eval.append(
                {
                    "reason": "uncertain_ignore",
                    "split": target_split,
                    "scenario_id": scenario,
                    "source": source,
                }
            )
            continue
        bbox = approved_boxes.get(int(source["image_id"]))
        candidate = _candidate_from_approved(
            source,
            identity_index,
            split="negative_control" if target_split == "guard" else target_split,
            bbox=bbox,
        )
        if status == "confirmed_negative":
            if target_split == "guard":
                guard_rows.append(candidate)
            else:
                eval_prefill[target_split].append(candidate)
            continue
        if status != "positive":
            raise ValueError(f"unsupported approved Task4 status {status!r}")
        decision = decisions.get(int(source["image_id"]))
        if decision is None:
            if target_split == "guard":
                guard_rows.append(candidate)
                continue
            pending.append(
                {
                    **source,
                    "sample_id": candidate["sample_id"],
                    "bbox_xyxy": bbox,
                    "protocol_split": target_split,
                    "scenario_id": scenario,
                }
            )
            continue
        applied = _apply_positive_qc(
            candidate,
            approved_boxes[int(source["image_id"])],
            decision,
            split="negative_control" if target_split == "guard" else target_split,
        )
        if applied is None:
            excluded_eval.append(
                {
                    "reason": "review_remove",
                    "split": target_split,
                    "scenario_id": scenario,
                    "source": source,
                }
            )
        elif target_split == "guard":
            guard_rows.append(applied)
        else:
            eval_prefill[target_split].append(applied)
    for split, source_episodes in (
        ("validation", range(150, 170)),
        ("test", range(170, 200)),
    ):
        wanted = set(source_episodes)
        task1_rows = [
            dict(row)
            for row in task1_evaluation
            if row["source_dataset"] == TASK1_DATASET
            and int(row["source_episode_index"]) in wanted
            and (int(row["source_frame_index"]) % STATIC_STRIDE == 0)
        ]
        if len(task1_rows) != EXPECTED_TASK1_EVAL[split]:
            raise ValueError(
                f"Task1 {split} exact-5 Hz count drift: {len(task1_rows)} != {EXPECTED_TASK1_EVAL[split]}"
            )
        for row in task1_rows:
            row["split"] = split
        eval_prefill[split].extend(task1_rows)
    actual_gap: dict[str, dict[str, int]] = {
        split: {scenario: 0 for scenario in SCENARIOS}
        for split in ("validation", "test")
    }
    for row in pending:
        actual_gap[str(row["protocol_split"])][str(row["scenario_id"])] += 1
    if actual_gap != EXPECTED_REVIEW_GAP:
        raise ValueError(
            f"evaluation review-gap drift: {actual_gap} != {EXPECTED_REVIEW_GAP}"
        )
    for split in ("validation", "test"):
        eval_prefill[split].sort(
            key=lambda row: (
                str(row["source_dataset"]),
                int(row["source_episode_index"]),
                int(row["source_frame_index"]),
            )
        )
        _validate_manifest_rows(eval_prefill[split], split=split)
    guard_rows.sort(
        key=lambda row: (
            int(row["source_episode_index"]),
            int(row["source_frame_index"]),
        )
    )
    if guard_rows:
        _validate_manifest_rows(guard_rows, split="negative_control")
    output_root.mkdir(parents=True)
    frame_manifest = output_root / "frame_manifest.jsonl"
    validation_prefill = output_root / "validation_prefill_manifest.jsonl"
    test_prefill = output_root / "test_prefill_manifest.jsonl"
    guard_manifest = output_root / "guard_manifest.jsonl"
    _write_jsonl(frame_manifest, train)
    _write_jsonl(validation_prefill, eval_prefill["validation"])
    _write_jsonl(test_prefill, eval_prefill["test"])
    _write_jsonl(guard_manifest, guard_rows)
    _write_jsonl(
        output_root / "excluded_frames.jsonl", [*removed_train, *excluded_eval]
    )
    _write_jsonl(output_root / "pending_eval_positive_review.jsonl", pending)
    for split, rows in (
        ("train", train),
        ("validation", eval_prefill["validation"]),
        ("test", eval_prefill["test"]),
    ):
        coco = _make_coco(rows, approved_by_image, split=split)
        _build_scenario_static_v1_atomic_text(
            output_root
            / f"{split}{('_prefill' if split != 'train' else '')}.coco.json",
            json.dumps(coco, indent=2, sort_keys=True) + "\n",
        )
    review_audit = _write_review_set(
        review_root=review_root,
        pending=pending,
        annotation_root=annotation_root,
        bbox_root=bbox_root,
        provenance_index=provenance_index,
        batch_size=batch_size,
    )
    scenario_definition = {
        scenario: {
            "description": definition["description"],
            "acquisition_groups": definition["acquisition_groups"],
            **{
                split: list(definition[split])
                for split in ("train", "validation", "test", "guard")
            },
        }
        for scenario, definition in SCENARIOS.items()
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "scenario-static-v1",
        "state": "evaluation_review_pending",
        "source_fps": build_scenario_static_v1_SOURCE_FPS,
        "static_fps": STATIC_FPS,
        "static_stride": STATIC_STRIDE,
        "scenario_definition": scenario_definition,
        "task1_episodes": {
            "train": list(range(0, 150)),
            "validation": list(range(150, 170)),
            "test": list(range(170, 200)),
        },
        "sampling": {
            "training_epoch_positive_fraction": 0.5,
            "positive_scenario_fraction": {scenario: 0.2 for scenario in SCENARIOS},
            "episode_balanced_within_scenario": True,
            "negative_task4_task1_fraction": {"Task4": 0.5, "Task1": 0.5},
        },
        "exclusion_policy": {
            "REMOVE": "exclude entire frame",
            "uncertain_ignore": "exclude entire frame",
            "excluded_frames_are_not_negatives": True,
        },
        "train_counts": _counts(train),
        "train_task4_scenario_counts": _scenario_counts(
            train, split="train", episode_sets=episode_sets
        ),
        "validation_prefill_counts": _counts(eval_prefill["validation"]),
        "validation_prefill_task4_scenario_counts": _scenario_counts(
            eval_prefill["validation"], split="validation", episode_sets=episode_sets
        ),
        "test_prefill_counts": _counts(eval_prefill["test"]),
        "test_prefill_task4_scenario_counts": _scenario_counts(
            eval_prefill["test"], split="test", episode_sets=episode_sets
        ),
        "source_eval_status_counts": {
            split: dict(counts) for split, counts in source_eval_counts.items()
        },
        "pending_review_counts": actual_gap,
        "guard_count": len(guard_rows),
        "review_root": str(review_root),
    }
    protocol_path = output_root / "protocol.json"
    _build_scenario_static_v1_atomic_text(
        protocol_path, json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    inputs = [
        approved_manifest,
        approved_instances,
        legacy_train_manifest,
        full_identity_manifest,
        evaluation_identity_manifest,
        task1_train_manifest,
        prior_review_root / "review_manifest.csv",
        prior_review_root / "review_decisions.json",
        prior_review_root / "REVIEW_COMPLETE.json",
    ]
    audit = {
        "schema_version": 1,
        "kind": "candidate_c_scenario_static_v1_prefill",
        "state": "evaluation_review_pending",
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "output_sha256": {
            str(path): _sha256(path)
            for path in (
                frame_manifest,
                validation_prefill,
                test_prefill,
                guard_manifest,
                protocol_path,
            )
        },
        "review_audit": review_audit,
        "leakage_checks": {
            "task4_episode_disjoint": True,
            "task1_episode_disjoint": True,
            "manifest_unique_physical_frames": True,
            "all_included_frames_exact_5hz": True,
            "remove_and_uncertain_absent": True,
        },
    }
    _build_scenario_static_v1_atomic_text(
        output_root / "provenance_audit.json",
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    return {"protocol": protocol, "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-root",
        type=Path,
        default=Path(
            "bbox_vla/annotation/task4_polyp_v1/review-qa/task4-polyp-bbox-v1-approved"
        ),
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("bbox_vla/annotation/task4_polyp_v1"),
    )
    parser.add_argument(
        "--legacy-train-manifest",
        type=Path,
        default=Path("bbox_vla/annotation/endo_bbox_vla/frame_manifest.jsonl"),
    )
    parser.add_argument(
        "--full-identity-manifest",
        type=Path,
        default=Path(
            "bbox_vla/annotation/endo_bbox_vla/full_positive/frame_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--evaluation-identity-manifest",
        type=Path,
        default=Path(
            "bbox_vla/annotation/endo_bbox_vla/evaluation_frame_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--task1-train-manifest",
        type=Path,
        default=Path("bbox_vla/annotation/task1_static_5hz_v1/manifest.jsonl"),
    )
    parser.add_argument(
        "--prior-review-root",
        type=Path,
        default=Path("annotations/model_used_review"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("bbox_vla/annotation/endo_bbox_vla/scenario_static_v1"),
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("bbox_vla/annotation/manual_qc/scenario_static_v1_eval_gap"),
    )
    parser.add_argument("--bbox-root", type=Path, default=Path("bbox_vla"))
    parser.add_argument("--batch-size", type=int, default=200)
    return parser


def build_scenario_static_v1_main(
    argv: _build_scenario_static_v1_Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    result = build(
        approved_root=args.approved_root.resolve(),
        annotation_root=args.annotation_root.resolve(),
        legacy_train_manifest=args.legacy_train_manifest.resolve(),
        full_identity_manifest=args.full_identity_manifest.resolve(),
        evaluation_identity_manifest=args.evaluation_identity_manifest.resolve(),
        task1_train_manifest=args.task1_train_manifest.resolve(),
        prior_review_root=args.prior_review_root.resolve(),
        output_root=args.output_root.resolve(),
        review_root=args.review_root.resolve(),
        bbox_root=args.bbox_root.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _physical_keys(
    rows: _finalize_scenario_static_v1_Sequence[
        _finalize_scenario_static_v1_Mapping[str, Any]
    ],
) -> set[tuple[str, int, int]]:
    return {
        (
            str(row["source_dataset"]),
            int(row["source_episode_index"]),
            int(row["source_frame_index"]),
        )
        for row in rows
    }


def finalize(
    *,
    draft_root: Path,
    review_root: Path,
    approved_root: Path,
    identity_manifests: _finalize_scenario_static_v1_Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"final output root already exists: {output_root}")
    decisions = _load_completed_review(review_root)
    pending = _read_jsonl(draft_root / "pending_eval_positive_review.jsonl")
    pending_ids = {int(row["image_id"]) for row in pending}
    if len(pending_ids) != len(pending) or set(decisions) != pending_ids:
        raise ValueError(
            "completed review does not exactly cover the pending evaluation positives"
        )
    approved_by_image, approved_boxes, _ = _approved_indexes(
        approved_root / "manifest.jsonl", approved_root / "instances.json"
    )
    identity_index = _candidate_identity_index(identity_manifests)
    episode_sets = _episode_sets()
    final_rows: dict[str, list[dict[str, Any]]] = {
        "validation": _read_jsonl(draft_root / "validation_prefill_manifest.jsonl"),
        "test": _read_jsonl(draft_root / "test_prefill_manifest.jsonl"),
    }
    new_decision_counts: dict[str, Counter[str]] = {
        "validation": Counter(),
        "test": Counter(),
    }
    newly_removed: list[dict[str, Any]] = []
    for source in pending:
        split = str(source["protocol_split"])
        image_id = int(source["image_id"])
        decision = decisions[image_id]
        new_decision_counts[split][str(decision["decision"])] += 1
        approved = approved_by_image[image_id]
        candidate = _candidate_from_approved(
            approved, identity_index, split=split, bbox=approved_boxes[image_id]
        )
        applied = _apply_positive_qc(
            candidate, approved_boxes[image_id], decision, split=split
        )
        if applied is None:
            newly_removed.append(
                {
                    "reason": "review_remove",
                    "split": split,
                    "scenario_id": source["scenario_id"],
                    "source": approved,
                }
            )
        else:
            final_rows[split].append(applied)
    for split, rows in final_rows.items():
        rows.sort(
            key=lambda row: (
                str(row["source_dataset"]),
                int(row["source_episode_index"]),
                int(row["source_frame_index"]),
            )
        )
        _validate_manifest_rows(rows, split=split)
        if any((int(row["source_frame_index"]) % STATIC_STRIDE for row in rows)):
            raise ValueError(f"non-5 Hz frame entered final {split}")
    train = _read_jsonl(draft_root / "frame_manifest.jsonl")
    train_keys = _physical_keys(train)
    validation_keys = _physical_keys(final_rows["validation"])
    test_keys = _physical_keys(final_rows["test"])
    if (
        train_keys & validation_keys
        or train_keys & test_keys
        or validation_keys & test_keys
    ):
        raise ValueError("physical frame leakage across final static splits")
    task4_episode_sets = {
        split: {
            int(row["source_episode_index"])
            for row in rows
            if row["source_dataset"] == TASK4_DATASET
        }
        for split, rows in {
            "train": train,
            "validation": final_rows["validation"],
            "test": final_rows["test"],
        }.items()
    }
    task1_episode_sets = {
        split: {
            int(row["source_episode_index"])
            for row in rows
            if row["source_dataset"] == TASK1_DATASET
        }
        for split, rows in {
            "train": train,
            "validation": final_rows["validation"],
            "test": final_rows["test"],
        }.items()
    }
    for sets, name in ((task4_episode_sets, "Task4"), (task1_episode_sets, "Task1")):
        if (
            sets["train"] & sets["validation"]
            or sets["train"] & sets["test"]
            or sets["validation"] & sets["test"]
        ):
            raise ValueError(f"{name} episode leakage across final splits")
    output_root.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for split in ("validation", "test"):
        path = output_root / f"{split}_manifest.jsonl"
        _write_jsonl(path, final_rows[split])
        paths[split] = path
        coco = _make_coco(final_rows[split], approved_by_image, split=split)
        coco["info"]["review_state"] = "complete"
        _build_scenario_static_v1_atomic_text(
            output_root / f"{split}.coco.json",
            json.dumps(coco, indent=2, sort_keys=True) + "\n",
        )
    _write_jsonl(output_root / "newly_removed_frames.jsonl", newly_removed)
    parent_protocol = _read_json(draft_root / "protocol.json")
    protocol = {
        **parent_protocol,
        "state": "complete",
        "parent_draft_root": str(draft_root),
        "completed_review_root": str(review_root),
        "new_review_decision_counts": {
            split: dict(counter) for split, counter in new_decision_counts.items()
        },
        "newly_removed_count": len(newly_removed),
        "validation_counts": _counts(final_rows["validation"]),
        "validation_task4_scenario_counts": _scenario_counts(
            final_rows["validation"], split="validation", episode_sets=episode_sets
        ),
        "test_counts": _counts(final_rows["test"]),
        "test_task4_scenario_counts": _scenario_counts(
            final_rows["test"], split="test", episode_sets=episode_sets
        ),
    }
    protocol_path = output_root / "protocol.json"
    _build_scenario_static_v1_atomic_text(
        protocol_path, json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "schema_version": 1,
        "kind": "candidate_c_scenario_static_v1_final",
        "state": "complete",
        "parent_protocol_sha256": _sha256(draft_root / "protocol.json"),
        "parent_provenance_sha256": _sha256(draft_root / "provenance_audit.json"),
        "review_manifest_sha256": _sha256(review_root / "review_manifest.csv"),
        "review_decisions_sha256": _sha256(review_root / "review_decisions.json"),
        "review_completion_sha256": _sha256(review_root / "REVIEW_COMPLETE.json"),
        "output_sha256": {
            path.name: _sha256(path) for path in (*paths.values(), protocol_path)
        },
        "leakage_checks": {
            "physical_frame_disjoint": True,
            "task4_episode_disjoint": True,
            "task1_episode_disjoint": True,
            "all_frames_exact_5hz": True,
            "remove_and_uncertain_absent": True,
        },
    }
    _build_scenario_static_v1_atomic_text(
        output_root / "provenance_audit.json",
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    return {"protocol": protocol, "audit": audit}


def finalize_scenario_static_v1_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft-root",
        type=Path,
        default=Path("bbox_vla/annotation/endo_bbox_vla/scenario_static_v1"),
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("bbox_vla/annotation/manual_qc/scenario_static_v1_eval_gap"),
    )
    parser.add_argument(
        "--approved-root",
        type=Path,
        default=Path(
            "bbox_vla/annotation/task4_polyp_v1/review-qa/task4-polyp-bbox-v1-approved"
        ),
    )
    parser.add_argument(
        "--identity-manifests",
        nargs="+",
        type=Path,
        default=[
            Path("bbox_vla/annotation/endo_bbox_vla/frame_manifest.jsonl"),
            Path(
                "bbox_vla/annotation/endo_bbox_vla/full_positive/frame_manifest.jsonl"
            ),
            Path("bbox_vla/annotation/endo_bbox_vla/evaluation_frame_manifest.jsonl"),
        ],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("bbox_vla/annotation/endo_bbox_vla/scenario_static_v1_final"),
    )
    return parser


def finalize_scenario_static_v1_main(
    argv: _finalize_scenario_static_v1_Sequence[str] | None = None,
) -> int:
    args = finalize_scenario_static_v1_build_parser().parse_args(argv)
    result = finalize(
        draft_root=args.draft_root.resolve(),
        review_root=args.review_root.resolve(),
        approved_root=args.approved_root.resolve(),
        identity_manifests=[path.resolve() for path in args.identity_manifests],
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def apply_label_proposals(
    records: Sequence[FrameRecord],
    proposals: Sequence[LabelProposal],
    *,
    review_version: str,
    require_all: bool = False,
) -> tuple[tuple[FrameRecord, ...], dict[str, Any]]:
    if not review_version.strip():
        raise ValueError("review_version must be non-empty")
    proposal_by_id = {proposal.image_id: proposal for proposal in proposals}
    record_ids = {record.image_id for record in records}
    unknown = set(proposal_by_id) - record_ids
    if unknown:
        raise ValueError(
            f"proposals reference image ids absent from manifest: {sorted(unknown)[:20]}"
        )
    if require_all and set(proposal_by_id) != record_ids:
        missing = record_ids - set(proposal_by_id)
        raise ValueError(
            f"require_all=True but {len(missing)} image(s) have no proposal"
        )
    updated: list[FrameRecord] = []
    bbox_by_id: dict[int, tuple[float, float, float, float]] = {}
    for record in records:
        proposal = proposal_by_id.get(record.image_id)
        if proposal is None:
            updated.append(record)
            continue
        if proposal.bbox is not None:
            x, y, width, height = proposal.bbox
            if x + width > record.width + 1e-06 or y + height > record.height + 1e-06:
                raise ValueError(
                    f"image {record.image_id}: bbox {proposal.bbox} exceeds {record.width}x{record.height}"
                )
            bbox_by_id[record.image_id] = proposal.bbox
        updated.append(
            replace(
                record,
                status=proposal.status,
                partial=proposal.partial,
                occluded=proposal.occluded,
                blurred=proposal.blurred,
                review_version=proposal.review_version or review_version,
            )
        )
    coco = build_empty_coco(updated, annotation_revision=review_version)
    coco["annotations"] = [
        {
            "id": record.image_id,
            "image_id": record.image_id,
            "category_id": 1,
            "bbox": list(bbox_by_id[record.image_id]),
            "area": bbox_by_id[record.image_id][2] * bbox_by_id[record.image_id][3],
            "iscrowd": 0,
        }
        for record in updated
        if record.status == AnnotationStatus.POSITIVE
    ]
    validate_coco_manifest(coco, updated)
    return (tuple(updated), coco)


def cli(argv=None):
    from endo_bbox_vla.utils import run_cli

    return run_cli(
        {
            "frames": main,
            "proposals": grounding_dino_candidates_main,
            "build": build_scenario_static_v1_main,
            "finalize": finalize_scenario_static_v1_main,
        },
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
