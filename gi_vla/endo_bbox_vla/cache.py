"""Frozen feature-cache schemas, identity checks and tensor storage."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from pathlib import Path as _ProjectPath
from typing import Any, Literal

import torch
from torch import Tensor

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


CACHE_SCHEMA_VERSION = 1

CACHE_KIND = "groot_n17_multilevel_static_cache"


@dataclass(frozen=True, slots=True)
class CacheRow:
    sample_id: str
    source_dataset: str
    source_episode_index: int
    source_frame_index: int
    dataset_index: int
    frame_id: int
    timestamp: float
    prompt_mode: str
    prompt_id: str
    selected_prompt_sha256: str
    detector_generation: int
    context_generation: int
    context_sha256: str
    tensor_path: str
    tensor_sha256: str

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.sample_id, self.prompt_id, self.context_generation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": CACHE_KIND,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheRow:
        payload = dict(value)
        if payload.pop("schema_version", None) != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported GR00T cache row schema")
        if payload.pop("kind", None) != CACHE_KIND:
            raise ValueError("unsupported GR00T cache row kind")
        return cls(**payload)


def _read_cache_manifest(path: Path) -> tuple[CacheRow, ...]:
    if not path.is_file():
        return ()
    rows = tuple(
        (
            CacheRow.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    identities = [row.identity for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("GR00T cache manifest contains duplicate identities")
    return rows


def _validated_tensor_path(root: Path, row: CacheRow) -> Path:
    from endo_bbox_vla.utils import sha256_file

    relative = Path(row.tensor_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"unsafe GR00T cache tensor path: {row.tensor_path}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"GR00T cache tensor escapes its root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"GR00T cache tensor is missing: {resolved}")
    if sha256_file(resolved) != row.tensor_sha256:
        raise ValueError(f"GR00T cache tensor hash mismatch: {resolved}")
    return resolved


_CacheKind = Literal["groot_temporal_e03", "groot_temporal_dit_conditioning"]

groot_temporal_cache_CACHE_SCHEMA_VERSION = 1


def _strict_non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    return value


_HEX = frozenset("0123456789abcdef")


def _check_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in _HEX for character in value))
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tensor_value_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    digest = hashlib.sha256(_canonical_bytes(header))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    value_sha256: str

    def __post_init__(self) -> None:
        _strict_non_empty(self.name, "tensor metadata name")
        if not self.shape or any(
            (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.shape
            )
        ):
            raise ValueError("tensor metadata shape must contain positive integers")
        _strict_non_empty(self.dtype, "tensor metadata dtype")
        _check_sha256(self.value_sha256, "tensor value_sha256")

    @classmethod
    def from_tensor(cls, name: str, tensor: Tensor) -> TensorMetadata:
        return cls(
            name=name,
            shape=tuple((int(value) for value in tensor.shape)),
            dtype=str(tensor.dtype),
            value_sha256=_tensor_value_sha256(tensor),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TensorMetadata:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("tensor metadata fields do not match the cache schema")
        return cls(
            name=payload["name"],
            shape=tuple(payload["shape"]),
            dtype=payload["dtype"],
            value_sha256=payload["value_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "value_sha256": self.value_sha256,
        }


VISUAL_CACHE_KIND = "groot_temporal_e03"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any((part in {"", ".", ".."} for part in path.parts))
    ):
        raise ValueError("cache tensor_path must be a safe relative POSIX path")
    return path.as_posix()


CONDITIONING_CACHE_KIND = "groot_temporal_dit_conditioning"


def _strict_generation(value: object, name: str = "context_generation") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class GrootTemporalCacheRecord:
    kind: _CacheKind
    sample_id: str
    prompt_id: str | None
    context_generation: int | None
    selected_prompt_sha256: str | None
    tensor_path: str
    tensor_file_sha256: str
    content_sha256: str
    tensors: tuple[TensorMetadata, ...]
    model_revision: str
    source_hook: str
    cache_schema_sha256: str
    schema_version: int = groot_temporal_cache_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _strict_non_empty(self.sample_id, "sample_id")
        if self.kind not in {VISUAL_CACHE_KIND, CONDITIONING_CACHE_KIND}:
            raise ValueError(f"unsupported GR00T temporal cache kind {self.kind!r}")
        if self.kind == VISUAL_CACHE_KIND:
            if any(
                (
                    value is not None
                    for value in (
                        self.prompt_id,
                        self.context_generation,
                        self.selected_prompt_sha256,
                    )
                )
            ):
                raise ValueError("visual cache identity is sample_id only")
        else:
            _strict_non_empty(self.prompt_id, "prompt_id")
            _strict_generation(self.context_generation)
            _check_sha256(self.selected_prompt_sha256, "selected_prompt_sha256")
        _safe_relative_path(self.tensor_path)
        for name, value in (
            ("tensor_file_sha256", self.tensor_file_sha256),
            ("content_sha256", self.content_sha256),
            ("cache_schema_sha256", self.cache_schema_sha256),
        ):
            _check_sha256(value, name)
        _strict_non_empty(self.model_revision, "model_revision")
        _strict_non_empty(self.source_hook, "source_hook")
        if not self.tensors or len({item.name for item in self.tensors}) != len(
            self.tensors
        ):
            raise ValueError("cache record tensors must have distinct names")
        if self.schema_version != groot_temporal_cache_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported GR00T temporal cache row schema")

    @property
    def logical_key(self) -> str | tuple[str, str, int]:
        if self.kind == VISUAL_CACHE_KIND:
            return self.sample_id
        assert self.prompt_id is not None and self.context_generation is not None
        return (self.sample_id, self.prompt_id, self.context_generation)

    @property
    def semantic_key(self) -> tuple[str, str] | None:
        if self.kind == VISUAL_CACHE_KIND:
            return None
        assert self.prompt_id is not None
        return (self.sample_id, self.prompt_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GrootTemporalCacheRecord:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("GR00T temporal cache row fields do not match the schema")
        converted = dict(payload)
        tensor_rows = converted["tensors"]
        if not isinstance(tensor_rows, list):
            raise TypeError("cache row tensors must be a list")
        converted["tensors"] = tuple(
            (TensorMetadata.from_dict(row) for row in tensor_rows)
        )
        return cls(**converted)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tensors"] = [tensor.to_dict() for tensor in self.tensors]
        return payload


@dataclass(frozen=True, slots=True)
class LoadedVisualFeatures:
    tokens: Tensor
    padding_mask: Tensor
    record: GrootTemporalCacheRecord


@dataclass(frozen=True, slots=True)
class LoadedConditioningFeatures:
    tokens: Tensor
    padding_mask: Tensor
    sequence_positions: Tensor
    record: GrootTemporalCacheRecord


GROOT_N17_FIVE_TASK_REVISION = "efbfbb0b4d13bf482734ead3d77d1ad2ceceac1b"

VISUAL_SOURCE_HOOK = "qwen3_vl_visual_block_11_pre_merger"

CONDITIONING_SOURCE_HOOK = "action_head_vlln_then_vl_self_attention_output"

_FLOAT_DTYPES = frozenset({"torch.bfloat16", "torch.float16", "torch.float32"})


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class GrootTemporalCacheSchema:
    """Pinned visual/VL feature contract; serialization retains recorded keys."""

    model_revision: str = GROOT_N17_FIVE_TASK_REVISION
    visual_shape: tuple[int, int] = (256, 1024)
    conditioning_shape: tuple[int, int] = (64, 2048)
    float_dtype: str = "torch.bfloat16"
    visual_source_hook: str = VISUAL_SOURCE_HOOK
    conditioning_source_hook: str = CONDITIONING_SOURCE_HOOK
    schema_version: int = groot_temporal_cache_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _strict_non_empty(self.model_revision, "model_revision")
        _strict_non_empty(self.visual_source_hook, "e03_source_hook")
        _strict_non_empty(self.conditioning_source_hook, "conditioning_source_hook")
        if (
            len(self.visual_shape) != 2
            or len(self.conditioning_shape) != 2
            or any(
                (value <= 0 for value in (*self.visual_shape, *self.conditioning_shape))
            )
        ):
            raise ValueError(
                "visual and conditioning shapes must each contain two positive values"
            )
        if self.float_dtype not in _FLOAT_DTYPES:
            raise ValueError(f"unsupported cache float dtype {self.float_dtype!r}")
        if self.schema_version != groot_temporal_cache_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported GR00T temporal cache schema")

    def to_dict(self) -> dict[str, Any]:
        # These keys are part of the hash-bound original cache schema.
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "e03_shape": list(self.visual_shape),
            "conditioning_shape": list(self.conditioning_shape),
            "float_dtype": self.float_dtype,
            "e03_source_hook": self.visual_source_hook,
            "conditioning_source_hook": self.conditioning_source_hook,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GrootTemporalCacheSchema:
        """Read the original serialized schema using descriptive Python fields."""
        values = dict(payload)
        for recorded, current in (
            ("e03_shape", "visual_shape"),
            ("e03_source_hook", "visual_source_hook"),
        ):
            if recorded in values:
                if current in values:
                    raise ValueError(f"duplicate cache schema field: {current}")
                values[current] = values.pop(recorded)
        for key in ("visual_shape", "conditioning_shape"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with suppress(OSError):
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _full_manifest_payload(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid GR00T temporal full manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("GR00T temporal full manifest root must be a mapping")
    return payload


@dataclass(frozen=True, slots=True)
class GrootTemporalPromptAssignment:
    """One authoritative frame assignment copied from the full manifest."""

    clip_id: str
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
    prompt_id: str
    selected_prompt: str
    context_generation: int

    def __post_init__(self) -> None:
        _strict_non_empty(self.clip_id, "clip_id")
        _strict_generation(self.frame_offset, "frame_offset")
        _strict_non_empty(self.sample_id, "sample_id")
        _strict_non_empty(self.source_dataset, "source_dataset")
        for name in (
            "source_episode_index",
            "source_frame_index",
            "dataset_index",
            "frame_id",
            "detector_generation",
        ):
            _strict_generation(getattr(self, name), name)
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or (not math.isfinite(self.timestamp))
            or (self.timestamp < 0)
        ):
            raise ValueError("timestamp must be finite and non-negative")
        _strict_non_empty(self.status, "status")
        if self.prompt_mode not in {"actual", "counterfactual", "neutral"}:
            raise ValueError(f"unsupported prompt_mode {self.prompt_mode!r}")
        _strict_non_empty(self.actual_prompt, "actual_prompt")
        _strict_non_empty(self.prompt_id, "prompt_id")
        _strict_non_empty(self.selected_prompt, "selected_prompt")
        _strict_generation(self.context_generation)

    @property
    def conditioning_key(self) -> tuple[str, str, int]:
        return (self.sample_id, self.prompt_id, self.context_generation)

    @property
    def selected_prompt_sha256(self) -> str:
        return hashlib.sha256(self.selected_prompt.encode("utf-8")).hexdigest()


def read_full_manifest_prompt_assignments(
    source: str | Path | Mapping[str, Any],
) -> tuple[GrootTemporalPromptAssignment, ...]:
    """Read prompt assignments verbatim from a serialized smoke/full plan.

    No assignment seed, prompt fractions, task prompt mapping, or planner is
    accepted here.  ``conditioning_key`` is redundant by design and is checked
    against the serialized frame fields to catch a partially edited plan.
    """
    payload = _full_manifest_payload(source)
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError(
            "GR00T temporal full manifest must contain a non-empty clips list"
        )
    assignments: list[GrootTemporalPromptAssignment] = []
    logical_prompts: dict[tuple[str, str, int], tuple[str, str]] = {}
    for clip_index, clip_row in enumerate(clips):
        if not isinstance(clip_row, Mapping):
            raise TypeError(f"full manifest clip {clip_index} must be a mapping")
        clip_metadata = clip_row.get("clip")
        if not isinstance(clip_metadata, Mapping):
            raise TypeError(f"full manifest clip {clip_index} is missing clip metadata")
        clip_id = _strict_non_empty(clip_metadata.get("clip_id"), "clip_id")
        frames = clip_row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(
                f"full manifest clip {clip_id!r} must contain frame assignments"
            )
        offsets: set[int] = set()
        for row_index, frame_row in enumerate(frames):
            if not isinstance(frame_row, Mapping):
                raise TypeError(
                    f"full manifest frame {clip_id}:{row_index} must be a mapping"
                )
            assignment = GrootTemporalPromptAssignment(
                clip_id=clip_id,
                frame_offset=frame_row.get("frame_offset"),
                sample_id=frame_row.get("sample_id"),
                source_dataset=frame_row.get("source_dataset"),
                source_episode_index=frame_row.get("source_episode_index"),
                source_frame_index=frame_row.get("source_frame_index"),
                dataset_index=frame_row.get("dataset_index"),
                frame_id=frame_row.get("frame_id"),
                timestamp=frame_row.get("timestamp"),
                detector_generation=frame_row.get("detector_generation"),
                status=frame_row.get("status"),
                prompt_mode=frame_row.get("prompt_mode"),
                actual_prompt=frame_row.get("actual_prompt"),
                prompt_id=frame_row.get("prompt_id"),
                selected_prompt=frame_row.get("selected_prompt"),
                context_generation=frame_row.get("context_generation"),
            )
            if assignment.frame_offset in offsets:
                raise ValueError(f"full manifest clip {clip_id!r} repeats frame_offset")
            offsets.add(assignment.frame_offset)
            if frame_row.get("local_cache_key") != assignment.sample_id:
                raise ValueError("serialized local_cache_key must equal sample_id")
            serialized_key = frame_row.get("conditioning_key")
            if not isinstance(serialized_key, Mapping):
                raise TypeError(
                    "each full-manifest frame must serialize conditioning_key"
                )
            expected_key = {
                "sample_id": assignment.sample_id,
                "prompt_id": assignment.prompt_id,
                "context_generation": assignment.context_generation,
            }
            if dict(serialized_key) != expected_key:
                raise ValueError(
                    "serialized conditioning_key disagrees with its frame assignment"
                )
            serialized_prompt_sha = frame_row.get("selected_prompt_sha256")
            if (
                serialized_prompt_sha is not None
                and _check_sha256(serialized_prompt_sha, "selected_prompt_sha256")
                != assignment.selected_prompt_sha256
            ):
                raise ValueError(
                    "selected_prompt_sha256 disagrees with selected_prompt"
                )
            prompt_identity = (
                assignment.prompt_mode,
                assignment.selected_prompt_sha256,
            )
            previous = logical_prompts.setdefault(
                assignment.conditioning_key, prompt_identity
            )
            if previous != prompt_identity:
                raise ValueError(
                    "one conditioning key has conflicting full-manifest prompts"
                )
            assignments.append(assignment)
    return tuple(assignments)


@dataclass(frozen=True, slots=True)
class LoadedAssignedTemporalFeatures:
    assignment: GrootTemporalPromptAssignment
    visual: LoadedVisualFeatures
    conditioning: LoadedConditioningFeatures


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, tensors: Mapping[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(tensors), temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_tensor(
    value: Tensor, *, name: str, shape: tuple[int, ...], dtype: str
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().cpu().contiguous()
    if tuple(result.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(result.shape)}")
    if str(result.dtype) != dtype:
        raise TypeError(f"{name} must use {dtype}, got {result.dtype}")
    if result.is_floating_point() and (not bool(torch.isfinite(result).all())):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


class GrootTemporalCache:
    """Atomic writer/resumable fail-closed loader for split GR00T features."""

    def __init__(
        self,
        root: str | Path,
        *,
        schema: GrootTemporalCacheSchema = GrootTemporalCacheSchema(),
    ) -> None:
        self.root = Path(root)
        self.schema = schema
        self.root.mkdir(parents=True, exist_ok=True)
        self.visual_manifest_path = self.root / "e03_manifest.jsonl"
        self.conditioning_manifest_path = self.root / "conditioning_manifest.jsonl"
        self._visual = self._read_manifest(self.visual_manifest_path, VISUAL_CACHE_KIND)
        self._conditioning = self._read_manifest(
            self.conditioning_manifest_path, CONDITIONING_CACHE_KIND
        )
        self._validate_conditioning_aliases(self._conditioning.values())
        self._conditioning_by_semantic: dict[
            tuple[str, str], GrootTemporalCacheRecord
        ] = {}
        for row in self._conditioning.values():
            semantic_key = row.semantic_key
            assert semantic_key is not None
            self._conditioning_by_semantic.setdefault(semantic_key, row)
        self._dirty_manifest_kinds: set[_CacheKind] = set()

    def _expected_names(self, kind: _CacheKind) -> tuple[str, ...]:
        if kind == VISUAL_CACHE_KIND:
            return ("mid_tokens", "mid_padding_mask")
        return (
            "dit_conditioning_tokens",
            "dit_conditioning_padding_mask",
            "dit_sequence_positions",
        )

    def _expected_metadata(
        self, kind: _CacheKind
    ) -> dict[str, tuple[tuple[int, ...], str]]:
        if kind == VISUAL_CACHE_KIND:
            count = self.schema.visual_shape[0]
            return {
                "mid_tokens": (self.schema.visual_shape, self.schema.float_dtype),
                "mid_padding_mask": ((count,), "torch.bool"),
            }
        count = self.schema.conditioning_shape[0]
        return {
            "dit_conditioning_tokens": (
                self.schema.conditioning_shape,
                self.schema.float_dtype,
            ),
            "dit_conditioning_padding_mask": ((count,), "torch.bool"),
            "dit_sequence_positions": ((count,), "torch.int64"),
        }

    def _source_hook(self, kind: _CacheKind) -> str:
        return (
            self.schema.visual_source_hook
            if kind == VISUAL_CACHE_KIND
            else self.schema.conditioning_source_hook
        )

    @staticmethod
    def _validate_payload_semantics(
        kind: _CacheKind, tensors: Mapping[str, Tensor]
    ) -> None:
        if kind == VISUAL_CACHE_KIND:
            if bool(tensors["mid_padding_mask"].all()):
                raise ValueError(
                    "visual cache row must contain at least one valid token"
                )
            return
        padding_mask = tensors["dit_conditioning_padding_mask"]
        positions = tensors["dit_sequence_positions"]
        if bool(padding_mask.all()):
            raise ValueError(
                "conditioning cache row must contain at least one valid token"
            )
        valid_positions = positions[~padding_mask]
        if bool((valid_positions < 0).any()) or (
            valid_positions.numel() > 1
            and bool((valid_positions[1:] <= valid_positions[:-1]).any())
        ):
            raise ValueError(
                "valid conditioning sequence positions must strictly increase"
            )

    def _validate_record_contract(self, record: GrootTemporalCacheRecord) -> None:
        if record.model_revision != self.schema.model_revision:
            raise ValueError(
                "cache row model revision differs from the requested GR00T revision"
            )
        if record.source_hook != self._source_hook(record.kind):
            raise ValueError(
                "cache row source hook differs from the requested extractor hook"
            )
        if record.cache_schema_sha256 != self.schema.sha256:
            raise ValueError(
                "cache row tensor schema differs from the requested cache schema"
            )
        expected = self._expected_metadata(record.kind)
        actual = {item.name: (item.shape, item.dtype) for item in record.tensors}
        if actual != expected:
            raise ValueError(f"cache row tensor shape/dtype contract changed: {actual}")
        content_payload = {
            "kind": record.kind,
            "model_revision": record.model_revision,
            "source_hook": record.source_hook,
            "cache_schema_sha256": record.cache_schema_sha256,
            "tensors": [item.to_dict() for item in record.tensors],
        }
        if _canonical_sha256(content_payload) != record.content_sha256:
            raise ValueError(
                "cache row content_sha256 does not match its tensor metadata"
            )
        expected_path = (
            PurePosixPath("tensors")
            / record.kind
            / record.content_sha256[:2]
            / f"{record.content_sha256}.pt"
        ).as_posix()
        if record.tensor_path != expected_path:
            raise ValueError(
                "cache row tensor_path is not its canonical content-addressed path"
            )

    def _read_manifest(
        self, path: Path, expected_kind: _CacheKind
    ) -> dict[str | tuple[str, str, int], GrootTemporalCacheRecord]:
        if not path.exists():
            return {}
        rows: dict[str | tuple[str, str, int], GrootTemporalCacheRecord] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(
                        f"blank row in GR00T temporal cache manifest {path}:{line_number}"
                    )
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, Mapping):
                        raise TypeError("cache row must be a mapping")
                    row = GrootTemporalCacheRecord.from_dict(payload)
                except Exception as exc:
                    raise ValueError(
                        f"invalid cache row {path}:{line_number}: {exc}"
                    ) from exc
                if row.kind != expected_kind:
                    raise ValueError(
                        f"cache manifest {path} contains the wrong feature kind"
                    )
                self._validate_record_contract(row)
                if row.logical_key in rows:
                    raise ValueError(f"cache manifest {path} repeats a logical key")
                rows[row.logical_key] = row
        return rows

    @staticmethod
    def _validate_conditioning_aliases(
        rows: Sequence[GrootTemporalCacheRecord],
    ) -> None:
        by_semantic_key: dict[tuple[str, str], GrootTemporalCacheRecord] = {}
        for row in rows:
            semantic_key = row.semantic_key
            assert semantic_key is not None
            prior = by_semantic_key.setdefault(semantic_key, row)
            if (
                prior.content_sha256 != row.content_sha256
                or prior.tensor_path != row.tensor_path
                or prior.tensor_file_sha256 != row.tensor_file_sha256
                or (prior.selected_prompt_sha256 != row.selected_prompt_sha256)
            ):
                raise ValueError(
                    "conditioning generations for one sample/prompt must alias identical content"
                )

    def _manifest_records(self, kind: _CacheKind) -> list[GrootTemporalCacheRecord]:
        rows = self._visual if kind == VISUAL_CACHE_KIND else self._conditioning

        def sort_key(row: GrootTemporalCacheRecord) -> tuple[str, ...]:
            key = row.logical_key
            return (
                (key,) if isinstance(key, str) else tuple((str(value) for value in key))
            )

        return sorted(rows.values(), key=sort_key)

    def _write_manifest(self, kind: _CacheKind) -> None:
        path = (
            self.visual_manifest_path
            if kind == VISUAL_CACHE_KIND
            else self.conditioning_manifest_path
        )
        payload = b"".join(
            (
                _canonical_bytes(record.to_dict()) + b"\n"
                for record in self._manifest_records(kind)
            )
        )
        _atomic_write_bytes(path, payload)

    def flush_manifests(self) -> None:
        """Atomically checkpoint all deferred logical rows.

        Payloads are already content-addressed and durable before a row enters
        the in-memory index.  A crash before this flush can therefore leave
        only reusable orphan payloads; reopening resumes from the last fully
        replaced manifest.
        """
        for kind in (VISUAL_CACHE_KIND, CONDITIONING_CACHE_KIND):
            if kind not in self._dirty_manifest_kinds:
                continue
            self._write_manifest(kind)
            self._dirty_manifest_kinds.remove(kind)

    @property
    def has_deferred_manifest_rows(self) -> bool:
        return bool(self._dirty_manifest_kinds)

    def _content_metadata(
        self, kind: _CacheKind, tensors: Mapping[str, Tensor]
    ) -> tuple[tuple[TensorMetadata, ...], str]:
        ordered = tuple(
            (
                TensorMetadata.from_tensor(name, tensors[name])
                for name in self._expected_names(kind)
            )
        )
        content_payload = {
            "kind": kind,
            "model_revision": self.schema.model_revision,
            "source_hook": self._source_hook(kind),
            "cache_schema_sha256": self.schema.sha256,
            "tensors": [item.to_dict() for item in ordered],
        }
        return (ordered, _canonical_sha256(content_payload))

    def _payload_path(self, kind: _CacheKind, content_sha256: str) -> tuple[str, Path]:
        relative = (
            PurePosixPath("tensors")
            / kind
            / content_sha256[:2]
            / f"{content_sha256}.pt"
        ).as_posix()
        return (relative, self.root / relative)

    def _load_payload(
        self, record: GrootTemporalCacheRecord, *, device: torch.device | str = "cpu"
    ) -> dict[str, Tensor]:
        self._validate_record_contract(record)
        relative = _safe_relative_path(record.tensor_path)
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("cache tensor path escapes its root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"missing GR00T temporal cache payload: {path}")
        if _sha256_file(path) != record.tensor_file_sha256:
            raise ValueError(f"GR00T temporal cache file SHA mismatch: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"cannot load GR00T temporal cache payload: {path}"
            ) from exc
        if not isinstance(payload, Mapping) or set(payload) != set(
            self._expected_names(record.kind)
        ):
            raise ValueError("GR00T temporal cache payload tensor names changed")
        metadata = {item.name: item for item in record.tensors}
        result: dict[str, Tensor] = {}
        for name in self._expected_names(record.kind):
            value = payload[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"cached {name} is not a torch.Tensor")
            expected = metadata[name]
            if (
                tuple(value.shape) != expected.shape
                or str(value.dtype) != expected.dtype
            ):
                raise ValueError(f"cached {name} shape/dtype differs from its manifest")
            if _tensor_value_sha256(value) != expected.value_sha256:
                raise ValueError(f"cached {name} value SHA differs from its manifest")
            result[name] = value.to(device=device)
        self._validate_payload_semantics(record.kind, result)
        return result

    def _ensure_payload(
        self,
        *,
        kind: _CacheKind,
        tensors: Mapping[str, Tensor],
        metadata: tuple[TensorMetadata, ...],
        content_sha256: str,
    ) -> tuple[str, str]:
        relative, path = self._payload_path(kind, content_sha256)
        if path.exists():
            provisional = GrootTemporalCacheRecord(
                kind=kind,
                sample_id="content-validation",
                prompt_id="content-validation"
                if kind == CONDITIONING_CACHE_KIND
                else None,
                context_generation=0 if kind == CONDITIONING_CACHE_KIND else None,
                selected_prompt_sha256="0" * 64
                if kind == CONDITIONING_CACHE_KIND
                else None,
                tensor_path=relative,
                tensor_file_sha256=_sha256_file(path),
                content_sha256=content_sha256,
                tensors=metadata,
                model_revision=self.schema.model_revision,
                source_hook=self._source_hook(kind),
                cache_schema_sha256=self.schema.sha256,
            )
            self._load_payload(provisional)
        else:
            _atomic_torch_save(path, tensors)
        return (relative, _sha256_file(path))

    def put_visual(
        self,
        sample_id: str,
        *,
        mid_tokens: Tensor,
        mid_padding_mask: Tensor,
        persist_manifest: bool = True,
    ) -> GrootTemporalCacheRecord:
        sample = _strict_non_empty(sample_id, "sample_id")
        tensors = {
            "mid_tokens": _normalise_tensor(
                mid_tokens,
                name="mid_tokens",
                shape=self.schema.visual_shape,
                dtype=self.schema.float_dtype,
            ),
            "mid_padding_mask": _normalise_tensor(
                mid_padding_mask,
                name="mid_padding_mask",
                shape=(self.schema.visual_shape[0],),
                dtype="torch.bool",
            ),
        }
        self._validate_payload_semantics(VISUAL_CACHE_KIND, tensors)
        metadata, content_sha = self._content_metadata(VISUAL_CACHE_KIND, tensors)
        existing = self._visual.get(sample)
        if existing is not None:
            if existing.content_sha256 != content_sha:
                raise ValueError(
                    "resume found different visual content for the same sample_id"
                )
            self._load_payload(existing)
            return existing
        relative, file_sha = self._ensure_payload(
            kind=VISUAL_CACHE_KIND,
            tensors=tensors,
            metadata=metadata,
            content_sha256=content_sha,
        )
        record = GrootTemporalCacheRecord(
            kind=VISUAL_CACHE_KIND,
            sample_id=sample,
            prompt_id=None,
            context_generation=None,
            selected_prompt_sha256=None,
            tensor_path=relative,
            tensor_file_sha256=file_sha,
            content_sha256=content_sha,
            tensors=metadata,
            model_revision=self.schema.model_revision,
            source_hook=self.schema.visual_source_hook,
            cache_schema_sha256=self.schema.sha256,
        )
        self._visual[sample] = record
        if persist_manifest:
            try:
                self._write_manifest(VISUAL_CACHE_KIND)
                self._dirty_manifest_kinds.discard(VISUAL_CACHE_KIND)
            except Exception:
                del self._visual[sample]
                raise
        else:
            self._dirty_manifest_kinds.add(VISUAL_CACHE_KIND)
        return record

    def put_conditioning(
        self,
        sample_id: str,
        prompt_id: str,
        context_generation: int,
        *,
        selected_prompt: str,
        conditioning_tokens: Tensor,
        conditioning_padding_mask: Tensor,
        sequence_positions: Tensor,
        persist_manifest: bool = True,
    ) -> GrootTemporalCacheRecord:
        sample = _strict_non_empty(sample_id, "sample_id")
        prompt = _strict_non_empty(prompt_id, "prompt_id")
        generation = _strict_generation(context_generation)
        selected = _strict_non_empty(selected_prompt, "selected_prompt")
        prompt_sha = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        count = self.schema.conditioning_shape[0]
        tensors = {
            "dit_conditioning_tokens": _normalise_tensor(
                conditioning_tokens,
                name="conditioning_tokens",
                shape=self.schema.conditioning_shape,
                dtype=self.schema.float_dtype,
            ),
            "dit_conditioning_padding_mask": _normalise_tensor(
                conditioning_padding_mask,
                name="conditioning_padding_mask",
                shape=(count,),
                dtype="torch.bool",
            ),
            "dit_sequence_positions": _normalise_tensor(
                sequence_positions,
                name="sequence_positions",
                shape=(count,),
                dtype="torch.int64",
            ),
        }
        self._validate_payload_semantics(CONDITIONING_CACHE_KIND, tensors)
        metadata, content_sha = self._content_metadata(CONDITIONING_CACHE_KIND, tensors)
        logical_key = (sample, prompt, generation)
        existing = self._conditioning.get(logical_key)
        if existing is not None:
            if (
                existing.content_sha256 != content_sha
                or existing.selected_prompt_sha256 != prompt_sha
            ):
                raise ValueError(
                    "resume found different conditioning for the same logical key"
                )
            self._load_payload(existing)
            return existing
        semantic_key = (sample, prompt)
        source = self._conditioning_by_semantic.get(semantic_key)
        if source is not None:
            if (
                source.content_sha256 != content_sha
                or source.selected_prompt_sha256 != prompt_sha
            ):
                raise ValueError(
                    "conditioning generations for one sample/prompt must have identical content"
                )
            self._load_payload(source)
            relative, file_sha = (source.tensor_path, source.tensor_file_sha256)
        else:
            relative, file_sha = self._ensure_payload(
                kind=CONDITIONING_CACHE_KIND,
                tensors=tensors,
                metadata=metadata,
                content_sha256=content_sha,
            )
        record = GrootTemporalCacheRecord(
            kind=CONDITIONING_CACHE_KIND,
            sample_id=sample,
            prompt_id=prompt,
            context_generation=generation,
            selected_prompt_sha256=prompt_sha,
            tensor_path=relative,
            tensor_file_sha256=file_sha,
            content_sha256=content_sha,
            tensors=metadata,
            model_revision=self.schema.model_revision,
            source_hook=self.schema.conditioning_source_hook,
            cache_schema_sha256=self.schema.sha256,
        )
        self._conditioning[logical_key] = record
        added_semantic = source is None
        if added_semantic:
            self._conditioning_by_semantic[semantic_key] = record
        try:
            if persist_manifest:
                self._write_manifest(CONDITIONING_CACHE_KIND)
                self._dirty_manifest_kinds.discard(CONDITIONING_CACHE_KIND)
            else:
                self._dirty_manifest_kinds.add(CONDITIONING_CACHE_KIND)
        except Exception:
            del self._conditioning[logical_key]
            if added_semantic:
                del self._conditioning_by_semantic[semantic_key]
            raise
        return record

    def alias_conditioning_generation(
        self,
        sample_id: str,
        prompt_id: str,
        *,
        source_generation: int,
        target_generation: int,
        persist_manifest: bool = True,
    ) -> GrootTemporalCacheRecord:
        source_key = (
            _strict_non_empty(sample_id, "sample_id"),
            _strict_non_empty(prompt_id, "prompt_id"),
            _strict_generation(source_generation, "source_generation"),
        )
        target_key = (
            source_key[0],
            source_key[1],
            _strict_generation(target_generation, "target_generation"),
        )
        if source_key not in self._conditioning:
            raise KeyError(f"missing source conditioning key {source_key!r}")
        source = self._conditioning[source_key]
        existing = self._conditioning.get(target_key)
        if existing is not None:
            if (
                existing.content_sha256 != source.content_sha256
                or existing.selected_prompt_sha256 != source.selected_prompt_sha256
            ):
                raise ValueError(
                    "target generation already refers to different conditioning"
                )
            self._load_payload(existing)
            return existing
        self._load_payload(source)
        record = GrootTemporalCacheRecord(
            **{
                **source.to_dict(),
                "context_generation": target_key[2],
                "tensors": source.tensors,
            }
        )
        self._conditioning[target_key] = record
        try:
            if persist_manifest:
                self._write_manifest(CONDITIONING_CACHE_KIND)
                self._dirty_manifest_kinds.discard(CONDITIONING_CACHE_KIND)
            else:
                self._dirty_manifest_kinds.add(CONDITIONING_CACHE_KIND)
        except Exception:
            del self._conditioning[target_key]
            raise
        return record

    def load_visual(
        self, sample_id: str, *, device: torch.device | str = "cpu"
    ) -> LoadedVisualFeatures:
        key = _strict_non_empty(sample_id, "sample_id")
        try:
            record = self._visual[key]
        except KeyError as exc:
            raise KeyError(f"missing visual cache key {key!r}") from exc
        payload = self._load_payload(record, device=device)
        return LoadedVisualFeatures(
            tokens=payload["mid_tokens"],
            padding_mask=payload["mid_padding_mask"],
            record=record,
        )

    def load_conditioning(
        self,
        key: tuple[str, str, int],
        *,
        selected_prompt: str | None = None,
        device: torch.device | str = "cpu",
    ) -> LoadedConditioningFeatures:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(
                "conditioning cache key must be (sample_id,prompt_id,generation)"
            )
        normalised = (
            _strict_non_empty(key[0], "sample_id"),
            _strict_non_empty(key[1], "prompt_id"),
            _strict_generation(key[2]),
        )
        try:
            record = self._conditioning[normalised]
        except KeyError as exc:
            raise KeyError(f"missing conditioning cache key {normalised!r}") from exc
        if selected_prompt is not None:
            selected = _strict_non_empty(selected_prompt, "selected_prompt")
            if (
                hashlib.sha256(selected.encode("utf-8")).hexdigest()
                != record.selected_prompt_sha256
            ):
                raise ValueError(
                    "full-manifest selected_prompt differs from cached conditioning"
                )
        payload = self._load_payload(record, device=device)
        return LoadedConditioningFeatures(
            tokens=payload["dit_conditioning_tokens"],
            padding_mask=payload["dit_conditioning_padding_mask"],
            sequence_positions=payload["dit_sequence_positions"],
            record=record,
        )

    def load_full_manifest(
        self,
        source: str | Path | Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[LoadedAssignedTemporalFeatures, ...]:
        """Load exact full-manifest keys without performing prompt assignment."""
        assignments = read_full_manifest_prompt_assignments(source)
        visual_values: dict[str, LoadedVisualFeatures] = {}
        conditioning_values: dict[tuple[str, str, int], LoadedConditioningFeatures] = {}
        loaded: list[LoadedAssignedTemporalFeatures] = []
        for assignment in assignments:
            visual = visual_values.get(assignment.sample_id)
            if visual is None:
                visual = self.load_visual(assignment.sample_id, device=device)
                visual_values[assignment.sample_id] = visual
            conditioning = conditioning_values.get(assignment.conditioning_key)
            if conditioning is None:
                conditioning = self.load_conditioning(
                    assignment.conditioning_key,
                    selected_prompt=assignment.selected_prompt,
                    device=device,
                )
                conditioning_values[assignment.conditioning_key] = conditioning
            loaded.append(
                LoadedAssignedTemporalFeatures(
                    assignment=assignment, visual=visual, conditioning=conditioning
                )
            )
        return tuple(loaded)

    def missing_full_manifest_keys(
        self, source: str | Path | Mapping[str, Any]
    ) -> dict[str, tuple[Any, ...]]:
        """Return exact resume work implied by an already-assigned full manifest."""
        assignments = read_full_manifest_prompt_assignments(source)
        missing_visual = sorted(
            {assignment.sample_id for assignment in assignments} - set(self._visual)
        )
        missing_conditioning = sorted(
            {assignment.conditioning_key for assignment in assignments}
            - set(self._conditioning)
        )
        return {
            "e03": tuple(missing_visual),
            "conditioning": tuple(missing_conditioning),
        }
