"""Shared serialization, hashes, reproducibility and small conversions."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from endo_bbox_vla.cache import GrootTemporalPromptAssignment
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.train import TrainingVariant
from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


def _resolve_repo_path(value: str | Path) -> Path:
    from endo_bbox_vla.config import resolve_path

    return resolve_path(value)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _variant_root(config: ExperimentConfig, variant: TrainingVariant) -> Path:
    return (
        config.path_value("workspace", "root") / "runs" / f"shared_{variant}" / "seed42"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json(value))


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["name"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    torch.cuda.set_rng_state_all(list(state["cuda"]))


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, b"".join((_canonical_json(dict(row)) for row in rows)))


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


type Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PolypPrediction:
    """Public sidecar result in original frame coordinates."""

    present: bool
    score: float
    bbox_xyxy: Box | None
    frame_id: int
    timestamp: float

    def to_dict(self) -> dict[str, bool | float | int | list[float] | None]:
        payload = asdict(self)
        payload["bbox_xyxy"] = (
            list(self.bbox_xyxy) if self.bbox_xyxy is not None else None
        )
        return payload


@dataclass(frozen=True, slots=True)
class LatencyStats:
    repeats: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float
    std_ms: float
    peak_incremental_allocated_bytes: int


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted((float(value) for value in values))
    if not ordered or not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid percentile input")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(
    values: Sequence[float], *, peak_incremental_allocated_bytes: int = 0
) -> LatencyStats:
    parsed = [float(value) for value in values]
    if not parsed or not all((math.isfinite(value) and value >= 0 for value in parsed)):
        raise ValueError("latencies must be non-empty, finite, and non-negative")
    return LatencyStats(
        repeats=len(parsed),
        mean_ms=statistics.fmean(parsed),
        p50_ms=statistics.median(parsed),
        p95_ms=_percentile(parsed, 0.95),
        minimum_ms=min(parsed),
        maximum_ms=max(parsed),
        std_ms=statistics.pstdev(parsed),
        peak_incremental_allocated_bytes=int(peak_incremental_allocated_bytes),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FullShardScope:
    num_shards: int
    shard_index: int
    assignments: tuple[GrootTemporalPromptAssignment, ...]
    visual_keys: frozenset[str]
    logical_conditioning_keys: frozenset[tuple[str, str, int]]
    physical_conditioning_keys: frozenset[tuple[str, str]]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "frame_occurrences": len(self.assignments),
            "e03_entries": len(self.visual_keys),
            "logical_conditioning_entries": len(self.logical_conditioning_keys),
            "physical_conditioning_forwards": len(self.physical_conditioning_keys),
        }


def physical_key_shard(key: tuple[str, str], *, num_shards: int) -> int:
    if (
        isinstance(num_shards, bool)
        or not isinstance(num_shards, int)
        or num_shards < 1
    ):
        raise ValueError("num_shards must be a positive integer")
    sample_id, prompt_id = key
    digest = hashlib.sha256()
    for value in (sample_id, prompt_id):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") % num_shards


def physical_conditioning_key(
    assignment: GrootTemporalPromptAssignment,
) -> tuple[str, str]:
    return (assignment.sample_id, assignment.prompt_id)


def build_full_shard_scope(
    assignments: Sequence[GrootTemporalPromptAssignment],
    *,
    num_shards: int,
    shard_index: int,
) -> FullShardScope:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    selected = tuple(
        (
            value
            for value in assignments
            if physical_key_shard(
                physical_conditioning_key(value), num_shards=num_shards
            )
            == shard_index
        )
    )
    if not selected:
        raise ValueError("deterministic full shard is empty")
    return FullShardScope(
        num_shards=num_shards,
        shard_index=shard_index,
        assignments=selected,
        visual_keys=frozenset((value.sample_id for value in selected)),
        logical_conditioning_keys=frozenset(
            (value.conditioning_key for value in selected)
        ),
        physical_conditioning_keys=frozenset(
            (physical_conditioning_key(value) for value in selected)
        ),
    )


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
    else:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    return np.asarray(value)


def _four_finite(values: Any) -> tuple[float, float, float, float]:
    array = _as_numpy(values).astype(np.float64, copy=False).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        raise ValueError(f"Expected four finite box values, got {values!r}.")
    return tuple((float(value) for value in array))


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height


def letterbox_geometry(
    original_size: tuple[int, int], target_size: tuple[int, int] = (224, 224)
) -> LetterboxGeometry:
    """Return integer resize and symmetric padding used by aspect-ratio letterboxing.

    Sizes are expressed as ``(width, height)``.  The integer geometry exactly
    represents the 640x480 -> 224x168 + 28/28 padding transform in Pi0.5.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    if min(original_width, original_height, target_width, target_height) <= 0:
        raise ValueError("Original and target dimensions must be positive.")
    ratio = max(original_width / target_width, original_height / target_height)
    resized_width = min(target_width, max(1, int(original_width / ratio)))
    resized_height = min(target_height, max(1, int(original_height / ratio)))
    return LetterboxGeometry(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=(target_width - resized_width) // 2,
        pad_top=(target_height - resized_height) // 2,
    )


def _clip_xyxy(box: Sequence[float], width: int, height: int) -> Box | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def letterbox_xyxy_to_original_xyxy(
    box_xyxy: Sequence[float],
    *,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (224, 224),
) -> Box | None:
    """Undo letterbox padding/resize and clip to the original image boundary."""
    geometry = letterbox_geometry(original_size, target_size)
    x1, y1, x2, y2 = _four_finite(box_xyxy)
    original = (
        (x1 - geometry.pad_left) / geometry.scale_x,
        (y1 - geometry.pad_top) / geometry.scale_y,
        (x2 - geometry.pad_left) / geometry.scale_x,
        (y2 - geometry.pad_top) / geometry.scale_y,
    )
    return _clip_xyxy(original, geometry.original_width, geometry.original_height)


def normalized_cxcywh_to_original_xyxy(
    box_cxcywh: Sequence[float],
    *,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (224, 224),
) -> Box | None:
    """Convert normalized letterbox ``cxcywh`` to clipped original pixels."""
    center_x, center_y, width, height = _four_finite(box_cxcywh)
    target_width, target_height = target_size
    absolute = (
        (center_x - width / 2.0) * target_width,
        (center_y - height / 2.0) * target_height,
        (center_x + width / 2.0) * target_width,
        (center_y + height / 2.0) * target_height,
    )
    return letterbox_xyxy_to_original_xyxy(
        absolute, original_size=original_size, target_size=target_size
    )


def run_cli(commands, argv=None, *, description="Run one task in this category."):
    """Forward a selected task's arguments to its original argument parser."""
    import argparse
    import inspect
    import sys

    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=tuple(commands))
    if not values or values[0] in {"-h", "--help"}:
        parser.parse_args(values)
    name = values.pop(0)
    if name not in commands:
        parser.error(f"unknown command: {name}")
    function = commands[name]
    if "argv" in inspect.signature(function).parameters:
        return function(values)
    previous = sys.argv
    try:
        sys.argv = [previous[0], *values]
        return function()
    finally:
        sys.argv = previous
