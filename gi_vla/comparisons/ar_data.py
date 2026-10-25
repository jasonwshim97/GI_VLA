"""comparisons — ar_data: consolidated research implementation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from annotation.prepare import FrameRecord
    from endo_bbox_vla.data import ScenarioStaticDataContract
from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
type IntBBoxXYWH = tuple[int, int, int, int]
type IntBBoxXYXY = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ARParseResult:
    """Non-throwing result returned for generated model text.

    ``present`` is ``None`` when the output is malformed.  ``bbox_xyxy`` uses
    half-open pixel edges, so ``(x, y, x + w, y + h)`` is lossless with the
    parsed integer ``bbox_xywh``.
    """

    text: str
    format_valid: bool
    present: bool | None
    bbox_xywh: IntBBoxXYWH | None
    bbox_xyxy: IntBBoxXYXY | None
    error: str | None
    canonical_text: str | None

    def __post_init__(self) -> None:
        if self.format_valid:
            if (
                self.present is None
                or self.error is not None
                or self.canonical_text is None
            ):
                raise ValueError(
                    "a valid parse result must contain presence and canonical text"
                )
            if self.present != (self.bbox_xywh is not None):
                raise ValueError("presence and bbox_xywh disagree")
            if self.present != (self.bbox_xyxy is not None):
                raise ValueError("presence and bbox_xyxy disagree")
        elif (
            self.present is not None
            or self.bbox_xywh is not None
            or self.bbox_xyxy is not None
        ):
            raise ValueError("an invalid parse result must not contain a detection")


@dataclass(frozen=True, slots=True)
class AnswerSpanMetadata:
    """Character spans for independently weighting detection and action loss.

    Spans are half-open offsets into the canonical assistant answer, before the
    chat template adds role tokens.  A tokenizer-aware collator can tokenize
    the exact segment strings to derive token spans without guessing around
    punctuation or negative signs.
    """

    detection: tuple[int, int]
    separator: tuple[int, int]
    action: tuple[int, int]
    action_numeric: tuple[tuple[int, int], ...]
    detection_text: str
    action_text: str

    def to_dict(self) -> dict[str, str | list[int]]:
        return {
            "detection": list(self.detection),
            "separator": list(self.separator),
            "action": list(self.action),
            "action_numeric": [list(span) for span in self.action_numeric],
            "detection_text": self.detection_text,
            "action_text": self.action_text,
        }


type FloatAction = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
ACTION_DIM = 16


@dataclass(frozen=True, slots=True)
class GroundedActionParseResult:
    """Non-throwing parse result for the bbox/presence plus 16D action."""

    text: str
    format_valid: bool
    present: bool | None
    bbox_xywh: IntBBoxXYWH | None
    bbox_xyxy: IntBBoxXYXY | None
    action: FloatAction | None
    error: str | None
    canonical_text: str | None
    spans: AnswerSpanMetadata | None

    def __post_init__(self) -> None:
        if self.format_valid:
            if (
                self.present is None
                or self.action is None
                or self.error is not None
                or (self.canonical_text is None)
                or (self.spans is None)
            ):
                raise ValueError("a valid grounded-action parse must be complete")
            if self.present != (self.bbox_xywh is not None):
                raise ValueError("presence and bbox_xywh disagree")
            if self.present != (self.bbox_xyxy is not None):
                raise ValueError("presence and bbox_xyxy disagree")
            if len(self.action) != ACTION_DIM:
                raise ValueError("grounded action has the wrong dimension")
        elif any(
            (
                value is not None
                for value in (
                    self.present,
                    self.bbox_xywh,
                    self.bbox_xyxy,
                    self.action,
                    self.canonical_text,
                    self.spans,
                )
            )
        ):
            raise ValueError("an invalid grounded-action parse must contain no target")


type FloatBBoxXYWH = tuple[float, float, float, float]
DEFAULT_IMAGE_SIZE = (640, 480)
type ImageSize = tuple[int, int]


def _coerce_image_size(image_size: Sequence[int]) -> ImageSize:
    if isinstance(image_size, str | bytes) or len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = image_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or (not isinstance(width, Integral))
        or (not isinstance(height, Integral))
    ):
        raise TypeError("image_size width and height must be integers")
    result = (int(width), int(height))
    if min(result) <= 0:
        raise ValueError("image_size width and height must be positive")
    return result


def _coerce_numeric_box(box: Sequence[Real], *, name: str) -> FloatBBoxXYWH:
    if isinstance(box, str | bytes) or len(box) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    values: list[float] = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} contains NaN or infinity")
        values.append(converted)
    return (values[0], values[1], values[2], values[3])


def validate_xywh(
    bbox_xywh: Sequence[Real],
    *,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    tolerance: float = 1e-06,
) -> FloatBBoxXYWH:
    """Validate a finite positive ``xywh`` box in an original image."""
    width, height = _coerce_image_size(image_size)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    x, y, box_width, box_height = _coerce_numeric_box(bbox_xywh, name="bbox_xywh")
    if x < -tolerance or y < -tolerance or box_width <= 0 or (box_height <= 0):
        raise ValueError(f"invalid xywh bbox {(x, y, box_width, box_height)}")
    if x + box_width > width + tolerance or y + box_height > height + tolerance:
        raise ValueError(
            f"xywh bbox {(x, y, box_width, box_height)} exceeds image size {(width, height)}"
        )
    return (x, y, box_width, box_height)


type FloatBBoxXYXY = tuple[float, float, float, float]


def xywh_to_xyxy(bbox_xywh: Sequence[Real]) -> FloatBBoxXYXY:
    """Convert top-left ``xywh`` to half-open ``xyxy`` without quantizing."""
    x, y, width, height = _coerce_numeric_box(bbox_xywh, name="bbox_xywh")
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    return (x, y, x + width, y + height)


def xyxy_to_xywh(bbox_xyxy: Sequence[Real]) -> FloatBBoxXYWH:
    """Convert half-open ``xyxy`` to top-left ``xywh`` without quantizing."""
    x1, y1, x2, y2 = _coerce_numeric_box(bbox_xyxy, name="bbox_xyxy")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have x2 > x1 and y2 > y1")
    return (x1, y1, x2 - x1, y2 - y1)


def scale_xywh(
    bbox_xywh: Sequence[Real], *, source_size: Sequence[int], target_size: Sequence[int]
) -> FloatBBoxXYWH:
    """Scale an ``xywh`` box between two unpadded image coordinate frames."""
    source_width, source_height = _coerce_image_size(source_size)
    target_width, target_height = _coerce_image_size(target_size)
    x, y, width, height = validate_xywh(
        bbox_xywh, image_size=(source_width, source_height)
    )
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return (x * scale_x, y * scale_y, width * scale_x, height * scale_y)


def _round_pixel_edge(value: float) -> int:
    return math.floor(value + 0.5)


def quantize_xywh(
    bbox_xywh: Sequence[Real], *, image_size: Sequence[int] = DEFAULT_IMAGE_SIZE
) -> IntBBoxXYWH:
    """Round both ``xyxy`` edges to pixels and retain a non-empty box.

    Rounding the two edges independently avoids accumulating the origin's
    rounding error into width/height.  Ground-truth tensors remain floating
    point; this function is only the deterministic text-label projection.
    """
    image_width, image_height = _coerce_image_size(image_size)
    x, y, width, height = validate_xywh(
        bbox_xywh, image_size=(image_width, image_height)
    )
    x1 = min(max(_round_pixel_edge(max(x, 0.0)), 0), image_width - 1)
    y1 = min(max(_round_pixel_edge(max(y, 0.0)), 0), image_height - 1)
    x2 = min(max(_round_pixel_edge(min(x + width, image_width)), x1 + 1), image_width)
    y2 = min(
        max(_round_pixel_edge(min(y + height, image_height)), y1 + 1), image_height
    )
    return (x1, y1, x2 - x1, y2 - y1)


NEGATIVE_TARGET = "N"
POSITIVE_TARGET_PREFIX = "Y"


def serialize_ar_target(
    bbox_xywh: Sequence[Real] | None = None,
    *,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
) -> str:
    """Serialize a positive floating COCO box or a confirmed negative."""
    resolved_size = _coerce_image_size(image_size)
    if bbox_xywh is None:
        return NEGATIVE_TARGET
    x, y, width, height = quantize_xywh(bbox_xywh, image_size=resolved_size)
    return f"{POSITIVE_TARGET_PREFIX} [{x}, {y}, {width}, {height}]"


def validate_action(action: object) -> FloatAction:
    """Return one finite canonical float action with exactly 16 elements."""
    value = action
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("action must be a one-dimensional numeric sequence")
    if len(value) != ACTION_DIM:
        raise ValueError(f"action must contain exactly {ACTION_DIM} values")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError("action values must be real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("action contains NaN or infinity")
        converted.append(number)
    return tuple(converted)


ACTION_PREFIX = "A"
ACTION_DECIMALS = 6


def _fixed_action_value(value: float) -> str:
    formatted = f"{value:.{ACTION_DECIMALS}f}"
    return "0.000000" if formatted == "-0.000000" else formatted


def serialize_action_target(action: object) -> str:
    """Serialize a continuous 16D action with a fixed six-decimal grammar."""
    values = validate_action(action)
    payload = ", ".join((_fixed_action_value(value) for value in values))
    return f"{ACTION_PREFIX} [{payload}]"


TARGET_SEPARATOR = " | "
_FIXED_ACTION_VALUE_PATTERN = "-?(?:0|[1-9][0-9]*)\\.[0-9]{6}"
_ACTION_TARGET_PATTERN = re.compile(
    f"A \\[{_FIXED_ACTION_VALUE_PATTERN}(?:, {_FIXED_ACTION_VALUE_PATTERN}){{{ACTION_DIM - 1}}}\\]"
)


def grounded_action_answer_spans(text: str) -> AnswerSpanMetadata:
    """Return loss-weighting spans for an already canonical combined target."""
    if not isinstance(text, str):
        raise TypeError("grounded-action target must be text")
    separator_start = text.find(TARGET_SEPARATOR)
    if separator_start <= 0 or text.find(TARGET_SEPARATOR, separator_start + 1) != -1:
        raise ValueError(
            "grounded-action target must contain exactly one ' | ' separator"
        )
    separator_end = separator_start + len(TARGET_SEPARATOR)
    detection_text = text[:separator_start]
    action_text = text[separator_end:]
    if not detection_text or not action_text:
        raise ValueError("grounded-action target segments must be non-empty")
    if _ACTION_TARGET_PATTERN.fullmatch(action_text) is None:
        raise ValueError("action segment is not canonical fixed-six-decimal text")
    numeric_spans = tuple(
        (
            (separator_end + match.start(), separator_end + match.end())
            for match in re.finditer(_FIXED_ACTION_VALUE_PATTERN, action_text)
        )
    )
    if len(numeric_spans) != ACTION_DIM:
        raise ValueError(
            f"action segment must contain exactly {ACTION_DIM} fixed-decimal values"
        )
    return AnswerSpanMetadata(
        detection=(0, separator_start),
        separator=(separator_start, separator_end),
        action=(separator_end, len(text)),
        action_numeric=numeric_spans,
        detection_text=detection_text,
        action_text=action_text,
    )


def serialize_grounded_action_target(
    bbox_xywh: Sequence[Real] | None,
    action: object,
    *,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
) -> str:
    """Serialize presence/bbox and continuous action into the active contract."""
    detection = serialize_ar_target(bbox_xywh, image_size=image_size)
    return detection + TARGET_SEPARATOR + serialize_action_target(action)


def _invalid_result(text: str, error: str) -> ARParseResult:
    return ARParseResult(
        text=text,
        format_valid=False,
        present=None,
        bbox_xywh=None,
        bbox_xyxy=None,
        error=error,
        canonical_text=None,
    )


_POSITIVE_PATTERN = re.compile(
    "Y\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]"
)


def _validate_integer_xywh(
    bbox_xywh: Sequence[int], *, image_size: Sequence[int]
) -> IntBBoxXYWH:
    if isinstance(bbox_xywh, str | bytes) or len(bbox_xywh) != 4:
        raise ValueError("bbox_xywh must contain exactly four integers")
    if any(
        (
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in bbox_xywh
        )
    ):
        raise TypeError("bbox_xywh values must be integers")
    x, y, width, height = (int(value) for value in bbox_xywh)
    validate_xywh((x, y, width, height), image_size=image_size)
    return (x, y, width, height)


def parse_ar_output(
    text: object, *, image_size: Sequence[int] = DEFAULT_IMAGE_SIZE
) -> ARParseResult:
    """Parse generated text without raising on malformed model output.

    Leading and trailing whitespace and whitespace inside a positive tuple are
    tolerated.  The grammar is otherwise exact: decimals, signs, additional
    boxes, Markdown fences, and explanatory prose are invalid.
    """
    resolved_size = _coerce_image_size(image_size)
    if not isinstance(text, str):
        return _invalid_result(repr(text), "output must be a string")
    stripped = text.strip()
    if not stripped:
        return _invalid_result(text, "output is empty")
    if stripped == NEGATIVE_TARGET:
        return ARParseResult(
            text=text,
            format_valid=True,
            present=False,
            bbox_xywh=None,
            bbox_xyxy=None,
            error=None,
            canonical_text=NEGATIVE_TARGET,
        )
    match = _POSITIVE_PATTERN.fullmatch(stripped)
    if match is None:
        return _invalid_result(text, "expected exactly 'N' or 'Y [x, y, w, h]'")
    try:
        bbox_xywh = tuple((int(value) for value in match.groups()))
        x, y, width, height = _validate_integer_xywh(
            bbox_xywh, image_size=resolved_size
        )
    except (TypeError, ValueError) as exc:
        return _invalid_result(text, str(exc))
    canonical_text = f"{POSITIVE_TARGET_PREFIX} [{x}, {y}, {width}, {height}]"
    return ARParseResult(
        text=text,
        format_valid=True,
        present=True,
        bbox_xywh=(x, y, width, height),
        bbox_xyxy=(x, y, x + width, y + height),
        error=None,
        canonical_text=canonical_text,
    )


def _invalid_grounded_result(text: str, error: str) -> GroundedActionParseResult:
    return GroundedActionParseResult(
        text=text,
        format_valid=False,
        present=None,
        bbox_xywh=None,
        bbox_xyxy=None,
        action=None,
        error=error,
        canonical_text=None,
        spans=None,
    )


_GROUNDED_ACTION_PATTERN = re.compile(
    f"(?P<detection>N|Y \\[[0-9]+, [0-9]+, [0-9]+, [0-9]+\\]) \\| A \\[(?P<action>{_FIXED_ACTION_VALUE_PATTERN}(?:, {_FIXED_ACTION_VALUE_PATTERN}){{{ACTION_DIM - 1}}})\\]"
)


def parse_grounded_action_output(
    text: object, *, image_size: Sequence[int] = DEFAULT_IMAGE_SIZE
) -> GroundedActionParseResult:
    """Parse the strict active ``detection | 16D action`` output contract.

    Leading and trailing whitespace are tolerated.  Internal spacing, bbox
    integer grammar, action dimension, and six-decimal formatting are exact.
    Malformed model output is represented as an invalid result rather than
    raising; invalid programmer-supplied ``image_size`` still raises.
    """
    resolved_size = _coerce_image_size(image_size)
    if not isinstance(text, str):
        return _invalid_grounded_result(repr(text), "output must be a string")
    stripped = text.strip()
    if not stripped:
        return _invalid_grounded_result(text, "output is empty")
    match = _GROUNDED_ACTION_PATTERN.fullmatch(stripped)
    if match is None:
        return _invalid_grounded_result(
            text,
            "expected canonical 'N | A [16 values]' or 'Y [x, y, w, h] | A [16 values]' with six decimals",
        )
    detection = parse_ar_output(match.group("detection"), image_size=resolved_size)
    if not detection.format_valid:
        return _invalid_grounded_result(
            text, detection.error or "invalid detection segment"
        )
    try:
        action = validate_action(
            tuple((float(value) for value in match.group("action").split(", ")))
        )
    except (TypeError, ValueError) as exc:
        return _invalid_grounded_result(text, str(exc))
    canonical = serialize_grounded_action_target(
        detection.bbox_xywh, action, image_size=resolved_size
    )
    spans = grounded_action_answer_spans(canonical)
    return GroundedActionParseResult(
        text=text,
        format_valid=True,
        present=detection.present,
        bbox_xywh=detection.bbox_xywh,
        bbox_xyxy=detection.bbox_xyxy,
        action=action,
        error=None,
        canonical_text=canonical,
        spans=spans,
    )


GROUNDED_ACTION_CONTRACT = "grounded_continuous_action_v1"
_VIEW_KEYS = (
    "train_coco",
    "calibration_coco",
    "negative_control_coco",
    "validation_coco",
    "test_coco",
)
PINNED_CAMERA_ORDER = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
_PRIMARY_DATA_KEYS = frozenset(
    {
        "annotations",
        "approved_payload_sha256",
        "coco_root",
        "protocol_id",
        "protocol_manifest",
        "dataset_revision",
        "image_root",
        "manifest",
        "train_coco",
        "calibration_coco",
        "negative_control_coco",
        "validation_coco",
        "test_coco",
        "original_width",
        "original_height",
    }
)
TARGET_CAMERA = "observation.images.cam_low"
GROUNDED_ACTION_OUTPUT_FORMAT = "Y [x,y,w,h] | A [a0,...,a15] OR N | A [a0,...,a15]"
PINNED_ACTION_FEATURE_ORDER = (
    "left_waist.pos",
    "left_shoulder.pos",
    "left_elbow.pos",
    "left_forearm_roll.pos",
    "left_wrist_angle.pos",
    "left_wrist_rotate.pos",
    "left_gripper.pos",
    "right_waist.pos",
    "right_shoulder.pos",
    "right_elbow.pos",
    "right_forearm_roll.pos",
    "right_wrist_angle.pos",
    "right_wrist_rotate.pos",
    "right_gripper.pos",
    "knob_ud.pos",
    "knob_lr.pos",
)
_SCENARIO_DATA_KEYS = frozenset(
    {
        "format",
        "source_study_config",
        "source_study_config_sha256",
        "protocol_id",
        "dataset_revision",
        "fold_id",
        "original_width",
        "original_height",
    }
)
_COMMIT_PATTERN = re.compile("[0-9a-f]{40}\\Z")
LEGACY_OUTPUT_FORMAT = "Y [x,y,w,h] | N"
_LEGACY_PATH_RELOCATIONS = (
    (
        Path("bbox_vla/workspace/annotations/task4_polyp_v1/review-qa"),
        Path("bbox_vla/annotation/task4_polyp_v1/review-qa"),
    ),
    (
        Path("bbox_vla/workspace/annotations/task4_polyp_v1"),
        Path("bbox_vla/raw_data/task4_polyp"),
    ),
    (
        Path("bbox_vla/workspace/protocols/task4-polyp-primary-v2"),
        Path("bbox_vla/annotation/protocols/task4-polyp-primary-v2"),
    ),
    (Path("bbox_vla/workspace/runs"), Path("bbox_vla/ar_qwen/artifacts/runs")),
)


def _relocate_legacy_path(path: Path) -> Path:
    for legacy_prefix, current_prefix in _LEGACY_PATH_RELOCATIONS:
        try:
            suffix = path.relative_to(legacy_prefix)
        except ValueError:
            continue
        return current_prefix / suffix
    return path


_SHA256_PATTERN = re.compile("[0-9a-f]{64}\\Z")


@dataclass(frozen=True, slots=True)
class ARExperimentConfig:
    """Validated configuration whose paths are resolved relative to the repo."""

    model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    generation: dict[str, Any]
    evaluation: dict[str, Any]
    source_path: Path

    @property
    def repo_root(self) -> Path:
        return PROJECT_ROOT

    def resolve_repo_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        from endo_bbox_vla.config import resolve_path

        return resolve_path(_relocate_legacy_path(path), root=self.repo_root)

    @property
    def data_format(self) -> str:
        return str(self.data.get("format", "primary_v2_coco"))

    @property
    def output_contract(self) -> str:
        return str(self.model.get("output_contract", "bbox_presence_v1"))

    @property
    def uses_grounded_action(self) -> bool:
        return self.output_contract == GROUNDED_ACTION_CONTRACT

    @property
    def uses_lossless_cache(self) -> bool:
        """Whether scenario-static samples must come from the verified PNG/action cache."""
        return (
            self.data_format == "scenario_static_v1"
            and self.data.get("cache_required") is True
        )

    @property
    def camera_order(self) -> tuple[str, ...]:
        if not self.uses_grounded_action:
            return (TARGET_CAMERA,)
        return tuple((str(value) for value in self.model["camera_order"]))

    @property
    def target_camera(self) -> str:
        return str(self.model.get("target_camera", TARGET_CAMERA))

    @property
    def camera_input_sizes(self) -> dict[str, tuple[int, int]]:
        if not self.uses_grounded_action:
            return {
                TARGET_CAMERA: (
                    int(self.data["original_width"]),
                    int(self.data["original_height"]),
                )
            }
        values = self.model.get("camera_input_sizes")
        if not isinstance(values, Mapping):
            raise TypeError(
                "grounded action model.camera_input_sizes must be a mapping"
            )
        return {
            key: (int(values[key][0]), int(values[key][1]))
            for key in PINNED_CAMERA_ORDER
        }

    @property
    def action_key(self) -> str:
        return str(self.model.get("action", {}).get("key", "action"))

    @property
    def action_dim(self) -> int:
        return int(self.model.get("action", {}).get("dim", 0))

    @property
    def action_decimals(self) -> int:
        return int(self.model.get("action", {}).get("decimals", 0))

    @property
    def approved_root(self) -> Path:
        if self.data_format != "primary_v2_coco":
            raise AttributeError("approved_root exists only for primary_v2_coco data")
        return self.resolve_repo_path(self.data["annotations"])

    @property
    def protocol_root(self) -> Path:
        if self.data_format != "primary_v2_coco":
            raise AttributeError("protocol_root exists only for primary_v2_coco data")
        return self.resolve_repo_path(self.data["coco_root"])

    @property
    def image_root(self) -> Path:
        if self.data_format != "primary_v2_coco":
            raise AttributeError("image_root exists only for primary_v2_coco data")
        return self.resolve_repo_path(self.data["image_root"])

    @property
    def scenario_contract(self) -> ScenarioStaticDataContract:
        from endo_bbox_vla.data import load_scenario_static_contract

        if self.data_format != "scenario_static_v1":
            raise AttributeError(
                "scenario_contract exists only for scenario_static_v1 data"
            )
        return load_scenario_static_contract(
            self.resolve_repo_path(self.data["source_study_config"])
        )

    @property
    def output_dir(self) -> Path:
        return self.resolve_repo_path(self.training["output_dir"])

    @property
    def cache_root(self) -> Path | None:
        value = self.data.get("cache_root")
        return None if value is None else self.resolve_repo_path(str(value))

    @property
    def cache_manifest_path(self) -> Path | None:
        root = self.cache_root
        value = self.data.get("cache_manifest")
        if root is None or value is None:
            return None
        name = Path(str(value))
        if name.is_absolute() or len(name.parts) != 1 or name.name != str(value):
            raise ValueError("data.cache_manifest must be one relative filename")
        return root / name

    @property
    def effective_batch_size(self) -> int:
        return int(self.training["batch_size"]) * int(
            self.training["gradient_accumulation_steps"]
        )

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.source_path.read_bytes()).hexdigest()

    @property
    def data_binding_revision(self) -> str:
        return str(self.data["dataset_revision"])

    @property
    def data_binding_sha256(self) -> str:
        key = (
            "approved_payload_sha256"
            if self.data_format == "primary_v2_coco"
            else "source_study_config_sha256"
        )
        return str(self.data[key])

    def coco_path(self, view: str) -> Path:
        if self.data_format != "primary_v2_coco":
            raise AttributeError("COCO views exist only for primary_v2_coco data")
        key = "test_coco" if view == "locked_test" else f"{view}_coco"
        if key not in _VIEW_KEYS:
            raise KeyError(f"unknown experiment view: {view!r}")
        return self.protocol_root / str(self.data[key])

    def validate(self) -> None:
        from endo_bbox_vla.data import SCENARIO_STATIC_PROTOCOL_ID

        if self.model.get("type") != "qwen2_vl_ar":
            raise ValueError("model.type must be 'qwen2_vl_ar'")
        if self.model.get("repo_id") != "Qwen/Qwen2-VL-7B-Instruct":
            raise ValueError(
                "the primary AR baseline must use Qwen/Qwen2-VL-7B-Instruct"
            )
        revision = str(self.model.get("revision", ""))
        if _COMMIT_PATTERN.fullmatch(revision) is None:
            raise ValueError(
                "model.revision must be a full immutable 40-character commit"
            )
        if self.model.get("processor_revision") != revision:
            raise ValueError(
                "model.processor_revision must equal the pinned model revision"
            )
        if self.model.get("coordinate_convention") != "original_pixel_top_left_xywh":
            raise ValueError(
                "AR boxes must use original-pixel top-left xywh coordinates"
            )
        if self.uses_grounded_action:
            if self.model.get("output_format") != GROUNDED_ACTION_OUTPUT_FORMAT:
                raise ValueError(
                    "grounded action model.output_format does not match the pinned 16D contract"
                )
            if self.camera_order != PINNED_CAMERA_ORDER:
                raise ValueError(
                    f"grounded action camera_order must be {PINNED_CAMERA_ORDER!r}"
                )
            if self.target_camera != TARGET_CAMERA:
                raise ValueError(
                    f"grounded action target_camera must be {TARGET_CAMERA!r}"
                )
            sizes = self.model.get("camera_input_sizes")
            if not isinstance(sizes, Mapping) or set(sizes) != set(PINNED_CAMERA_ORDER):
                raise ValueError(
                    "grounded action camera_input_sizes must bind exactly the four cameras"
                )
            for camera_key in PINNED_CAMERA_ORDER:
                value = sizes[camera_key]
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(
                        (
                            isinstance(item, bool) or not isinstance(item, int)
                            for item in value
                        )
                    )
                    or (min(value) <= 0)
                ):
                    raise ValueError(
                        f"invalid camera_input_sizes entry for {camera_key}"
                    )
            original_size = (640, 480)
            if tuple(sizes[TARGET_CAMERA]) != original_size:
                raise ValueError("grounded action cam_low input must remain 640x480")
            auxiliary_sizes = {
                tuple(sizes[key]) for key in PINNED_CAMERA_ORDER if key != TARGET_CAMERA
            }
            if auxiliary_sizes not in ({original_size}, {(320, 240)}):
                raise ValueError(
                    "auxiliary camera inputs must be uniformly 640x480 or fallback 320x240"
                )
            action = self.model.get("action")
            if not isinstance(action, Mapping):
                raise TypeError("grounded action model.action must be a mapping")
            if set(action) != {"key", "dim", "dtype", "decimals", "feature_names"}:
                raise ValueError(
                    "grounded action model.action keys do not match the contract"
                )
            if self.action_key != "action":
                raise ValueError("grounded action model.action.key must be 'action'")
            if self.action_dim != 16:
                raise ValueError("grounded action model.action.dim must be 16")
            if action.get("dtype") != "float32":
                raise ValueError("grounded action model.action.dtype must be float32")
            if self.action_decimals != 6:
                raise ValueError("grounded action model.action.decimals must be 6")
            if tuple(action.get("feature_names", ())) != PINNED_ACTION_FEATURE_ORDER:
                raise ValueError("grounded action feature_names/order changed")
        elif self.output_contract != "bbox_presence_v1":
            raise ValueError(
                f"unsupported model.output_contract: {self.output_contract!r}"
            )
        elif self.model.get("output_format") != LEGACY_OUTPUT_FORMAT:
            raise ValueError(f"model.output_format must be {LEGACY_OUTPUT_FORMAT!r}")
        if self.model.get("freeze_vision_encoder") is not True:
            raise ValueError("the vision encoder must remain frozen")
        if self.model.get("train_vision_projector") is not True:
            raise ValueError("the Qwen visual merger/projector must remain trainable")
        processor = self.model.get("processor")
        if not isinstance(processor, Mapping):
            raise TypeError("model.processor must be a mapping")
        if processor.get("use_fast") is not False:
            raise ValueError("model.processor.use_fast must be explicitly false")
        if int(processor.get("min_pixels", 0)) <= 0:
            raise ValueError("model.processor.min_pixels must be positive")
        if int(processor.get("max_pixels", 0)) < int(processor["min_pixels"]):
            raise ValueError("model.processor.max_pixels must be >= min_pixels")
        if self.uses_grounded_action:
            smallest_pixels = min(
                (width * height for width, height in self.camera_input_sizes.values())
            )
            if int(processor["min_pixels"]) > smallest_pixels:
                raise ValueError(
                    "processor.min_pixels would upscale a checkpoint-bound camera input"
                )
        lora = self.model.get("lora")
        if not isinstance(lora, Mapping):
            raise TypeError("model.lora must be a mapping")
        if int(lora.get("rank", 0)) <= 0 or int(lora.get("alpha", 0)) <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        dropout = float(lora.get("dropout", -1))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.lora.dropout must be in [0, 1)")
        targets = lora.get("target_modules")
        if (
            not isinstance(targets, list)
            or not targets
            or (not all((isinstance(x, str) for x in targets)))
        ):
            raise TypeError("model.lora.target_modules must be a non-empty string list")
        if self.data_format == "primary_v2_coco":
            required_data = _PRIMARY_DATA_KEYS
        elif self.data_format == "scenario_static_v1":
            required_data = _SCENARIO_DATA_KEYS
        else:
            raise ValueError(f"unsupported data.format: {self.data_format!r}")
        missing_data = required_data - self.data.keys()
        if missing_data:
            raise ValueError(f"missing data settings: {sorted(missing_data)}")
        if self.data_format == "primary_v2_coco":
            if self.data.get("logical_views") is not True:
                raise ValueError("data.logical_views must be true")
            if self.data.get("protocol_manifest") != "protocol.json":
                raise ValueError("data.protocol_manifest must be protocol.json")
            if self.data.get("protocol_id") != "task4-polyp-primary-v2":
                raise ValueError("data.protocol_id must be task4-polyp-primary-v2")
            if self.data.get("dataset_revision") != "task4-polyp-bbox-v1-approved":
                raise ValueError(
                    "the AR baseline must use the approved Task4 polyp revision"
                )
            approved_payload_sha256 = str(self.data.get("approved_payload_sha256", ""))
            if _SHA256_PATTERN.fullmatch(approved_payload_sha256) is None:
                raise ValueError(
                    "data.approved_payload_sha256 must be a 64-character SHA-256"
                )
            filenames = [str(self.data[key]) for key in _VIEW_KEYS]
            if len(set(filenames)) != len(filenames):
                raise ValueError("all five experiment COCO views must be distinct")
        else:
            if self.data.get("protocol_id") != SCENARIO_STATIC_PROTOCOL_ID:
                raise ValueError(
                    f"data.protocol_id must be {SCENARIO_STATIC_PROTOCOL_ID}"
                )
            configured_hash = str(self.data.get("source_study_config_sha256", ""))
            if _SHA256_PATTERN.fullmatch(configured_hash) is None:
                raise ValueError("data.source_study_config_sha256 must be a SHA-256")
            contract = self.scenario_contract
            if contract.source_config_sha256 != configured_hash:
                raise ValueError("paper block-11 visual source config hash mismatch")
            if self.data.get("dataset_revision") != contract.dataset_revision:
                raise ValueError(
                    "Qwen scenario data revision differs from paper block-11 visual"
                )
            cache_required = self.data.get("cache_required", False)
            if not isinstance(cache_required, bool):
                raise TypeError("scenario-static data.cache_required must be a boolean")
            if self.uses_grounded_action and (not cache_required):
                raise ValueError("grounded action data.cache_required must be true")
            if cache_required and (
                self.cache_root is None or self.cache_manifest_path is None
            ):
                raise ValueError(
                    "cached scenario data must configure cache_root/cache_manifest"
                )
        if (
            int(self.data["original_width"]) != 640
            or int(self.data["original_height"]) != 480
        ):
            raise ValueError("Task4 cam_low geometry must remain 640x480")
        if int(self.training.get("epochs", 0)) != 1:
            raise ValueError("EndoVLA-style SFT must run for exactly one epoch")
        expected_batch_size = 1 if self.uses_grounded_action else 2
        if int(self.training.get("batch_size", 0)) != expected_batch_size:
            raise ValueError(
                f"training.batch_size must be {expected_batch_size} for {self.output_contract}"
            )
        if int(self.training.get("gradient_accumulation_steps", 0)) <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if str(self.training.get("precision", "")).lower() not in {"bf16", "bfloat16"}:
            raise ValueError("training.precision must be BF16")
        if float(self.training.get("learning_rate", -1)) != 0.0002:
            raise ValueError("the reported EndoVLA SFT learning rate is 2e-4")
        if self.training.get("optimizer") != "adamw":
            raise ValueError("training.optimizer must be adamw")
        if self.training.get("schedule") != "linear":
            raise ValueError("training.schedule must be linear")
        if not 0.0 <= float(self.training.get("warmup_fraction", -1)) < 1.0:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if int(self.training.get("max_sequence_length", 0)) <= 0:
            raise ValueError("training.max_sequence_length must be positive")
        if self.uses_grounded_action:
            if int(self.training["max_sequence_length"]) < 2304:
                raise ValueError(
                    "four-camera grounded action training requires at least 2304 tokens"
                )
            if int(self.training["gradient_accumulation_steps"]) != 2:
                raise ValueError(
                    "four-camera SFT gradient_accumulation_steps must be 2"
                )
            if float(self.training.get("action_numeric_loss_weight", -1)) != 0.1:
                raise ValueError("training.action_numeric_loss_weight must be 0.1")
        if int(self.training.get("seed", -1)) < 0:
            raise ValueError("training.seed must be non-negative")
        if self.training.get("balanced_sampling") is not True:
            raise ValueError(
                "balanced_sampling must remain enabled for the detector comparison"
            )
        if self.generation.get("do_sample") is not False:
            raise ValueError("bbox generation must be deterministic")
        if int(self.generation.get("num_beams", 0)) != 1:
            raise ValueError("bbox generation must use greedy decoding (num_beams=1)")
        if self.generation.get("force_positive_bbox_candidate") is not True:
            raise ValueError(
                "force_positive_bbox_candidate must be true so AP has one scored candidate per frame"
            )
        if int(self.generation.get("max_new_tokens", 0)) <= 0:
            raise ValueError("generation.max_new_tokens must be positive")
        if self.uses_grounded_action and int(self.generation["max_new_tokens"]) < 224:
            raise ValueError(
                "grounded 16D action generation requires at least 224 new tokens"
            )
        if (
            float(
                self.evaluation.get("max_false_positives_per_100_negative_frames", -1)
            )
            < 0
        ):
            raise ValueError("the false-positive budget must be non-negative")


def _resolve_image_size(image_size: Sequence[int]) -> ImageSize:
    if isinstance(image_size, str | bytes) or len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = image_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or (not isinstance(width, Integral))
        or (not isinstance(height, Integral))
    ):
        raise TypeError("image_size values must be integers")
    resolved = (int(width), int(height))
    if min(resolved) <= 0:
        raise ValueError("image_size values must be positive")
    return resolved


GROUNDED_ACTION_PROMPT_TEMPLATE = "Use the following four synchronized robot camera views in their labeled order.\nDetect the Task4 polyp in cam_low and output the same-frame absolute continuous 16D action recorded in the dataset.\nReturn exactly one line in one of these forms:\nN | A [a0, a1, ..., a15]\nY [x, y, w, h] | A [a0, a1, ..., a15]\nUse N only when no polyp is present in cam_low. For Y, use integer pixel coordinates in the original cam_low {width}x{height} image with top-left xywh convention. Do not use coordinates from another camera or normalized, resized, or padded coordinates. A must contain exactly 16 finite values in source action order, each written with exactly six digits after the decimal point. Return no explanation or additional text."


def format_grounded_action_prompt(
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
) -> str:
    """Render the active four-camera bbox/action output contract."""
    width, height = _resolve_image_size(image_size)
    return GROUNDED_ACTION_PROMPT_TEMPLATE.format(width=width, height=height)


DEFAULT_GROUNDED_ACTION_PROMPT = format_grounded_action_prompt()
BBOX_CAMERA_KEY = "observation.images.cam_low"
QWEN_CAMERA_ORDER = (
    "observation.images.cam_high",
    BBOX_CAMERA_KEY,
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def resolve_camera_input_sizes(
    value: Mapping[str, Sequence[int]] | None,
    *,
    source_image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, ImageSize]:
    """Validate deterministic per-camera processor input geometry.

    The supervised ``cam_low`` frame may never be resized because bbox text is
    expressed in its original pixels.  Auxiliary views may be downsampled by a
    checkpoint-bound config while the lossless cache remains at source size.
    """
    source_size = _resolve_image_size(source_image_size)
    if value is None:
        return {key: source_size for key in QWEN_CAMERA_ORDER}
    if not isinstance(value, Mapping) or set(value) != set(QWEN_CAMERA_ORDER):
        raise ValueError(f"camera_input_sizes must bind exactly {QWEN_CAMERA_ORDER!r}")
    result = {key: _resolve_image_size(value[key]) for key in QWEN_CAMERA_ORDER}
    if result[BBOX_CAMERA_KEY] != source_size:
        raise ValueError(
            f"{BBOX_CAMERA_KEY} input size must remain source geometry {source_size}"
        )
    return result


CAMERA_LABELS = {
    "observation.images.cam_high": "cam_high",
    BBOX_CAMERA_KEY: "cam_low (bbox coordinate target)",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}


def build_multicamera_qwen_messages(
    images: Sequence[Image.Image],
    *,
    answer: str | None = None,
    prompt: str = DEFAULT_GROUNDED_ACTION_PROMPT,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    camera_order: Sequence[str] = QWEN_CAMERA_ORDER,
    camera_input_sizes: Mapping[str, Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """Build one semantically labeled, ordered four-camera Qwen chat."""
    resolved_size = _resolve_image_size(image_size)
    order = tuple((str(value) for value in camera_order))
    if order != QWEN_CAMERA_ORDER:
        raise ValueError(f"camera_order must be exactly {QWEN_CAMERA_ORDER!r}")
    if isinstance(images, Image.Image | str | bytes) or len(images) != len(order):
        raise ValueError(
            f"images must contain exactly {len(order)} ordered camera frames"
        )
    expected_sizes = resolve_camera_input_sizes(
        camera_input_sizes, source_image_size=resolved_size
    )
    resolved_images: list[Image.Image] = []
    for key, image in zip(order, images, strict=True):
        if not isinstance(image, Image.Image):
            raise TypeError(f"{key} must be a PIL.Image.Image")
        expected_size = expected_sizes[key]
        if image.mode != "RGB" or image.size != expected_size:
            raise ValueError(
                f"{key} must be RGB at {expected_size}, got {image.mode} {image.size}"
            )
        resolved_images.append(image)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    content: list[dict[str, Any]] = []
    for key, image in zip(order, resolved_images, strict=True):
        content.extend(
            (
                {"type": "text", "text": f"{CAMERA_LABELS[key]}:\n"},
                {"type": "image", "image": image},
            )
        )
    content.append({"type": "text", "text": prompt})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        parsed = parse_grounded_action_output(answer, image_size=resolved_size)
        if not parsed.format_valid or parsed.canonical_text != answer:
            raise ValueError("answer must be a canonical grounded bbox/action target")
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


POLYP_PROMPT_TEMPLATE = "Detect the Task4 polyp in this endoscopic image.\nReturn exactly one line in one of these forms:\nN\nY [x, y, w, h]\nUse N only when no polyp is present. For Y, use integer pixel coordinates in the original {width}x{height} image with top-left xywh convention. Do not use normalized, resized, or padded coordinates. Return no explanation or additional text."


def format_polyp_prompt(image_size: Sequence[int] = DEFAULT_IMAGE_SIZE) -> str:
    """Render the fixed output contract into the user instruction."""
    width, height = _resolve_image_size(image_size)
    return POLYP_PROMPT_TEMPLATE.format(width=width, height=height)


DEFAULT_POLYP_PROMPT = format_polyp_prompt()


def build_qwen_messages(
    image: Image.Image,
    *,
    answer: str | None = None,
    prompt: str = DEFAULT_POLYP_PROMPT,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
) -> list[dict[str, Any]]:
    """Build a Qwen-compatible user message and optional assistant target."""
    resolved_size = _resolve_image_size(image_size)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image.mode != "RGB":
        raise ValueError("image must be converted to RGB before building messages")
    if image.size != resolved_size:
        raise ValueError(f"image has size {image.size}, expected {resolved_size}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        parsed = parse_ar_output(answer, image_size=resolved_size)
        if not parsed.format_valid or parsed.canonical_text != answer:
            raise ValueError("answer must be canonical 'N' or 'Y [x, y, w, h]' text")
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def assistant_only_labels(
    input_ids: Sequence[int],
    prompt_input_ids: Sequence[int],
    *,
    attention_mask: Sequence[int] | None = None,
    ignore_index: int = -100,
) -> list[int]:
    """Mask a tokenized chat prefix without importing Transformers.

    The caller tokenizes the user chat with ``add_generation_prompt=True`` and
    the corresponding full user+assistant chat.  This helper fails closed if
    the former is not an exact prefix of the latter, preventing accidental
    supervision on image or instruction tokens.
    """
    full = [int(token) for token in input_ids]
    prompt = [int(token) for token in prompt_input_ids]
    if not full:
        raise ValueError("input_ids must not be empty")
    if not prompt:
        raise ValueError("prompt_input_ids must not be empty")
    if len(prompt) >= len(full):
        raise ValueError(
            "full input must contain at least one assistant token after the prompt"
        )
    if full[: len(prompt)] != prompt:
        raise ValueError("prompt_input_ids must be an exact prefix of input_ids")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")
    labels = list(full)
    labels[: len(prompt)] = [ignore_index] * len(prompt)
    if attention_mask is not None:
        mask = [int(value) for value in attention_mask]
        if len(mask) != len(full) or any((value not in (0, 1) for value in mask)):
            raise ValueError(
                "attention_mask must contain one 0/1 value per input token"
            )
        labels = [
            label if keep else ignore_index
            for label, keep in zip(labels, mask, strict=True)
        ]
    if all((label == ignore_index for label in labels)):
        raise ValueError("assistant-only masking left no supervised tokens")
    return labels


def _keep_pil(
    image: Image.Image, target: dict[str, Any]
) -> tuple[Image.Image, dict[str, Any]]:
    return (image, target)


class QwenPolypARDataset(Dataset[dict[str, Any]]):
    """Create SFT chat examples from one approved ``primary_v2`` COCO view.

    Non-trainable manifest states are always excluded.  Positive frames must
    have exactly one box and confirmed negatives must have none; every item is
    checked again at access time even when construction verification is
    disabled for a targeted diagnostic.
    """

    def __init__(
        self,
        coco_path: str | Path,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        split: str | None = None,
        prompt: str | None = None,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        verify: bool = True,
        enforce_manifest_split: bool = False,
    ) -> None:
        from annotation.prepare import AnnotationStatus
        from endo_bbox_vla.data import COCOPolypDataset

        self.image_size = _resolve_image_size(image_size)
        self.prompt = format_polyp_prompt(self.image_size) if prompt is None else prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.base_dataset = COCOPolypDataset(
            coco_path,
            manifest_path,
            image_root,
            split=split,
            statuses=(AnnotationStatus.POSITIVE, AnnotationStatus.CONFIRMED_NEGATIVE),
            transform=_keep_pil,
            verify=verify,
            enforce_manifest_split=enforce_manifest_split,
        )
        self.image_root = Path(image_root)
        allowed = {AnnotationStatus.POSITIVE, AnnotationStatus.CONFIRMED_NEGATIVE}
        for record in self.base_dataset.records:
            if record.status not in allowed:
                raise RuntimeError(
                    f"non-trainable status leaked into AR dataset: {record.status.value}"
                )
            if (record.width, record.height) != self.image_size:
                raise ValueError(
                    f"image {record.image_id} uses {(record.width, record.height)}, expected authoritative size {self.image_size}"
                )
        self.ignored_record_count = len(self.base_dataset.coco["images"]) - len(
            self.base_dataset.records
        )
        if self.ignored_record_count < 0:
            raise RuntimeError(
                "AR dataset selected more records than the COCO view contains"
            )

    @property
    def records(self) -> tuple[FrameRecord, ...]:
        return self.base_dataset.records

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from annotation.prepare import AnnotationStatus

        image, target, record = self.base_dataset[index]
        if (
            not isinstance(image, Image.Image)
            or image.mode != "RGB"
            or image.size != self.image_size
        ):
            raise RuntimeError(
                f"image {record.image_id} violated the RGB {self.image_size} contract"
            )
        boxes: Tensor = target["boxes"]
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise RuntimeError(f"image {record.image_id} boxes must have shape [N, 4]")
        is_positive = record.status == AnnotationStatus.POSITIVE
        expected_box_count = 1 if is_positive else 0
        if len(boxes) != expected_box_count:
            raise RuntimeError(
                f"image {record.image_id} status {record.status.value} requires {expected_box_count} box(es), got {len(boxes)}"
            )
        presence = float(target["presence"].item())
        if presence != float(is_positive):
            raise RuntimeError(
                f"image {record.image_id} presence disagrees with reviewed status"
            )
        bbox_xywh: FloatBBoxXYWH | None = None
        ar_bbox_xywh: IntBBoxXYWH | None = None
        if is_positive:
            bbox_xyxy = tuple((float(value) for value in boxes[0].tolist()))
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            ar_bbox_xywh = quantize_xywh(bbox_xywh, image_size=self.image_size)
        answer = serialize_ar_target(bbox_xywh, image_size=self.image_size)
        prompt_messages = build_qwen_messages(
            image, prompt=self.prompt, image_size=self.image_size
        )
        messages = [
            *prompt_messages,
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return {
            "image": image,
            "messages": messages,
            "prompt_messages": prompt_messages,
            "answer": answer,
            "present": is_positive,
            "bbox_xywh": bbox_xywh,
            "ar_bbox_xywh": ar_bbox_xywh,
            "target": target,
            "record": record,
            "image_id": record.image_id,
            "image_path": self.image_root / record.image_path,
        }

    def balanced_sample_weights(self, *, balance_groups: bool = True) -> Tensor:
        return self.base_dataset.balanced_sample_weights(balance_groups=balance_groups)


def rgb_pil_from_sample(
    value: Any, *, image_size: ImageSize, camera_key: str = "camera"
) -> Image.Image:
    """Canonicalize a decoded source frame to RGB uint8 without resizing."""
    if isinstance(value, Image.Image):
        if value.mode != "RGB":
            raise ValueError(
                f"{camera_key} PIL image must use RGB mode, got {value.mode!r}"
            )
        image = value.copy()
    elif isinstance(value, Tensor) or hasattr(value, "shape"):
        tensor = (
            value.detach().cpu()
            if isinstance(value, Tensor)
            else torch.as_tensor(value)
        )
        if tuple(tensor.shape) == (3, image_size[1], image_size[0]):
            tensor = tensor.permute(1, 2, 0)
        elif tuple(tensor.shape) != (image_size[1], image_size[0], 3):
            raise ValueError(
                f"{camera_key} must be CHW or HWC RGB at {image_size}, got {tuple(tensor.shape)}"
            )
        if tensor.is_floating_point():
            if not bool(tensor.isfinite().all()):
                raise ValueError(f"{camera_key} contains NaN or infinity")
            if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
                raise ValueError(f"floating {camera_key} must be in [0, 1]")
            tensor = tensor.mul(255).round().to(dtype=torch.uint8)
        elif tensor.dtype != torch.uint8:
            raise TypeError(f"integer {camera_key} must use uint8")
        image = Image.fromarray(tensor.numpy(), mode="RGB")
    else:
        raise TypeError(f"{camera_key} must be a PIL image, tensor, or ndarray")
    if image.size != image_size:
        raise ValueError(f"{camera_key} has size {image.size}, expected {image_size}")
    return image


def coerce_action_tensor(value: Any) -> Tensor:
    """Validate and canonicalize one source action to finite float32 ``[16]``."""
    action = torch.tensor(validate_action(value), dtype=torch.float32)
    if tuple(action.shape) != (ACTION_DIM,) or not bool(torch.isfinite(action).all()):
        raise RuntimeError("canonical action tensor violated the finite 16D contract")
    return action


CACHE_MANIFEST_FILENAME = "cache_manifest.json"
CACHE_COMPLETE_FILENAME = "CACHE_COMPLETE.json"


def _safe_payload_path(root: Path, value: Any, *, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes cache root") from exc
    return resolved


_cache_SHA256_PATTERN = re.compile("[0-9a-f]{64}\\Z")


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if _cache_SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


ACTION_KEY = "action"
CACHE_SCHEMA_VERSION = 1


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


CACHE_KIND = "bbox_vla_qwen_four_camera_action_cache"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class LosslessARFrameCache:
    """Sparse frame source with eager verification of every cached payload."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_dataset_repo: str | None = None,
        expected_dataset_revision: str | None = None,
        expected_view: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_completion_sha256: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        completion_path = self.root / CACHE_COMPLETE_FILENAME
        if not completion_path.is_file():
            raise FileNotFoundError(
                f"cache has no completion authority: {completion_path}"
            )
        completion_bytes = completion_path.read_bytes()
        if expected_completion_sha256 is not None and _sha256_bytes(
            completion_bytes
        ) != _require_sha256(
            expected_completion_sha256, name="expected_completion_sha256"
        ):
            raise ValueError("cache completion SHA-256 mismatch")
        completion = json.loads(completion_bytes)
        if (
            completion.get("schema_version") != CACHE_SCHEMA_VERSION
            or completion.get("kind") != f"{CACHE_KIND}_complete"
            or completion.get("state") != "complete"
        ):
            raise ValueError("cache completion authority is invalid")
        cache_manifest_path = _safe_payload_path(
            self.root,
            completion.get("cache_manifest_file"),
            name="completion.cache_manifest_file",
        )
        cache_manifest_sha256 = _require_sha256(
            completion.get("cache_manifest_sha256"),
            name="completion.cache_manifest_sha256",
        )
        if _sha256_file(cache_manifest_path) != cache_manifest_sha256:
            raise ValueError("cache manifest SHA-256 mismatch")
        metadata = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != CACHE_SCHEMA_VERSION
            or metadata.get("kind") != CACHE_KIND
        ):
            raise ValueError("cache manifest contract is invalid")
        expected_values = {
            "dataset_repo": expected_dataset_repo,
            "dataset_revision": expected_dataset_revision,
            "view": expected_view,
            "source_manifest_sha256": expected_manifest_sha256,
        }
        mismatches = {
            name: {"expected": expected, "actual": metadata.get(name)}
            for name, expected in expected_values.items()
            if expected is not None and metadata.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"cache source identity mismatch: {mismatches}")
        if tuple(metadata.get("camera_order", ())) != QWEN_CAMERA_ORDER:
            raise ValueError("cache camera order mismatch")
        if metadata.get("bbox_camera_key") != BBOX_CAMERA_KEY:
            raise ValueError("cache bbox camera mismatch")
        if metadata.get("action_key") != ACTION_KEY or metadata.get("action_shape") != [
            16
        ]:
            raise ValueError("cache action contract mismatch")
        if metadata.get("action_dtype") != "float32":
            raise ValueError("cache action dtype mismatch")
        if metadata.get("pixel_hash_encoding") != "sha256_rgb_uint8_hwc_c_order":
            raise ValueError("cache pixel hash contract mismatch")
        image_size = tuple(metadata.get("image_size", ()))
        if len(image_size) != 2 or any(
            (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in image_size
            )
        ):
            raise ValueError("cache image_size is invalid")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.repo_id = str(metadata["dataset_repo"])
        self.revision = str(metadata["dataset_revision"])
        self.view = str(metadata["view"])
        self.source_manifest_sha256 = _require_sha256(
            metadata["source_manifest_sha256"], name="metadata.source_manifest_sha256"
        )
        self.source_length = int(metadata["source_length"])
        if self.source_length <= 0:
            raise ValueError("cache source_length must be positive")
        records_path = _safe_payload_path(
            self.root, metadata.get("records_file"), name="metadata.records_file"
        )
        records_sha256 = _require_sha256(
            metadata.get("records_sha256"), name="metadata.records_sha256"
        )
        if records_sha256 != completion.get("records_sha256"):
            raise ValueError("metadata and completion records hashes disagree")
        if _sha256_file(records_path) != records_sha256:
            raise ValueError("cache records manifest SHA-256 mismatch")
        self.entries: dict[int, dict[str, Any]] = {}
        with records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank cache manifest row at line {line_number}")
                entry = json.loads(line)
                dataset_index = int(entry["dataset_index"])
                if dataset_index in self.entries:
                    raise ValueError(f"duplicate cached dataset_index {dataset_index}")
                if not 0 <= dataset_index < self.source_length:
                    raise ValueError(
                        f"cached dataset_index {dataset_index} is outside source"
                    )
                self._verify_entry(entry, line_number=line_number)
                self.entries[dataset_index] = entry
        record_count = int(metadata["record_count"])
        if record_count != len(self.entries) or record_count != int(
            completion["record_count"]
        ):
            raise ValueError("cache record count mismatch")
        self.features = {
            key: {
                "dtype": "image",
                "shape": [self.image_size[1], self.image_size[0], 3],
            }
            for key in QWEN_CAMERA_ORDER
        }
        self.features[ACTION_KEY] = {"dtype": "float32", "shape": [16]}

    def __len__(self) -> int:
        return self.source_length

    def _verify_payload(self, payload: Mapping[str, Any], *, name: str) -> Path:
        path = _safe_payload_path(self.root, payload.get("path"), name=f"{name}.path")
        if not path.is_file():
            raise FileNotFoundError(f"missing cached payload: {path}")
        expected_size = int(payload.get("bytes", -1))
        if expected_size < 0 or path.stat().st_size != expected_size:
            raise ValueError(f"{name} byte-size mismatch")
        expected_hash = _require_sha256(payload.get("sha256"), name=f"{name}.sha256")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"{name} SHA-256 mismatch")
        return path

    def _verify_entry(self, entry: Mapping[str, Any], *, line_number: int) -> None:
        cameras = entry.get("cameras")
        if not isinstance(cameras, Mapping) or set(cameras) != set(QWEN_CAMERA_ORDER):
            raise ValueError(f"cache row {line_number} camera order mismatch")
        for camera_key in QWEN_CAMERA_ORDER:
            payload = cameras[camera_key]
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"cache row {line_number} {camera_key} payload must be a mapping"
                )
            image_path = self._verify_payload(
                payload, name=f"row {line_number} {camera_key}"
            )
            if payload.get("pixel_dtype") != "uint8" or payload.get("pixel_shape") != [
                self.image_size[1],
                self.image_size[0],
                3,
            ]:
                raise ValueError(
                    f"cache row {line_number} {camera_key} pixel metadata mismatch"
                )
            expected_pixel_hash = _require_sha256(
                payload.get("pixel_bytes_sha256"),
                name=f"row {line_number} {camera_key}.pixel_bytes_sha256",
            )
            with Image.open(image_path) as decoded:
                decoded.load()
                if decoded.mode != "RGB" or decoded.size != self.image_size:
                    raise ValueError(
                        f"cache row {line_number} {camera_key} decoded geometry mismatch"
                    )
                pixels = np.asarray(decoded, dtype=np.uint8)
            if (
                _sha256_bytes(np.ascontiguousarray(pixels).tobytes(order="C"))
                != expected_pixel_hash
            ):
                raise ValueError(
                    f"cache row {line_number} {camera_key} pixel SHA-256 mismatch"
                )
        action = entry.get("action")
        if not isinstance(action, Mapping):
            raise TypeError(f"cache row {line_number} action payload must be a mapping")
        if action.get("dtype") != "float32" or action.get("shape") != [16]:
            raise ValueError(f"cache row {line_number} action metadata mismatch")
        action_path = self._verify_payload(action, name=f"row {line_number} action")
        expected_array_hash = _require_sha256(
            action.get("array_bytes_sha256"),
            name=f"row {line_number} action.array_bytes_sha256",
        )
        with action_path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
        if (
            array.dtype != np.float32
            or array.shape != (16,)
            or (not np.isfinite(array).all())
        ):
            raise ValueError(f"cache row {line_number} action array contract mismatch")
        if _sha256_bytes(array.tobytes(order="C")) != expected_array_hash:
            raise ValueError(
                f"cache row {line_number} action array-byte SHA-256 mismatch"
            )

    def __getitem__(self, dataset_index: int) -> Mapping[str, Any]:
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
            raise TypeError("cache dataset_index must be an integer")
        try:
            entry = self.entries[dataset_index]
        except KeyError as exc:
            raise KeyError(
                f"dataset_index {dataset_index} is not materialized in this cache"
            ) from exc
        cameras: dict[str, Image.Image] = {}
        for camera_key in QWEN_CAMERA_ORDER:
            path = _safe_payload_path(
                self.root,
                entry["cameras"][camera_key]["path"],
                name=f"{camera_key}.path",
            )
            with Image.open(path) as source_image:
                source_image.load()
                if source_image.mode != "RGB" or source_image.size != self.image_size:
                    raise ValueError(
                        f"cached {camera_key} decoded with wrong mode or size"
                    )
                cameras[camera_key] = source_image.copy()
        action_path = _safe_payload_path(
            self.root, entry["action"]["path"], name="action.path"
        )
        with action_path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
        if (
            array.dtype != np.float32
            or array.shape != (16,)
            or (not np.isfinite(array).all())
        ):
            raise ValueError("cached action violates finite float32 [16] contract")
        return {
            **cameras,
            ACTION_KEY: torch.from_numpy(array.copy()),
            "index": torch.tensor(dataset_index, dtype=torch.int64),
            "episode_index": torch.tensor(
                int(entry["episode_index"]), dtype=torch.int64
            ),
            "frame_index": torch.tensor(int(entry["frame_index"]), dtype=torch.int64),
            "timestamp": torch.tensor(float(entry["timestamp"]), dtype=torch.float32),
        }


@dataclass(frozen=True, slots=True)
class CacheViewBinding:
    """Resolved child-cache identity from either a view root or bundle root."""

    root: Path
    view_manifest_sha256: str
    completion_sha256: str | None
    aggregate_manifest_sha256: str | None


CACHE_VIEW_ORDER = ("train", "validation", "test")


def _json_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return value


CACHE_BUNDLE_KIND = "bbox_vla_qwen_four_camera_action_cache_bundle"


def resolve_cache_view_binding(
    cache_root: str | Path,
    view: str,
    *,
    manifest_filename: str = CACHE_MANIFEST_FILENAME,
) -> CacheViewBinding:
    """Resolve and hash-bind a child view from the configured cache root."""
    if manifest_filename != CACHE_MANIFEST_FILENAME:
        raise ValueError(
            f"manifest_filename must be {CACHE_MANIFEST_FILENAME!r}, got {manifest_filename!r}"
        )
    if view not in CACHE_VIEW_ORDER:
        raise ValueError(f"unsupported cache view: {view!r}")
    root = Path(cache_root).resolve()
    manifest_path = root / manifest_filename
    if not manifest_path.is_file():
        if (root / view).exists():
            raise FileNotFoundError(
                f"cache bundle is missing aggregate identity: {manifest_path}"
            )
        raise FileNotFoundError(f"cache manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _json_mapping(manifest_path, name="cache manifest")
    kind = manifest.get("kind")
    if kind == CACHE_KIND:
        if manifest.get("view") != view:
            raise ValueError(
                f"direct cache contains view {manifest.get('view')!r}, expected {view!r}"
            )
        return CacheViewBinding(
            root=root,
            view_manifest_sha256=_sha256_bytes(manifest_bytes),
            completion_sha256=None,
            aggregate_manifest_sha256=None,
        )
    if (
        kind != CACHE_BUNDLE_KIND
        or manifest.get("schema_version") != CACHE_SCHEMA_VERSION
    ):
        raise ValueError(
            "cache manifest is neither a valid view nor aggregate contract"
        )
    if tuple(manifest.get("view_order", ())) != CACHE_VIEW_ORDER:
        raise ValueError("cache aggregate view order mismatch")
    if tuple(manifest.get("camera_order", ())) != QWEN_CAMERA_ORDER:
        raise ValueError("cache aggregate camera order mismatch")
    if (
        manifest.get("bbox_camera_key") != BBOX_CAMERA_KEY
        or manifest.get("pixel_hash_encoding") != "sha256_rgb_uint8_hwc_c_order"
        or manifest.get("action_key") != ACTION_KEY
        or (manifest.get("action_dtype") != "float32")
        or (manifest.get("action_shape") != [16])
    ):
        raise ValueError("cache aggregate camera/action contract mismatch")
    views = manifest.get("views")
    if not isinstance(views, Mapping) or set(views) != set(CACHE_VIEW_ORDER):
        raise ValueError("cache aggregate must bind exactly train/validation/test")
    binding = views[view]
    if not isinstance(binding, Mapping):
        raise TypeError(f"cache aggregate {view} binding must be a mapping")
    expected_manifest_relative = f"{view}/{CACHE_MANIFEST_FILENAME}"
    expected_completion_relative = f"{view}/{CACHE_COMPLETE_FILENAME}"
    if binding.get("cache_manifest") != expected_manifest_relative:
        raise ValueError(f"cache aggregate {view} manifest path mismatch")
    if binding.get("completion") != expected_completion_relative:
        raise ValueError(f"cache aggregate {view} completion path mismatch")
    child_manifest_path = _safe_payload_path(
        root, binding["cache_manifest"], name=f"aggregate.views.{view}.cache_manifest"
    )
    child_completion_path = _safe_payload_path(
        root, binding["completion"], name=f"aggregate.views.{view}.completion"
    )
    expected_child_manifest_sha256 = _require_sha256(
        binding.get("cache_manifest_sha256"),
        name=f"aggregate.views.{view}.cache_manifest_sha256",
    )
    expected_completion_sha256 = _require_sha256(
        binding.get("completion_sha256"),
        name=f"aggregate.views.{view}.completion_sha256",
    )
    if _sha256_file(child_manifest_path) != expected_child_manifest_sha256:
        raise ValueError(f"cache aggregate {view} child manifest SHA-256 mismatch")
    if _sha256_file(child_completion_path) != expected_completion_sha256:
        raise ValueError(f"cache aggregate {view} completion SHA-256 mismatch")
    child_manifest = _json_mapping(
        child_manifest_path, name=f"aggregate {view} child manifest"
    )
    aggregate_child_identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "view": view,
        "dataset_repo": manifest.get("dataset_repo"),
        "dataset_revision": manifest.get("dataset_revision"),
        "source_manifest_sha256": binding.get("source_manifest_sha256"),
        "camera_order": list(QWEN_CAMERA_ORDER),
        "bbox_camera_key": BBOX_CAMERA_KEY,
        "action_key": ACTION_KEY,
        "action_dtype": "float32",
        "action_shape": [16],
        "pixel_hash_encoding": "sha256_rgb_uint8_hwc_c_order",
    }
    if any(
        (
            child_manifest.get(key) != expected
            for key, expected in aggregate_child_identity.items()
        )
    ):
        raise ValueError(
            f"cache aggregate {view} identity disagrees with child manifest"
        )
    return CacheViewBinding(
        root=child_manifest_path.parent,
        view_manifest_sha256=expected_child_manifest_sha256,
        completion_sha256=expected_completion_sha256,
        aggregate_manifest_sha256=_sha256_bytes(manifest_bytes),
    )


class QwenScenarioStaticARDataset(Dataset[dict[str, Any]]):
    """Qwen view over the exact reviewed manifests used by block-11 visual.

    Legacy mode (the default) preserves the bbox-only, single-``cam_low``
    contract.  ``grounded_action=True`` activates four synchronized cameras
    and the same-frame absolute 16D dataset action.  Every bbox remains defined
    in original ``cam_low`` pixels; auxiliary views never change its frame.
    """

    def __init__(
        self,
        contract: ScenarioStaticDataContract,
        view: str,
        *,
        source: Any | None = None,
        cache_root: str | Path | None = None,
        cache_manifest: str = "cache_manifest.json",
        cache_completion_sha256: str | None = None,
        grounded_action: bool = False,
        camera_input_sizes: Mapping[str, Sequence[int]] | None = None,
        prompt: str | None = None,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        from endo_bbox_vla.data import LeRobotFrameSource

        if view not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported scenario-static Qwen view: {view!r}")
        self.contract = contract
        self.view = view
        self.image_size = _resolve_image_size(image_size)
        if not isinstance(grounded_action, bool):
            raise TypeError("grounded_action must be a boolean")
        self.grounded_action = grounded_action
        if camera_input_sizes is not None and (not self.grounded_action):
            raise ValueError("camera_input_sizes requires grounded_action=True")
        self.camera_input_sizes = resolve_camera_input_sizes(
            camera_input_sizes, source_image_size=self.image_size
        )
        if prompt is None:
            self.prompt = (
                format_grounded_action_prompt(self.image_size)
                if self.grounded_action
                else format_polyp_prompt(self.image_size)
            )
        else:
            self.prompt = prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.contract.camera_key != BBOX_CAMERA_KEY:
            raise ValueError(
                f"scenario-static bbox camera must be {BBOX_CAMERA_KEY!r}, got {self.contract.camera_key!r}"
            )
        self.records = contract.records(view)
        if any(
            (
                record.status not in {"positive", "confirmed_negative"}
                for record in self.records
            )
        ):
            raise ValueError(f"scenario-static {view} contains non-trainable records")
        if any(
            (
                (record.width, record.height) != self.image_size
                for record in self.records
            )
        ):
            raise ValueError(
                f"scenario-static {view} contains non-{self.image_size} frames"
            )
        if source is not None and cache_root is not None:
            raise ValueError("source and cache_root are mutually exclusive")
        if cache_root is not None:
            from comparisons.ar_data import (
                CACHE_MANIFEST_FILENAME as CACHE_MANIFEST_FILENAME,
            )
            from comparisons.ar_data import LosslessARFrameCache as LosslessARFrameCache
            from comparisons.ar_data import (
                resolve_cache_view_binding as resolve_cache_view_binding,
            )

            if cache_manifest != CACHE_MANIFEST_FILENAME:
                raise ValueError(
                    f"cache_manifest must be {CACHE_MANIFEST_FILENAME!r}, got {cache_manifest!r}"
                )
            binding = resolve_cache_view_binding(
                cache_root, view, manifest_filename=cache_manifest
            )
            if (
                cache_completion_sha256 is not None
                and binding.completion_sha256 is not None
                and (cache_completion_sha256 != binding.completion_sha256)
            ):
                raise ValueError(
                    "configured cache completion SHA-256 disagrees with aggregate binding"
                )
            self.source = LosslessARFrameCache(
                binding.root,
                expected_dataset_repo=contract.dataset_repo,
                expected_dataset_revision=contract.dataset_revision,
                expected_view=view,
                expected_manifest_sha256=contract.manifest_sha256[view],
                expected_completion_sha256=cache_completion_sha256
                or binding.completion_sha256,
            )
        else:
            self.source = (
                source
                if source is not None
                else LeRobotFrameSource(
                    contract.dataset_root,
                    repo_id=contract.dataset_repo,
                    revision=contract.dataset_revision,
                    video_backend=contract.video_backend,
                )
            )
        self.ignored_record_count = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from endo_bbox_vla.data import load_record_sample

        record = self.records[index]
        sample = load_record_sample(self.source, record)
        required = (
            (*QWEN_CAMERA_ORDER, ACTION_KEY)
            if self.grounded_action
            else (BBOX_CAMERA_KEY,)
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(
                f"source sample {record.dataset_index} lacks required Qwen fields: {missing}"
            )
        is_positive = record.status == "positive"
        bbox_xywh: FloatBBoxXYWH | None = None
        ar_bbox_xywh: IntBBoxXYWH | None = None
        if is_positive:
            if record.bbox_xyxy is None:
                raise RuntimeError(
                    f"positive frame {record.sample_id} has no reviewed bbox"
                )
            bbox_xywh = xyxy_to_xywh(record.bbox_xyxy)
            ar_bbox_xywh = quantize_xywh(bbox_xywh, image_size=self.image_size)
        elif record.bbox_xyxy is not None:
            raise RuntimeError(
                f"negative frame {record.sample_id} unexpectedly has a bbox"
            )
        boxes = (
            torch.tensor([record.bbox_xyxy], dtype=torch.float32)
            if record.bbox_xyxy is not None
            else torch.empty((0, 4), dtype=torch.float32)
        )
        if not self.grounded_action:
            image = rgb_pil_from_sample(
                sample[BBOX_CAMERA_KEY],
                image_size=self.image_size,
                camera_key=BBOX_CAMERA_KEY,
            )
            answer = serialize_ar_target(bbox_xywh, image_size=self.image_size)
            prompt_messages = build_qwen_messages(
                image, prompt=self.prompt, image_size=self.image_size
            )
            messages = [
                *prompt_messages,
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            return {
                "image": image,
                "messages": messages,
                "prompt_messages": prompt_messages,
                "answer": answer,
                "present": is_positive,
                "bbox_xywh": bbox_xywh,
                "ar_bbox_xywh": ar_bbox_xywh,
                "target": {
                    "boxes": boxes,
                    "presence": torch.tensor(float(is_positive), dtype=torch.float32),
                },
                "record": record,
                "image_id": record.image_id,
                "image_path": None,
            }
        source_camera_images = {
            key: rgb_pil_from_sample(
                sample[key], image_size=self.image_size, camera_key=key
            )
            for key in QWEN_CAMERA_ORDER
        }
        camera_images = {
            key: source_camera_images[key]
            if source_camera_images[key].size == self.camera_input_sizes[key]
            else source_camera_images[key].resize(
                self.camera_input_sizes[key], resample=Image.Resampling.BICUBIC
            )
            for key in QWEN_CAMERA_ORDER
        }
        images = tuple((camera_images[key] for key in QWEN_CAMERA_ORDER))
        action = coerce_action_tensor(sample[ACTION_KEY])
        answer = serialize_grounded_action_target(
            bbox_xywh, action, image_size=self.image_size
        )
        answer_spans = grounded_action_answer_spans(answer)
        prompt_messages = build_multicamera_qwen_messages(
            images,
            prompt=self.prompt,
            image_size=self.image_size,
            camera_input_sizes=self.camera_input_sizes,
        )
        messages = [
            *prompt_messages,
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return {
            "image": camera_images[BBOX_CAMERA_KEY],
            "images": images,
            "camera_images": camera_images,
            "camera_order": QWEN_CAMERA_ORDER,
            "camera_input_sizes": dict(self.camera_input_sizes),
            "bbox_camera_key": BBOX_CAMERA_KEY,
            "action": action,
            "messages": messages,
            "prompt_messages": prompt_messages,
            "answer": answer,
            "answer_spans": answer_spans.to_dict(),
            "action_numeric_spans": answer_spans.action_numeric,
            "loss_weight_segments": {
                "detection": answer_spans.detection_text,
                "action": answer_spans.action_text,
            },
            "present": is_positive,
            "bbox_xywh": bbox_xywh,
            "ar_bbox_xywh": ar_bbox_xywh,
            "target": {
                "boxes": boxes,
                "presence": torch.tensor(float(is_positive), dtype=torch.float32),
                "action": action.clone(),
            },
            "record": record,
            "image_id": record.image_id,
            "image_path": None,
        }

    def balanced_sample_weights(self, *, balance_groups: bool = True) -> Tensor:
        """Provide deterministic diagnostic weights without changing full-run sampling."""
        del balance_groups
        branches = [
            (record.status, record.source_dataset, record.group_id)
            for record in self.records
        ]
        counts: dict[tuple[str, str, int | None], int] = {}
        for branch in branches:
            counts[branch] = counts.get(branch, 0) + 1
        return torch.tensor(
            [1.0 / counts[branch] for branch in branches], dtype=torch.double
        )


@dataclass(frozen=True, slots=True)
class CacheMaterializationReport:
    root: str
    record_count: int
    completion_sha256: str
    records_sha256: str
    cache_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CacheBundleMaterializationReport:
    root: str
    aggregate_manifest_sha256: str
    views: Mapping[str, CacheMaterializationReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "aggregate_manifest_sha256": self.aggregate_manifest_sha256,
            "views": {view: self.views[view].to_dict() for view in CACHE_VIEW_ORDER},
        }


CACHE_RECORDS_FILENAME = "records.jsonl"


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _scalar_float(value: Any, *, name: str) -> float:
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError(f"{name} must contain one scalar")
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _lossless_source_action(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        source = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        source = torch.from_numpy(value)
    else:
        raise TypeError("cache source action must be a torch.Tensor or NumPy array")
    if source.dtype != torch.float32:
        raise TypeError(f"cache source action must use float32, got {source.dtype}")
    action = coerce_action_tensor(source)
    if not torch.equal(action, source):
        raise RuntimeError(
            "cache action canonicalization changed source float32 values"
        )
    return action


def _write_payload(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _npy_bytes(action: torch.Tensor) -> bytes:
    if action.dtype != torch.float32 or tuple(action.shape) != (16,):
        raise ValueError("cache NPY action source must be float32 [16]")
    buffer = io.BytesIO()
    source = action.detach().cpu().numpy()
    np.save(buffer, source, allow_pickle=False)
    payload = buffer.getvalue()
    decoded = np.load(io.BytesIO(payload), allow_pickle=False)
    if decoded.dtype != np.float32 or decoded.shape != (16,):
        raise RuntimeError("NPY cache round-trip changed action dtype or shape")
    if not np.array_equal(decoded, source):
        raise RuntimeError("NPY cache round-trip changed action values")
    return payload


_CAMERA_FILENAMES = {
    "observation.images.cam_high": "cam_high.png",
    "observation.images.cam_low": "cam_low.png",
    "observation.images.cam_left_wrist": "cam_left_wrist.png",
    "observation.images.cam_right_wrist": "cam_right_wrist.png",
}


def _png_bytes(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        raise ValueError("cache PNG source must be RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=1)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        decoded.load()
        if decoded.mode != "RGB" or decoded.size != image.size:
            raise RuntimeError("PNG cache round-trip changed image mode or size")
        if not np.array_equal(
            np.asarray(decoded, dtype=np.uint8), np.asarray(image, dtype=np.uint8)
        ):
            raise RuntimeError("PNG cache round-trip changed RGB pixels")
    return payload


def materialize_lossless_cache(
    destination: str | Path,
    *,
    source: Any,
    records: Sequence[Any],
    view: str,
    dataset_repo: str,
    dataset_revision: str,
    source_manifest_sha256: str,
    image_size: tuple[int, int] = (640, 480),
) -> CacheMaterializationReport:
    """Materialize sparse manifest records without importing LeRobot here.

    ``source`` must implement the ordinary frame-source ``__len__`` and
    ``__getitem__`` protocol.  Destination reuse is rejected so a completed
    cache can never be partially overwritten.
    """
    from endo_bbox_vla.data import load_record_sample

    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite cache path: {root}")
    if not records:
        raise ValueError("cache materialization requires at least one record")
    if not isinstance(view, str) or not view.strip():
        raise ValueError("view must be a non-empty string")
    if not isinstance(dataset_repo, str) or not dataset_repo.strip():
        raise ValueError("dataset_repo must be a non-empty string")
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError("dataset_revision must be a non-empty string")
    source_manifest_sha256 = _require_sha256(
        source_manifest_sha256, name="source_manifest_sha256"
    )
    source_length = len(source)
    if source_length <= 0:
        raise ValueError("source must contain at least one frame")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    root.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    seen_dataset_indices: set[int] = set()
    for ordinal, record in enumerate(records):
        dataset_index = int(record.dataset_index)
        if dataset_index in seen_dataset_indices:
            raise ValueError(
                f"duplicate dataset_index in cache records: {dataset_index}"
            )
        seen_dataset_indices.add(dataset_index)
        sample = load_record_sample(source, record)
        missing = [
            key
            for key in (*QWEN_CAMERA_ORDER, ACTION_KEY, "timestamp")
            if key not in sample
        ]
        if missing:
            raise KeyError(
                f"source sample {dataset_index} lacks cache fields: {missing}"
            )
        record_timestamp = _scalar_float(record.timestamp, name="record.timestamp")
        sample_timestamp = _scalar_float(sample["timestamp"], name="sample.timestamp")
        if not math.isclose(
            sample_timestamp, record_timestamp, rel_tol=0.0, abs_tol=1e-05
        ):
            raise ValueError(
                f"source timestamp mismatch for {record.sample_id}: expected {record_timestamp}, got {sample_timestamp}"
            )
        entry_root = root / "entries" / f"{ordinal:06d}"
        camera_payloads: dict[str, dict[str, Any]] = {}
        for camera_key in QWEN_CAMERA_ORDER:
            image = rgb_pil_from_sample(
                sample[camera_key], image_size=(width, height), camera_key=camera_key
            )
            pixels = np.asarray(image, dtype=np.uint8)
            if pixels.shape != (height, width, 3) or not pixels.flags.c_contiguous:
                pixels = np.ascontiguousarray(pixels)
            path = entry_root / _CAMERA_FILENAMES[camera_key]
            payload = _write_payload(path, _png_bytes(image))
            payload["path"] = path.relative_to(root).as_posix()
            payload.update(
                {
                    "pixel_dtype": "uint8",
                    "pixel_shape": [height, width, 3],
                    "pixel_bytes_sha256": _sha256_bytes(pixels.tobytes(order="C")),
                }
            )
            camera_payloads[camera_key] = payload
        action = _lossless_source_action(sample[ACTION_KEY])
        action_path = entry_root / "action.npy"
        action_payload = _write_payload(action_path, _npy_bytes(action))
        action_payload["path"] = action_path.relative_to(root).as_posix()
        action_payload.update(
            {
                "dtype": "float32",
                "shape": [16],
                "array_bytes_sha256": _sha256_bytes(
                    action.detach().cpu().numpy().tobytes(order="C")
                ),
            }
        )
        timestamp = record_timestamp
        if timestamp < 0:
            raise ValueError(f"record {record.sample_id} has invalid timestamp")
        entries.append(
            {
                "sample_id": str(record.sample_id),
                "image_id": int(record.image_id),
                "dataset_index": dataset_index,
                "episode_index": int(record.episode_index),
                "frame_index": int(record.frame_index),
                "timestamp": timestamp,
                "cameras": camera_payloads,
                "action": action_payload,
            }
        )
    records_path = root / CACHE_RECORDS_FILENAME
    records_bytes = b"".join((_json_bytes(entry) for entry in entries))
    records_path.write_bytes(records_bytes)
    records_sha256 = _sha256_bytes(records_bytes)
    cache_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "view": view,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "source_length": source_length,
        "source_manifest_sha256": source_manifest_sha256,
        "record_count": len(entries),
        "camera_order": list(QWEN_CAMERA_ORDER),
        "bbox_camera_key": BBOX_CAMERA_KEY,
        "image_size": [width, height],
        "image_encoding": "png_rgb_uint8_lossless",
        "pixel_hash_encoding": "sha256_rgb_uint8_hwc_c_order",
        "action_key": ACTION_KEY,
        "action_dtype": "float32",
        "action_shape": [16],
        "records_file": CACHE_RECORDS_FILENAME,
        "records_sha256": records_sha256,
    }
    cache_manifest_path = root / CACHE_MANIFEST_FILENAME
    cache_manifest_bytes = _json_bytes(cache_manifest)
    cache_manifest_path.write_bytes(cache_manifest_bytes)
    cache_manifest_sha256 = _sha256_bytes(cache_manifest_bytes)
    completion = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": f"{CACHE_KIND}_complete",
        "state": "complete",
        "record_count": len(entries),
        "cache_manifest_file": CACHE_MANIFEST_FILENAME,
        "cache_manifest_sha256": cache_manifest_sha256,
        "records_file": CACHE_RECORDS_FILENAME,
        "records_sha256": records_sha256,
    }
    completion_path = root / CACHE_COMPLETE_FILENAME
    completion_bytes = _json_bytes(completion)
    completion_path.write_bytes(completion_bytes)
    return CacheMaterializationReport(
        root=str(root),
        record_count=len(entries),
        completion_sha256=_sha256_bytes(completion_bytes),
        records_sha256=records_sha256,
        cache_manifest_sha256=cache_manifest_sha256,
    )


def write_cache_aggregate_manifest(
    cache_root: str | Path,
    *,
    dataset_repo: str,
    dataset_revision: str,
    source_manifest_sha256: Mapping[str, str],
) -> str:
    """Verify all three view caches and bind them into one deterministic identity.

    The aggregate manifest is written only after every child completion record,
    child manifest, records JSONL, and payload hash has been checked.  Its own
    SHA-256 is the split-complete cache identity that model metadata should pin.
    """
    root = Path(cache_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"cache bundle root does not exist: {root}")
    aggregate_path = root / CACHE_MANIFEST_FILENAME
    if aggregate_path.exists():
        raise FileExistsError(
            f"refusing to overwrite aggregate cache manifest: {aggregate_path}"
        )
    if not isinstance(dataset_repo, str) or not dataset_repo.strip():
        raise ValueError("dataset_repo must be a non-empty string")
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError("dataset_revision must be a non-empty string")
    if set(source_manifest_sha256) != set(CACHE_VIEW_ORDER):
        raise ValueError(
            f"source_manifest_sha256 must contain exactly {CACHE_VIEW_ORDER!r}"
        )
    view_bindings: dict[str, dict[str, Any]] = {}
    shared_source_length: int | None = None
    shared_image_size: list[int] | None = None
    for view in CACHE_VIEW_ORDER:
        expected_source_hash = _require_sha256(
            source_manifest_sha256[view], name=f"source_manifest_sha256[{view!r}]"
        )
        child_root = root / view
        child_manifest_path = child_root / CACHE_MANIFEST_FILENAME
        child_completion_path = child_root / CACHE_COMPLETE_FILENAME
        child_manifest_bytes = child_manifest_path.read_bytes()
        child_completion_bytes = child_completion_path.read_bytes()
        child_manifest_sha256 = _sha256_bytes(child_manifest_bytes)
        child_completion_sha256 = _sha256_bytes(child_completion_bytes)
        child_manifest = _json_mapping(
            child_manifest_path, name=f"{view} cache manifest"
        )
        expected_identity = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "view": view,
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "source_manifest_sha256": expected_source_hash,
            "camera_order": list(QWEN_CAMERA_ORDER),
            "bbox_camera_key": BBOX_CAMERA_KEY,
            "action_key": ACTION_KEY,
            "action_dtype": "float32",
            "action_shape": [16],
            "pixel_hash_encoding": "sha256_rgb_uint8_hwc_c_order",
        }
        mismatches = {
            key: {"expected": expected, "actual": child_manifest.get(key)}
            for key, expected in expected_identity.items()
            if child_manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"{view} child cache identity mismatch: {mismatches}")
        LosslessARFrameCache(
            child_root,
            expected_dataset_repo=dataset_repo,
            expected_dataset_revision=dataset_revision,
            expected_view=view,
            expected_manifest_sha256=expected_source_hash,
            expected_completion_sha256=child_completion_sha256,
        )
        source_length = int(child_manifest["source_length"])
        image_size = [int(value) for value in child_manifest["image_size"]]
        if shared_source_length is None:
            shared_source_length = source_length
            shared_image_size = image_size
        elif source_length != shared_source_length or image_size != shared_image_size:
            raise ValueError("child caches disagree on source length or image size")
        view_bindings[view] = {
            "source_manifest_sha256": expected_source_hash,
            "record_count": int(child_manifest["record_count"]),
            "cache_manifest": f"{view}/{CACHE_MANIFEST_FILENAME}",
            "cache_manifest_sha256": child_manifest_sha256,
            "completion": f"{view}/{CACHE_COMPLETE_FILENAME}",
            "completion_sha256": child_completion_sha256,
        }
    aggregate = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_BUNDLE_KIND,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "source_length": shared_source_length,
        "camera_order": list(QWEN_CAMERA_ORDER),
        "bbox_camera_key": BBOX_CAMERA_KEY,
        "image_size": shared_image_size,
        "image_encoding": "png_rgb_uint8_lossless",
        "pixel_hash_encoding": "sha256_rgb_uint8_hwc_c_order",
        "action_key": ACTION_KEY,
        "action_dtype": "float32",
        "action_shape": [16],
        "view_order": list(CACHE_VIEW_ORDER),
        "views": view_bindings,
    }
    aggregate_bytes = _json_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_bytes)
    return _sha256_bytes(aggregate_bytes)


def materialize_lossless_cache_bundle(
    destination: str | Path,
    *,
    source: Any,
    records_by_view: Mapping[str, Sequence[Any]],
    dataset_repo: str,
    dataset_revision: str,
    source_manifest_sha256: Mapping[str, str],
    image_size: tuple[int, int] = (640, 480),
) -> CacheBundleMaterializationReport:
    """Materialize train/validation/test children plus the aggregate identity."""
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite cache bundle path: {root}")
    if set(records_by_view) != set(CACHE_VIEW_ORDER):
        raise ValueError(f"records_by_view must contain exactly {CACHE_VIEW_ORDER!r}")
    if set(source_manifest_sha256) != set(CACHE_VIEW_ORDER):
        raise ValueError(
            f"source_manifest_sha256 must contain exactly {CACHE_VIEW_ORDER!r}"
        )
    root.mkdir(parents=True)
    reports: dict[str, CacheMaterializationReport] = {}
    for view in CACHE_VIEW_ORDER:
        reports[view] = materialize_lossless_cache(
            root / view,
            source=source,
            records=records_by_view[view],
            view=view,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            source_manifest_sha256=source_manifest_sha256[view],
            image_size=image_size,
        )
    aggregate_sha256 = write_cache_aggregate_manifest(
        root,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        source_manifest_sha256=source_manifest_sha256,
    )
    return CacheBundleMaterializationReport(
        root=str(root), aggregate_manifest_sha256=aggregate_sha256, views=reports
    )


def load_ar_config(path: str | Path) -> ARExperimentConfig:
    source_path = Path(path).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("AR experiment config must be a mapping")
    required = {"model", "data", "training", "generation", "evaluation"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    config = ARExperimentConfig(
        model=dict(raw["model"]),
        data=dict(raw["data"]),
        training=dict(raw["training"]),
        generation=dict(raw["generation"]),
        evaluation=dict(raw["evaluation"]),
        source_path=source_path,
    )
    config.validate()
    return config


@dataclass(frozen=True, slots=True)
class DataBindingReport:
    annotation_revision: str
    approved_payload_sha256: str
    protocol_id: str
    view_counts: dict[str, int]
    data_format: str = "primary_v2_coco"
    manifest_sha256: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_revision": self.annotation_revision,
            "approved_payload_sha256": self.approved_payload_sha256,
            "protocol_id": self.protocol_id,
            "view_counts": dict(self.view_counts),
            "data_format": self.data_format,
            "manifest_sha256": None
            if self.manifest_sha256 is None
            else dict(self.manifest_sha256),
        }


def verify_data_bindings(config: ARExperimentConfig) -> DataBindingReport:
    """Verify the append-only annotation and content-bound experiment views."""
    from annotation.prepare import verify_locked_revision
    from endo_bbox_vla.data import verify_primary_v2_views

    if config.data_format == "scenario_static_v1":
        contract = config.scenario_contract
        return DataBindingReport(
            annotation_revision=contract.dataset_revision,
            approved_payload_sha256=contract.source_config_sha256,
            protocol_id=contract.protocol_id,
            view_counts={
                name: counts["total"] for name, counts in contract.counts.items()
            },
            data_format=config.data_format,
            manifest_sha256=contract.manifest_sha256,
        )
    approval = verify_locked_revision(config.approved_root, verify_source=False)
    protocol = verify_primary_v2_views(config.protocol_root, config.approved_root)
    if approval.revision != config.data["dataset_revision"]:
        raise ValueError(
            "configured dataset revision does not match the approved revision"
        )
    if approval.approved_payload_sha256 != config.data["approved_payload_sha256"]:
        raise ValueError(
            "configured approved payload hash does not match the locked revision"
        )
    if protocol["annotation_revision"] != approval.revision:
        raise ValueError("protocol and approval annotation revisions differ")
    if protocol["approved_payload_sha256"] != approval.approved_payload_sha256:
        raise ValueError("protocol and approval payload hashes differ")
    if protocol["protocol_id"] != config.data["protocol_id"]:
        raise ValueError("configured protocol ID does not match the verified protocol")
    view_counts = {
        str(name): int(metadata["trainable_count"])
        for name, metadata in protocol["views"].items()
    }
    return DataBindingReport(
        annotation_revision=approval.revision,
        approved_payload_sha256=approval.approved_payload_sha256,
        protocol_id=str(protocol["protocol_id"]),
        view_counts=view_counts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for materializing the configured three-view grounded-action cache."""
    import argparse

    from comparisons.ar_data import load_ar_config as load_ar_config
    from comparisons.ar_data import verify_data_bindings as verify_data_bindings
    from endo_bbox_vla.data import LeRobotFrameSource as LeRobotFrameSource

    parser = argparse.ArgumentParser(
        description="Materialize the lossless block-11 visual four-camera continuous-action cache."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Override data.cache_root (destination must not already exist).",
    )
    args = parser.parse_args(argv)
    config = load_ar_config(args.config)
    verify_data_bindings(config)
    if config.data_format != "scenario_static_v1" or not config.uses_grounded_action:
        raise ValueError(
            "cache materialization requires a grounded scenario_static_v1 config"
        )
    destination = args.destination or config.cache_root
    if destination is None:
        raise ValueError("config must define data.cache_root or --destination")
    contract = config.scenario_contract
    source = LeRobotFrameSource(
        contract.dataset_root,
        repo_id=contract.dataset_repo,
        revision=contract.dataset_revision,
        video_backend=contract.video_backend,
    )
    report = materialize_lossless_cache_bundle(
        destination,
        source=source,
        records_by_view={view: contract.records(view) for view in CACHE_VIEW_ORDER},
        dataset_repo=contract.dataset_repo,
        dataset_revision=contract.dataset_revision,
        source_manifest_sha256={
            view: contract.manifest_sha256[view] for view in CACHE_VIEW_ORDER
        },
        image_size=(
            int(config.data["original_width"]),
            int(config.data["original_height"]),
        ),
    )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    return 0


def config_as_json(config: ARExperimentConfig) -> str:
    """Return the experiment payload in a deterministic, metadata-safe form."""
    payload = {
        "model": config.model,
        "data": config.data,
        "training": config.training,
        "generation": config.generation,
        "evaluation": config.evaluation,
        "config_sha256": config.config_sha256,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def cli(argv=None):
    from endo_bbox_vla.utils import run_cli

    return run_cli({"cache": main}, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
