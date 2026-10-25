"""comparisons — ar_evaluate: consolidated research implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from comparisons.ar_model import ARGenerationOutput
    from endo_bbox_vla.metrics import Detection, GroundTruth, SourceThresholdSelection
from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
type ImageId = str | int


@dataclass(frozen=True, slots=True)
class ARGenerationRecord:
    """One raw AR output associated with an original camera frame."""

    image_id: ImageId
    text: str
    present_score: float
    width: int = 640
    height: int = 480
    generated_tokens: int | None = None
    latency_ms: float | None = None
    target_action: tuple[float, ...] | None = None
    source_dataset: str | None = None
    source_role: str | None = None
    scenario: str | None = None
    source_episode_index: int | None = None
    source_frame_index: int | None = None

    @property
    def output(self) -> ARGenerationOutput:
        from comparisons.ar_model import ARGenerationOutput

        return ARGenerationOutput(
            text=self.text,
            present_score=self.present_score,
            generated_tokens=self.generated_tokens,
            latency_ms=self.latency_ms,
        )


type Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ARGenerationDiagnostic:
    """Parser result retained independently from thresholded detector metrics."""

    image_id: ImageId
    text: str
    present_score: float
    format_valid: bool
    parsed_present: bool | None
    bbox_xyxy: Box | None
    error: str | None
    canonical_text: str | None
    generated_tokens: int | None = None
    latency_ms: float | None = None
    strict_joint_format_expected: bool = False
    strict_joint_format_valid: bool | None = None
    action_parse_valid: bool | None = None
    predicted_action: tuple[float, ...] | None = None
    target_action: tuple[float, ...] | None = None
    action_mae: float | None = None
    source_dataset: str | None = None
    source_role: str | None = None
    scenario: str | None = None
    source_episode_index: int | None = None
    source_frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.bbox_xyxy is not None:
            payload["bbox_xyxy"] = list(self.bbox_xyxy)
        if self.predicted_action is not None:
            payload["predicted_action"] = list(self.predicted_action)
        if self.target_action is not None:
            payload["target_action"] = list(self.target_action)
        return payload


@dataclass(frozen=True, slots=True)
class ARConversionResult:
    """Raw detections plus format diagnostics for one logical split."""

    detections: tuple[Detection, ...]
    diagnostics: tuple[ARGenerationDiagnostic, ...]

    def summary(self) -> dict[str, Any]:
        total = len(self.diagnostics)
        format_valid_count = sum((item.format_valid for item in self.diagnostics))
        positive_count = sum(
            (
                item.format_valid and item.parsed_present is True
                for item in self.diagnostics
            )
        )
        negative_count = sum(
            (
                item.format_valid and item.parsed_present is False
                for item in self.diagnostics
            )
        )
        errors = Counter(
            (
                item.error or "invalid_parser_result"
                for item in self.diagnostics
                if not item.format_valid
            )
        )
        token_counts = [
            item.generated_tokens
            for item in self.diagnostics
            if item.generated_tokens is not None
        ]
        latencies = [
            item.latency_ms for item in self.diagnostics if item.latency_ms is not None
        ]
        joint = [item for item in self.diagnostics if item.strict_joint_format_expected]
        action_targets = [item for item in joint if item.target_action is not None]
        parsed_actions = [
            item
            for item in action_targets
            if item.action_parse_valid is True and item.predicted_action is not None
        ]
        action_summary: dict[str, Any] = {
            "target_count": len(action_targets),
            "parse_valid_count": len(parsed_actions),
            "parse_invalid_count": len(action_targets) - len(parsed_actions),
            "parse_valid_rate": len(parsed_actions) / len(action_targets)
            if action_targets
            else None,
            "mae": None,
            "nmae": None,
            "nmae_normalization": "per_dimension_target_range",
            "normalized_dimension_count": 0,
            "per_dimension_mae": None,
            "per_dimension_target_range": None,
        }
        if parsed_actions:
            targets = np.asarray(
                [item.target_action for item in action_targets], dtype=np.float64
            )
            paired_targets = np.asarray(
                [item.target_action for item in parsed_actions], dtype=np.float64
            )
            predictions = np.asarray(
                [item.predicted_action for item in parsed_actions], dtype=np.float64
            )
            absolute_error = np.abs(predictions - paired_targets)
            target_range = np.ptp(targets, axis=0)
            variable = target_range > np.finfo(np.float64).eps
            per_dimension_mae = np.mean(absolute_error, axis=0)
            action_summary.update(
                {
                    "mae": float(np.mean(absolute_error)),
                    "nmae": float(
                        np.mean(absolute_error[:, variable] / target_range[variable])
                    )
                    if np.any(variable)
                    else None,
                    "normalized_dimension_count": int(np.count_nonzero(variable)),
                    "per_dimension_mae": [float(value) for value in per_dimension_mae],
                    "per_dimension_target_range": [
                        float(value) for value in target_range
                    ],
                }
            )
        return {
            "generation_count": total,
            "format_valid_count": format_valid_count,
            "format_invalid_count": total - format_valid_count,
            "format_valid_rate": format_valid_count / total if total else 0.0,
            "positive_output_count": positive_count,
            "negative_output_count": negative_count,
            "candidate_detection_count": len(self.detections),
            "parse_errors": dict(sorted(errors.items())),
            "strict_joint_format": {
                "expected_count": len(joint),
                "valid_count": sum(
                    (item.strict_joint_format_valid is True for item in joint)
                ),
                "invalid_count": sum(
                    (item.strict_joint_format_valid is False for item in joint)
                ),
                "valid_rate": sum(
                    (item.strict_joint_format_valid is True for item in joint)
                )
                / len(joint)
                if joint
                else None,
            },
            "action": action_summary,
            "generated_tokens": {
                "observed_count": len(token_counts),
                "mean": float(np.mean(token_counts)) if token_counts else None,
                "max": max(token_counts, default=None),
            },
            "latency_ms": {
                "observed_count": len(latencies),
                "mean": float(np.mean(latencies)) if latencies else None,
                "p50": float(np.percentile(latencies, 50)) if latencies else None,
                "p95": float(np.percentile(latencies, 95)) if latencies else None,
            },
        }


def _get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _image_id(value: Any) -> ImageId:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"image_id must contain one value, got shape {array.shape}.")
    scalar = array.reshape(-1)[0].item()
    if not isinstance(scalar, str | int | np.integer):
        raise TypeError(
            f"image_id must be a string or integer, got {type(scalar).__name__}."
        )
    return int(scalar) if isinstance(scalar, np.integer) else scalar


def _optional_action(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    from comparisons.ar_data import validate_action as validate_action

    return tuple(validate_action(value))


def _optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string when supplied.")
    return value


def _optional_nonnegative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer when supplied.")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative when supplied.")
    return converted


def _coerce_record(
    record: ARGenerationRecord | Mapping[str, Any] | Any,
) -> ARGenerationRecord:
    from comparisons.ar_model import coerce_generation_output

    if isinstance(record, ARGenerationRecord):
        candidate = record
    else:
        image_id = _get(record, "image_id")
        if image_id is None:
            raise ValueError("Every AR generation record needs image_id.")
        nested_output = _get(record, "output")
        output = coerce_generation_output(
            record if nested_output is None else nested_output
        )
        image_size = _get(record, "image_size")
        if image_size is not None:
            dimensions = tuple((int(item) for item in image_size))
            if len(dimensions) != 2:
                raise ValueError("image_size must be (width, height).")
            width, height = dimensions
        else:
            width = int(_get(record, "width", 640))
            height = int(_get(record, "height", 480))
        candidate = ARGenerationRecord(
            image_id=_image_id(image_id),
            text=output.text,
            present_score=output.present_score,
            width=width,
            height=height,
            generated_tokens=output.generated_tokens,
            latency_ms=output.latency_ms,
            target_action=_optional_action(_get(record, "target_action")),
            source_dataset=_optional_text(
                _get(record, "source_dataset"), name="source_dataset"
            ),
            source_role=_optional_text(_get(record, "source_role"), name="source_role"),
            scenario=_optional_text(_get(record, "scenario"), name="scenario"),
            source_episode_index=_optional_nonnegative_int(
                _get(record, "source_episode_index"), name="source_episode_index"
            ),
            source_frame_index=_optional_nonnegative_int(
                _get(record, "source_frame_index"), name="source_frame_index"
            ),
        )
    validated_output = coerce_generation_output(candidate.output)
    if candidate.width <= 0 or candidate.height <= 0:
        raise ValueError("AR generation image dimensions must be positive.")
    return ARGenerationRecord(
        image_id=_image_id(candidate.image_id),
        text=validated_output.text,
        present_score=validated_output.present_score,
        width=int(candidate.width),
        height=int(candidate.height),
        generated_tokens=validated_output.generated_tokens,
        latency_ms=validated_output.latency_ms,
        target_action=_optional_action(candidate.target_action),
        source_dataset=_optional_text(candidate.source_dataset, name="source_dataset"),
        source_role=_optional_text(candidate.source_role, name="source_role"),
        scenario=_optional_text(candidate.scenario, name="scenario"),
        source_episode_index=_optional_nonnegative_int(
            candidate.source_episode_index, name="source_episode_index"
        ),
        source_frame_index=_optional_nonnegative_int(
            candidate.source_frame_index, name="source_frame_index"
        ),
    )


def _default_parse(text: str, *, image_size: tuple[int, int]) -> Any:
    from comparisons.ar_data import parse_ar_output as parse_ar_output

    return parse_ar_output(text, image_size=image_size)


def _parsed_action(parsed: Any) -> tuple[float, ...] | None:
    value = _get(parsed, "action")
    if value is None:
        return None
    try:
        return _optional_action(value)
    except (TypeError, ValueError):
        return None


type ParseCallable = Callable[..., Any]


def _parsed_box(parsed: Any, *, width: int, height: int) -> Box | None:
    value = _get(parsed, "bbox_xyxy")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    x1, y1, x2, y2 = (float(item) for item in array)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return (x1, y1, x2, y2)


def adapt_ar_generation(
    record: ARGenerationRecord | Mapping[str, Any] | Any,
    *,
    parser: ParseCallable | None = None,
) -> tuple[Detection | None, ARGenerationDiagnostic]:
    """Parse one raw generation and optionally create one scored detection."""
    from endo_bbox_vla.metrics import Detection

    generation = _coerce_record(record)
    parse = parser or _default_parse
    parsed = parse(generation.text, image_size=(generation.width, generation.height))
    format_valid = bool(_get(parsed, "format_valid", False))
    joint_expected = generation.target_action is not None
    predicted_action = _parsed_action(parsed)
    action_parse_valid = (
        format_valid and predicted_action is not None if joint_expected else None
    )
    present_value = _get(parsed, "present")
    parsed_present = present_value if isinstance(present_value, bool) else None
    bbox = (
        _parsed_box(parsed, width=generation.width, height=generation.height)
        if format_valid and parsed_present is True
        else None
    )
    error_value = _get(parsed, "error")
    error = None if error_value is None else str(error_value)
    if format_valid and parsed_present is None:
        format_valid = False
        error = error or "valid output has no Boolean presence decision"
    if format_valid and parsed_present is True and (bbox is None):
        format_valid = False
        error = error or "valid positive output has no usable in-bounds bbox"
    if (
        format_valid
        and parsed_present is False
        and (_get(parsed, "bbox_xyxy") is not None)
    ):
        format_valid = False
        error = error or "valid negative output unexpectedly contains a bbox"
    if joint_expected and (not action_parse_valid):
        format_valid = False
        error = error or "strict joint output has no valid 16D action"
    joint_format_valid = format_valid if joint_expected else None
    action_mae = (
        float(
            np.mean(
                np.abs(
                    np.asarray(predicted_action, dtype=np.float64)
                    - np.asarray(generation.target_action, dtype=np.float64)
                )
            )
        )
        if action_parse_valid
        and predicted_action is not None
        and (generation.target_action is not None)
        else None
    )
    detection = (
        Detection(generation.image_id, float(generation.present_score), bbox)
        if format_valid and parsed_present is True and (bbox is not None)
        else None
    )
    canonical_value = _get(parsed, "canonical_text")
    diagnostic = ARGenerationDiagnostic(
        image_id=generation.image_id,
        text=generation.text,
        present_score=float(generation.present_score),
        format_valid=format_valid,
        parsed_present=parsed_present,
        bbox_xyxy=bbox if format_valid else None,
        error=error,
        canonical_text=None if canonical_value is None else str(canonical_value),
        generated_tokens=generation.generated_tokens,
        latency_ms=generation.latency_ms,
        strict_joint_format_expected=joint_expected,
        strict_joint_format_valid=joint_format_valid,
        action_parse_valid=action_parse_valid,
        predicted_action=predicted_action if action_parse_valid else None,
        target_action=generation.target_action,
        action_mae=action_mae,
        source_dataset=generation.source_dataset,
        source_role=generation.source_role,
        scenario=generation.scenario,
        source_episode_index=generation.source_episode_index,
        source_frame_index=generation.source_frame_index,
    )
    return (detection, diagnostic)


def generation_to_detection(
    record: ARGenerationRecord | Mapping[str, Any] | Any,
    *,
    parser: ParseCallable | None = None,
) -> Detection | None:
    """Return only the shared-evaluator detection for one AR generation."""
    return adapt_ar_generation(record, parser=parser)[0]


def convert_ar_generations(
    records: Iterable[ARGenerationRecord | Mapping[str, Any] | Any],
    *,
    parser: ParseCallable | None = None,
) -> ARConversionResult:
    """Convert exactly one AR output per image into raw bbox detections."""
    detections: list[Detection] = []
    diagnostics: list[ARGenerationDiagnostic] = []
    seen: set[ImageId] = set()
    for record in records:
        detection, diagnostic = adapt_ar_generation(record, parser=parser)
        if diagnostic.image_id in seen:
            raise ValueError(
                f"More than one AR generation for image_id={diagnostic.image_id!r}."
            )
        seen.add(diagnostic.image_id)
        diagnostics.append(diagnostic)
        if detection is not None:
            detections.append(detection)
    return ARConversionResult(tuple(detections), tuple(diagnostics))


def convert_generation_records(
    records: Iterable[ARGenerationRecord | Mapping[str, Any] | Any],
    *,
    parser: ParseCallable | None = None,
) -> ARConversionResult:
    """Compatibility alias with an explicit record-oriented name."""
    return convert_ar_generations(records, parser=parser)


def _validate_generation_coverage(
    conversion: ARConversionResult,
    ground_truths: list[GroundTruth | Mapping[str, Any] | Any],
) -> None:
    generation_ids = {item.image_id for item in conversion.diagnostics}
    target_ids: list[ImageId] = []
    for target in ground_truths:
        image_id = _get(target, "image_id")
        if image_id is None:
            raise ValueError("Every ground-truth record needs image_id.")
        target_ids.append(_image_id(image_id))
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Ground truth must contain exactly one record per image_id.")
    expected_ids = set(target_ids)
    missing = sorted((str(value) for value in expected_ids - generation_ids))
    extra = sorted((str(value) for value in generation_ids - expected_ids))
    if missing or extra:
        raise ValueError(
            f"AR evaluation requires exactly one generation per ground-truth frame; missing={missing[:5]!r}, extra={extra[:5]!r}."
        )


def compute_ar_detection_metrics(
    generations: Iterable[ARGenerationRecord | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    confidence_threshold: float,
    parser: ParseCallable | None = None,
    **metric_kwargs: Any,
) -> dict[str, Any]:
    """Run the shared bbox metrics and append AR format/latency statistics."""
    from endo_bbox_vla.metrics import (
        compute_detection_metric_breakdowns,
        compute_detection_metrics,
    )

    conversion = convert_ar_generations(generations, parser=parser)
    ground_truth_list = list(ground_truths)
    _validate_generation_coverage(conversion, ground_truth_list)
    metrics = compute_detection_metrics(
        conversion.detections,
        ground_truth_list,
        confidence_threshold=confidence_threshold,
        **metric_kwargs,
    )
    metrics["ar_generation"] = conversion.summary()
    metadata = {
        item.image_id: {
            "source_dataset": item.source_dataset,
            "source_role": item.source_role,
            "scenario": item.scenario,
        }
        for item in conversion.diagnostics
    }
    metrics["breakdowns"] = compute_detection_metric_breakdowns(
        conversion.detections,
        ground_truth_list,
        metadata,
        confidence_threshold=confidence_threshold,
        **metric_kwargs,
    )
    return metrics


def select_ar_confidence_threshold(
    generations: Iterable[ARGenerationRecord | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    parser: ParseCallable | None = None,
    max_fp_per_100_confirmed_negative: float = 5.0,
    match_iou_threshold: float = 0.5,
) -> SourceThresholdSelection:
    """Fit the AR model's own threshold on the calibration split only."""
    from endo_bbox_vla.metrics import select_source_constrained_threshold

    conversion = convert_ar_generations(generations, parser=parser)
    ground_truth_list = list(ground_truths)
    _validate_generation_coverage(conversion, ground_truth_list)
    source_by_image = {
        item.image_id: item.source_dataset for item in conversion.diagnostics
    }
    return select_source_constrained_threshold(
        conversion.detections,
        ground_truth_list,
        source_by_image=source_by_image,
        max_fp_per_100_confirmed_negative=max_fp_per_100_confirmed_negative,
        match_iou_threshold=match_iou_threshold,
    )


AR_GENERATION_DIAGNOSTICS_SCHEMA_VERSION = 2


def write_ar_prediction_bundle(
    path: str | Path,
    *,
    split: str,
    generations: Iterable[ARGenerationRecord | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    model_identity: Mapping[str, Any],
    parser: ParseCallable | None = None,
    diagnostics_path: str | Path | None = None,
) -> Path:
    """Write an evaluator-compatible raw bundle and optional AR diagnostics.

    The primary file is created by :func:`bbox_vla.detr.evaluate.write_prediction_bundle`
    without extending its schema, so existing calibration/scoring commands can
    read it unchanged.  Text, parser validity, token count, and latency can be
    retained in a separate sidecar when ``diagnostics_path`` is supplied.
    """
    from endo_bbox_vla.metrics import write_prediction_bundle

    conversion = convert_ar_generations(generations, parser=parser)
    ground_truth_list = list(ground_truths)
    _validate_generation_coverage(conversion, ground_truth_list)
    destination = write_prediction_bundle(
        path,
        split=split,
        predictions=conversion.detections,
        ground_truths=ground_truth_list,
        model_identity=model_identity,
    )
    if diagnostics_path is not None:
        report_path = Path(diagnostics_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": AR_GENERATION_DIAGNOSTICS_SCHEMA_VERSION,
            "kind": "bbox_vla_ar_generation_diagnostics",
            "split": split,
            "model": dict(model_identity),
            "summary": conversion.summary(),
            "generations": [item.to_dict() for item in conversion.diagnostics],
        }
        report_path.write_text(
            json.dumps(
                payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return destination


@dataclass(frozen=True, slots=True)
class MetricSpec:
    key: str
    label: str
    decimals: int = 4


METRIC_SPECS = (
    MetricSpec("ap50", "AP50"),
    MetricSpec("ap50_95", "AP50:95"),
    MetricSpec("recall", "Recall"),
    MetricSpec("mean_iou_all_positive", "Mean IoU"),
    MetricSpec("mean_center_error_normalized", "Center error", 6),
    MetricSpec("fp_per_100_confirmed_negative", "FP / 100 negatives"),
)


def _as_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


_OPTIONAL_PROTOCOL_FIELDS = (
    "protocol_id",
    "annotation_revision",
    "approved_payload_sha256",
)
_REQUIRED_PROTOCOL_FIELDS = ("split", "fold_id", "seed")
_IDENTITY_ALIASES = {
    "fold_id": ("fold_id",),
    "seed": ("seed",),
    "protocol_id": ("protocol_id",),
    "annotation_revision": ("annotation_revision", "dataset_revision"),
    "approved_payload_sha256": ("approved_payload_sha256",),
}


def _nested_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _identity_candidates(score: Mapping[str, Any], field: str) -> list[tuple[str, Any]]:
    """Find identity values in current and legacy/nested score layouts."""
    aliases = _IDENTITY_ALIASES[field]
    candidates: list[tuple[str, Any]] = []
    for alias in aliases:
        for prefix, path in (
            ("model", ("model", alias)),
            ("model.protocol", ("model", "protocol", alias)),
            ("model.data", ("model", "data", alias)),
            ("protocol", ("protocol", alias)),
            ("root", (alias,)),
        ):
            value = _nested_value(score, path)
            if value is not None:
                candidates.append((f"{prefix}.{alias}", value))
    return candidates


def _identity_value(score: Mapping[str, Any], field: str, role: str) -> Any:
    candidates = _identity_candidates(score, field)
    if not candidates:
        return None
    first_value = candidates[0][1]
    conflicts = [
        (location, value) for location, value in candidates[1:] if value != first_value
    ]
    if conflicts:
        rendered = ", ".join(
            (f"{location}={value!r}" for location, value in candidates)
        )
        raise ValueError(f"{role} has internally inconsistent {field}: {rendered}")
    return first_value


def _validate_identity_value(field: str, value: Any, context: str) -> Any:
    if field == "seed":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{context} seed must be an integer")
        return value
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} {field} must be a non-empty string")
    if field == "approved_payload_sha256":
        lowered = value.lower()
        if len(lowered) != 64 or any(
            (character not in "0123456789abcdef" for character in lowered)
        ):
            raise ValueError(
                f"{context} approved_payload_sha256 must be a 64-character hex digest"
            )
        return lowered
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _validate_score(score: Mapping[str, Any], role: str) -> None:
    split = score.get("split")
    if not isinstance(split, str) or not split:
        raise TypeError(f"{role}.split must be a non-empty string")
    model = _as_object(score.get("model"), f"{role}.model")
    for field in ("model_type", "checkpoint_sha256"):
        value = model.get(field)
        if not isinstance(value, str) or not value:
            raise TypeError(f"{role}.model.{field} must be a non-empty string")
    for field in _REQUIRED_PROTOCOL_FIELDS[1:]:
        value = _identity_value(score, field, role)
        if value is None:
            raise ValueError(
                f"{role} is missing required protocol identity field {field!r}"
            )
        _validate_identity_value(field, value, role)
    for field in _OPTIONAL_PROTOCOL_FIELDS:
        value = _identity_value(score, field, role)
        if value is not None:
            _validate_identity_value(field, value, role)
    metrics = _as_object(score.get("metrics"), f"{role}.metrics")
    for spec in METRIC_SPECS:
        if spec.key not in metrics:
            raise ValueError(f"{role}.metrics is missing {spec.key!r}")
        _finite_number(metrics[spec.key], f"{role}.metrics.{spec.key}")


def _protocol_value(score: Mapping[str, Any], field: str, role: str) -> Any:
    if field == "split":
        return score.get("split")
    value = _identity_value(score, field, role)
    return None if value is None else _validate_identity_value(field, value, role)


def _build_protocol(
    detr_score: Mapping[str, Any], ar_score: Mapping[str, Any]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    sources = {"detr_score": detr_score, "qwen_ar_score": ar_score}
    for field in (*_REQUIRED_PROTOCOL_FIELDS, *_OPTIONAL_PROTOCOL_FIELDS):
        values = {
            role: _protocol_value(score, field, role) for role, score in sources.items()
        }
        present = {role: value for role, value in values.items() if value is not None}
        missing = sorted(set(values) - set(present))
        if field in _REQUIRED_PROTOCOL_FIELDS and missing:
            raise ValueError(
                f"required protocol identity field {field!r} is missing from {', '.join(missing)}"
            )
        distinct = {json.dumps(value, sort_keys=True) for value in present.values()}
        if len(distinct) > 1:
            rendered = ", ".join(
                (f"{role}={value!r}" for role, value in present.items())
            )
            raise ValueError(f"protocol identity mismatch for {field}: {rendered}")
        value = next(iter(present.values()), None)
        if not present:
            status = "not_available"
        elif missing:
            status = "partial"
        else:
            status = "matched"
        checks[field] = {"status": status, "value": value, "missing_from": missing}
    return {
        "split": checks["split"]["value"],
        "fold_id": checks["fold_id"]["value"],
        "seed": checks["seed"]["value"],
        "protocol_id": checks["protocol_id"]["value"],
        "annotation_revision": checks["annotation_revision"]["value"],
        "approved_payload_sha256": checks["approved_payload_sha256"]["value"],
        "checks": checks,
    }


COMPARISON_SCHEMA_VERSION = 1


def _extract_ar_diagnostics(
    diagnostics: Mapping[str, Any], ar_score: Mapping[str, Any]
) -> dict[str, Any]:
    kind = diagnostics.get("kind")
    if kind is not None and kind != "bbox_vla_ar_generation_diagnostics":
        raise ValueError(f"unexpected AR diagnostics kind: {kind!r}")
    if diagnostics.get("split") != ar_score.get("split"):
        raise ValueError(
            f"AR diagnostics split mismatch: diagnostics={diagnostics.get('split')!r}, score={ar_score.get('split')!r}"
        )
    diagnostic_model = _as_object(diagnostics.get("model"), "ar_diagnostics.model")
    score_model = _as_object(ar_score.get("model"), "qwen_ar_score.model")
    if dict(diagnostic_model) != dict(score_model):
        raise ValueError(
            "AR diagnostics model identity does not match the Qwen-AR score"
        )
    summary = _as_object(diagnostics.get("summary"), "ar_diagnostics.summary")
    rate = _finite_number(
        summary.get("format_valid_rate"), "ar_diagnostics.format_valid_rate"
    )
    if not 0.0 <= rate <= 1.0:
        raise ValueError("ar_diagnostics.format_valid_rate must be in [0, 1]")
    result: dict[str, Any] = {"format_valid_rate": rate}
    for field in (
        "generation_count",
        "format_valid_count",
        "format_invalid_count",
        "positive_output_count",
        "negative_output_count",
        "candidate_detection_count",
    ):
        if field in summary:
            value = summary[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"ar_diagnostics.{field} must be a non-negative integer"
                )
            result[field] = value
    for section_name in ("generated_tokens", "latency_ms"):
        raw_section = summary.get(section_name)
        if raw_section is None:
            result[section_name] = None
            continue
        section = _as_object(raw_section, f"ar_diagnostics.{section_name}")
        parsed: dict[str, Any] = {}
        observed = section.get("observed_count")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise TypeError(
                f"ar_diagnostics.{section_name}.observed_count must be a non-negative integer"
            )
        parsed["observed_count"] = observed
        numeric_fields = (
            ("mean", "max")
            if section_name == "generated_tokens"
            else ("mean", "p50", "p95")
        )
        for field in numeric_fields:
            value = section.get(field)
            parsed[field] = (
                None
                if value is None
                else _finite_number(value, f"ar_diagnostics.{section_name}.{field}")
            )
        result[section_name] = parsed
    return result


def _confidence_interval(
    score: Mapping[str, Any], key: str, role: str
) -> dict[str, float] | None:
    collection = score.get("episode_bootstrap_95ci")
    if collection is None:
        return None
    intervals = _as_object(collection, f"{role}.episode_bootstrap_95ci")
    raw = intervals.get(key)
    if raw is None:
        return None
    interval = _as_object(raw, f"{role}.episode_bootstrap_95ci.{key}")
    parsed = {
        bound: _finite_number(
            interval.get(bound), f"{role}.episode_bootstrap_95ci.{key}.{bound}"
        )
        for bound in ("point", "low", "high")
    }
    if not parsed["low"] <= parsed["point"] <= parsed["high"]:
        raise ValueError(
            f"{role} confidence interval for {key} does not contain its point"
        )
    metric_point = _finite_number(
        _as_object(score["metrics"], f"{role}.metrics")[key], f"{role}.metrics.{key}"
    )
    if not math.isclose(parsed["point"], metric_point, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{role} confidence-interval point differs from metric {key}")
    return parsed


def _threshold_summary(score: Mapping[str, Any], role: str) -> dict[str, Any]:
    threshold = score.get("threshold")
    return {
        "source": score.get("threshold_source"),
        "value": None
        if threshold is None
        else _finite_number(threshold, f"{role}.threshold"),
    }


def build_comparison_report(
    detr_score: Mapping[str, Any],
    ar_score: Mapping[str, Any],
    *,
    ar_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate two score objects and return a serialisable comparison."""
    detr_score = _as_object(detr_score, "detr_score")
    ar_score = _as_object(ar_score, "qwen_ar_score")
    _validate_score(detr_score, "detr_score")
    _validate_score(ar_score, "qwen_ar_score")
    protocol = _build_protocol(detr_score, ar_score)
    detr_metrics = _as_object(detr_score["metrics"], "detr_score.metrics")
    ar_metrics = _as_object(ar_score["metrics"], "qwen_ar_score.metrics")
    metrics: dict[str, Any] = {}
    for spec in METRIC_SPECS:
        detr_value = _finite_number(
            detr_metrics[spec.key], f"detr_score.metrics.{spec.key}"
        )
        ar_value = _finite_number(
            ar_metrics[spec.key], f"qwen_ar_score.metrics.{spec.key}"
        )
        metrics[spec.key] = {
            "label": spec.label,
            "detr": detr_value,
            "qwen_ar": ar_value,
            "delta_qwen_ar_minus_detr": ar_value - detr_value,
            "detr_episode_bootstrap_95ci": _confidence_interval(
                detr_score, spec.key, "detr_score"
            ),
            "qwen_ar_episode_bootstrap_95ci": _confidence_interval(
                ar_score, spec.key, "qwen_ar_score"
            ),
        }
    diagnostics_summary = (
        None
        if ar_diagnostics is None
        else _extract_ar_diagnostics(
            _as_object(ar_diagnostics, "ar_diagnostics"), ar_score
        )
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "kind": "bbox_vla_detr_qwen_ar_comparison",
        "protocol": protocol,
        "models": {
            "detr": dict(_as_object(detr_score["model"], "detr_score.model")),
            "qwen_ar": dict(_as_object(ar_score["model"], "qwen_ar_score.model")),
        },
        "thresholds": {
            "detr": _threshold_summary(detr_score, "detr_score"),
            "qwen_ar": _threshold_summary(ar_score, "qwen_ar_score"),
        },
        "metrics": metrics,
        "ar_diagnostics": diagnostics_summary,
    }


def _protocol_markdown_line(check: Mapping[str, Any]) -> str:
    value = check.get("value")
    rendered = "not available" if value is None else f"`{value}`"
    if check.get("status") == "partial":
        missing = ", ".join((str(item) for item in check.get("missing_from", ())))
        rendered += f" (partial; missing from {missing})"
    return rendered


def _format_value(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _format_metric_cell(metric: Mapping[str, Any], role: str, decimals: int) -> str:
    value = _finite_number(metric[role], f"report.metrics.{role}")
    interval = metric.get(f"{role}_episode_bootstrap_95ci")
    rendered = _format_value(value, decimals)
    if interval is not None:
        interval = _as_object(interval, f"report.metrics.{role}.interval")
        low = _finite_number(interval["low"], "confidence interval low")
        high = _finite_number(interval["high"], "confidence interval high")
        rendered += (
            f" [{_format_value(low, decimals)}, {_format_value(high, decimals)}]"
        )
    return rendered


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the comparison as a concise, reviewable Markdown document."""
    protocol = _as_object(report.get("protocol"), "report.protocol")
    checks = _as_object(protocol.get("checks"), "report.protocol.checks")
    models = _as_object(report.get("models"), "report.models")
    detr_model = _as_object(models.get("detr"), "report.models.detr")
    ar_model = _as_object(models.get("qwen_ar"), "report.models.qwen_ar")
    lines = [
        f"# DETR vs Qwen2-VL AR — {protocol.get('split')}",
        "",
        f"- Fold: {_protocol_markdown_line(_as_object(checks['fold_id'], 'fold check'))}",
        f"- Seed: {_protocol_markdown_line(_as_object(checks['seed'], 'seed check'))}",
        "- Protocol: "
        + _protocol_markdown_line(_as_object(checks["protocol_id"], "protocol check")),
        "- Annotation revision: "
        + _protocol_markdown_line(
            _as_object(checks["annotation_revision"], "annotation revision check")
        ),
        "- Approved payload: "
        + _protocol_markdown_line(
            _as_object(checks["approved_payload_sha256"], "payload check")
        ),
        "",
        "95% episode-bootstrap intervals are shown in brackets when present.",
        "",
        "| Metric | DETR (`{}`) | Qwen2-VL AR (`{}`) | Δ AR - DETR |".format(
            detr_model.get("model_type"), ar_model.get("model_type")
        ),
        "|---|---:|---:|---:|",
    ]
    metrics = _as_object(report.get("metrics"), "report.metrics")
    for spec in METRIC_SPECS:
        metric = _as_object(metrics.get(spec.key), f"report.metrics.{spec.key}")
        delta = _finite_number(
            metric["delta_qwen_ar_minus_detr"],
            f"report.metrics.{spec.key}.delta_qwen_ar_minus_detr",
        )
        lines.append(
            "| {} | {} | {} | {:+.{precision}f} |".format(
                spec.label,
                _format_metric_cell(metric, "detr", spec.decimals),
                _format_metric_cell(metric, "qwen_ar", spec.decimals),
                delta,
                precision=spec.decimals,
            )
        )
    lines.extend(["", "## Qwen-AR generation diagnostics", ""])
    diagnostics = report.get("ar_diagnostics")
    if diagnostics is None:
        lines.append("No AR diagnostics sidecar was supplied.")
    else:
        diagnostics = _as_object(diagnostics, "report.ar_diagnostics")
        rate = _finite_number(diagnostics["format_valid_rate"], "format-valid rate")
        valid = diagnostics.get("format_valid_count")
        total = diagnostics.get("generation_count")
        count_suffix = (
            f" ({valid}/{total})" if valid is not None and total is not None else ""
        )
        lines.append(f"- Strict format-valid rate: {rate:.2%}{count_suffix}")
        latency = diagnostics.get("latency_ms")
        if isinstance(latency, Mapping) and latency.get("p50") is not None:
            p50 = _finite_number(latency["p50"], "latency p50")
            p95 = latency.get("p95")
            mean = latency.get("mean")
            rendered = f"- End-to-end latency: p50 {p50:.2f} ms"
            if p95 is not None:
                rendered += f", p95 {_finite_number(p95, 'latency p95'):.2f} ms"
            if mean is not None:
                rendered += f", mean {_finite_number(mean, 'latency mean'):.2f} ms"
            lines.append(rendered)
        else:
            lines.append("- End-to-end latency: not observed")
    return "\n".join(lines) + "\n"


def _read_json_object(path: str | Path, context: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {context} JSON from {source}: {exc}") from exc
    return dict(_as_object(payload, f"{context} root"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_score_files(
    detr_score_path: str | Path,
    ar_score_path: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    ar_diagnostics_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read, compare, and write content-bound JSON and Markdown reports."""
    detr_path = Path(detr_score_path).resolve()
    ar_path = Path(ar_score_path).resolve()
    diagnostics_path = (
        None if ar_diagnostics_path is None else Path(ar_diagnostics_path).resolve()
    )
    json_path = Path(output_json).resolve()
    markdown_path = Path(output_markdown).resolve()
    inputs = {detr_path, ar_path}
    if diagnostics_path is not None:
        inputs.add(diagnostics_path)
    if json_path == markdown_path:
        raise ValueError("JSON and Markdown outputs must be different paths")
    if json_path in inputs or markdown_path in inputs:
        raise ValueError("comparison output paths must not overwrite input artifacts")
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "refusing to overwrite existing comparison output(s): "
            + ", ".join((str(path) for path in existing))
        )
    detr_score = _read_json_object(detr_path, "DETR score")
    ar_score = _read_json_object(ar_path, "Qwen-AR score")
    diagnostics = (
        None
        if diagnostics_path is None
        else _read_json_object(diagnostics_path, "Qwen-AR diagnostics")
    )
    report = build_comparison_report(detr_score, ar_score, ar_diagnostics=diagnostics)
    report["inputs"] = {
        "detr_score": {"path": str(detr_path), "sha256": _sha256_file(detr_path)},
        "qwen_ar_score": {"path": str(ar_path), "sha256": _sha256_file(ar_path)},
        "qwen_ar_diagnostics": None
        if diagnostics_path is None
        else {"path": str(diagnostics_path), "sha256": _sha256_file(diagnostics_path)},
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detr-score", type=Path, required=True)
    parser.add_argument("--ar-score", type=Path, required=True)
    parser.add_argument("--ar-diagnostics", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = compare_score_files(
        args.detr_score,
        args.ar_score,
        ar_diagnostics_path=args.ar_diagnostics,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        overwrite=args.overwrite,
    )
    print(render_markdown(report), end="")
    return 0


def cli(argv=None):
    from endo_bbox_vla.utils import run_cli

    return run_cli({"compare": main}, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
