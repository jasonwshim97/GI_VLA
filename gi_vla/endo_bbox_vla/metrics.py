"""Bounding-box metrics, source-constrained thresholds and frame scoring."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from pathlib import Path as _ProjectPath
from typing import Any

import numpy as np

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


type ImageId = str | int

type Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """A scored box prediction in original image coordinates."""

    image_id: ImageId
    score: float
    bbox_xyxy: Box


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """One reviewed frame and its (at most one) polyp annotation."""

    image_id: ImageId
    status: str
    bbox_xyxy: Box | None = None
    width: int = 640
    height: int = 480
    episode_index: int | str | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    partial: bool = False
    occluded: bool = False
    blurred: bool = False


@dataclass(frozen=True, slots=True)
class SourceThresholdSelection:
    """AR operating point selected under pooled or per-source FP constraints."""

    threshold: float
    mean_iou_all_positive: float
    recall: float
    fp_per_100_confirmed_negative: float
    matched_positives: int
    positive_count: int
    false_positive_frames: int
    confirmed_negative_count: int
    per_source_negative_constraints: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_box(value: Any) -> Box:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        raise ValueError(f"A box must contain four finite values, got {value!r}.")
    box = tuple((float(item) for item in array))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid xyxy box with non-positive area: {box!r}.")
    return box


def _get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _to_scalar(value: Any) -> Any:
    """Convert one-element numpy/torch-like values without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one scalar value, got shape {array.shape}.")
    return array.reshape(-1)[0].item()


def _coerce_detection(record: Detection | Mapping[str, Any] | Any) -> Detection:
    if isinstance(record, Detection):
        detection = record
    else:
        image_id = _get(record, "image_id")
        if image_id is None:
            raise ValueError("Every prediction needs image_id.")
        bbox = _get(record, "bbox_xyxy", _get(record, "box"))
        if bbox is None:
            raise ValueError(f"Prediction for image {image_id!r} has no bbox_xyxy.")
        detection = Detection(
            image_id=image_id,
            score=float(_to_scalar(_get(record, "score"))),
            bbox_xyxy=_to_box(bbox),
        )
    if not math.isfinite(detection.score) or not 0.0 <= detection.score <= 1.0:
        raise ValueError(
            f"Prediction score must be finite and in [0, 1], got {detection.score!r}."
        )
    _to_box(detection.bbox_xyxy)
    return detection


EVALUATED_STATUSES = frozenset({"positive", "confirmed_negative"})

IGNORED_STATUSES = frozenset({"uncertain_ignore", "unreviewed"})


def xywh_to_xyxy(box: Sequence[float]) -> Box:
    x, y, width, height = (float(item) for item in box)
    return _to_box((x, y, x + width, y + height))


def _coerce_ground_truth(record: GroundTruth | Mapping[str, Any] | Any) -> GroundTruth:
    if isinstance(record, GroundTruth):
        ground_truth = record
    else:
        image_id = _get(record, "image_id")
        if image_id is None:
            raise ValueError("Every ground-truth record needs image_id.")
        status = str(_get(record, "status", "positive"))
        bbox = _get(record, "bbox_xyxy")
        if bbox is None:
            coco_bbox = _get(record, "bbox")
            bbox = xywh_to_xyxy(coco_bbox) if coco_bbox is not None else None
        else:
            bbox = _to_box(bbox)
        ground_truth = GroundTruth(
            image_id=image_id,
            status=status,
            bbox_xyxy=bbox,
            width=int(_get(record, "width", 640)),
            height=int(_get(record, "height", 480)),
            episode_index=_get(record, "episode_index"),
            frame_index=None
            if _get(record, "frame_index") is None
            else int(_get(record, "frame_index")),
            timestamp=None
            if _get(record, "timestamp") is None
            else float(_get(record, "timestamp")),
            partial=bool(_get(record, "partial", False)),
            occluded=bool(_get(record, "occluded", False)),
            blurred=bool(_get(record, "blurred", False)),
        )
    if ground_truth.status not in EVALUATED_STATUSES | IGNORED_STATUSES:
        raise ValueError(f"Unknown annotation status {ground_truth.status!r}.")
    if ground_truth.width <= 0 or ground_truth.height <= 0:
        raise ValueError("Ground-truth image dimensions must be positive.")
    if ground_truth.status == "positive" and ground_truth.bbox_xyxy is None:
        raise ValueError(f"Positive image {ground_truth.image_id!r} has no bbox.")
    if (
        ground_truth.status == "confirmed_negative"
        and ground_truth.bbox_xyxy is not None
    ):
        raise ValueError(
            f"Confirmed-negative image {ground_truth.image_id!r} must not have a bbox."
        )
    if ground_truth.bbox_xyxy is not None:
        _to_box(ground_truth.bbox_xyxy)
    return ground_truth


def _normalise_inputs(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
) -> tuple[list[Detection], list[GroundTruth]]:
    detections = [_coerce_detection(item) for item in predictions]
    targets = [_coerce_ground_truth(item) for item in ground_truths]
    ids = [target.image_id for target in targets]
    if len(set(ids)) != len(ids):
        raise ValueError("Ground truth must contain exactly one record per image_id.")
    known_ids = set(ids)
    unknown = sorted(
        {str(item.image_id) for item in detections if item.image_id not in known_ids}
    )
    if unknown:
        raise ValueError(
            f"Predictions contain image_ids absent from ground truth: {unknown[:5]!r}."
        )
    return (detections, targets)


def _detections_by_image(
    detections: Sequence[Detection],
) -> dict[ImageId, list[Detection]]:
    grouped: dict[ImageId, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.image_id].append(detection)
    for values in grouped.values():
        values.sort(key=lambda item: item.score, reverse=True)
    return grouped


def _subgroup_recall(
    targets: Sequence[GroundTruth],
    matched: Mapping[ImageId, bool],
    key_fn: Any,
    keys: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    positives = [target for target in targets if target.status == "positive"]
    for key in keys:
        members = [target for target in positives if key_fn(target) == key]
        count = len(members)
        hits = sum((bool(matched.get(target.image_id, False)) for target in members))
        output[key] = {
            "recall": hits / count if count else 0.0,
            "matched": hits,
            "count": count,
        }
    return output


def box_iou(first: Box | Sequence[float], second: Box | Sequence[float]) -> float:
    """Intersection over union for two original-coordinate xyxy boxes."""
    a = _to_box(first)
    b = _to_box(second)
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _best_detection(
    candidates: Sequence[Detection], target: GroundTruth, confidence_threshold: float
) -> tuple[Detection | None, float]:
    above_threshold = [
        item for item in candidates if item.score >= confidence_threshold
    ]
    if not above_threshold or target.bbox_xyxy is None:
        return (None, 0.0)
    scored = [
        (box_iou(item.bbox_xyxy, target.bbox_xyxy), item) for item in above_threshold
    ]
    overlap, detection = max(scored, key=lambda pair: (pair[0], pair[1].score))
    return (detection, overlap)


def _average_precision(
    detections: Sequence[Detection],
    targets: Sequence[GroundTruth],
    iou_threshold: float,
) -> float:
    evaluated = {
        target.image_id: target
        for target in targets
        if target.status in EVALUATED_STATUSES
    }
    positive_count = sum((target.status == "positive" for target in evaluated.values()))
    if positive_count == 0:
        return 0.0
    ordered = sorted(
        (item for item in detections if item.image_id in evaluated),
        key=lambda item: (-item.score, str(item.image_id), item.bbox_xyxy),
    )
    matched: set[ImageId] = set()
    true_positives = np.zeros(len(ordered), dtype=np.float64)
    false_positives = np.zeros(len(ordered), dtype=np.float64)
    for index, detection in enumerate(ordered):
        target = evaluated[detection.image_id]
        if (
            target.status == "positive"
            and detection.image_id not in matched
            and (box_iou(detection.bbox_xyxy, target.bbox_xyxy) >= iou_threshold)
        ):
            true_positives[index] = 1.0
            matched.add(detection.image_id)
        else:
            false_positives[index] = 1.0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / positive_count
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, np.finfo(float).eps
    )
    samples = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= value])) if np.any(recall >= value) else 0.0
        for value in samples
    ]
    return float(np.mean(interpolated))


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _temporal_metrics(
    targets: Sequence[GroundTruth],
    matched: Mapping[ImageId, bool],
    selected: Mapping[ImageId, Detection | None],
    fps: float,
) -> dict[str, float | int]:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    episodes: dict[Any, list[GroundTruth]] = defaultdict(list)
    for target in targets:
        if target.status in EVALUATED_STATUSES:
            episode_key = (
                target.episode_index
                if target.episode_index is not None
                else target.image_id
            )
            episodes[episode_key].append(target)
    miss_runs: list[int] = []
    residual_jitter: list[float] = []
    predicted_steps: list[float] = []
    for episode_targets in episodes.values():
        episode_targets.sort(
            key=lambda item: (
                item.frame_index if item.frame_index is not None else math.inf,
                item.timestamp if item.timestamp is not None else math.inf,
                str(item.image_id),
            )
        )
        current_miss_run = 0
        previous: GroundTruth | None = None
        previous_detection: Detection | None = None
        for target in episode_targets:
            timestamp_gap = (
                target.timestamp - previous.timestamp
                if previous is not None
                and target.timestamp is not None
                and (previous.timestamp is not None)
                else None
            )
            contiguous = timestamp_gap is None or timestamp_gap <= 1.5 / fps + 1e-09
            if target.status != "positive" or not contiguous:
                if current_miss_run:
                    miss_runs.append(current_miss_run)
                current_miss_run = 0
                previous = None if not contiguous else target
                previous_detection = None
                continue
            detection = selected.get(target.image_id)
            is_match = bool(matched.get(target.image_id, False))
            if not is_match:
                current_miss_run += 1
            elif current_miss_run:
                miss_runs.append(current_miss_run)
                current_miss_run = 0
            if (
                is_match
                and detection is not None
                and (previous is not None)
                and (previous_detection is not None)
            ):
                assert target.bbox_xyxy is not None and previous.bbox_xyxy is not None
                diagonal = math.hypot(target.width, target.height)
                current_pred = _center(detection.bbox_xyxy)
                previous_pred = _center(previous_detection.bbox_xyxy)
                current_gt = _center(target.bbox_xyxy)
                previous_gt = _center(previous.bbox_xyxy)
                predicted_steps.append(
                    math.hypot(
                        current_pred[0] - previous_pred[0],
                        current_pred[1] - previous_pred[1],
                    )
                    / diagonal
                )
                residual_jitter.append(
                    math.hypot(
                        current_pred[0]
                        - current_gt[0]
                        - (previous_pred[0] - previous_gt[0]),
                        current_pred[1]
                        - current_gt[1]
                        - (previous_pred[1] - previous_gt[1]),
                    )
                    / diagonal
                )
            previous = target if is_match else None
            previous_detection = detection if is_match else None
        if current_miss_run:
            miss_runs.append(current_miss_run)
    miss_seconds = [run / fps for run in miss_runs]
    return {
        "miss_run_count": len(miss_runs),
        "mean_consecutive_miss_seconds": float(np.mean(miss_seconds))
        if miss_seconds
        else 0.0,
        "max_consecutive_miss_seconds": max(miss_seconds, default=0.0),
        "center_jitter_mean": float(np.mean(residual_jitter))
        if residual_jitter
        else 0.0,
        "center_jitter_p95": float(np.percentile(residual_jitter, 95))
        if residual_jitter
        else 0.0,
        "predicted_center_step_mean": float(np.mean(predicted_steps))
        if predicted_steps
        else 0.0,
        "jitter_pair_count": len(residual_jitter),
    }


def _size_bucket(target: GroundTruth, detector_input_size: int) -> str:
    assert target.bbox_xyxy is not None
    scale = min(detector_input_size / target.width, detector_input_size / target.height)
    transformed_short_side = (
        min(
            target.bbox_xyxy[2] - target.bbox_xyxy[0],
            target.bbox_xyxy[3] - target.bbox_xyxy[1],
        )
        * scale
    )
    if transformed_short_side < 14:
        return "lt_14"
    if transformed_short_side <= 28:
        return "14_to_28"
    return "gt_28"


def _center_error(predicted: Box, expected: Box, width: int, height: int) -> float:
    predicted_center = _center(predicted)
    expected_center = _center(expected)
    distance = math.hypot(
        predicted_center[0] - expected_center[0],
        predicted_center[1] - expected_center[1],
    )
    return distance / math.hypot(width, height)


def _is_central(target: GroundTruth) -> bool:
    assert target.bbox_xyxy is not None
    center_x, center_y = _center(target.bbox_xyxy)
    return (
        0.25 * target.width <= center_x <= 0.75 * target.width
        and 0.25 * target.height <= center_y <= 0.75 * target.height
    )


def compute_detection_metrics(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    confidence_threshold: float,
    match_iou_threshold: float = 0.5,
    detector_input_size: int = 224,
    fps: float = 15.0,
    ap_iou_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute all primary, subgroup, and temporal metrics as JSON-safe values."""
    if not math.isfinite(confidence_threshold) or confidence_threshold < 0.0:
        raise ValueError("confidence_threshold must be finite and non-negative.")
    if not 0 < match_iou_threshold <= 1:
        raise ValueError("match_iou_threshold must be in (0, 1].")
    if detector_input_size <= 0:
        raise ValueError("detector_input_size must be positive.")
    iou_thresholds = tuple(ap_iou_thresholds or np.arange(0.5, 0.951, 0.05).tolist())
    if any((not 0 < value <= 1 for value in iou_thresholds)):
        raise ValueError("All AP IoU thresholds must be in (0, 1].")
    detections, targets = _normalise_inputs(predictions, ground_truths)
    evaluated_targets = [item for item in targets if item.status in EVALUATED_STATUSES]
    ignored_count = len(targets) - len(evaluated_targets)
    grouped = _detections_by_image(detections)
    matched: dict[ImageId, bool] = {}
    selected: dict[ImageId, Detection | None] = {}
    best_ious: list[float] = []
    detected_ious: list[float] = []
    center_errors: list[float] = []
    for target in evaluated_targets:
        if target.status != "positive":
            continue
        detection, overlap = _best_detection(
            grouped.get(target.image_id, ()), target, confidence_threshold
        )
        is_match = detection is not None and overlap >= match_iou_threshold
        matched[target.image_id] = is_match
        selected[target.image_id] = detection
        best_ious.append(overlap)
        if detection is not None:
            detected_ious.append(overlap)
            assert target.bbox_xyxy is not None
            center_errors.append(
                _center_error(
                    detection.bbox_xyxy, target.bbox_xyxy, target.width, target.height
                )
            )
    positive_targets = [item for item in evaluated_targets if item.status == "positive"]
    negative_targets = [
        item for item in evaluated_targets if item.status == "confirmed_negative"
    ]
    false_positive_frames = sum(
        (
            any(
                (
                    item.score >= confidence_threshold
                    for item in grouped.get(target.image_id, ())
                )
            )
            for target in negative_targets
        )
    )
    fp_per_100 = (
        100.0 * false_positive_frames / len(negative_targets)
        if negative_targets
        else 0.0
    )
    matched_count = sum(
        (bool(matched.get(item.image_id, False)) for item in positive_targets)
    )
    ap_by_iou = {
        f"{threshold:.2f}": _average_precision(detections, evaluated_targets, threshold)
        for threshold in iou_thresholds
    }
    ap50_key = min(ap_by_iou, key=lambda key: abs(float(key) - 0.5))
    size_recall = _subgroup_recall(
        evaluated_targets,
        matched,
        lambda target: _size_bucket(target, detector_input_size),
        ("lt_14", "14_to_28", "gt_28"),
    )
    location_recall = _subgroup_recall(
        evaluated_targets,
        matched,
        lambda target: "central" if _is_central(target) else "peripheral",
        ("central", "peripheral"),
    )
    flag_recall = {
        flag: _subgroup_recall(
            evaluated_targets,
            matched,
            lambda target, field=flag: "true"
            if bool(getattr(target, field))
            else "false",
            ("true",),
        )["true"]
        for flag in ("partial", "occluded", "blurred")
    }
    return {
        "ap50": ap_by_iou[ap50_key],
        "ap50_95": float(np.mean(tuple(ap_by_iou.values()))) if ap_by_iou else 0.0,
        "ap_by_iou": ap_by_iou,
        "recall": matched_count / len(positive_targets) if positive_targets else 0.0,
        "matched_positives": matched_count,
        "positive_count": len(positive_targets),
        "false_positive_frames": false_positive_frames,
        "confirmed_negative_count": len(negative_targets),
        "fp_per_100_confirmed_negative": fp_per_100,
        "mean_iou_all_positive": float(np.mean(best_ious)) if best_ious else 0.0,
        "mean_iou_detected_positive": float(np.mean(detected_ious))
        if detected_ious
        else 0.0,
        "mean_center_error_normalized": float(np.mean(center_errors))
        if center_errors
        else 0.0,
        "size_recall": size_recall,
        "location_recall": location_recall,
        "flag_recall": flag_recall,
        "temporal": _temporal_metrics(evaluated_targets, matched, selected, fps),
        "counts": {
            "images": len(targets),
            "evaluated": len(evaluated_targets),
            "ignored": ignored_count,
            "predictions": len(detections),
        },
        "settings": {
            "confidence_threshold": float(confidence_threshold),
            "match_iou_threshold": float(match_iou_threshold),
            "detector_input_size": detector_input_size,
            "fps": float(fps),
        },
    }


def _thresholds_get(record: Any, name: str, default: Any = None) -> Any:
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


def _selection_detection(record: Detection | Mapping[str, Any] | Any) -> Detection:
    if isinstance(record, Detection):
        candidate = record
    else:
        image_id = _thresholds_get(record, "image_id")
        bbox = _thresholds_get(record, "bbox_xyxy", _thresholds_get(record, "box"))
        if image_id is None or bbox is None:
            raise ValueError(
                "Every threshold-selection prediction needs image_id and bbox_xyxy."
            )
        score = float(_thresholds_get(record, "score"))
        values = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if values.size != 4 or not np.isfinite(values).all():
            raise ValueError(
                "Threshold-selection bbox_xyxy must contain four finite values."
            )
        candidate = Detection(
            _image_id(image_id), score, tuple((float(x) for x in values))
        )
    if not math.isfinite(candidate.score) or not 0.0 <= candidate.score <= 1.0:
        raise ValueError("Threshold-selection scores must be finite and in [0, 1].")
    box_iou(candidate.bbox_xyxy, candidate.bbox_xyxy)
    return candidate


def _selection_ground_truth(
    record: GroundTruth | Mapping[str, Any] | Any,
) -> GroundTruth:
    if isinstance(record, GroundTruth):
        return record
    image_id = _thresholds_get(record, "image_id")
    if image_id is None:
        raise ValueError("Every threshold-selection ground truth needs image_id.")
    bbox = _thresholds_get(record, "bbox_xyxy")
    values = None
    if bbox is not None:
        array = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if array.size != 4 or not np.isfinite(array).all():
            raise ValueError(
                "Threshold-selection bbox_xyxy must contain four finite values."
            )
        values = tuple((float(x) for x in array))
    target = GroundTruth(
        image_id=_image_id(image_id),
        status=str(_thresholds_get(record, "status", "positive")),
        bbox_xyxy=values,
        width=int(_thresholds_get(record, "width", 640)),
        height=int(_thresholds_get(record, "height", 480)),
    )
    return target


TASK1_NEGATIVE_ROLE = "Task1-negative"

TASK4_NEGATIVE_ROLE = "Task4-negative"


def negative_source_role(source_dataset: str | None) -> str:
    """Map preserved dataset provenance to the preregistered negative arms."""
    if source_dataset is None:
        return "unspecified"
    lowered = source_dataset.casefold()
    if "task1" in lowered or "withdraw" in lowered:
        return TASK1_NEGATIVE_ROLE
    if "task4" in lowered or "polyp" in lowered:
        return TASK4_NEGATIVE_ROLE
    return source_dataset


def select_source_constrained_threshold(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    source_by_image: Mapping[ImageId, str | None] | None = None,
    required_negative_sources: Iterable[str] = (),
    max_fp_per_100_confirmed_negative: float = 5.0,
    match_iou_threshold: float = 0.5,
) -> SourceThresholdSelection:
    """Maximize positive-frame mean IoU under independent negative-source FP5 gates.

    Ties are resolved by higher recall, fewer total false-positive frames, and
    finally a higher threshold.  The implementation walks score events once,
    so thousands of validation frames do not require recomputing AP for every
    candidate threshold.
    """
    if (
        not math.isfinite(max_fp_per_100_confirmed_negative)
        or max_fp_per_100_confirmed_negative < 0.0
    ):
        raise ValueError("The false-positive budget must be finite and non-negative.")
    if not 0.0 < match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be in (0, 1].")
    detections = [_selection_detection(item) for item in predictions]
    targets = [_selection_ground_truth(item) for item in ground_truths]
    compute_detection_metrics(
        detections,
        targets,
        confidence_threshold=0.0,
        match_iou_threshold=match_iou_threshold,
    )
    target_ids = [target.image_id for target in targets]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Ground truth must contain exactly one record per image_id.")
    target_by_id = {target.image_id: target for target in targets}
    positives = [target for target in targets if target.status == "positive"]
    negatives = [target for target in targets if target.status == "confirmed_negative"]
    if not positives:
        raise ValueError(
            "Threshold selection requires at least one reviewed positive frame."
        )
    if not negatives:
        raise ValueError(
            "Threshold selection requires at least one confirmed-negative frame."
        )
    raw_sources = source_by_image or {}
    negative_role: dict[ImageId, str] = {
        target.image_id: negative_source_role(raw_sources.get(target.image_id))
        for target in negatives
    }
    negative_counts = Counter(negative_role.values())
    required = set(required_negative_sources)
    missing_sources = sorted(required - negative_counts.keys())
    if missing_sources:
        raise ValueError(
            f"Threshold selection lacks preregistered negative sources: {missing_sources!r}."
        )
    best_iou = {target.image_id: 0.0 for target in positives}
    matched = {target.image_id: False for target in positives}
    false_positive_ids: set[ImageId] = set()
    false_positive_by_source: Counter[str] = Counter()

    def candidate(threshold: float) -> SourceThresholdSelection | None:
        per_source: dict[str, dict[str, float | int]] = {}
        for source, count in sorted(negative_counts.items()):
            false_positives = int(false_positive_by_source[source])
            rate = 100.0 * false_positives / count
            per_source[source] = {
                "confirmed_negative_count": int(count),
                "false_positive_frames": false_positives,
                "fp_per_100_confirmed_negative": rate,
            }
        if any(
            (
                float(row["fp_per_100_confirmed_negative"])
                > max_fp_per_100_confirmed_negative + 1e-12
                for row in per_source.values()
            )
        ):
            return None
        total_fp = len(false_positive_ids)
        return SourceThresholdSelection(
            threshold=float(threshold),
            mean_iou_all_positive=float(np.mean(tuple(best_iou.values()))),
            recall=sum(matched.values()) / len(positives),
            fp_per_100_confirmed_negative=100.0 * total_fp / len(negatives),
            matched_positives=sum(matched.values()),
            positive_count=len(positives),
            false_positive_frames=total_fp,
            confirmed_negative_count=len(negatives),
            per_source_negative_constraints=per_source,
        )

    selected: SourceThresholdSelection | None = None

    def consider(value: SourceThresholdSelection | None) -> None:
        nonlocal selected
        if value is None:
            return
        key = (
            value.mean_iou_all_positive,
            value.recall,
            -value.false_positive_frames,
            value.threshold,
        )
        if selected is None or key > (
            selected.mean_iou_all_positive,
            selected.recall,
            -selected.false_positive_frames,
            selected.threshold,
        ):
            selected = value

    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    if not ordered:
        consider(candidate(1.0))
    else:
        maximum = ordered[0].score
        no_detection_threshold = math.nextafter(maximum, math.inf)
        if no_detection_threshold <= 1.0:
            consider(candidate(no_detection_threshold))
        index = 0
        while index < len(ordered):
            score = ordered[index].score
            end = index
            while end < len(ordered) and ordered[end].score == score:
                detection = ordered[end]
                target = target_by_id[detection.image_id]
                if target.status == "positive":
                    assert target.bbox_xyxy is not None
                    overlap = box_iou(detection.bbox_xyxy, target.bbox_xyxy)
                    if overlap > best_iou[target.image_id]:
                        best_iou[target.image_id] = overlap
                        matched[target.image_id] = overlap >= match_iou_threshold
                elif (
                    target.status == "confirmed_negative"
                    and target.image_id not in false_positive_ids
                ):
                    false_positive_ids.add(target.image_id)
                    false_positive_by_source[negative_role[target.image_id]] += 1
                end += 1
            consider(candidate(score))
            index = end
    if selected is None:
        raise ValueError(
            "No threshold in [0, 1] satisfies every negative-source false-positive budget."
        )
    return selected


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    result = tuple((float(item) for item in array))
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def evaluate_frame_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.5,
    detector_input_size: int = 224,
    fps: float = 15.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("frame prediction evaluation is empty")
    identities = [
        (
            row.get("sample_id"),
            row.get("prompt_id", row.get("prompt_mode")),
            row.get("context_generation", 0),
        )
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "frame prediction rows repeat sample/prompt/generation identities"
        )
    for row in rows:
        score = float(row.get("score", float("nan")))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("frame prediction score must lie in [0,1]")
        if str(row.get("status")) not in {"positive", "confirmed_negative"}:
            raise ValueError("frame predictions must exclude uncertain supervision")
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row.get("prompt_mode", "unspecified"))].append(row)
    groups: dict[str, Any] = {}
    for prompt_mode, group in sorted(by_prompt.items()):
        ground_truths = [
            {
                "image_id": row["image_id"],
                "status": row["status"],
                "bbox_xyxy": row.get("gt_bbox_xyxy"),
                "width": row.get("width", 640),
                "height": row.get("height", 480),
                "episode_index": row.get("episode_index"),
                "frame_index": row.get("frame_index"),
                "timestamp": row.get("timestamp"),
                "partial": row.get("partial", False),
                "occluded": row.get("occluded", False),
                "blurred": row.get("blurred", False),
            }
            for row in group
        ]
        predictions = [
            {
                "image_id": row["image_id"],
                "score": row["score"],
                "bbox_xyxy": row["bbox_xyxy"],
            }
            for row in group
            if bool(row.get("bbox_valid")) and _box(row.get("bbox_xyxy")) is not None
        ]
        groups[prompt_mode] = compute_detection_metrics(
            predictions,
            ground_truths,
            confidence_threshold=confidence_threshold,
            detector_input_size=detector_input_size,
            fps=fps,
        )
    return {
        "schema_version": 1,
        "metric": "shared_bbox_detection_metrics",
        "confidence_threshold": confidence_threshold,
        "detector_input_size": detector_input_size,
        "fps": fps,
        "prompt_groups": groups,
    }


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Validation result for a confidence threshold selected under an FP budget."""

    threshold: float
    recall: float
    fp_per_100_confirmed_negative: float
    matched_positives: int
    positive_count: int
    false_positive_frames: int
    confirmed_negative_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    point: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def select_confidence_threshold(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    max_fp_per_100_confirmed_negative: float = 5.0,
    match_iou_threshold: float = 0.5,
) -> ThresholdSelection:
    """Maximise recall subject to the reviewed-negative false-positive budget.

    Ties are resolved toward fewer false-positive frames and then the higher
    (more conservative) threshold.  A threshold just above the maximum score is
    included when it remains inside the probability range.
    """
    if max_fp_per_100_confirmed_negative < 0:
        raise ValueError("The false-positive budget cannot be negative.")
    detections, targets = _normalise_inputs(predictions, ground_truths)
    grouped = _detections_by_image(detections)
    positives = [target for target in targets if target.status == "positive"]
    negatives = [target for target in targets if target.status == "confirmed_negative"]
    if not positives:
        raise ValueError(
            "Confidence-threshold calibration requires at least one reviewed positive frame."
        )
    if not negatives:
        raise ValueError(
            "Confidence-threshold calibration requires at least one confirmed-negative frame; do not tune a threshold on an all-positive validation split."
        )
    match_scores = sorted(
        (
            max(
                (
                    detection.score
                    for detection in grouped.get(target.image_id, ())
                    if box_iou(detection.bbox_xyxy, target.bbox_xyxy)
                    >= match_iou_threshold
                ),
                default=-math.inf,
            )
            for target in positives
        )
    )
    match_scores = [score for score in match_scores if math.isfinite(score)]
    negative_max_scores = sorted(
        (
            max(
                (detection.score for detection in grouped.get(target.image_id, ())),
                default=-math.inf,
            )
            for target in negatives
        )
    )
    negative_max_scores = [
        score for score in negative_max_scores if math.isfinite(score)
    ]
    event_scores = sorted(set(match_scores) | set(negative_max_scores))
    if event_scores:
        margin = max(1e-12, abs(event_scores[-1]) * 1e-12)
        no_detection_threshold = min(1.0, event_scores[-1] + margin)
        candidates = [no_detection_threshold, *event_scores]
    else:
        candidates = [1.0]
    best: ThresholdSelection | None = None
    for threshold in candidates:
        matched_count = len(match_scores) - bisect_left(match_scores, threshold)
        false_positive_frames = len(negative_max_scores) - bisect_left(
            negative_max_scores, threshold
        )
        fp_rate = 100.0 * false_positive_frames / len(negatives)
        if fp_rate > max_fp_per_100_confirmed_negative + 1e-12:
            continue
        candidate = ThresholdSelection(
            threshold=float(threshold),
            recall=matched_count / len(positives),
            fp_per_100_confirmed_negative=fp_rate,
            matched_positives=matched_count,
            positive_count=len(positives),
            false_positive_frames=false_positive_frames,
            confirmed_negative_count=len(negatives),
        )
        candidate_key = (
            candidate.recall,
            -candidate.fp_per_100_confirmed_negative,
            candidate.threshold,
        )
        if best is None or candidate_key > (
            best.recall,
            -best.fp_per_100_confirmed_negative,
            best.threshold,
        ):
            best = candidate
    if best is None:
        raise ValueError(
            "No threshold in [0, 1] satisfies the false-positive budget; at least one confirmed-negative prediction has score 1.0"
        )
    return best


def episode_bootstrap(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    *,
    confidence_threshold: float,
    resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    metric_names: Sequence[str] = (
        "ap50",
        "ap50_95",
        "recall",
        "fp_per_100_confirmed_negative",
        "mean_iou_all_positive",
        "mean_center_error_normalized",
    ),
    match_iou_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Episode-level non-parametric bootstrap confidence intervals."""
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    detections, targets = _normalise_inputs(predictions, ground_truths)
    grouped_targets: dict[Any, list[GroundTruth]] = defaultdict(list)
    for target in targets:
        episode = target.episode_index
        if episode is None:
            episode = ("image", str(target.image_id))
        grouped_targets[episode].append(target)
    if not grouped_targets:
        raise ValueError("Cannot bootstrap an empty evaluation set.")
    grouped_detections = _detections_by_image(detections)
    episode_ids = list(grouped_targets)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(resamples):
        sampled_ids = rng.choice(len(episode_ids), size=len(episode_ids), replace=True)
        sampled_targets: list[GroundTruth] = []
        sampled_detections: list[Detection] = []
        for draw_index, episode_position in enumerate(sampled_ids):
            episode = episode_ids[int(episode_position)]
            for target in grouped_targets[episode]:
                cloned_id = f"bootstrap:{draw_index}:{target.image_id}"
                sampled_targets.append(replace(target, image_id=cloned_id))
                sampled_detections.extend(
                    (
                        replace(detection, image_id=cloned_id)
                        for detection in grouped_detections.get(target.image_id, ())
                    )
                )
        result = compute_detection_metrics(
            sampled_detections,
            sampled_targets,
            confidence_threshold=confidence_threshold,
            match_iou_threshold=match_iou_threshold,
        )
        for name in metric_names:
            if name not in result or not isinstance(result[name], (int, float)):
                raise ValueError(
                    f"Bootstrap metric {name!r} is not a scalar top-level metric."
                )
            samples[name].append(float(result[name]))
    point_metrics = compute_detection_metrics(
        detections,
        targets,
        confidence_threshold=confidence_threshold,
        match_iou_threshold=match_iou_threshold,
    )
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        interval = BootstrapInterval(
            point=float(point_metrics[name]),
            low=float(np.quantile(values, alpha)),
            high=float(np.quantile(values, 1.0 - alpha)),
        )
        intervals[name] = interval.to_dict()
    return intervals


def ground_truth_from_coco(
    coco: Mapping[str, Any],
    manifest_records: Iterable[Mapping[str, Any] | Any],
    *,
    split: str | None = None,
) -> list[GroundTruth]:
    """Join a split COCO document with records from the global manifest.

    ``split`` defaults to ``coco['info']['split']`` when it names one concrete
    split.  Image-id filtering remains authoritative, making older COCO files
    without the info field compatible with the global JSONL manifest.
    """
    images = {image["id"]: image for image in coco.get("images", ())}
    declared_split = coco.get("info", {}).get("split")
    is_logical_view = bool(coco.get("info", {}).get("experiment_protocol"))
    if (
        split is None
        and (not is_logical_view)
        and (declared_split not in (None, "", "mixed"))
    ):
        split = str(declared_split)
    boxes: dict[ImageId, Box] = {}
    for annotation in coco.get("annotations", ()):
        image_id = annotation["image_id"]
        if image_id in boxes:
            raise ValueError(f"More than one polyp annotation for image {image_id!r}.")
        boxes[image_id] = xywh_to_xyxy(annotation["bbox"])
    targets: list[GroundTruth] = []
    for record in manifest_records:
        image_id = _get(record, "image_id")
        if split is not None and _get(record, "split") != split:
            continue
        if image_id not in images:
            continue
        image = images[image_id]
        status = str(_get(record, "status"))
        targets.append(
            GroundTruth(
                image_id=image_id,
                status=status,
                bbox_xyxy=boxes.get(image_id),
                width=int(image.get("width", _get(record, "width", 640))),
                height=int(image.get("height", _get(record, "height", 480))),
                episode_index=_get(record, "episode_index"),
                frame_index=int(_get(record, "frame_index")),
                timestamp=float(_get(record, "timestamp")),
                partial=bool(_get(record, "partial", False)),
                occluded=bool(_get(record, "occluded", False)),
                blurred=bool(_get(record, "blurred", False)),
            )
        )
    joined_ids = {target.image_id for target in targets}
    missing = sorted(
        (str(image_id) for image_id in images if image_id not in joined_ids)
    )
    if missing:
        raise ValueError(
            f"COCO images are missing from the selected manifest records: {missing[:5]!r}."
        )
    return [_coerce_ground_truth(item) for item in targets]


PREDICTION_BUNDLE_SCHEMA_VERSION = 1

FIXED_EVALUATION_SPLITS = frozenset({"negative_control", "validation", "locked_test"})


def write_prediction_bundle(
    path: str | Path,
    *,
    split: str,
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    model_identity: Mapping[str, Any],
) -> Path:
    """Persist raw, unthresholded outputs with a model/split identity."""
    if split not in {"calibration", *FIXED_EVALUATION_SPLITS}:
        raise ValueError(f"Unsupported prediction-bundle split: {split!r}")
    detections, targets = _normalise_inputs(predictions, ground_truths)
    required_identity = {"model_type", "checkpoint_sha256", "fold_id", "seed"}
    missing = required_identity - model_identity.keys()
    if missing:
        raise ValueError(
            f"Prediction bundle model identity is missing {sorted(missing)}"
        )
    payload = {
        "schema_version": PREDICTION_BUNDLE_SCHEMA_VERSION,
        "kind": "bbox_vla_raw_predictions",
        "split": split,
        "model": dict(model_identity),
        "predictions": [asdict(item) for item in detections],
        "ground_truths": [asdict(item) for item in targets],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def read_prediction_bundle(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Prediction bundle root must be an object")
    if payload.get("schema_version") != PREDICTION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported prediction-bundle schema_version")
    if payload.get("kind") != "bbox_vla_raw_predictions":
        raise ValueError("Input is not a bbox_vla raw prediction bundle")
    if payload.get("split") not in {"calibration", *FIXED_EVALUATION_SPLITS}:
        raise ValueError(f"Invalid prediction-bundle split: {payload.get('split')!r}")
    model_identity = payload.get("model")
    if not isinstance(model_identity, dict):
        raise TypeError("Prediction bundle model identity must be an object")
    _normalise_inputs(payload.get("predictions", ()), payload.get("ground_truths", ()))
    return payload


THRESHOLD_ARTIFACT_SCHEMA_VERSION = 1


def fit_threshold_artifact(
    bundle: Mapping[str, Any],
    *,
    max_fp_per_100_confirmed_negative: float = 5.0,
    calibration_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit exactly once on the acquisition-disjoint calibration split."""
    if bundle.get("split") != "calibration":
        raise ValueError(
            "Confidence threshold may only be fitted on split='calibration'"
        )
    selection = select_confidence_threshold(
        bundle.get("predictions", ()),
        bundle.get("ground_truths", ()),
        max_fp_per_100_confirmed_negative=max_fp_per_100_confirmed_negative,
    )
    return {
        "schema_version": THRESHOLD_ARTIFACT_SCHEMA_VERSION,
        "kind": "bbox_vla_confidence_threshold",
        "selected_on_split": "calibration",
        "model": dict(bundle.get("model", {})),
        "calibration_bundle_sha256": calibration_bundle_sha256,
        "max_fp_per_100_confirmed_negative": float(max_fp_per_100_confirmed_negative),
        "threshold": selection.threshold,
        "selection": selection.to_dict(),
    }


def _fixed_threshold_for_bundle(
    bundle: Mapping[str, Any], artifact: Mapping[str, Any], *, expected_split: str
) -> float:
    if expected_split not in FIXED_EVALUATION_SPLITS:
        raise ValueError(
            f"Fixed scoring is only allowed for {sorted(FIXED_EVALUATION_SPLITS)}"
        )
    if bundle.get("split") != expected_split:
        raise ValueError(
            f"Prediction bundle split={bundle.get('split')!r}, expected {expected_split!r}"
        )
    if artifact.get("schema_version") != THRESHOLD_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported threshold-artifact schema_version")
    if artifact.get("kind") != "bbox_vla_confidence_threshold":
        raise ValueError("Input is not a bbox_vla confidence-threshold artifact")
    if artifact.get("selected_on_split") != "calibration":
        raise ValueError("Threshold artifact was not selected on calibration")
    if artifact.get("model") != bundle.get("model"):
        raise ValueError(
            "Threshold artifact and prediction bundle refer to different model runs"
        )
    threshold = float(artifact.get("threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold artifact contains invalid threshold {threshold!r}")
    return threshold


def score_prediction_bundle(
    bundle: Mapping[str, Any],
    threshold_artifact: Mapping[str, Any],
    *,
    expected_split: str,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate validation/test with an immutable calibration-derived threshold."""
    threshold = _fixed_threshold_for_bundle(
        bundle, threshold_artifact, expected_split=expected_split
    )
    predictions = bundle.get("predictions", ())
    ground_truths = bundle.get("ground_truths", ())
    metrics = compute_detection_metrics(
        predictions, ground_truths, confidence_threshold=threshold
    )
    intervals = None
    if bootstrap_samples:
        intervals = episode_bootstrap(
            predictions,
            ground_truths,
            confidence_threshold=threshold,
            resamples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    return {
        "split": expected_split,
        "threshold_source": "calibration",
        "threshold": threshold,
        "model": dict(bundle.get("model", {})),
        "metrics": metrics,
        "episode_bootstrap_95ci": intervals,
    }


def compute_detection_metric_breakdowns(
    predictions: Iterable[Detection | Mapping[str, Any] | Any],
    ground_truths: Iterable[GroundTruth | Mapping[str, Any] | Any],
    metadata_by_image: Mapping[ImageId, Mapping[str, Any]],
    *,
    confidence_threshold: float,
    **metric_kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute shared detector metrics for each preserved source and scenario."""
    prediction_list = list(predictions)
    target_list = list(ground_truths)
    target_ids = {
        _image_id(_thresholds_get(target, "image_id")) for target in target_list
    }
    prediction_by_id: dict[ImageId, list[Any]] = defaultdict(list)
    for prediction in prediction_list:
        prediction_by_id[_image_id(_thresholds_get(prediction, "image_id"))].append(
            prediction
        )
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for field, output_name in (
        ("source_dataset", "by_source"),
        ("source_role", "by_source_role"),
        ("scenario", "by_scenario"),
    ):
        groups: dict[str, set[ImageId]] = defaultdict(set)
        for image_id in target_ids:
            metadata = metadata_by_image.get(image_id, {})
            value = metadata.get(field)
            groups["unspecified" if value is None else str(value)].add(image_id)
        output: dict[str, dict[str, Any]] = {}
        for name, image_ids in sorted(groups.items()):
            group_targets = [
                target
                for target in target_list
                if _image_id(_thresholds_get(target, "image_id")) in image_ids
            ]
            group_predictions = [
                prediction
                for image_id in image_ids
                for prediction in prediction_by_id.get(image_id, ())
            ]
            output[name] = compute_detection_metrics(
                group_predictions,
                group_targets,
                confidence_threshold=confidence_threshold,
                **metric_kwargs,
            )
        result[output_name] = output
    return result
