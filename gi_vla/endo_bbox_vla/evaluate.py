"""Validation selection and scoring with frozen operating thresholds."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.data import SharedFeatureTable, StaticFrameRecord
    from endo_bbox_vla.metrics import Box, Detection, GroundTruth
    from endo_bbox_vla.network import SharedDetectorHead
    from endo_bbox_vla.train import TrainingVariant

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


def _prediction_row(
    record: StaticFrameRecord, *, score: float, box: Tensor, variant: TrainingVariant
) -> dict[str, Any]:
    from endo_bbox_vla.data import _actual_training_prompt
    from endo_bbox_vla.train import _cxcywh_to_original, _visible_gt

    prompt = _actual_training_prompt(record)
    bbox = _cxcywh_to_original(box)
    visible_gt, visible_fraction, crop_clipped = _visible_gt(record)
    return {
        "schema_version": 1,
        "kind": "shared_e03_frame_prediction",
        "sample_id": record.sample_id,
        "image_id": record.image_id,
        "source_dataset": record.source_dataset,
        "source_episode_index": record.source_episode_index,
        "source_frame_index": record.source_frame_index,
        "dataset_index": record.dataset_index,
        "episode_index": record.episode_index,
        "frame_index": record.frame_index,
        "timestamp": record.timestamp,
        "source_video": record.source_video,
        "split": record.split,
        "status": record.status,
        "width": record.width,
        "height": record.height,
        "partial": record.partial,
        "occluded": record.occluded,
        "blurred": record.blurred,
        "group_id": record.group_id,
        "gt_bbox_xyxy": None if record.bbox_xyxy is None else list(record.bbox_xyxy),
        "gt_bbox_visible_xyxy": visible_gt,
        "crop_visible_fraction": visible_fraction,
        "crop_clipped": crop_clipped,
        "present": bbox is not None,
        "score": score,
        "bbox_valid": bbox is not None,
        "bbox_xyxy": bbox,
        "bbox_cxcywh_normalized": [float(value) for value in box.detach().cpu()],
        "prompt": prompt,
        "prompt_mode": "actual",
        "prompt_id": "actual:" + hashlib.sha256(prompt.encode()).hexdigest(),
        "context_generation": 0,
        "context_condition": "same_gr00t_action_forward_block11",
        "variant": variant,
    }


def _predict(
    head: SharedDetectorHead,
    table: SharedFeatureTable,
    *,
    variant: TrainingVariant,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    from endo_bbox_vla.network import select_single_prediction

    rows: list[dict[str, Any]] = []
    head.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(table), batch_size):
            stop = min(start + batch_size, len(table))
            outputs = head(
                table.features[start:stop].to(device, non_blocking=True),
                table.padding_masks[start:stop].to(device, non_blocking=True),
            )
            scores, boxes = select_single_prediction(outputs)
            for offset, record in enumerate(table.records[start:stop]):
                rows.append(
                    _prediction_row(
                        record,
                        score=float(scores[offset].float().cpu()),
                        box=boxes[offset].float().cpu(),
                        variant=variant,
                    )
                )
    return rows


def _metric_inputs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Detection], list[GroundTruth], dict[int, str]]:
    from endo_bbox_vla.metrics import Detection, GroundTruth

    predictions: list[Detection] = []
    targets: list[GroundTruth] = []
    sources: dict[int, str] = {}
    for row in rows:
        image_id = int(row["image_id"])
        bbox = row.get("bbox_xyxy")
        if bbox is not None:
            predictions.append(
                Detection(
                    image_id,
                    float(row["score"]),
                    tuple((float(value) for value in bbox)),
                )
            )
        gt = row.get("gt_bbox_xyxy")
        targets.append(
            GroundTruth(
                image_id=image_id,
                status=str(row["status"]),
                bbox_xyxy=None if gt is None else tuple((float(value) for value in gt)),
                width=int(row["width"]),
                height=int(row["height"]),
                episode_index=int(row["episode_index"]),
                frame_index=int(row["frame_index"]),
                timestamp=float(row["timestamp"]),
                partial=bool(row["partial"]),
                occluded=bool(row["occluded"]),
                blurred=bool(row["blurred"]),
            )
        )
        sources[image_id] = str(row["source_dataset"])
    return (predictions, targets, sources)


def _validation_report(
    config: ExperimentConfig, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from endo_bbox_vla.data import GROOT_N17_GEOMETRY
    from endo_bbox_vla.metrics import (
        TASK1_NEGATIVE_ROLE,
        TASK4_NEGATIVE_ROLE,
        evaluate_frame_predictions,
        select_source_constrained_threshold,
    )

    predictions, targets, sources = _metric_inputs(rows)
    threshold = select_source_constrained_threshold(
        predictions,
        targets,
        source_by_image=sources,
        required_negative_sources=(TASK1_NEGATIVE_ROLE, TASK4_NEGATIVE_ROLE),
        max_fp_per_100_confirmed_negative=float(
            config.section("evaluation")["max_fp_per_100_per_negative_source"]
        ),
    )
    evaluation = evaluate_frame_predictions(
        rows,
        confidence_threshold=threshold.threshold,
        detector_input_size=GROOT_N17_GEOMETRY.output_size[0],
        fps=float(config.section("evaluation")["detector_fps"]),
    )
    return {
        "threshold_selection": threshold.to_dict(),
        "metrics": evaluation["prompt_groups"]["actual"],
        "full_evaluation": evaluation,
    }


def _selection_key(report: Mapping[str, Any]) -> tuple[float, float, float, int]:
    metrics = report["metrics"]
    return (
        float(metrics["mean_iou_all_positive"]),
        float(metrics["ap50_95"]),
        float(metrics["recall"]),
        -int(metrics["false_positive_frames"]),
    )


def compare_variants(config: ExperimentConfig) -> dict[str, Any]:
    from endo_bbox_vla.config import _variant_config, _variant_names
    from endo_bbox_vla.utils import _atomic_json, _variant_root, sha256_file

    rows: list[dict[str, Any]] = []
    variants = _variant_names(config)
    for variant in variants:
        root = _variant_root(config, variant)
        report_path = root / "train_report.json"
        best_path = root / "best.pt"
        if not report_path.is_file() or not best_path.is_file():
            rows.append({"variant": variant, "status": "pending"})
            continue
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        validation = checkpoint["metadata"]["validation"]
        metrics = validation["metrics"]
        threshold = validation["threshold_selection"]
        settings = _variant_config(config, variant)
        rows.append(
            {
                "variant": variant,
                "description": str(settings.get("description", variant)),
                "status": "complete",
                "head_kind": str(settings.get("head_kind", "location_first")),
                "num_queries": int(checkpoint["metadata"].get("num_queries", 4)),
                "sampler": str(settings["sampler"]),
                "checkpoint": str(best_path),
                "checkpoint_sha256": sha256_file(best_path),
                "epoch": int(validation["epoch"]),
                "threshold": float(threshold["threshold"]),
                "mean_iou_all_positive": float(metrics["mean_iou_all_positive"]),
                "recall": float(metrics["recall"]),
                "ap50": float(metrics["ap50"]),
                "ap75": float(metrics["ap_by_iou"]["0.75"]),
                "ap50_95": float(metrics["ap50_95"]),
                "false_positive_frames": int(metrics["false_positive_frames"]),
                "selection_key": list(_selection_key(validation)),
            }
        )
    completed = [row for row in rows if row["status"] == "complete"]
    winner = (
        max(completed, key=lambda row: tuple(row["selection_key"]))["variant"]
        if len(completed) == len(variants)
        else None
    )
    result = {
        "schema_version": 1,
        "kind": "shared_e03_validation_comparison",
        "selection_split": "validation",
        "internal_test_used": False,
        "winner": winner,
        "rows": rows,
    }
    _atomic_json(config.path_value("workspace", "root") / "comparison.json", result)
    return result


def _fixed_threshold_report(
    config: ExperimentConfig, rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    from endo_bbox_vla.data import GROOT_N17_GEOMETRY
    from endo_bbox_vla.metrics import evaluate_frame_predictions

    evaluation = evaluate_frame_predictions(
        rows,
        confidence_threshold=threshold,
        detector_input_size=GROOT_N17_GEOMETRY.output_size[0],
        fps=float(config.section("evaluation")["detector_fps"]),
    )
    return {
        "threshold": threshold,
        "metrics": evaluation["prompt_groups"]["actual"],
        "full_evaluation": evaluation,
    }


def evaluate_winner_test(config: ExperimentConfig, *, device: str) -> dict[str, Any]:
    from endo_bbox_vla.config import _variant_names
    from endo_bbox_vla.data import (
        SharedFeatureTable,
        _evaluation_bindings,
        _load_cache_contract,
    )
    from endo_bbox_vla.network import _build_head
    from endo_bbox_vla.utils import (
        _atomic_json,
        _atomic_jsonl,
        _variant_root,
        sha256_file,
    )

    root = config.path_value("workspace", "root") / "internal_test"
    report_path = root / "test.report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    comparison_path = config.path_value("workspace", "root") / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    winner = comparison.get("winner")
    if winner not in set(_variant_names(config)):
        raise ValueError(
            "validation comparison has no completed shared detector winner"
        )
    variant: TrainingVariant = str(winner)
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("shared detector internal-test evaluation requires CUDA")
    checkpoint_path = _variant_root(config, variant) / "best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    metadata = checkpoint["metadata"]
    if (
        metadata.get("config_sha256") != config.sha256
        or metadata.get("internal_test_used") is not False
        or metadata.get("feature_capture") != "same_action_forward"
    ):
        raise ValueError("shared detector winner checkpoint provenance changed")
    threshold = float(metadata["validation"]["threshold_selection"]["threshold"])
    contract = _load_cache_contract(config, evaluation=True)
    table = SharedFeatureTable(
        contract,
        _evaluation_bindings(config, contract, "test"),
        progress_label="internal_test",
    )
    head = _build_head(config, variant, torch_device)
    head.load_state_dict(checkpoint["head"], strict=True)
    started = time.monotonic()
    rows = _predict(
        head,
        table,
        variant=variant,
        device=torch_device,
        batch_size=int(config.section("training")["evaluation_batch_size"]),
    )
    evaluation = _fixed_threshold_report(config, rows, threshold=threshold)
    result = {
        "schema_version": 1,
        "kind": "shared_e03_internal_test_report",
        "split": "test",
        "label": "internal_test",
        "variant": variant,
        "selection_split": "validation",
        "threshold": threshold,
        "threshold_source_epoch": int(metadata["validation"]["epoch"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config.sha256,
        "feature_capture": "same_action_forward",
        "visual_parameters_trainable": False,
        "prediction_rows": len(rows),
        "metrics": evaluation["metrics"],
        "full_evaluation": evaluation["full_evaluation"],
        "elapsed_seconds": time.monotonic() - started,
    }
    root.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(root / "test.predictions.jsonl", rows)
    _atomic_json(report_path, result)
    return result


def evaluate_all_variants_test(
    config: ExperimentConfig, *, device: str
) -> dict[str, Any]:
    """Evaluate every validation-frozen ablation arm on the internal test."""
    from endo_bbox_vla.config import _variant_config, _variant_names
    from endo_bbox_vla.data import (
        SharedFeatureTable,
        _evaluation_bindings,
        _load_cache_contract,
    )
    from endo_bbox_vla.network import _build_head
    from endo_bbox_vla.utils import (
        _atomic_json,
        _atomic_jsonl,
        _variant_root,
        sha256_file,
    )

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("shared detector ablation evaluation requires CUDA")
    variants = _variant_names(config)
    missing = [
        variant
        for variant in variants
        if not (_variant_root(config, variant) / "best.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"ablation checkpoints are missing: {missing}")
    contract = _load_cache_contract(config, evaluation=True)
    table = SharedFeatureTable(
        contract,
        _evaluation_bindings(config, contract, "test"),
        progress_label="internal_test_all_variants",
    )
    output_root = config.path_value("workspace", "root") / "internal_test_ablation"
    results: list[dict[str, Any]] = []
    for variant in variants:
        variant_root = output_root / variant
        report_path = variant_root / "test.report.json"
        if report_path.is_file():
            results.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        checkpoint_path = _variant_root(config, variant) / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=torch_device, weights_only=True
        )
        metadata = checkpoint["metadata"]
        if (
            metadata.get("config_sha256") != config.sha256
            or metadata.get("internal_test_used") is not False
            or metadata.get("feature_capture") != "same_action_forward"
            or (metadata.get("variant") != variant)
        ):
            raise ValueError(f"ablation checkpoint provenance changed: {variant}")
        threshold = float(metadata["validation"]["threshold_selection"]["threshold"])
        head = _build_head(config, variant, torch_device)
        head.load_state_dict(checkpoint["head"], strict=True)
        started = time.monotonic()
        rows = _predict(
            head,
            table,
            variant=variant,
            device=torch_device,
            batch_size=int(config.section("training")["evaluation_batch_size"]),
        )
        evaluation = _fixed_threshold_report(config, rows, threshold=threshold)
        result = {
            "schema_version": 1,
            "kind": "shared_e03_ablation_internal_test_report",
            "split": "test",
            "label": "internal_test",
            "variant": variant,
            "description": str(
                _variant_config(config, variant).get("description", variant)
            ),
            "selection_split": "validation",
            "threshold": threshold,
            "threshold_source_epoch": int(metadata["validation"]["epoch"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config_sha256": config.sha256,
            "feature_capture": "same_action_forward",
            "visual_parameters_trainable": False,
            "head_kind": metadata["head_kind"],
            "num_queries": int(metadata["num_queries"]),
            "prediction_rows": len(rows),
            "metrics": evaluation["metrics"],
            "full_evaluation": evaluation["full_evaluation"],
            "elapsed_seconds": time.monotonic() - started,
        }
        variant_root.mkdir(parents=True, exist_ok=True)
        _atomic_jsonl(variant_root / "test.predictions.jsonl", rows)
        _atomic_json(report_path, result)
        results.append(result)
    aggregate = {
        "schema_version": 1,
        "kind": "shared_e03_ablation_internal_test_comparison",
        "selection_split": "validation",
        "evaluation_split": "internal_test",
        "config": str(config.path),
        "config_sha256": config.sha256,
        "variants": list(variants),
        "results": results,
    }
    _atomic_json(output_root / "comparison.json", aggregate)
    return aggregate


def _paper_metric_row(
    config: ExperimentConfig, variant: TrainingVariant, report: Mapping[str, Any]
) -> dict[str, Any]:
    from endo_bbox_vla.config import _variant_config
    from endo_bbox_vla.utils import _variant_root

    settings = _variant_config(config, variant)
    checkpoint_path = _variant_root(config, variant) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["head"]
    parameters = sum((int(tensor.numel()) for tensor in state.values()))
    metrics = report["metrics"]
    kind = str(settings.get("head_kind", "location_first"))
    sampler = str(settings["sampler"])
    return {
        "variant": variant,
        "model": str(settings.get("paper_label", settings.get("description", variant))),
        "num_queries": int(checkpoint["metadata"]["num_queries"]),
        "location_first": kind == "location_first",
        "hard_positive_finetuning": sampler == "size_location_balanced",
        "head_parameters": parameters,
        "threshold": float(report["threshold"]),
        "mIoU": float(metrics["mean_iou_all_positive"]),
        "Recall@IoU50": float(metrics["recall"]),
        "AP50": float(metrics["ap50"]),
        "AP75": float(metrics["ap_by_iou"]["0.75"]),
        "AP50:95": float(metrics["ap50_95"]),
        "FP/100": float(metrics["fp_per_100_confirmed_negative"]),
    }


def write_ablation_tables(config: ExperimentConfig) -> dict[str, Any]:
    """Write paper-ready internal-test ablation tables from sealed reports."""
    from endo_bbox_vla.config import _variant_names
    from endo_bbox_vla.utils import _atomic_json, _atomic_write

    variants = _variant_names(config)
    report_root = config.path_value("workspace", "root") / "internal_test_ablation"
    rows: list[dict[str, Any]] = []
    for variant in variants:
        report_path = report_root / variant / "test.report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"ablation test report is missing: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("config_sha256") != config.sha256:
            raise ValueError(f"ablation report config changed: {variant}")
        rows.append(_paper_metric_row(config, variant, report))
    paper_root = config.path_value("workspace", "root") / "paper"
    payload = {
        "schema_version": 1,
        "kind": "shared_e03_ablation_paper_table",
        "split": "internal_test",
        "selection_split": "validation",
        "config": str(config.path),
        "config_sha256": config.sha256,
        "rows": rows,
    }
    _atomic_json(paper_root / "shared_e03_ablation_internal_test.json", payload)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(
        paper_root / "shared_e03_ablation_internal_test.csv",
        stream.getvalue().encode("utf-8"),
    )
    lines = [
        "# shared detector ablation — internal test",
        "",
        "Every arm uses the same frozen same-forward cam_low block-11 token cache. Thresholds were fixed independently on validation under the per-source FP/100 constraint before internal-test scoring.",
        "",
        "| Model | Queries | Location-first | Hard-positive fine-tuning | Head params (M) | mIoU | Recall@IoU50 | AP50 | AP75 | AP50:95 | FP/100 |",
        "|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['num_queries']} | {('✓' if row['location_first'] else '–')} | {('✓' if row['hard_positive_finetuning'] else '–')} | {row['head_parameters'] / 1000000.0:.2f} | {row['mIoU']:.3f} | {row['Recall@IoU50']:.3f} | {row['AP50']:.3f} | {row['AP75']:.3f} | {row['AP50:95']:.3f} | {row['FP/100']:.3f} |"
        )
    _atomic_write(
        paper_root / "shared_e03_ablation_internal_test.md",
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    return payload


def _split_batch_items(value: Any, batch_size: int) -> list[Any]:
    if isinstance(value, Mapping):
        records: list[dict[str, Any]] = [dict() for _ in range(batch_size)]
        for key, field in value.items():
            if isinstance(field, (str, bytes)) or not hasattr(field, "__len__"):
                for record in records:
                    record[key] = field
                continue
            if len(field) != batch_size:
                for record in records:
                    record[key] = field
            else:
                for index, record in enumerate(records):
                    record[key] = field[index]
        return records
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        if len(value) != batch_size:
            raise ValueError(
                f"Batch field has {len(value)} items, expected {batch_size}."
            )
        return list(value)
    if batch_size != 1:
        raise ValueError("A scalar batch field is only valid for batch size one.")
    return [value]


def _ground_truth_from_dataset_item(
    target: Mapping[str, Any], record: Any
) -> GroundTruth:
    from endo_bbox_vla.metrics import _coerce_ground_truth, _get, _to_box, _to_scalar

    boxes = target.get("boxes")
    box: Box | None = None
    if boxes is not None:
        if hasattr(boxes, "detach"):
            boxes = boxes.detach()
        array = np.asarray(boxes.cpu() if hasattr(boxes, "cpu") else boxes)
        if array.size:
            values = array.reshape(-1, 4)[0].astype(np.float64, copy=True)
            spec = target.get("letterbox_spec")
            if spec is not None:
                values[[0, 2]] = (values[[0, 2]] - float(spec.pad_left)) / float(
                    spec.scale_x
                )
                values[[1, 3]] = (values[[1, 3]] - float(spec.pad_top)) / float(
                    spec.scale_y
                )
                original_width, original_height = spec.original_size
                values[[0, 2]] = np.clip(values[[0, 2]], 0, original_width)
                values[[1, 3]] = np.clip(values[[1, 3]], 0, original_height)
            box = _to_box(values)
    image_id = _get(record, "image_id", target.get("image_id"))
    if hasattr(image_id, "numel"):
        image_id = _to_scalar(image_id)
    return _coerce_ground_truth(
        {
            "image_id": image_id,
            "status": _get(
                record,
                "status",
                "positive" if box is not None else "confirmed_negative",
            ),
            "bbox_xyxy": box,
            "width": _get(record, "width", 640),
            "height": _get(record, "height", 480),
            "episode_index": _get(record, "episode_index"),
            "frame_index": _get(record, "frame_index"),
            "timestamp": _get(record, "timestamp"),
            "partial": _get(record, "partial", False),
            "occluded": _get(record, "occluded", False),
            "blurred": _get(record, "blurred", False),
        }
    )


def collect_detector_records(
    model: Any,
    dataloader: Iterable[Any],
    model_type: str,
    device: Any,
    *,
    precision: str = "float32",
) -> tuple[list[Detection], list[GroundTruth]]:
    """Run either detector family and retain original-coordinate raw records.

    The loader is expected to yield ``(images, targets, FrameRecord objects)``.
    Both list-style detection collation and a stacked shared-model tensor are
    accepted.  No confidence threshold is applied here, which lets one immutable
    prediction bundle feed calibration and fixed-threshold scoring.
    """
    import torch

    from endo_bbox_vla.metrics import Detection, _to_box
    from endo_bbox_vla.utils import (
        normalized_cxcywh_to_original_xyxy as normalized_cxcywh_to_original_xyxy,
    )

    family = model_type.lower().replace("-", "_")
    if family not in {"shared", "shared_pi05", "torchvision", "highres", "faster_rcnn"}:
        raise ValueError(f"Unsupported model_type {model_type!r}.")
    shared = family in {"shared", "shared_pi05"}
    normalized_precision = precision.lower()
    if normalized_precision not in {"float32", "fp32", "bfloat16", "bf16"}:
        raise ValueError(f"Unsupported evaluation precision: {precision!r}")
    model.eval()
    predictions: list[Detection] = []
    ground_truths: list[GroundTruth] = []
    with torch.inference_mode():
        for batch in dataloader:
            if isinstance(batch, Mapping):
                images, targets, records = (
                    batch["images"],
                    batch["targets"],
                    batch["records"],
                )
            elif isinstance(batch, Sequence) and len(batch) == 3:
                images, targets, records = batch
            else:
                raise ValueError("Dataloader must yield (images, targets, records).")
            if isinstance(images, torch.Tensor):
                image_list = list(images) if images.ndim == 4 else [images]
            else:
                image_list = list(images)
            batch_size = len(image_list)
            target_list = _split_batch_items(targets, batch_size)
            record_list = _split_batch_items(records, batch_size)
            batch_ground_truths = [
                _ground_truth_from_dataset_item(target, record)
                for target, record in zip(target_list, record_list, strict=True)
            ]
            ground_truths.extend(batch_ground_truths)
            device_images = [image.to(device) for image in image_list]
            if normalized_precision in {"bfloat16", "bf16"}:
                autocast = torch.autocast(
                    device_type=torch.device(device).type, dtype=torch.bfloat16
                )
            else:
                from contextlib import nullcontext

                autocast = nullcontext()
            if shared:
                with autocast:
                    outputs = model(torch.stack(device_images))
                if not isinstance(outputs, Mapping):
                    raise TypeError("Shared detector must return a mapping.")
                logits = outputs["presence_logits"].detach().reshape(-1).cpu()
                boxes = outputs["boxes_cxcywh"].detach().reshape(-1, 4).cpu()
                if len(logits) != batch_size or len(boxes) != batch_size:
                    raise ValueError(
                        "Shared detector output batch size does not match its input."
                    )
                scores = torch.sigmoid(logits).tolist()
                for ground_truth, score, box in zip(
                    batch_ground_truths, scores, boxes.tolist(), strict=True
                ):
                    converted = normalized_cxcywh_to_original_xyxy(
                        box, original_size=(ground_truth.width, ground_truth.height)
                    )
                    if converted is not None:
                        predictions.append(
                            Detection(ground_truth.image_id, float(score), converted)
                        )
            else:
                with autocast:
                    outputs = model(device_images)
                if len(outputs) != batch_size:
                    raise ValueError(
                        "torchvision detector output batch size does not match its input."
                    )
                for ground_truth, output in zip(
                    batch_ground_truths, outputs, strict=True
                ):
                    scores = output["scores"].detach().cpu().reshape(-1).tolist()
                    boxes = output["boxes"].detach().cpu().reshape(-1, 4).tolist()
                    labels = output.get("labels")
                    if labels is None:
                        labels_list = [1] * len(scores)
                    else:
                        labels_list = labels.detach().cpu().reshape(-1).tolist()
                    for score, box, label in zip(
                        scores, boxes, labels_list, strict=True
                    ):
                        if int(label) == 1:
                            predictions.append(
                                Detection(
                                    ground_truth.image_id, float(score), _to_box(box)
                                )
                            )
    return (predictions, ground_truths)


def evaluate_detector(
    model: Any,
    dataloader: Iterable[Any],
    model_type: str,
    device: Any,
    threshold: float = 0.0,
    *,
    match_iou_threshold: float = 0.5,
    precision: str = "float32",
) -> dict[str, Any]:
    """Collect raw detections and evaluate one explicitly fixed threshold."""
    from endo_bbox_vla.metrics import compute_detection_metrics

    predictions, ground_truths = collect_detector_records(
        model, dataloader, model_type, device, precision=precision
    )
    return compute_detection_metrics(
        predictions,
        ground_truths,
        confidence_threshold=threshold,
        match_iou_threshold=match_iou_threshold,
    )


def main(argv=None) -> int:
    """Evaluate saved predictions or a frozen static experiment configuration."""
    import argparse
    from pathlib import Path

    from endo_bbox_vla.data import load_config
    from endo_bbox_vla.metrics import evaluate_frame_predictions

    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    saved = commands.add_parser("predictions")
    saved.add_argument("--input", type=Path, required=True)
    saved.add_argument("--threshold", type=float, required=True)
    saved.add_argument("--fps", type=float, default=5.0)
    saved.add_argument("--output", type=Path, required=True)
    static = commands.add_parser("static")
    static.add_argument("--config", type=Path, required=True)
    static.add_argument(
        "--phase", choices=("validation", "internal-test"), default="validation"
    )
    static.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "predictions":
        rows = [
            json.loads(line)
            for line in args.input.read_text().splitlines()
            if line.strip()
        ]
        result = evaluate_frame_predictions(
            rows,
            confidence_threshold=args.threshold,
            detector_input_size=256,
            fps=args.fps,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif args.phase == "validation":
        compare_variants(load_config(args.config))
    else:
        evaluate_winner_test(load_config(args.config), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
