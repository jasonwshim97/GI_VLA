"""comparisons — detectors: consolidated research implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from endo_bbox_vla.data import ScenarioEpisodeBalancedSampler, StaticFrameRecord
    from endo_bbox_vla.metrics import Detection, GroundTruth
from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT


def _resolve_repo_path(value: str | Path) -> Path:
    from endo_bbox_vla.config import resolve_path

    return resolve_path(value)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path
    sha256: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise TypeError(f"config section {name!r} must be a mapping")
        return dict(value)

    def path_value(self, section: str, key: str) -> Path:
        return _resolve_repo_path(self.section(section)[key])


ModelName = Literal["faster_rcnn", "detr_r50"]
visual_separate_detector_pipeline_REPO_ROOT = PROJECT_ROOT


def _resolve_baseline_path(value: str | Path) -> Path:
    from endo_bbox_vla.config import resolve_path

    return resolve_path(value)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    raw: dict[str, Any]
    path: Path
    sha256: str
    source: ExperimentConfig

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise TypeError(f"config section {name!r} must be a mapping")
        return dict(value)

    def model(self, name: ModelName) -> dict[str, Any]:
        value = self.section("models").get(name)
        if not isinstance(value, Mapping):
            raise TypeError(f"model section {name!r} must be a mapping")
        return dict(value)

    def workspace(self) -> Path:
        return _resolve_baseline_path(self.section("workspace")["root"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _manifest_for_split(config: ExperimentConfig, split: str) -> Path:
    if split == "train":
        return config.path_value("data", "train_manifest")
    if split == "validation":
        return config.path_value("data", "validation_manifest")
    if split == "test":
        return config.path_value("data", "test_manifest")
    raise ValueError(f"unsupported split {split!r}")


CAMERA_KEY = "observation.images.cam_low"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class VerifiedFrame:
    record: StaticFrameRecord
    image_path: Path
    image_sha256: str
    pixel_sha256: str


def load_verified_frames(
    config: ExperimentConfig, split: str, *, verify_image_files: bool
) -> tuple[VerifiedFrame, ...]:
    from endo_bbox_vla.data import read_frame_manifest
    from endo_bbox_vla.utils import sha256_file

    manifest_path = _manifest_for_split(config, split)
    records = tuple(
        (
            record
            for record in read_frame_manifest(manifest_path)
            if record.split == split
            and record.status in {"positive", "confirmed_negative"}
        )
    )
    if split == "test" and (not records):
        records = tuple(
            (
                record
                for record in read_frame_manifest(manifest_path)
                if record.status in {"positive", "confirmed_negative"}
            )
        )
    if not records:
        raise ValueError(f"no trainable records found for {split}")
    record_by_id = {record.sample_id: record for record in records}
    view_root = config.path_value("data", "verified_cache_root") / split
    cache_manifest_path = view_root / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("kind") != "bbox_vla_qwen_four_camera_action_cache":
        raise ValueError("verified PNG cache has the wrong artifact kind")
    expected_size = list(config.section("data")["original_size"])
    if cache_manifest.get("image_size") != expected_size:
        raise ValueError("verified PNG cache has the wrong image size")
    if cache_manifest.get("bbox_camera_key") != CAMERA_KEY:
        raise ValueError("verified PNG cache does not bind cam_low")
    cache_records_path = view_root / str(cache_manifest["records_file"])
    if cache_manifest.get("records_sha256") != sha256_file(cache_records_path):
        raise ValueError("verified PNG cache records SHA changed")
    if split != "train" and cache_manifest.get("source_manifest_sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("verified PNG cache source manifest SHA changed")
    cached = _read_jsonl(cache_records_path)
    if len(cached) != len(records) or int(
        cache_manifest.get("record_count", -1)
    ) != len(records):
        raise ValueError(
            "verified PNG cache count differs from block-11 visual manifest"
        )
    if {str(row.get("sample_id")) for row in cached} != set(record_by_id):
        raise ValueError(
            "verified PNG cache sample IDs differ from block-11 visual manifest"
        )
    output: list[VerifiedFrame] = []
    for row in cached:
        sample_id = str(row["sample_id"])
        camera = row.get("cameras", {}).get(CAMERA_KEY)
        if not isinstance(camera, Mapping):
            raise ValueError(f"cam_low cache metadata missing for {sample_id}")
        relative = Path(str(camera["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe cached image path for {sample_id}")
        image_path = view_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        expected_sha = str(camera["sha256"])
        if verify_image_files and sha256_file(image_path) != expected_sha:
            raise ValueError(f"cam_low PNG SHA mismatch for {sample_id}")
        output.append(
            VerifiedFrame(
                record=record_by_id[sample_id],
                image_path=image_path,
                image_sha256=expected_sha,
                pixel_sha256=str(camera["pixel_bytes_sha256"]),
            )
        )
    return tuple(output)


class StandaloneDetectorDataset(Dataset[tuple[Tensor, dict[str, Tensor], int]]):
    """Original-resolution RGB frames with one-or-zero detection targets."""

    def __init__(self, frames: Sequence[VerifiedFrame]) -> None:
        self.frames = tuple(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        frame = self.frames[index]
        with Image.open(frame.image_path) as source:
            image = source.convert("RGB")
            if image.size != (640, 480):
                raise ValueError(f"cached frame changed size: {frame.image_path}")
            array = np.array(image, dtype=np.uint8, copy=True)
        pixels = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        record = frame.record
        if record.bbox_xyxy is None:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor([record.bbox_xyxy], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
        area = (
            torch.empty((0,), dtype=torch.float32)
            if boxes.numel() == 0
            else (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(record.image_id, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return (pixels, target, index)


def _collate(
    batch: Sequence[tuple[Tensor, dict[str, Tensor], int]],
) -> tuple[list[Tensor], list[dict[str, Tensor]], list[int]]:
    images, targets, indices = zip(*batch, strict=True)
    return (list(images), list(targets), [int(index) for index in indices])


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _loader(
    config: BaselineConfig,
    dataset: StandaloneDetectorDataset,
    *,
    batch_size: int,
    sampler: ScenarioEpisodeBalancedSampler | None,
) -> DataLoader[Any]:
    workers = int(config.section("training")["num_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
    )


def build_faster_rcnn_detector(
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    min_size: int = 480,
    max_size: int = 640,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Build Faster R-CNN ResNet50-FPN v2 with a polyp/background head.

    Torchvision detection imports and optional COCO-weight downloads happen
    only when this function is called.
    """
    if num_classes < 2:
        raise ValueError(
            "num_classes must include background and at least one foreground class"
        )
    try:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "The high-resolution baseline requires a working torchvision detection installation"
        ) from error
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        progress=progress,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model


def _detr_snapshot(config: BaselineConfig) -> Path:
    from endo_bbox_vla.utils import sha256_file

    model_config = config.model("detr_r50")
    snapshot = Path(str(model_config["local_snapshot"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    pins = {
        "model.safetensors": "model_sha256",
        "config.json": "config_sha256",
        "preprocessor_config.json": "preprocessor_sha256",
    }
    for filename, key in pins.items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != str(model_config[key]):
            raise ValueError(f"pinned DETR artifact changed: {path}")
    return snapshot


def build_model(
    config: BaselineConfig, name: ModelName, *, device: torch.device, pretrained: bool
) -> nn.Module:
    model_config = config.model(name)
    if name == "faster_rcnn":
        model = build_faster_rcnn_detector(
            num_classes=int(model_config["num_classes_including_background"]),
            pretrained=pretrained,
            trainable_backbone_layers=5,
            min_size=int(model_config["min_size"]),
            max_size=int(model_config["max_size"]),
            progress=False,
        )
        model.roi_heads.score_thresh = 0.0
    else:
        from transformers import DetrConfig, DetrForObjectDetection

        snapshot = _detr_snapshot(config)
        if pretrained:
            model = DetrForObjectDetection.from_pretrained(
                snapshot,
                num_labels=int(model_config["num_labels"]),
                id2label={0: "polyp"},
                label2id={"polyp": 0},
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
        else:
            detr_config = DetrConfig.from_pretrained(snapshot, local_files_only=True)
            detr_config.num_labels = int(model_config["num_labels"])
            detr_config.id2label = {0: "polyp"}
            detr_config.label2id = {"polyp": 0}
            model = DetrForObjectDetection(detr_config)
        if int(model.config.num_queries) != int(model_config["num_queries"]):
            raise ValueError("DETR query count differs from the pinned baseline")
    total = sum((parameter.numel() for parameter in model.parameters()))
    if total != int(model_config["expected_total_parameters"]):
        raise ValueError(
            f"{name} parameter count {total} != {model_config['expected_total_parameters']}"
        )
    return model.to(device)


def faster_rcnn_parameter_groups(
    model: nn.Module, *, backbone_lr: float = 2e-05, head_lr: float = 0.0001
) -> list[dict[str, Any]]:
    """Split trainable parameters into disjoint backbone and detection groups."""
    if backbone_lr <= 0 or head_lr <= 0:
        raise ValueError("learning rates must be positive")
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "backbone" or name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    if not backbone_parameters:
        raise ValueError("model has no trainable backbone parameters")
    if not head_parameters:
        raise ValueError("model has no trainable RPN/ROI head parameters")
    return [
        {"params": backbone_parameters, "lr": backbone_lr, "group_name": "backbone"},
        {"params": head_parameters, "lr": head_lr, "group_name": "rpn_roi_heads"},
    ]


def _optimizer(
    config: BaselineConfig, name: ModelName, model: nn.Module
) -> torch.optim.AdamW:
    model_config = config.model(name)
    if name == "faster_rcnn":
        groups = faster_rcnn_parameter_groups(
            model,
            backbone_lr=float(model_config["backbone_learning_rate"]),
            head_lr=float(model_config["head_learning_rate"]),
        )
    else:
        backbone: list[nn.Parameter] = []
        heads: list[nn.Parameter] = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if "backbone" in parameter_name else heads).append(parameter)
        if not backbone or not heads:
            raise ValueError("DETR optimizer parameter split is empty")
        groups = [
            {
                "params": backbone,
                "lr": float(model_config["backbone_learning_rate"]),
                "group_name": "backbone",
            },
            {
                "params": heads,
                "lr": float(model_config["head_learning_rate"]),
                "group_name": "transformer_and_heads",
            },
        ]
    return torch.optim.AdamW(
        groups,
        weight_decay=float(config.section("training")["weight_decay"]),
        fused=True,
    )


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


def _run_root(config: BaselineConfig, name: ModelName) -> Path:
    return config.workspace() / "runs" / name / "seed42"


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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
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
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    torch.cuda.set_rng_state_all(
        [torch.as_tensor(value, dtype=torch.uint8).cpu() for value in state["cuda"]]
    )


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _detr_pixels(
    config: BaselineConfig, images: Sequence[Tensor], device: torch.device
) -> tuple[Tensor, Tensor]:
    model_config = config.model("detr_r50")
    pixels = torch.stack(images).to(device, non_blocking=True)
    mean = torch.tensor(model_config["image_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model_config["image_std"], device=device).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    mask = torch.ones(
        (len(images), pixels.shape[-2], pixels.shape[-1]),
        dtype=torch.long,
        device=device,
    )
    return (pixels, mask)


def _actual_prompt(record: StaticFrameRecord) -> str:
    from endo_bbox_vla.config import CANONICAL_TASK_PROMPTS
    from endo_bbox_vla.data import TASK1_DATASET, TASK4_DATASET

    if record.source_dataset == TASK1_DATASET:
        return CANONICAL_TASK_PROMPTS[0]
    if record.source_dataset == TASK4_DATASET:
        return CANONICAL_TASK_PROMPTS[3]
    raise ValueError(f"unsupported block-11 visual source: {record.source_dataset}")


def _xyxy_to_normalized_cxcywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        (x1 + x2) / (2.0 * 640.0),
        (y1 + y2) / (2.0 * 480.0),
        (x2 - x1) / 640.0,
        (y2 - y1) / 480.0,
    ]


def _prediction_row(
    record: StaticFrameRecord,
    *,
    name: ModelName,
    score: float,
    bbox_xyxy: Sequence[float] | None,
) -> dict[str, Any]:
    prompt = _actual_prompt(record)
    bbox = None if bbox_xyxy is None else [float(value) for value in bbox_xyxy]
    return {
        "schema_version": 1,
        "kind": "e03_separate_detector_frame_prediction",
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
        "present": bbox is not None,
        "score": float(score),
        "bbox_valid": bbox is not None,
        "bbox_xyxy": bbox,
        "bbox_cxcywh_normalized": None
        if bbox is None
        else _xyxy_to_normalized_cxcywh(bbox),
        "prompt": prompt,
        "prompt_mode": "actual",
        "prompt_id": "actual:" + hashlib.sha256(prompt.encode()).hexdigest(),
        "context_generation": 0,
        "context_condition": "independent_cam_low_full_frame",
        "model": name,
    }


def _valid_box(box: Tensor) -> list[float] | None:
    values = box.float().detach().cpu().tolist()
    x1 = min(max(float(values[0]), 0.0), 640.0)
    y1 = min(max(float(values[1]), 0.0), 480.0)
    x2 = min(max(float(values[2]), 0.0), 640.0)
    y2 = min(max(float(values[3]), 0.0), 480.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _predict(
    config: BaselineConfig,
    name: ModelName,
    model: nn.Module,
    dataset: StandaloneDetectorDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    loader = _loader(config, dataset, batch_size=batch_size, sampler=None)
    rows: list[dict[str, Any]] = []
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        for images, _, indices in loader:
            if name == "faster_rcnn":
                outputs = model(
                    [image.to(device, non_blocking=True) for image in images]
                )
                candidates = []
                for output in outputs:
                    if len(output["scores"]) == 0:
                        candidates.append((0.0, None))
                        continue
                    selected = int(torch.argmax(output["scores"]))
                    candidates.append(
                        (
                            float(output["scores"][selected].float().cpu()),
                            _valid_box(output["boxes"][selected]),
                        )
                    )
            else:
                pixels, mask = _detr_pixels(config, images, device)
                outputs = model(pixel_values=pixels, pixel_mask=mask)
                lesion_scores = outputs.logits.float().softmax(dim=-1)[..., 0]
                best_scores, best_indices = lesion_scores.max(dim=1)
                boxes = outputs.pred_boxes.float()
                candidates = []
                for offset in range(len(images)):
                    box = boxes[offset, best_indices[offset]]
                    cx, cy, width, height = box.unbind()
                    xyxy = torch.stack(
                        (
                            (cx - width / 2.0) * 640.0,
                            (cy - height / 2.0) * 480.0,
                            (cx + width / 2.0) * 640.0,
                            (cy + height / 2.0) * 480.0,
                        )
                    )
                    candidates.append(
                        (float(best_scores[offset].cpu()), _valid_box(xyxy))
                    )
            for index, (score, bbox) in zip(indices, candidates, strict=True):
                rows.append(
                    _prediction_row(
                        dataset.frames[index].record,
                        name=name,
                        score=score,
                        bbox_xyxy=bbox,
                    )
                )
    if len(rows) != len(dataset):
        raise RuntimeError("prediction count changed")
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
                    image_id=image_id,
                    score=float(row["score"]),
                    bbox_xyxy=tuple((float(value) for value in bbox)),
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


def validation_report(
    config: BaselineConfig, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
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
        detector_input_size=640,
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


def _detr_labels(
    targets: Sequence[Mapping[str, Tensor]], device: torch.device
) -> list[dict[str, Tensor]]:
    labels: list[dict[str, Tensor]] = []
    for target in targets:
        boxes = target["boxes"].to(device, non_blocking=True)
        if boxes.numel():
            x1, y1, x2, y2 = boxes.unbind(dim=-1)
            normalized = torch.stack(
                (
                    (x1 + x2) / (2.0 * 640.0),
                    (y1 + y2) / (2.0 * 480.0),
                    (x2 - x1) / 640.0,
                    (y2 - y1) / 480.0,
                ),
                dim=-1,
            )
        else:
            normalized = torch.empty((0, 4), dtype=torch.float32, device=device)
        labels.append(
            {
                "class_labels": torch.zeros(
                    (len(normalized),), dtype=torch.long, device=device
                ),
                "boxes": normalized,
            }
        )
    return labels


def _training_loss(
    config: BaselineConfig,
    name: ModelName,
    model: nn.Module,
    images: Sequence[Tensor],
    targets: Sequence[Mapping[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    if name == "faster_rcnn":
        device_images = [image.to(device, non_blocking=True) for image in images]
        device_targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        losses = model(device_images, device_targets)
        return (sum(losses.values()), dict(losses))
    pixels, mask = _detr_pixels(config, images, device)
    outputs = model(
        pixel_values=pixels, pixel_mask=mask, labels=_detr_labels(targets, device)
    )
    if outputs.loss is None:
        raise RuntimeError("DETR training returned no loss")
    return (outputs.loss, dict(outputs.loss_dict or {}))


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


def _model_metadata(
    config: BaselineConfig,
    name: ModelName,
    model: nn.Module,
    *,
    epoch: int,
    global_step: int,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "e03_separate_detector_checkpoint",
        "model": name,
        "epoch": epoch,
        "global_step": global_step,
        "config_sha256": config.sha256,
        "source_e03_config_sha256": config.source.sha256,
        "validation": report,
        "total_parameters": sum(
            (parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameters": sum(
            (
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "selection_split": "validation",
        "internal_test_used": False,
    }


def train(
    config: BaselineConfig, *, name: ModelName, device: str, resume: bool
) -> dict[str, Any]:
    from endo_bbox_vla.data import ScenarioEpisodeBalancedSampler
    from endo_bbox_vla.utils import sha256_file

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("specialist detector training requires CUDA")
    training = config.section("training")
    model_config = config.model(name)
    seed = int(training["seed"])
    _seed_everything(seed)
    train_frames = load_verified_frames(
        config.source, "train", verify_image_files=False
    )
    validation_frames = load_verified_frames(
        config.source, "validation", verify_image_files=False
    )
    train_dataset = StandaloneDetectorDataset(train_frames)
    validation_dataset = StandaloneDetectorDataset(validation_frames)
    sampler = ScenarioEpisodeBalancedSampler(
        [frame.record for frame in train_frames],
        seed=seed,
        samples_per_epoch=int(training["samples_per_epoch"]),
    )
    batch_size = int(model_config["batch_size"])
    accumulation = int(model_config["gradient_accumulation"])
    loader = _loader(config, train_dataset, batch_size=batch_size, sampler=sampler)
    model = build_model(config, name, device=torch_device, pretrained=True)
    optimizer = _optimizer(config, name, model)
    epochs = int(training["epochs"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    scheduler = _scheduler(
        optimizer,
        total_steps=epochs * updates_per_epoch,
        warmup_fraction=float(training["warmup_fraction"]),
    )
    root = _run_root(config, name)
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
    stale_validations = 0
    if last_path.is_file():
        if not resume:
            raise FileExistsError(f"partial {name} training exists: {last_path}")
        state = torch.load(last_path, map_location=torch_device, weights_only=True)
        if state.get("config_sha256") != config.sha256:
            raise ValueError("resume config SHA changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_key = None if state["best_key"] is None else tuple(state["best_key"])
        stale_validations = int(state["stale_validations"])
        _restore_rng_state(state["rng"])
    elif best_path.is_file():
        raise FileExistsError(f"orphaned best checkpoint without report: {best_path}")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    termination = "epochs_complete"
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for batch_index, (images, targets, _) in enumerate(loader):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=bool(training["bf16"])
            ):
                loss, components = _training_loss(
                    config, name, model, images, targets, torch_device
                )
                scaled = loss / accumulation
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError(f"non-finite {name} loss")
            scaled.backward()
            batches += 1
            totals["loss"] += float(loss.detach())
            for key, value in components.items():
                if isinstance(value, Tensor) and value.numel() == 1:
                    totals[key] += float(value.detach())
            update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                loader
            )
            if update:
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(gradient)):
                    raise FloatingPointError(f"non-finite {name} gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        validate = (epoch + 1) % int(
            training["validate_every_epochs"]
        ) == 0 or epoch + 1 == epochs
        validation: dict[str, Any] | None = None
        improved = False
        if validate:
            rows = _predict(
                config,
                name,
                model,
                validation_dataset,
                device=torch_device,
                batch_size=int(model_config["evaluation_batch_size"]),
            )
            validation = validation_report(config, rows)
            validation.update(
                {"model": name, "epoch": epoch, "global_step": global_step}
            )
            _atomic_json(root / f"validation_epoch_{epoch + 1:03d}.json", validation)
            key = _selection_key(validation)
            if best_key is None or key > best_key:
                best_key = key
                stale_validations = 0
                improved = True
                _atomic_torch_save(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "metadata": _model_metadata(
                            config,
                            name,
                            model,
                            epoch=epoch,
                            global_step=global_step,
                            report=validation,
                        ),
                    },
                )
                _atomic_jsonl(root / "best_validation_predictions.jsonl", rows)
            else:
                stale_validations += 1
        epoch_row = {
            "model": name,
            "epoch": epoch,
            "global_step": global_step,
            "batches": batches,
            **{key: value / max(batches, 1) for key, value in totals.items()},
            "learning_rate": [float(value) for value in scheduler.get_last_lr()],
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "validation": validation,
            "validation_improved": improved,
            "stale_validations": stale_validations,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        _atomic_torch_save(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_key": best_key,
                "stale_validations": stale_validations,
                "rng": _rng_state(),
                "config_sha256": config.sha256,
            },
        )
        print(json.dumps(epoch_row, sort_keys=True, allow_nan=False), flush=True)
        if stale_validations >= int(training["early_stopping_validation_patience"]):
            termination = "validation_early_stopping"
            break
    if not best_path.is_file():
        raise RuntimeError(f"{name} produced no validation-selected checkpoint")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    report = {
        "schema_version": 1,
        "kind": "e03_separate_detector_training_report",
        "model": name,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "source_e03_config_sha256": config.source.sha256,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation": checkpoint["metadata"]["validation"],
        "global_step": global_step,
        "termination_reason": termination,
        "sampler": sampler.audit(),
        "total_parameters": checkpoint["metadata"]["total_parameters"],
        "trainable_parameters": checkpoint["metadata"]["trainable_parameters"],
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "internal_test_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(report_path, report)
    return report


def _fixed_threshold_report(
    config: BaselineConfig, rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    from endo_bbox_vla.metrics import evaluate_frame_predictions

    evaluation = evaluate_frame_predictions(
        rows,
        confidence_threshold=threshold,
        detector_input_size=640,
        fps=float(config.section("evaluation")["detector_fps"]),
    )
    return {
        "threshold": threshold,
        "metrics": evaluation["prompt_groups"]["actual"],
        "full_evaluation": evaluation,
    }


def evaluate_internal_test(
    config: BaselineConfig, *, name: ModelName, device: str
) -> dict[str, Any]:
    from endo_bbox_vla.utils import sha256_file

    root = config.workspace() / "internal_test" / name
    report_path = root / "test.report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("internal-test evaluation requires CUDA")
    checkpoint_path = _run_root(config, name) / "best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    metadata = checkpoint["metadata"]
    if (
        metadata.get("config_sha256") != config.sha256
        or metadata.get("model") != name
        or metadata.get("internal_test_used") is not False
    ):
        raise ValueError("validation-selected checkpoint provenance changed")
    threshold = float(metadata["validation"]["threshold_selection"]["threshold"])
    model = build_model(config, name, device=torch_device, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    frames = load_verified_frames(config.source, "test", verify_image_files=False)
    dataset = StandaloneDetectorDataset(frames)
    started = time.monotonic()
    rows = _predict(
        config,
        name,
        model,
        dataset,
        device=torch_device,
        batch_size=int(config.model(name)["evaluation_batch_size"]),
    )
    evaluation = _fixed_threshold_report(config, rows, threshold=threshold)
    result = {
        "schema_version": 1,
        "kind": "e03_separate_detector_internal_test_report",
        "split": "test",
        "split_label": "internal_test",
        "model": name,
        "selection_split": "validation",
        "threshold": threshold,
        "threshold_source_epoch": int(metadata["validation"]["epoch"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config.sha256,
        "prediction_rows": len(rows),
        "metrics": evaluation["metrics"],
        "full_evaluation": evaluation["full_evaluation"],
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_jsonl(root / "test.predictions.jsonl", rows)
    _atomic_json(report_path, result)
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    source = _resolve_repo_path(path)
    payload = source.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(
            "high-resolution spatial-query config must be a schema-v1 mapping"
        )
    detector_size = tuple(raw["data"]["detector_size"])
    if detector_size != (512, 384):
        raise ValueError(
            "the preregistered A--D study requires detector_size [512,384]"
        )
    if raw["data"]["camera_key"] != CAMERA_KEY:
        raise ValueError(
            "high-resolution spatial-query must use cam_low as its only detector input"
        )
    if int(raw["backbone"]["visual_block_stop"]) != 11:
        raise ValueError(
            "high-resolution spatial-query must use zero-based visual block 11"
        )
    return ExperimentConfig(
        raw=raw, path=source, sha256=hashlib.sha256(payload).hexdigest()
    )


def load_baseline_config(path: str | Path) -> BaselineConfig:
    from endo_bbox_vla.utils import sha256_file

    source_path = _resolve_baseline_path(path)
    payload = source_path.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("separate-detector config must be a schema-v1 mapping")
    visual_path = _resolve_baseline_path(raw["source_e03_config"])
    if sha256_file(visual_path) != str(raw["source_e03_config_sha256"]):
        raise ValueError("source block-11 visual config SHA-256 changed")
    source = load_config(visual_path)
    if tuple(raw["data"]["input_size"]) != (640, 480):
        raise ValueError("specialist baselines must use the original 640x480 frame")
    if raw["data"]["camera_key"] != "observation.images.cam_low":
        raise ValueError("specialist baselines must use cam_low")
    if int(raw["models"]["detr_r50"]["num_queries"]) != 100:
        raise ValueError("canonical DETR baseline must retain its 100-query set")
    return BaselineConfig(
        raw=raw,
        path=source_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source=source,
    )


DEFAULT_CONFIG = (
    visual_separate_detector_pipeline_REPO_ROOT / "configs/standalone_detectors.yaml"
)


def main(argv=None):
    """Train or evaluate the completed standalone detector comparisons."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("train", "internal-test"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", choices=("faster_rcnn", "detr_r50"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_baseline_config(args.config)
    if args.command == "train":
        train(config, name=args.name, device=args.device, resume=args.resume)
    else:
        evaluate_internal_test(config, name=args.name, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
