"""Presence, box, heatmap and temporal training objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.network import (
        DecoderLayerOutput,
        SpatialQueryKind,
        TemporalDetectorOutput,
    )
    from endo_bbox_vla.train import TrainingVariant

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


def aligned_generalized_box_iou(boxes1_xyxy: Tensor, boxes2_xyxy: Tensor) -> Tensor:
    """Generalized IoU for aligned ``[N, 4]`` box pairs."""
    if (
        boxes1_xyxy.shape != boxes2_xyxy.shape
        or boxes1_xyxy.ndim != 2
        or boxes1_xyxy.shape[-1] != 4
    ):
        raise ValueError("aligned GIoU inputs must have identical [N, 4] shapes")
    top_left = torch.maximum(boxes1_xyxy[:, :2], boxes2_xyxy[:, :2])
    bottom_right = torch.minimum(boxes1_xyxy[:, 2:], boxes2_xyxy[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area1 = (boxes1_xyxy[:, 2:] - boxes1_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2_xyxy[:, 2:] - boxes2_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=torch.finfo(union.dtype).eps)
    enclosing_top_left = torch.minimum(boxes1_xyxy[:, :2], boxes2_xyxy[:, :2])
    enclosing_bottom_right = torch.maximum(boxes1_xyxy[:, 2:], boxes2_xyxy[:, 2:])
    enclosing_area = (
        (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(dim=-1)
    )
    return iou - (enclosing_area - union) / enclosing_area.clamp(
        min=torch.finfo(enclosing_area.dtype).eps
    )


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


class SpatialDetectionCriterion(nn.Module):
    """Single-target matching with optional dense center supervision."""

    def __init__(
        self,
        *,
        kind: SpatialQueryKind,
        grid_size: tuple[int, int],
        presence_weight: float = 1.0,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        heatmap_weight: float = 1.0,
        focal_presence: bool = False,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.grid_size = grid_size
        self.presence_weight = float(presence_weight)
        self.l1_weight = float(l1_weight)
        self.giou_weight = float(giou_weight)
        self.heatmap_weight = float(heatmap_weight)
        self.focal_presence = bool(focal_presence)

    def forward(
        self, outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        from endo_bbox_vla.network import _gaussian_heatmaps, _sigmoid_focal_loss

        logits = outputs["presence_logits"]
        predicted_boxes = outputs["boxes_cxcywh"]
        presence = (
            targets["presence"].to(device=logits.device, dtype=logits.dtype).flatten()
        )
        wanted_boxes = targets["boxes_cxcywh"].to(
            device=predicted_boxes.device, dtype=predicted_boxes.dtype
        )
        if logits.ndim != 2 or predicted_boxes.shape != (*logits.shape, 4):
            raise ValueError(
                "high-resolution spatial-query outputs require presence [B,K] and boxes [B,K,4]"
            )
        if tuple(wanted_boxes.shape) != (logits.shape[0], 4):
            raise ValueError(
                "high-resolution spatial-query target boxes must have shape [B,4]"
            )
        positive = presence > 0.5
        query_targets = torch.zeros_like(logits)
        selected_boxes = predicted_boxes[:, 0]
        selected_indices = torch.zeros(
            logits.shape[0], dtype=torch.long, device=logits.device
        )
        if bool(positive.any()):
            candidates = predicted_boxes[positive]
            wanted = wanted_boxes[positive][:, None, :].expand_as(candidates)
            l1_cost = (candidates - wanted).abs().sum(dim=-1)
            giou_cost = 1.0 - aligned_generalized_box_iou(
                cxcywh_to_xyxy(candidates.reshape(-1, 4)),
                cxcywh_to_xyxy(wanted.reshape(-1, 4)),
            ).reshape(candidates.shape[:2])
            matched = (l1_cost + 2.0 * giou_cost).detach().argmin(dim=1)
            positive_rows = positive.nonzero(as_tuple=False).flatten()
            query_targets[positive_rows, matched] = 1.0
            selected_indices[positive_rows] = matched
            selected_boxes = predicted_boxes[
                torch.arange(logits.shape[0], device=logits.device), selected_indices
            ]
        presence_loss = (
            _sigmoid_focal_loss(logits, query_targets)
            if self.focal_presence
            else F.binary_cross_entropy_with_logits(logits, query_targets)
        )
        if bool(positive.any()):
            predicted = selected_boxes[positive]
            wanted = wanted_boxes[positive]
            sample_weights = targets.get("box_weights")
            if sample_weights is None:
                weights = torch.ones(
                    len(predicted), device=predicted.device, dtype=predicted.dtype
                )
            else:
                weights = sample_weights.to(
                    device=predicted.device, dtype=predicted.dtype
                )[positive]
            weights = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
            l1_per_sample = (predicted - wanted).abs().mean(dim=-1)
            giou_per_sample = 1.0 - aligned_generalized_box_iou(
                cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(wanted)
            )
            l1_loss = (l1_per_sample * weights).mean()
            giou_loss = (giou_per_sample * weights).mean()
        else:
            connected_zero = predicted_boxes.sum() * 0.0
            l1_loss = connected_zero
            giou_loss = connected_zero
        heatmap_logits = outputs["heatmap_logits"]
        if self.kind == "location_first":
            heatmap_targets = _gaussian_heatmaps(
                wanted_boxes,
                presence.to(device=wanted_boxes.device, dtype=wanted_boxes.dtype),
                self.grid_size,
            ).flatten(1)
            heatmap_loss = _sigmoid_focal_loss(
                heatmap_logits, heatmap_targets, alpha=0.5
            )
        else:
            heatmap_loss = predicted_boxes.sum() * 0.0
        loss = (
            self.presence_weight * presence_loss
            + self.l1_weight * l1_loss
            + self.giou_weight * giou_loss
            + self.heatmap_weight * heatmap_loss
        )
        return {
            "loss": loss,
            "presence_loss": presence_loss,
            "l1_loss": l1_loss,
            "giou_loss": giou_loss,
            "heatmap_loss": heatmap_loss,
            "num_positive": positive.sum(),
        }


def build_shared_detection_criterion(
    *,
    kind: SpatialQueryKind = "location_first",
    grid_size: tuple[int, int] = (16, 16),
    presence_weight: float = 1.0,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    heatmap_weight: float = 1.0,
    focal_presence: bool = True,
) -> SpatialDetectionCriterion:
    """Build the shared-token detector loss for one ablation arm."""
    return SpatialDetectionCriterion(
        kind=kind,
        grid_size=grid_size,
        presence_weight=presence_weight,
        l1_weight=l1_weight,
        giou_weight=giou_weight,
        heatmap_weight=heatmap_weight,
        focal_presence=focal_presence,
    )


def _build_criterion(
    config: ExperimentConfig, variant: TrainingVariant
) -> torch.nn.Module:
    from endo_bbox_vla.config import _variant_config

    loss = config.section("loss")
    settings = _variant_config(config, variant)
    kind = str(settings.get("head_kind", "location_first"))
    heatmap_weight = float(
        settings.get(
            "heatmap_weight", loss["heatmap"] if kind == "location_first" else 0.0
        )
    )
    return build_shared_detection_criterion(
        kind=kind,
        grid_size=tuple(config.section("backbone")["grid_size"]),
        presence_weight=float(loss["presence"]),
        l1_weight=float(loss["l1"]),
        giou_weight=float(loss["giou"]),
        heatmap_weight=heatmap_weight,
        focal_presence=bool(settings.get("focal_presence", True)),
    )


def temporal_deformable_aligned_generalized_box_iou(
    boxes1: Tensor, boxes2: Tensor
) -> Tensor:
    """Aligned generalized IoU for equally-shaped ``[...,4]`` tensors."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(1e-07)
    enclosing_top_left = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosing_bottom_right = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing = (enclosing_bottom_right - enclosing_top_left).clamp_min(0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-07)


def temporal_deformable_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    centers, sizes = (boxes[..., :2], boxes[..., 2:])
    half = sizes / 2
    return torch.cat((centers - half, centers + half), dim=-1)


class TemporalDetectionCriterion(nn.Module):
    """One-GT Hungarian matching plus spatial/temporal auxiliary losses."""

    def __init__(
        self,
        *,
        presence_weight: float = 1.0,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        heatmap_weight: float = 1.0,
        validity_weight: float = 0.5,
        motion_weight: float = 0.25,
        auxiliary_weight: float = 0.5,
        matching_l1_weight: float = 5.0,
        matching_giou_weight: float = 2.0,
    ) -> None:
        super().__init__()
        values = {
            "presence_weight": presence_weight,
            "l1_weight": l1_weight,
            "giou_weight": giou_weight,
            "heatmap_weight": heatmap_weight,
            "validity_weight": validity_weight,
            "motion_weight": motion_weight,
            "auxiliary_weight": auxiliary_weight,
            "matching_l1_weight": matching_l1_weight,
            "matching_giou_weight": matching_giou_weight,
        }
        if any((not math.isfinite(value) or value < 0 for value in values.values())):
            raise ValueError("criterion weights must be finite and non-negative")
        if not any(
            (values[key] > 0 for key in ("presence_weight", "l1_weight", "giou_weight"))
        ):
            raise ValueError("at least one primary detection loss must be positive")
        for name, value in values.items():
            setattr(self, name, float(value))

    @staticmethod
    def _targets(
        targets: Mapping[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        presence_value = targets.get("presence", targets.get("present"))
        boxes_value = targets.get("boxes_cxcywh", targets.get("boxes"))
        if presence_value is None or boxes_value is None:
            raise KeyError("targets require presence and boxes_cxcywh")
        presence = torch.as_tensor(presence_value, device=device, dtype=dtype).flatten()
        boxes = torch.as_tensor(boxes_value, device=device, dtype=dtype)
        validity = torch.as_tensor(
            targets.get("validity", presence), device=device, dtype=dtype
        ).flatten()
        box_mask = torch.as_tensor(
            targets.get("gt_box_mask", targets.get("box_mask", presence > 0.5)),
            device=device,
            dtype=torch.bool,
        ).flatten()
        previous_boxes = torch.as_tensor(
            targets.get("previous_boxes_cxcywh", torch.zeros_like(boxes)),
            device=device,
            dtype=dtype,
        )
        previous_validity = torch.as_tensor(
            targets.get(
                "previous_box_validity", torch.zeros(batch_size, device=device)
            ),
            device=device,
            dtype=torch.bool,
        ).flatten()
        expected = {
            "presence": (presence, (batch_size,)),
            "boxes": (boxes, (batch_size, 4)),
            "validity": (validity, (batch_size,)),
            "box_mask": (box_mask, (batch_size,)),
            "previous_boxes": (previous_boxes, (batch_size, 4)),
            "previous_validity": (previous_validity, (batch_size,)),
        }
        for name, (value, shape) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"target {name} must have shape {shape}")
            if value.is_floating_point() and (not bool(torch.isfinite(value).all())):
                raise ValueError(f"target {name} contains NaN or infinity")
        if bool(((presence < 0) | (presence > 1)).any()) or bool(
            ((validity < 0) | (validity > 1)).any()
        ):
            raise ValueError("presence and validity targets must lie in [0,1]")
        return {name: value for name, (value, _) in expected.items()}

    def _query_losses(
        self, prediction: DecoderLayerOutput, target: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        logits = prediction.presence_logits
        boxes = prediction.boxes_cxcywh
        batch_size, num_queries = logits.shape
        positive = (target["presence"] > 0.5) & target["box_mask"]
        matched = torch.full((batch_size,), -1, device=logits.device, dtype=torch.long)
        positive_indices = positive.nonzero(as_tuple=False).flatten()
        if len(positive_indices):
            with torch.no_grad():
                predicted = boxes[positive_indices]
                wanted = target["boxes"][positive_indices, None].expand(
                    -1, num_queries, -1
                )
                l1_cost = (predicted - wanted).abs().mean(dim=-1)
                giou_cost = 1 - temporal_deformable_aligned_generalized_box_iou(
                    temporal_deformable_cxcywh_to_xyxy(predicted.reshape(-1, 4)),
                    temporal_deformable_cxcywh_to_xyxy(wanted.reshape(-1, 4)),
                ).reshape(-1, num_queries)
                cost = (
                    self.matching_l1_weight * l1_cost
                    + self.matching_giou_weight * giou_cost
                )
                matched[positive_indices] = cost.argmin(dim=1)
        query_presence = torch.zeros_like(logits)
        query_validity = torch.zeros_like(prediction.validity_logits)
        if len(positive_indices):
            chosen = matched[positive_indices]
            query_presence[positive_indices, chosen] = target["presence"][
                positive_indices
            ]
            query_validity[positive_indices, chosen] = target["validity"][
                positive_indices
            ]
        presence_loss = F.binary_cross_entropy_with_logits(logits, query_presence)
        validity_loss = F.binary_cross_entropy_with_logits(
            prediction.validity_logits, query_validity
        )
        if len(positive_indices):
            chosen_boxes = boxes[positive_indices, matched[positive_indices]]
            wanted_boxes = target["boxes"][positive_indices]
            l1_loss = F.l1_loss(chosen_boxes, wanted_boxes)
            giou_loss = (
                1
                - temporal_deformable_aligned_generalized_box_iou(
                    temporal_deformable_cxcywh_to_xyxy(chosen_boxes),
                    temporal_deformable_cxcywh_to_xyxy(wanted_boxes),
                )
            ).mean()
        else:
            l1_loss = boxes.sum() * 0
            giou_loss = boxes.sum() * 0
        motion_mask = positive & target["previous_validity"]
        motion_indices = motion_mask.nonzero(as_tuple=False).flatten()
        if len(motion_indices):
            chosen_motion = prediction.motion_residual[
                motion_indices, matched[motion_indices]
            ]
            motion_target = (
                target["boxes"][motion_indices]
                - target["previous_boxes"][motion_indices]
            )
            motion_loss = F.smooth_l1_loss(chosen_motion, motion_target)
        else:
            motion_loss = prediction.motion_residual.sum() * 0
        return (
            {
                "presence": presence_loss,
                "validity": validity_loss,
                "l1": l1_loss,
                "giou": giou_loss,
                "motion": motion_loss,
                "num_positive": positive.sum(),
                "num_motion": motion_mask.sum(),
            },
            matched,
        )

    @staticmethod
    def _heatmap_target(logits: Tensor, boxes: Tensor, positive: Tensor) -> Tensor:
        batch_size, height, width = logits.shape
        target = torch.zeros_like(logits)
        indices = positive.nonzero(as_tuple=False).flatten()
        if len(indices):
            x = (boxes[indices, 0].clamp(0, 1 - 1e-07) * width).long()
            y = (boxes[indices, 1].clamp(0, 1 - 1e-07) * height).long()
            target[indices, y, x] = 1
        return target

    def forward(
        self, outputs: TemporalDetectorOutput, targets: Mapping[str, Any]
    ) -> dict[str, Tensor]:
        from endo_bbox_vla.network import DecoderLayerOutput

        batch_size = outputs.presence_logits.shape[0]
        target = self._targets(
            targets,
            batch_size=batch_size,
            device=outputs.presence_logits.device,
            dtype=outputs.presence_logits.dtype,
        )
        main_prediction = DecoderLayerOutput(
            presence_logits=outputs.presence_logits,
            validity_logits=outputs.validity_logits,
            boxes_cxcywh=outputs.boxes_cxcywh,
            motion_residual=outputs.motion_residual,
            reference_points=outputs.reference_points,
        )
        main, matched = self._query_losses(main_prediction, target)
        positive = (target["presence"] > 0.5) & target["box_mask"]
        heatmap_target = self._heatmap_target(
            outputs.coarse_heatmap_logits, target["boxes"], positive
        )
        heatmap_loss = F.binary_cross_entropy_with_logits(
            outputs.coarse_heatmap_logits, heatmap_target
        )
        auxiliary_loss = outputs.presence_logits.sum() * 0
        for layer_prediction in outputs.decoder_layer_outputs[:-1]:
            auxiliary, _ = self._query_losses(layer_prediction, target)
            auxiliary_loss = auxiliary_loss + (
                self.presence_weight * auxiliary["presence"]
                + self.validity_weight * auxiliary["validity"]
                + self.l1_weight * auxiliary["l1"]
                + self.giou_weight * auxiliary["giou"]
                + self.motion_weight * auxiliary["motion"]
            )
        total = (
            self.presence_weight * main["presence"]
            + self.validity_weight * main["validity"]
            + self.l1_weight * main["l1"]
            + self.giou_weight * main["giou"]
            + self.heatmap_weight * heatmap_loss
            + self.motion_weight * main["motion"]
            + self.auxiliary_weight * auxiliary_loss
        )
        return {
            "loss": total,
            "presence_loss": main["presence"],
            "validity_loss": main["validity"],
            "l1_loss": main["l1"],
            "giou_loss": main["giou"],
            "heatmap_loss": heatmap_loss,
            "motion_loss": main["motion"],
            "auxiliary_loss": auxiliary_loss,
            "matched_query_indices": matched,
            "num_positive": main["num_positive"],
            "num_motion": main["num_motion"],
        }


class FP32BoxTemporalDetectionCriterion(TemporalDetectionCriterion):
    """Keep presence/heatmap under AMP but force matching and box losses to FP32."""

    @staticmethod
    def _targets(
        targets: Mapping[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        result = TemporalDetectionCriterion._targets(
            targets, batch_size=batch_size, device=device, dtype=dtype
        )
        boxes_value = targets.get("boxes_cxcywh", targets.get("boxes"))
        result["boxes"] = torch.as_tensor(
            boxes_value, device=device, dtype=torch.float32
        ).reshape(batch_size, 4)
        result["previous_boxes"] = torch.as_tensor(
            targets.get("previous_boxes_cxcywh", torch.zeros_like(result["boxes"])),
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, 4)
        for name in ("boxes", "previous_boxes"):
            if not bool(torch.isfinite(result[name]).all()):
                raise ValueError(f"target {name} contains NaN or infinity")
        return result

    def _query_losses(
        self, prediction: DecoderLayerOutput, target: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        logits = prediction.presence_logits
        batch_size, num_queries = logits.shape
        positive = (target["presence"] > 0.5) & target["box_mask"]
        positive_indices = positive.nonzero(as_tuple=False).flatten()
        matched = torch.full((batch_size,), -1, device=logits.device, dtype=torch.long)
        query_presence = torch.zeros_like(logits)
        query_validity = torch.zeros_like(prediction.validity_logits)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            boxes = prediction.boxes_cxcywh.float()
            wanted_all = target["boxes"].float()
            if len(positive_indices):
                with torch.no_grad():
                    predicted = boxes[positive_indices]
                    wanted = wanted_all[positive_indices, None].expand(
                        -1, num_queries, -1
                    )
                    l1_cost = (predicted - wanted).abs().mean(dim=-1)
                    giou_cost = 1 - temporal_deformable_aligned_generalized_box_iou(
                        temporal_deformable_cxcywh_to_xyxy(predicted.reshape(-1, 4)),
                        temporal_deformable_cxcywh_to_xyxy(wanted.reshape(-1, 4)),
                    ).reshape(-1, num_queries)
                    cost = (
                        self.matching_l1_weight * l1_cost
                        + self.matching_giou_weight * giou_cost
                    )
                    matched[positive_indices] = cost.argmin(dim=1)
                chosen_boxes = boxes[positive_indices, matched[positive_indices]]
                wanted_boxes = wanted_all[positive_indices]
                l1_loss = F.l1_loss(chosen_boxes, wanted_boxes)
                giou_loss = (
                    1
                    - temporal_deformable_aligned_generalized_box_iou(
                        temporal_deformable_cxcywh_to_xyxy(chosen_boxes),
                        temporal_deformable_cxcywh_to_xyxy(wanted_boxes),
                    )
                ).mean()
            else:
                l1_loss = boxes.sum() * 0
                giou_loss = boxes.sum() * 0
            motion_mask = positive & target["previous_validity"]
            motion_indices = motion_mask.nonzero(as_tuple=False).flatten()
            motion = prediction.motion_residual.float()
            if len(motion_indices):
                chosen_motion = motion[motion_indices, matched[motion_indices]]
                motion_target = (
                    wanted_all[motion_indices]
                    - target["previous_boxes"].float()[motion_indices]
                )
                motion_loss = F.smooth_l1_loss(chosen_motion, motion_target)
            else:
                motion_loss = motion.sum() * 0
        if len(positive_indices):
            chosen = matched[positive_indices]
            query_presence[positive_indices, chosen] = target["presence"][
                positive_indices
            ]
            query_validity[positive_indices, chosen] = target["validity"][
                positive_indices
            ]
        presence_loss = F.binary_cross_entropy_with_logits(logits, query_presence)
        validity_loss = F.binary_cross_entropy_with_logits(
            prediction.validity_logits, query_validity
        )
        for name, value in {
            "l1": l1_loss,
            "giou": giou_loss,
            "motion": motion_loss,
        }.items():
            if value.dtype != torch.float32:
                raise RuntimeError(f"{name} loss escaped the explicit FP32 path")
        return (
            {
                "presence": presence_loss,
                "validity": validity_loss,
                "l1": l1_loss,
                "giou": giou_loss,
                "motion": motion_loss,
                "num_positive": positive.sum(),
                "num_motion": motion_mask.sum(),
            },
            matched,
        )


class ProjectedOneQueryDetectionCriterion(nn.Module):
    """Original baseline objective: presence + 5 L1 + 2 GIoU."""

    def __init__(
        self,
        *,
        presence_weight: float = 1.0,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
    ) -> None:
        super().__init__()
        values = (presence_weight, l1_weight, giou_weight)
        if any((not math.isfinite(value) or value < 0 for value in values)):
            raise ValueError("baseline loss weights must be finite and non-negative")
        if not any((value > 0 for value in values)):
            raise ValueError("at least one baseline loss weight must be positive")
        self.presence_weight = float(presence_weight)
        self.l1_weight = float(l1_weight)
        self.giou_weight = float(giou_weight)

    def forward(
        self, outputs: Mapping[str, Tensor], targets: Mapping[str, Any]
    ) -> dict[str, Tensor]:
        logits = outputs["presence_logits"]
        boxes = outputs["boxes_cxcywh"]
        if logits.ndim != 1 or tuple(boxes.shape) != (logits.shape[0], 4):
            raise ValueError("baseline outputs must be presence [B] and boxes [B,4]")
        presence_value = targets.get("presence", targets.get("present"))
        target_boxes_value = targets.get("boxes_cxcywh", targets.get("boxes"))
        if presence_value is None or target_boxes_value is None:
            raise KeyError("baseline targets require presence and boxes_cxcywh")
        presence = torch.as_tensor(
            presence_value, device=logits.device, dtype=logits.dtype
        ).flatten()
        target_boxes = torch.as_tensor(
            target_boxes_value, device=boxes.device, dtype=boxes.dtype
        )
        if tuple(presence.shape) != (logits.shape[0],) or tuple(
            target_boxes.shape
        ) != tuple(boxes.shape):
            raise ValueError("baseline target shapes do not match outputs")
        presence_loss = F.binary_cross_entropy_with_logits(logits, presence)
        positive = presence.to(device=boxes.device) > 0.5
        if bool(positive.any()):
            predicted = boxes[positive]
            wanted = target_boxes[positive]
            l1_loss = F.l1_loss(predicted, wanted)
            giou_loss = (
                1
                - aligned_generalized_box_iou(
                    cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(wanted)
                )
            ).mean()
        else:
            l1_loss = boxes.sum() * 0
            giou_loss = boxes.sum() * 0
        loss = (
            self.presence_weight * presence_loss
            + self.l1_weight * l1_loss
            + self.giou_weight * giou_loss
        )
        return {
            "loss": loss,
            "presence_loss": presence_loss,
            "l1_loss": l1_loss,
            "giou_loss": giou_loss,
            "num_positive": positive.sum(),
        }
