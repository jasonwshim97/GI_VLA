"""Static and temporal detection networks and tensor packet types."""

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from endo_bbox_vla.config import ExperimentConfig
    from endo_bbox_vla.train import TrainingVariant

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


SpatialQueryKind = Literal["one_query", "learned_queries", "location_first"]


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, layers: int
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        modules: list[nn.Module] = []
        for index in range(layers):
            modules.append(
                nn.Linear(
                    input_dim if index == 0 else hidden_dim,
                    output_dim if index == layers - 1 else hidden_dim,
                )
            )
            if index != layers - 1:
                modules.append(nn.GELU())
        self.layers = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


def sine_position_encoding_2d(
    rows: int,
    columns: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a deterministic ``[1, rows*columns, hidden_dim]`` encoding."""
    if min(rows, columns, hidden_dim) <= 0 or hidden_dim % 4:
        raise ValueError(
            "rows/columns must be positive and hidden_dim divisible by four"
        )
    quarter = hidden_dim // 4
    frequencies = torch.arange(quarter, device=device, dtype=torch.float32)
    frequencies = 10000.0 ** (-frequencies / max(quarter - 1, 1))
    y = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(columns, device=device, dtype=torch.float32) + 0.5) / columns
    y_angles = 2.0 * math.pi * y[:, None] * frequencies[None, :]
    x_angles = 2.0 * math.pi * x[:, None] * frequencies[None, :]
    y_embedding = torch.cat((y_angles.sin(), y_angles.cos()), dim=-1)
    x_embedding = torch.cat((x_angles.sin(), x_angles.cos()), dim=-1)
    position = torch.cat(
        (
            y_embedding[:, None, :].expand(rows, columns, -1),
            x_embedding[None, :, :].expand(rows, columns, -1),
        ),
        dim=-1,
    )
    return position.reshape(1, rows * columns, hidden_dim).to(dtype=dtype)


def _inverse_sigmoid(value: Tensor, epsilon: float = 1e-05) -> Tensor:
    clipped = value.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log(clipped / (1.0 - clipped))


class SpatialQueryHead(nn.Module):
    """Learned-query or location-first single-lesion head."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        grid_size: tuple[int, int] = (24, 32),
        kind: SpatialQueryKind = "one_query",
        num_queries: int = 4,
        decoder_layers: int = 3,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kind not in {"one_query", "learned_queries", "location_first"}:
            raise ValueError(
                f"unsupported high-resolution spatial-query head kind: {kind!r}"
            )
        if min(input_dim, hidden_dim, *grid_size, decoder_layers, attention_heads) <= 0:
            raise ValueError("head dimensions must be positive")
        if hidden_dim % attention_heads or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by attention_heads and four")
        if kind == "one_query":
            num_queries = 1
        if num_queries <= 0 or num_queries > grid_size[0] * grid_size[1]:
            raise ValueError("num_queries must be within the spatial grid")
        self.kind = kind
        self.grid_size = tuple((int(value) for value in grid_size))
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.query_embedding = nn.Embedding(self.num_queries, hidden_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, layers=3)
        self.heatmap_head = (
            nn.Linear(hidden_dim, 1) if kind == "location_first" else None
        )

    @property
    def num_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def _spatial_references(self, indices: Tensor) -> Tensor:
        rows, columns = self.grid_size
        reference_x = (indices.remainder(columns).to(torch.float32) + 0.5) / columns
        reference_y = (
            indices.div(columns, rounding_mode="floor").to(torch.float32) + 0.5
        ) / rows
        return torch.stack((reference_x, reference_y), dim=-1)

    def forward(
        self, tokens: Tensor, padding_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"expected tokens [B,{self.num_tokens},D], got {tuple(tokens.shape)}"
            )
        batch_size = tokens.shape[0]
        memory = self.input_projection(tokens)
        memory = memory + sine_position_encoding_2d(
            *self.grid_size, self.hidden_dim, device=memory.device, dtype=memory.dtype
        )
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, self.num_tokens), dtype=torch.bool, device=memory.device
            )
        if tuple(padding_mask.shape) != (batch_size, self.num_tokens):
            raise ValueError(
                "padding mask does not match high-resolution spatial-query token grid"
            )
        padding_mask = padding_mask.to(device=memory.device, dtype=torch.bool)
        heatmap_logits: Tensor
        query_indices: Tensor
        references: Tensor
        if self.kind == "location_first":
            assert self.heatmap_head is not None
            heatmap_logits = self.heatmap_head(memory).squeeze(-1)
            masked_logits = heatmap_logits.masked_fill(padding_mask, float("-inf"))
            query_indices = masked_logits.topk(self.num_queries, dim=1).indices
            gathered = memory.gather(
                1, query_indices[:, :, None].expand(-1, -1, self.hidden_dim)
            )
            query = gathered + self.query_embedding.weight[None, :, :]
            references = self._spatial_references(query_indices).to(
                device=memory.device, dtype=memory.dtype
            )
        else:
            heatmap_logits = memory.new_empty((batch_size, 0))
            query_indices = torch.full(
                (batch_size, self.num_queries),
                -1 if self.kind == "learned_queries" else 0,
                dtype=torch.long,
                device=memory.device,
            )
            query = self.query_embedding.weight[None, :, :].expand(batch_size, -1, -1)
            references = memory.new_full((batch_size, self.num_queries, 2), 0.5)
        decoded = self.decoder(query, memory, memory_key_padding_mask=padding_mask)
        raw_boxes = self.box_head(decoded)
        if self.kind == "location_first":
            centers = torch.sigmoid(raw_boxes[..., :2] + _inverse_sigmoid(references))
            boxes = torch.cat((centers, raw_boxes[..., 2:].sigmoid()), dim=-1)
        else:
            boxes = raw_boxes.sigmoid()
        return {
            "presence_logits": self.presence_head(decoded).squeeze(-1),
            "boxes_cxcywh": boxes,
            "heatmap_logits": heatmap_logits,
            "query_indices": query_indices,
            "query_references": references,
        }


GROOT_N17_FIVE_TASK_CHECKPOINT = (
    "Jason-Doldol-Research/task1_5_100_groot_n17_relact_005000"
)

GROOT_N17_FIVE_TASK_REVISION = "efbfbb0b4d13bf482734ead3d77d1ad2ceceac1b"


@dataclass(frozen=True, slots=True)
class GrootCheckpointIdentity:
    """Immutable model identity carried into every token artifact."""

    repo_id: str = GROOT_N17_FIVE_TASK_CHECKPOINT
    revision: str = GROOT_N17_FIVE_TASK_REVISION

    def __post_init__(self) -> None:
        if self.repo_id != GROOT_N17_FIVE_TASK_CHECKPOINT:
            raise ValueError(
                "GR00T feature capture requires the pinned five-task checkpoint"
            )
        if self.revision != GROOT_N17_FIVE_TASK_REVISION:
            raise ValueError(
                "GR00T feature capture requires the pinned immutable revision"
            )


GROOT_N17_MID_BLOCK_INDEX = 11

GROOT_N17_SPATIAL_MERGE_SIZE = 2


@dataclass(frozen=True, slots=True)
class GrootImageTokenSpan:
    """Raw and merged offsets for one selected image in packed Qwen3-VL tokens."""

    batch_index: int
    timestep_index: int
    view_index: int
    camera_key: str
    image_ordinal: int
    grid_thw: tuple[int, int, int]
    raw_start: int
    raw_stop: int
    projected_start: int
    projected_stop: int

    @property
    def raw_spatial_shape(self) -> tuple[int, int]:
        return (self.grid_thw[1], self.grid_thw[2])

    @property
    def raw_num_tokens(self) -> int:
        return self.raw_stop - self.raw_start

    def projected_spatial_shape(self, merge_size: int) -> tuple[int, int]:
        return (self.grid_thw[1] // merge_size, self.grid_thw[2] // merge_size)

    @property
    def projected_num_tokens(self) -> int:
        return self.projected_stop - self.projected_start


_Level = Literal["mid", "projected", "e02", "e03"]


@dataclass(frozen=True, slots=True)
class GrootFeaturePacket:
    """Cam-low raw-mid and final-projected tokens from one GR00T forward."""

    mid_tokens: Tensor
    projected_tokens: Tensor
    mid_padding_mask: Tensor
    projected_padding_mask: Tensor
    mid_spatial_shape: tuple[int, int]
    projected_spatial_shape: tuple[int, int]
    selected_grid_thw: tuple[tuple[int, int, int], ...]
    spans: tuple[GrootImageTokenSpan, ...]
    policy_camera_order: tuple[str, ...]
    camera_key: str
    camera_view_index: int
    timestep_index: int
    identity: GrootCheckpointIdentity
    mid_layer_index: int = GROOT_N17_MID_BLOCK_INDEX
    merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE

    def __post_init__(self) -> None:
        if self.mid_tokens.ndim != 3 or self.projected_tokens.ndim != 3:
            raise ValueError("GR00T mid/projected tokens must both have shape [B,N,D]")
        batch_size = self.mid_tokens.shape[0]
        if (
            self.projected_tokens.shape[0] != batch_size
            or len(self.spans) != batch_size
        ):
            raise ValueError("GR00T packet batch dimensions disagree")
        mid_n = self.mid_tokens.shape[1]
        projected_n = self.projected_tokens.shape[1]
        if self.mid_spatial_shape[0] * self.mid_spatial_shape[1] != mid_n:
            raise ValueError("mid_spatial_shape does not match mid token count")
        if (
            self.projected_spatial_shape[0] * self.projected_spatial_shape[1]
            != projected_n
        ):
            raise ValueError(
                "projected_spatial_shape does not match projected token count"
            )
        expected_mid_mask = (batch_size, mid_n)
        expected_projected_mask = (batch_size, projected_n)
        if (
            self.mid_padding_mask.dtype != torch.bool
            or tuple(self.mid_padding_mask.shape) != expected_mid_mask
        ):
            raise ValueError(f"mid_padding_mask must be bool {expected_mid_mask}")
        if (
            self.projected_padding_mask.dtype != torch.bool
            or tuple(self.projected_padding_mask.shape) != expected_projected_mask
        ):
            raise ValueError(
                f"projected_padding_mask must be bool {expected_projected_mask}"
            )
        tensors = (
            self.mid_tokens,
            self.projected_tokens,
            self.mid_padding_mask,
            self.projected_padding_mask,
        )
        if any((tensor.device != self.mid_tokens.device for tensor in tensors)):
            raise ValueError("all GR00T packet tensors must share a device")
        if self.mid_tokens.requires_grad or self.projected_tokens.requires_grad:
            raise ValueError("captured GR00T features must be detached")
        if (
            not self.mid_tokens.is_floating_point()
            or not self.projected_tokens.is_floating_point()
        ):
            raise TypeError("captured GR00T features must use floating-point dtypes")
        if not bool(torch.isfinite(self.mid_tokens).all()) or not bool(
            torch.isfinite(self.projected_tokens).all()
        ):
            raise ValueError("captured GR00T features must be finite")
        if self.policy_camera_order[self.camera_view_index] != self.camera_key:
            raise ValueError("camera_view_index does not identify camera_key")
        if self.mid_layer_index != GROOT_N17_MID_BLOCK_INDEX:
            raise ValueError(
                "the first GR00T study is pinned to zero-based visual block 11"
            )

    @staticmethod
    def _canonical_level(level: _Level) -> Literal["mid", "projected"]:
        if level in ("projected", "e02"):
            return "projected"
        if level in ("mid", "e03"):
            return "mid"
        raise ValueError(f"unknown GR00T feature level {level!r}")

    def head_inputs(self, level: _Level) -> tuple[Tensor, Tensor]:
        """Return tensors accepted by the existing :class:`OneQueryDetrHead`."""
        canonical = self._canonical_level(level)
        if canonical == "projected":
            return (self.projected_tokens, self.projected_padding_mask)
        return (self.mid_tokens, self.mid_padding_mask)

    def head_kwargs(
        self, level: _Level, *, padded_rows: tuple[int, int] = (0, 0)
    ) -> dict[str, Any]:
        """Return shape kwargs for an projected/block-11-compatible one-query head."""
        canonical = self._canonical_level(level)
        if canonical == "projected":
            input_dim = int(self.projected_tokens.shape[-1])
            grid_size = self.projected_spatial_shape
        else:
            input_dim = int(self.mid_tokens.shape[-1])
            grid_size = self.mid_spatial_shape
        return {
            "input_dim": input_dim,
            "grid_size": grid_size,
            "padded_rows": padded_rows,
        }

    @property
    def shape_metadata(self) -> dict[str, Any]:
        """Serializable schema fields for a separate GR00T token cache."""
        return {
            "feature_family": "groot_n17_qwen3_vl",
            "base_checkpoint": self.identity.repo_id,
            "base_revision": self.identity.revision,
            "camera_key": self.camera_key,
            "policy_camera_order": list(self.policy_camera_order),
            "cam_low_view_index": self.camera_view_index,
            "cam_low_image_call_index": self.camera_view_index,
            "timestep_index": self.timestep_index,
            "selected_grid_thw": [list(value) for value in self.selected_grid_thw],
            "mid_layer_indices": [self.mid_layer_index],
            "mid_shapes": [
                [int(self.mid_tokens.shape[1]), int(self.mid_tokens.shape[2])]
            ],
            "mid_spatial_shapes": [list(self.mid_spatial_shape)],
            "mid_token_order": "time_major_row_major_after_qwen_merge_block_unshuffle",
            "projected_shape": [
                int(self.projected_tokens.shape[1]),
                int(self.projected_tokens.shape[2]),
            ],
            "projected_spatial_shape": list(self.projected_spatial_shape),
            "projected_token_order": "time_major_row_major_merge_grid",
            "mid_padding_mask_shape": [int(self.mid_padding_mask.shape[1])],
            "projected_padding_mask_shape": [int(self.projected_padding_mask.shape[1])],
            "spatial_merge_size": self.merge_size,
            "e02_level": "projected",
            "e03_level": "mid",
        }


class SharedDetectorHead(SpatialQueryHead):
    """Configurable ablation head for a detached GR00T block-11 visual token grid."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        grid_size: tuple[int, int] = (16, 16),
        kind: SpatialQueryKind = "location_first",
        num_queries: int = 4,
        decoder_layers: int = 3,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            kind=kind,
            num_queries=num_queries,
            decoder_layers=decoder_layers,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.input_dim = int(input_dim)

    def forward_packet(self, packet: GrootFeaturePacket) -> dict[str, Tensor]:
        """Run on the block-11 visual feature from the same GR00T action forward."""
        tokens, padding_mask = packet.head_inputs("mid")
        if packet.mid_spatial_shape != self.grid_size:
            raise ValueError(
                f"shared detector packet/head grid mismatch: {packet.mid_spatial_shape} != {self.grid_size}"
            )
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"shared detector packet/head width mismatch: {tokens.shape[-1]} != {self.input_dim}"
            )
        if tokens.requires_grad:
            raise ValueError("shared detector action-forward tokens must be detached")
        return self(tokens, padding_mask)


class SpatiallyInitializedDetectorHead(SharedDetectorHead):
    """Four-query location-first production head retained for compatibility."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        grid_size: tuple[int, int] = (16, 16),
        num_queries: int = 4,
        decoder_layers: int = 3,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            kind="location_first",
            num_queries=num_queries,
            decoder_layers=decoder_layers,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )


def _sigmoid_focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25) -> Tensor:
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = cross_entropy * (1.0 - probability_t).pow(2)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * loss).mean()


def _gaussian_heatmaps(
    boxes: Tensor, presence: Tensor, grid_size: tuple[int, int]
) -> Tensor:
    rows, columns = grid_size
    y = (torch.arange(rows, device=boxes.device, dtype=boxes.dtype) + 0.5) / rows
    x = (torch.arange(columns, device=boxes.device, dtype=boxes.dtype) + 0.5) / columns
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    center_x = boxes[:, 0, None, None]
    center_y = boxes[:, 1, None, None]
    sigma_x = (boxes[:, 2, None, None] / 6.0).clamp_min(0.75 / columns)
    sigma_y = (boxes[:, 3, None, None] / 6.0).clamp_min(0.75 / rows)
    squared = ((grid_x - center_x) / sigma_x).square() + (
        (grid_y - center_y) / sigma_y
    ).square()
    heatmaps = torch.exp(-0.5 * squared)
    return heatmaps * presence[:, None, None]


GROOT_CAM_LOW_KEY = "observation.images.cam_low"

GROOT_N17_MID_DIM = 1024

GROOT_N17_PROJECTED_DIM = 2048


def _camera_view_index(
    camera_order: Sequence[str], camera_key: str
) -> tuple[tuple[str, ...], int]:
    order = tuple((str(value) for value in camera_order))
    if not order or any((not value for value in order)):
        raise ValueError("camera_order must contain non-empty camera names")
    if len(set(order)) != len(order):
        raise ValueError("camera_order must not contain duplicate camera names")
    matches = [index for index, value in enumerate(order) if value == camera_key]
    if len(matches) != 1:
        raise ValueError(
            f"camera_order must contain {camera_key!r} exactly once, got {order}"
        )
    return (order, matches[0])


@dataclass(slots=True)
class _CapturedVisual:
    mid: list[Tensor]
    projected: list[Tensor]


def plan_groot_camera_token_spans(
    image_grid_thw: Tensor,
    *,
    camera_order: Sequence[str],
    camera_key: str,
    batch_size: int,
    timestep_index: int = -1,
    merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE,
) -> tuple[GrootImageTokenSpan, ...]:
    """Resolve one camera by name in Qwen's batch -> time -> view image order.

    ``image_grid_thw`` is authoritative for token counts.  ``batch_size`` and
    ``camera_order`` provide the labels which Qwen's flat grid tensor does not
    retain.  Every selected item must be a still image (grid ``t == 1``) because
    the detection heads consume a two-dimensional spatial grid.
    """
    if not isinstance(image_grid_thw, Tensor):
        raise TypeError("image_grid_thw must be a torch.Tensor")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            f"image_grid_thw must have shape [num_images,3], got {tuple(image_grid_thw.shape)}"
        )
    if image_grid_thw.dtype == torch.bool or image_grid_thw.is_floating_point():
        raise TypeError("image_grid_thw must use an integer dtype")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    order, view_index = _camera_view_index(camera_order, camera_key)
    rows = image_grid_thw.detach().to(device="cpu", dtype=torch.int64)
    if bool((rows <= 0).any()):
        raise ValueError("every image_grid_thw value must be positive")
    number_of_images = int(rows.shape[0])
    images_per_timestep = batch_size * len(order)
    if number_of_images % images_per_timestep:
        raise ValueError(
            f"image_grid_thw row count is incompatible with batch_size and camera_order: {number_of_images} rows for B={batch_size}, V={len(order)}"
        )
    number_of_timesteps = number_of_images // images_per_timestep
    selected_timestep = timestep_index
    if selected_timestep < 0:
        selected_timestep += number_of_timesteps
    if not 0 <= selected_timestep < number_of_timesteps:
        raise IndexError(
            f"timestep_index {timestep_index} is outside the packed horizon {number_of_timesteps}"
        )
    grid_rows = [tuple((int(value) for value in row.tolist())) for row in rows]
    raw_counts: list[int] = []
    projected_counts: list[int] = []
    for grid_t, grid_h, grid_w in grid_rows:
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"Qwen grid {(grid_t, grid_h, grid_w)} is not divisible by merge_size={merge_size}"
            )
        raw_counts.append(grid_t * grid_h * grid_w)
        projected_counts.append(
            grid_t * (grid_h // merge_size) * (grid_w // merge_size)
        )
    raw_offsets = [0]
    projected_offsets = [0]
    for raw_count, projected_count in zip(raw_counts, projected_counts, strict=True):
        raw_offsets.append(raw_offsets[-1] + raw_count)
        projected_offsets.append(projected_offsets[-1] + projected_count)
    spans: list[GrootImageTokenSpan] = []
    for batch_index in range(batch_size):
        image_ordinal = (
            batch_index * number_of_timesteps * len(order)
            + selected_timestep * len(order)
            + view_index
        )
        grid = grid_rows[image_ordinal]
        if grid[0] != 1:
            raise ValueError(
                f"selected {camera_key!r} item must have image grid t=1, got {grid} at ordinal {image_ordinal}"
            )
        spans.append(
            GrootImageTokenSpan(
                batch_index=batch_index,
                timestep_index=selected_timestep,
                view_index=view_index,
                camera_key=camera_key,
                image_ordinal=image_ordinal,
                grid_thw=grid,
                raw_start=raw_offsets[image_ordinal],
                raw_stop=raw_offsets[image_ordinal + 1],
                projected_start=projected_offsets[image_ordinal],
                projected_stop=projected_offsets[image_ordinal + 1],
            )
        )
    return tuple(spans)


_Result = TypeVar("_Result")


def unshuffle_qwen_raw_tokens(
    tokens: Tensor,
    *,
    grid_thw: tuple[int, int, int],
    merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE,
) -> Tensor:
    """Convert Qwen's merge-block token order to time-major row-major order."""
    if tokens.ndim != 2:
        raise ValueError(
            f"raw visual tokens must have shape [N,D], got {tuple(tokens.shape)}"
        )
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    if min(grid_t, grid_h, grid_w, merge_size) <= 0:
        raise ValueError("grid_thw and merge_size must be positive")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("grid height and width must be divisible by merge_size")
    expected_tokens = grid_t * grid_h * grid_w
    if tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} raw tokens for grid {grid_thw}, got {tokens.shape[0]}"
        )
    width = tokens.shape[-1]
    return (
        tokens.reshape(
            grid_t,
            grid_h // merge_size,
            grid_w // merge_size,
            merge_size,
            merge_size,
            width,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(grid_t * grid_h * grid_w, width)
        .contiguous()
    )


def _require_tensor(name: str, value: object, *, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tuple(value.shape)}"
        )
    return value


def _require_finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and (not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} contains NaN or infinity")


def _require_integer(name: str, value: Tensor) -> None:
    if value.dtype == torch.bool or value.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype")


@dataclass(frozen=True, slots=True)
class GrootTemporalFeaturePacket:
    """Pinned visual spatial tokens plus optional exact DiT conditioning.

    PyTorch padding-mask convention is used: ``True`` means padding.  Mid and
    conditioning masks and spatial shapes are independent because Qwen's
    spatial merger changes 16 x 16 visual tokens into 8 x 8 language-sequence
    slots.  Conditioning source metadata remains independent from current
    frame metadata so a runtime can explicitly reuse a stale semantic context.
    """

    mid_tokens: Tensor
    mid_padding_mask: Tensor
    mid_spatial_shape: tuple[int, int]
    frame_ids: Tensor
    timestamps: Tensor
    prompt_id: tuple[str, ...]
    detector_generation: Tensor
    dit_conditioning_tokens: Tensor | None = None
    dit_conditioning_padding_mask: Tensor | None = None
    dit_conditioning_spatial_shape: tuple[int, int] | None = None
    dit_sequence_positions: Tensor | None = None
    dit_source_frame_ids: Tensor | None = None
    dit_source_timestamps: Tensor | None = None
    dit_prompt_id: tuple[str, ...] | None = None
    dit_context_generation: Tensor | None = None
    selected_grid_thw: tuple[tuple[int, int, int], ...] = ()
    spans: tuple[GrootImageTokenSpan, ...] = ()
    policy_camera_order: tuple[str, ...] = ()
    camera_key: str = GROOT_CAM_LOW_KEY
    camera_view_index: int = -1
    timestep_index: int = -1
    identity: GrootCheckpointIdentity = GrootCheckpointIdentity()
    mid_layer_index: int = GROOT_N17_MID_BLOCK_INDEX
    merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE
    mid_feature_dim: int = GROOT_N17_MID_DIM
    dit_conditioning_dim: int = GROOT_N17_PROJECTED_DIM

    def __post_init__(self) -> None:
        mid = _require_tensor("mid_tokens", self.mid_tokens, ndim=3)
        batch_size, num_mid_tokens, mid_dim = mid.shape
        if mid_dim != self.mid_feature_dim:
            raise ValueError(
                f"mid_tokens width must be {self.mid_feature_dim}, got {mid_dim}"
            )
        rows, columns = self.mid_spatial_shape
        if min(rows, columns) <= 0 or rows * columns != num_mid_tokens:
            raise ValueError("mid_spatial_shape product must equal the mid token count")
        mid_mask = _require_tensor("mid_padding_mask", self.mid_padding_mask, ndim=2)
        if mid_mask.dtype != torch.bool or tuple(mid_mask.shape) != (
            batch_size,
            num_mid_tokens,
        ):
            raise ValueError("mid_padding_mask must be bool [B,N_mid]")
        if mid_mask.device != mid.device:
            raise ValueError("mid_padding_mask and mid_tokens must share a device")
        if bool(mid_mask.all(dim=1).any()):
            raise ValueError(
                "each mid feature row must contain at least one valid token"
            )
        if mid.requires_grad:
            raise ValueError("captured GR00T mid features must be detached")
        if not mid.is_floating_point():
            raise TypeError("mid_tokens must use a floating-point dtype")
        _require_finite("mid_tokens", mid)
        metadata = {
            "frame_ids": _require_tensor("frame_ids", self.frame_ids, ndim=1),
            "timestamps": _require_tensor("timestamps", self.timestamps, ndim=1),
            "detector_generation": _require_tensor(
                "detector_generation", self.detector_generation, ndim=1
            ),
        }
        for name, value in metadata.items():
            if tuple(value.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
            if value.device != mid.device:
                raise ValueError(f"{name} and mid_tokens must share a device")
        _require_integer("frame_ids", metadata["frame_ids"])
        _require_integer("detector_generation", metadata["detector_generation"])
        if not metadata["timestamps"].is_floating_point():
            raise TypeError("timestamps must use a floating-point dtype")
        _require_finite("timestamps", metadata["timestamps"])
        if bool((metadata["timestamps"] < 0).any()):
            raise ValueError("timestamps must be non-negative")
        if not isinstance(self.prompt_id, tuple) or len(self.prompt_id) != batch_size:
            raise ValueError("prompt_id must contain one string per batch element")
        if any((not isinstance(value, str) or not value for value in self.prompt_id)):
            raise ValueError("prompt_id entries must be non-empty strings")
        conditioning_fields = (
            self.dit_conditioning_tokens,
            self.dit_conditioning_padding_mask,
            self.dit_conditioning_spatial_shape,
            self.dit_sequence_positions,
            self.dit_source_frame_ids,
            self.dit_source_timestamps,
            self.dit_prompt_id,
            self.dit_context_generation,
        )
        if self.dit_conditioning_tokens is None:
            if any((value is not None for value in conditioning_fields[1:])):
                raise ValueError(
                    "DiT conditioning metadata must be absent when tokens are absent"
                )
        else:
            if any((value is None for value in conditioning_fields[1:])):
                raise ValueError(
                    "DiT conditioning requires its independent mask/grid/source metadata"
                )
            conditioning = _require_tensor(
                "dit_conditioning_tokens", self.dit_conditioning_tokens, ndim=3
            )
            if conditioning.shape[0] != batch_size:
                raise ValueError("mid and DiT conditioning batch dimensions differ")
            num_conditioning_tokens = int(conditioning.shape[1])
            if conditioning.shape[2] != self.dit_conditioning_dim:
                raise ValueError(
                    f"dit_conditioning_tokens width must be {self.dit_conditioning_dim}, got {conditioning.shape[2]}"
                )
            assert self.dit_conditioning_spatial_shape is not None
            dit_rows, dit_columns = self.dit_conditioning_spatial_shape
            if (
                min(dit_rows, dit_columns) <= 0
                or dit_rows * dit_columns != num_conditioning_tokens
            ):
                raise ValueError(
                    "dit_conditioning_spatial_shape product must equal the conditioning token count"
                )
            conditioning_mask = _require_tensor(
                "dit_conditioning_padding_mask",
                self.dit_conditioning_padding_mask,
                ndim=2,
            )
            if conditioning_mask.dtype != torch.bool or tuple(
                conditioning_mask.shape
            ) != (batch_size, num_conditioning_tokens):
                raise ValueError("dit_conditioning_padding_mask must be bool [B,N_dit]")
            sequence_positions = _require_tensor(
                "dit_sequence_positions", self.dit_sequence_positions, ndim=2
            )
            if tuple(sequence_positions.shape) != (batch_size, num_conditioning_tokens):
                raise ValueError("dit_sequence_positions must have shape [B,N_dit]")
            _require_integer("dit_sequence_positions", sequence_positions)
            if bool((sequence_positions[~conditioning_mask] < 0).any()):
                raise ValueError(
                    "valid DiT tokens require non-negative source sequence positions"
                )
            for batch_index in range(batch_size):
                valid_positions = sequence_positions[
                    batch_index, ~conditioning_mask[batch_index]
                ]
                if valid_positions.numel() > 1 and bool(
                    (valid_positions[1:] <= valid_positions[:-1]).any()
                ):
                    raise ValueError(
                        "valid dit_sequence_positions must preserve strictly increasing Qwen order"
                    )
            source_metadata = {
                "dit_source_frame_ids": _require_tensor(
                    "dit_source_frame_ids", self.dit_source_frame_ids, ndim=1
                ),
                "dit_source_timestamps": _require_tensor(
                    "dit_source_timestamps", self.dit_source_timestamps, ndim=1
                ),
                "dit_context_generation": _require_tensor(
                    "dit_context_generation", self.dit_context_generation, ndim=1
                ),
            }
            for name, value in source_metadata.items():
                if tuple(value.shape) != (batch_size,):
                    raise ValueError(f"{name} must have shape [B]")
            _require_integer(
                "dit_source_frame_ids", source_metadata["dit_source_frame_ids"]
            )
            _require_integer(
                "dit_context_generation", source_metadata["dit_context_generation"]
            )
            if not source_metadata["dit_source_timestamps"].is_floating_point():
                raise TypeError("dit_source_timestamps must use a floating-point dtype")
            if (
                not isinstance(self.dit_prompt_id, tuple)
                or len(self.dit_prompt_id) != batch_size
            ):
                raise ValueError(
                    "dit_prompt_id must contain one string per batch element"
                )
            if any(
                (
                    not isinstance(value, str) or not value
                    for value in self.dit_prompt_id
                )
            ):
                raise ValueError("dit_prompt_id entries must be non-empty strings")
            tensors = (
                conditioning,
                conditioning_mask,
                sequence_positions,
                *source_metadata.values(),
            )
            if any((value.device != mid.device for value in tensors)):
                raise ValueError(
                    "all GR00T temporal packet tensors must share a device"
                )
            if conditioning.requires_grad:
                raise ValueError("captured DiT conditioning features must be detached")
            if not conditioning.is_floating_point():
                raise TypeError(
                    "dit_conditioning_tokens must use a floating-point dtype"
                )
            _require_finite("dit_conditioning_tokens", conditioning)
            _require_finite(
                "dit_source_timestamps", source_metadata["dit_source_timestamps"]
            )
            age = self.context_age
            available = (~conditioning_mask).any(dim=1)
            if bool((age[available] < 0).any()):
                raise ValueError(
                    "available DiT conditioning cannot come from the future"
                )
            if bool(
                (
                    source_metadata["dit_source_frame_ids"][available]
                    > metadata["frame_ids"][available]
                ).any()
            ):
                raise ValueError(
                    "available DiT conditioning cannot come from a future frame"
                )
            if bool((age[~available] != -1).any()):
                raise ValueError(
                    "fully padded DiT conditioning rows must have context_age -1"
                )
        if self.selected_grid_thw and len(self.selected_grid_thw) != batch_size:
            raise ValueError(
                "selected_grid_thw must be empty or contain one grid per batch row"
            )
        if self.spans and len(self.spans) != batch_size:
            raise ValueError("spans must be empty or contain one span per batch row")
        if self.mid_layer_index != GROOT_N17_MID_BLOCK_INDEX:
            raise ValueError(
                "GR00T temporal capture is pinned to zero-based visual block 11"
            )
        if min(self.merge_size, self.mid_feature_dim, self.dit_conditioning_dim) <= 0:
            raise ValueError("feature dimensions and merge_size must be positive")
        if self.policy_camera_order:
            if not 0 <= self.camera_view_index < len(self.policy_camera_order):
                raise ValueError("camera_view_index is outside policy_camera_order")
            if self.policy_camera_order[self.camera_view_index] != self.camera_key:
                raise ValueError("camera_view_index does not identify camera_key")

    @property
    def batch_size(self) -> int:
        return int(self.mid_tokens.shape[0])

    @property
    def num_tokens(self) -> int:
        """Number of spatial detector tokens (the visual mid grid)."""
        return int(self.mid_tokens.shape[1])

    @property
    def conditioning_available(self) -> Tensor:
        if self.dit_conditioning_tokens is None:
            return torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.mid_tokens.device
            )
        assert self.dit_conditioning_padding_mask is not None
        assert self.dit_prompt_id is not None
        token_available = (~self.dit_conditioning_padding_mask).any(dim=1)
        prompt_matches = torch.tensor(
            [
                current == source
                for current, source in zip(self.prompt_id, self.dit_prompt_id)
            ],
            dtype=torch.bool,
            device=self.mid_tokens.device,
        )
        return token_available & prompt_matches

    @property
    def context_age(self) -> Tensor:
        if self.dit_source_timestamps is None:
            return self.timestamps.new_full((self.batch_size,), -1.0)
        age = self.timestamps - self.dit_source_timestamps.to(
            dtype=self.timestamps.dtype
        )
        if self.dit_conditioning_padding_mask is not None:
            available = (~self.dit_conditioning_padding_mask).any(dim=1)
            age = torch.where(available, age, age.new_full((), -1.0))
        return age

    @property
    def context_generation(self) -> Tensor:
        """Conditioning generation, or ``-1`` when no conditioning is present."""
        if self.dit_context_generation is None:
            return self.detector_generation.new_full((self.batch_size,), -1)
        return self.dit_context_generation

    @property
    def shape_metadata(self) -> dict[str, Any]:
        conditioning_shape = None
        if self.dit_conditioning_tokens is not None:
            conditioning_shape = [
                int(self.dit_conditioning_tokens.shape[1]),
                int(self.dit_conditioning_tokens.shape[2]),
            ]
        return {
            "feature_family": "groot_n17_temporal_qwen3_vl",
            "base_checkpoint": self.identity.repo_id,
            "base_revision": self.identity.revision,
            "camera_key": self.camera_key,
            "policy_camera_order": list(self.policy_camera_order),
            "cam_low_view_index": self.camera_view_index,
            "timestep_index": self.timestep_index,
            "mid_layer_indices": [self.mid_layer_index],
            "mid_shape": [int(self.mid_tokens.shape[1]), int(self.mid_tokens.shape[2])],
            "mid_spatial_shape": list(self.mid_spatial_shape),
            "mid_source": "qwen3_vl_visual_block_11_pre_merger",
            "mid_token_order": "time_major_row_major_after_qwen_merge_block_unshuffle",
            "dit_conditioning_shape": conditioning_shape,
            "dit_conditioning_spatial_shape": None
            if self.dit_conditioning_spatial_shape is None
            else list(self.dit_conditioning_spatial_shape),
            "dit_conditioning_source": "action_head_vlln_then_vl_self_attention_output",
            "dit_conditioning_is_exact_diffusion_encoder_hidden_states": True,
            "spatial_merge_size": self.merge_size,
        }

    def detached(self) -> GrootTemporalFeaturePacket:
        tensor_fields = {
            "mid_tokens": self.mid_tokens.detach(),
            "mid_padding_mask": self.mid_padding_mask.detach(),
            "frame_ids": self.frame_ids.detach(),
            "timestamps": self.timestamps.detach(),
            "detector_generation": self.detector_generation.detach(),
        }
        for name in (
            "dit_conditioning_tokens",
            "dit_conditioning_padding_mask",
            "dit_sequence_positions",
            "dit_source_frame_ids",
            "dit_source_timestamps",
            "dit_context_generation",
        ):
            value = getattr(self, name)
            tensor_fields[name] = None if value is None else value.detach()
        return replace(self, **tensor_fields)

    def without_dit_conditioning(self) -> GrootTemporalFeaturePacket:
        """Return a local-only packet without manufacturing semantic tokens."""
        return replace(
            self,
            dit_conditioning_tokens=None,
            dit_conditioning_padding_mask=None,
            dit_conditioning_spatial_shape=None,
            dit_sequence_positions=None,
            dit_source_frame_ids=None,
            dit_source_timestamps=None,
            dit_prompt_id=None,
            dit_context_generation=None,
        )


def _normalise_camera_order(
    camera_order: Sequence[str], camera_key: str
) -> tuple[tuple[str, ...], int]:
    order = tuple((str(value) for value in camera_order))
    if not order or any((not value for value in order)):
        raise ValueError("camera_order must contain non-empty camera names")
    if len(set(order)) != len(order):
        raise ValueError("camera_order must not contain duplicate camera names")
    matches = [index for index, value in enumerate(order) if value == camera_key]
    if len(matches) != 1:
        raise ValueError(
            f"camera_order must contain {camera_key!r} exactly once, got {order}"
        )
    return (order, matches[0])


@dataclass(slots=True)
class _CapturedTemporalFeatures:
    mid: list[Tensor]
    dit_conditioning: list[Tensor]


@dataclass(frozen=True, slots=True)
class _CapturedBackboneInput:
    """The exact multimodal tensors seen at Qwen3Backbone.forward entry."""

    input_ids: Tensor
    image_grid_thw: Tensor
    attention_mask: Tensor
    invocation_id: int


_groot_temporal_Result = TypeVar("_Result")

_POLICY_CAPTURE_LOCKS_GUARD = threading.Lock()

_POLICY_CAPTURE_LOCKS: weakref.WeakKeyDictionary[nn.Module, Any] = (
    weakref.WeakKeyDictionary()
)


def _policy_capture_lock(policy: nn.Module) -> Any:
    with _POLICY_CAPTURE_LOCKS_GUARD:
        lock = _POLICY_CAPTURE_LOCKS.get(policy)
        if lock is None:
            lock = threading.RLock()
            _POLICY_CAPTURE_LOCKS[policy] = lock
        return lock


def _tensor(name: str, value: object, *, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tuple(value.shape)}"
        )
    return value


def _finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and (not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} contains NaN or infinity")


@dataclass(frozen=True, slots=True)
class DecoderLayerOutput:
    """Auxiliary prediction tensors emitted after one decoder layer."""

    presence_logits: Tensor
    validity_logits: Tensor
    boxes_cxcywh: Tensor
    motion_residual: Tensor
    reference_points: Tensor

    def __post_init__(self) -> None:
        presence = _tensor("presence_logits", self.presence_logits, ndim=2)
        batch_size, num_queries = presence.shape
        if num_queries not in (1, 2):
            raise ValueError("endo_bbox_vla supports one or two queries")
        shapes = {
            "validity_logits": (batch_size, num_queries),
            "boxes_cxcywh": (batch_size, num_queries, 4),
            "motion_residual": (batch_size, num_queries, 4),
            "reference_points": (batch_size, num_queries, 2),
        }
        for name, shape in shapes.items():
            value = _tensor(name, getattr(self, name))
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            _finite(name, value)
        if bool(((self.boxes_cxcywh < 0) | (self.boxes_cxcywh > 1)).any()):
            raise ValueError("decoder boxes must lie in [0,1]")


DecoderType = Literal["dense", "deformable"]

FeatureMode = Literal[
    "mid_only", "projected_baseline", "mid_projected", "true_post_vla", "post_only"
]


class detr_head_MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_layers):
            in_features = input_dim if index == 0 else hidden_dim
            out_features = output_dim if index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            if index < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


@dataclass(frozen=True, slots=True)
class AdaptedFeaturePyramid:
    levels: tuple[Tensor, ...]
    masks: tuple[Tensor, ...]
    fused_tokens: Tensor
    semantic_context: Tensor
    fusion_gate: Tensor
    semantic_gate: Tensor
    spatial_feature_available: Tensor


class TemporalFeaturePacket(Protocol):
    """Metadata contract consumed by the temporal part of the detector.

    Feature-family-specific tensors remain private to the injected feature
    adapter.  This intentionally small structural contract lets the temporal
    query logic operate on Pi0.5 ``FeaturePacket`` instances and on packets
    whose token geometry differs (for example GR00T), without weakening the
    validation guarantees of either packet type.
    """

    frame_ids: Tensor
    timestamps: Tensor
    detector_generation: Tensor

    @property
    def batch_size(self) -> int: ...


def _integer(name: str, value: Tensor) -> None:
    if value.dtype == torch.bool or value.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype")


@dataclass(frozen=True, slots=True)
class TemporalQueryState:
    """Explicit causal lesion-query state; no mutable state lives in the head."""

    query: Tensor
    reference_points: Tensor
    reference_box: Tensor
    confidence: Tensor
    validity: Tensor
    age: Tensor
    generation: Tensor
    frame_ids: Tensor
    timestamps: Tensor

    def __post_init__(self) -> None:
        query = _tensor("query", self.query, ndim=2)
        batch_size = int(query.shape[0])
        values = {
            "reference_points": _tensor(
                "reference_points", self.reference_points, ndim=2
            ),
            "reference_box": _tensor("reference_box", self.reference_box, ndim=2),
            "confidence": _tensor("confidence", self.confidence, ndim=1),
            "validity": _tensor("validity", self.validity, ndim=1),
            "age": _tensor("age", self.age, ndim=1),
            "generation": _tensor("generation", self.generation, ndim=1),
            "frame_ids": _tensor("frame_ids", self.frame_ids, ndim=1),
            "timestamps": _tensor("timestamps", self.timestamps, ndim=1),
        }
        if tuple(values["reference_points"].shape) != (batch_size, 2):
            raise ValueError("reference_points must have shape [B,2]")
        if tuple(values["reference_box"].shape) != (batch_size, 4):
            raise ValueError("reference_box must have shape [B,4]")
        for name in (
            "confidence",
            "validity",
            "age",
            "generation",
            "frame_ids",
            "timestamps",
        ):
            if tuple(values[name].shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if any((value.device != query.device for value in values.values())):
            raise ValueError("all temporal-state tensors must share a device")
        if values["validity"].dtype != torch.bool:
            raise TypeError("validity must use bool dtype")
        _integer("generation", values["generation"])
        _integer("frame_ids", values["frame_ids"])
        for name in ("confidence", "age", "timestamps"):
            if not values[name].is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
        for name, value in {"query": query, **values}.items():
            _finite(name, value)
        for name in ("reference_points", "reference_box", "confidence"):
            if bool(((values[name] < 0) | (values[name] > 1)).any()):
                raise ValueError(f"{name} must lie in [0,1]")
        if bool((values["age"] < 0).any()):
            raise ValueError("age must be non-negative")

    @property
    def batch_size(self) -> int:
        return int(self.query.shape[0])

    @property
    def valid(self) -> Tensor:
        return self.validity

    def detached(self) -> TemporalQueryState:
        return TemporalQueryState(
            **{field.name: getattr(self, field.name).detach() for field in fields(self)}
        )


@dataclass(frozen=True, slots=True)
class TemporalDetectorOutput:
    """Detector predictions; the next state is returned separately."""

    presence_logits: Tensor
    validity_logits: Tensor
    boxes_cxcywh: Tensor
    motion_residual: Tensor
    queries: Tensor
    reference_points: Tensor
    coarse_reference_points: Tensor
    memory_gate: Tensor
    fusion_gate: Tensor
    semantic_gate: Tensor
    coarse_heatmap_logits: Tensor
    spatial_feature_available: Tensor
    decoder_layer_outputs: tuple[DecoderLayerOutput, ...]

    def __post_init__(self) -> None:
        logits = _tensor("presence_logits", self.presence_logits, ndim=2)
        batch_size, num_queries = logits.shape
        if num_queries not in (1, 2):
            raise ValueError("endo_bbox_vla supports one or two queries")
        queries = _tensor("queries", self.queries, ndim=3)
        fusion_gate = _tensor("fusion_gate", self.fusion_gate, ndim=3)
        heatmap = _tensor("coarse_heatmap_logits", self.coarse_heatmap_logits, ndim=3)
        shapes = {
            "validity_logits": (batch_size, num_queries),
            "boxes_cxcywh": (batch_size, num_queries, 4),
            "motion_residual": (batch_size, num_queries, 4),
            "queries": (batch_size, num_queries, queries.shape[-1]),
            "reference_points": (batch_size, num_queries, 2),
            "coarse_reference_points": (batch_size, 2),
            "memory_gate": (batch_size,),
            "fusion_gate": (batch_size, fusion_gate.shape[1], 1),
            "semantic_gate": (batch_size,),
            "coarse_heatmap_logits": tuple(heatmap.shape),
            "spatial_feature_available": (batch_size,),
        }
        if tuple(heatmap.shape[:1]) != (batch_size,):
            raise ValueError("coarse_heatmap_logits must have shape [B,H,W]")
        for name, shape in shapes.items():
            value = _tensor(name, getattr(self, name))
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(value.shape)}"
                )
            _finite(name, value)
        if self.spatial_feature_available.dtype != torch.bool:
            raise TypeError("spatial_feature_available must use bool dtype")
        for name in (
            "boxes_cxcywh",
            "reference_points",
            "memory_gate",
            "fusion_gate",
            "semantic_gate",
        ):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"{name} must lie in [0,1]")
        if (
            not isinstance(self.decoder_layer_outputs, tuple)
            or not self.decoder_layer_outputs
        ):
            raise ValueError("decoder_layer_outputs must be a non-empty tuple")
        if any(
            (
                value.presence_logits.shape != logits.shape
                for value in self.decoder_layer_outputs
            )
        ):
            raise ValueError(
                "decoder-layer outputs must match final batch/query dimensions"
            )

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class PooledVLAContext:
    """Post-PaliGemma semantics plus exact prompt/source provenance.

    ``fresh=True`` means the context came from the same action-prefix forward.
    ``as_stale`` is the only supported conversion for inter-frame reuse.
    """

    pooled_context: Tensor
    source_frame_ids: Tensor
    source_timestamps: Tensor
    prompt_id: tuple[str, ...]
    context_generation: Tensor
    fresh: Tensor

    def __post_init__(self) -> None:
        pooled = _tensor("pooled_context", self.pooled_context, ndim=2)
        batch_size = int(pooled.shape[0])
        values = {
            "source_frame_ids": _tensor(
                "source_frame_ids", self.source_frame_ids, ndim=1
            ),
            "source_timestamps": _tensor(
                "source_timestamps", self.source_timestamps, ndim=1
            ),
            "context_generation": _tensor(
                "context_generation", self.context_generation, ndim=1
            ),
            "fresh": _tensor("fresh", self.fresh, ndim=1),
        }
        for name, value in values.items():
            if tuple(value.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
            if value.device != pooled.device:
                raise ValueError(f"{name} and pooled_context must share a device")
        _integer("source_frame_ids", values["source_frame_ids"])
        _integer("context_generation", values["context_generation"])
        if not values["source_timestamps"].is_floating_point():
            raise TypeError("source_timestamps must use a floating-point dtype")
        if values["fresh"].dtype != torch.bool:
            raise TypeError("fresh must use bool dtype")
        if not isinstance(self.prompt_id, tuple) or len(self.prompt_id) != batch_size:
            raise ValueError("prompt_id must contain one string per batch element")
        if any((not isinstance(value, str) or not value for value in self.prompt_id)):
            raise ValueError("prompt_id entries must be non-empty strings")
        _finite("pooled_context", pooled)
        _finite("source_timestamps", values["source_timestamps"])

    @property
    def batch_size(self) -> int:
        return int(self.pooled_context.shape[0])

    def as_stale(self) -> PooledVLAContext:
        return PooledVLAContext(
            pooled_context=self.pooled_context.detach(),
            source_frame_ids=self.source_frame_ids.detach(),
            source_timestamps=self.source_timestamps.detach(),
            prompt_id=self.prompt_id,
            context_generation=self.context_generation.detach(),
            fresh=torch.zeros_like(self.fresh),
        )

    def detached(self) -> PooledVLAContext:
        return PooledVLAContext(
            pooled_context=self.pooled_context.detach(),
            source_frame_ids=self.source_frame_ids.detach(),
            source_timestamps=self.source_timestamps.detach(),
            prompt_id=self.prompt_id,
            context_generation=self.context_generation.detach(),
            fresh=self.fresh.detach(),
        )


@dataclass(frozen=True, slots=True)
class FeaturePacket:
    """Frozen features for one detector call.

    ``mid_tokens`` are exact outputs of the listed zero-based SigLIP encoder
    blocks. ``projected_tokens`` are pre-LM multimodal-projector tokens.
    ``post_vla_tokens`` are the cam-low slice of the final PaliGemma prefix
    hidden state and exist only when computed for this same frame/prompt.
    """

    mid_tokens: tuple[Tensor, ...]
    mid_layer_indices: tuple[int, ...]
    projected_tokens: Tensor
    padding_mask: Tensor
    spatial_shape: tuple[int, int]
    frame_ids: Tensor
    timestamps: Tensor
    context_age: Tensor
    prompt_id: tuple[str, ...]
    detector_generation: Tensor
    context_generation: Tensor
    post_vla_tokens: Tensor | None = None
    vla_context: PooledVLAContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mid_tokens, tuple):
            raise TypeError("mid_tokens must be a tuple")
        if (
            not isinstance(self.mid_layer_indices, tuple)
            or len(self.mid_layer_indices) != len(self.mid_tokens)
            or len(set(self.mid_layer_indices)) != len(self.mid_layer_indices)
            or any(
                (
                    not isinstance(value, int) or value < 0
                    for value in self.mid_layer_indices
                )
            )
        ):
            raise ValueError(
                "mid_layer_indices must identify every mid-token level exactly once"
            )
        projected = _tensor("projected_tokens", self.projected_tokens, ndim=3)
        batch_size, num_tokens, _ = projected.shape
        rows, columns = self.spatial_shape
        if rows <= 0 or columns <= 0 or rows * columns != num_tokens:
            raise ValueError("spatial_shape product must equal projected token count")
        for index, tokens in enumerate(self.mid_tokens):
            tokens = _tensor(f"mid_tokens[{index}]", tokens, ndim=3)
            if tokens.shape[:2] != (batch_size, num_tokens):
                raise ValueError(f"mid_tokens[{index}] must have leading shape [B,N]")
            if tokens.device != projected.device:
                raise ValueError("all feature tensors must share a device")
            _finite(f"mid_tokens[{index}]", tokens)
        mask = _tensor("padding_mask", self.padding_mask, ndim=2)
        if tuple(mask.shape) != (batch_size, num_tokens) or mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [B,N]")
        metadata = {
            "frame_ids": _tensor("frame_ids", self.frame_ids, ndim=1),
            "timestamps": _tensor("timestamps", self.timestamps, ndim=1),
            "context_age": _tensor("context_age", self.context_age, ndim=1),
            "detector_generation": _tensor(
                "detector_generation", self.detector_generation, ndim=1
            ),
            "context_generation": _tensor(
                "context_generation", self.context_generation, ndim=1
            ),
        }
        for name, value in metadata.items():
            if tuple(value.shape) != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
            if value.device != projected.device:
                raise ValueError(f"{name} and projected_tokens must share a device")
        _integer("frame_ids", metadata["frame_ids"])
        _integer("detector_generation", metadata["detector_generation"])
        _integer("context_generation", metadata["context_generation"])
        if not metadata["timestamps"].is_floating_point():
            raise TypeError("timestamps must use a floating-point dtype")
        if not metadata["context_age"].is_floating_point():
            raise TypeError("context_age must use a floating-point dtype")
        if bool((metadata["timestamps"] < 0).any()):
            raise ValueError("timestamps must be non-negative")
        age = metadata["context_age"]
        valid_age = (age == -1) | (age >= 0)
        if not bool(torch.isfinite(age).all()) or not bool(valid_age.all()):
            raise ValueError("context_age must be -1 (unavailable) or non-negative")
        if not isinstance(self.prompt_id, tuple) or len(self.prompt_id) != batch_size:
            raise ValueError("prompt_id must contain one string per batch element")
        if any((not isinstance(value, str) or not value for value in self.prompt_id)):
            raise ValueError("prompt_id entries must be non-empty strings")
        if self.post_vla_tokens is not None:
            post = _tensor("post_vla_tokens", self.post_vla_tokens, ndim=3)
            if post.shape[:2] != (batch_size, num_tokens):
                raise ValueError("post_vla_tokens must have leading shape [B,N]")
            if post.device != projected.device:
                raise ValueError(
                    "post_vla_tokens and projected_tokens must share a device"
                )
            _finite("post_vla_tokens", post)
        if self.vla_context is None:
            if bool((age != -1).any()):
                raise ValueError("context_age must be -1 without vla_context")
        else:
            context = self.vla_context
            if (
                context.batch_size != batch_size
                or context.pooled_context.device != projected.device
            ):
                raise ValueError("vla_context batch/device must match feature packet")
            if context.prompt_id != self.prompt_id:
                raise ValueError("feature packet and VLA context prompt_id differ")
            if not torch.equal(
                context.context_generation, metadata["context_generation"]
            ):
                raise ValueError("feature packet and VLA context generation differ")
            expected_age = metadata["timestamps"] - context.source_timestamps.to(
                metadata["timestamps"]
            )
            if bool((expected_age < 0).any()) or not torch.allclose(
                age, expected_age, atol=1e-05, rtol=1e-05
            ):
                raise ValueError(
                    "context_age must equal current minus context source timestamp"
                )
            if self.post_vla_tokens is not None and (not bool(context.fresh.all())):
                raise ValueError(
                    "spatial post_vla_tokens require fresh same-forward context"
                )
        _finite("projected_tokens", projected)
        _finite("timestamps", metadata["timestamps"])

    @property
    def batch_size(self) -> int:
        return int(self.projected_tokens.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.projected_tokens.shape[1])

    def detached(self) -> FeaturePacket:
        return FeaturePacket(
            mid_tokens=tuple((value.detach() for value in self.mid_tokens)),
            mid_layer_indices=self.mid_layer_indices,
            projected_tokens=self.projected_tokens.detach(),
            padding_mask=self.padding_mask.detach(),
            spatial_shape=self.spatial_shape,
            frame_ids=self.frame_ids.detach(),
            timestamps=self.timestamps.detach(),
            context_age=self.context_age.detach(),
            prompt_id=self.prompt_id,
            detector_generation=self.detector_generation.detach(),
            context_generation=self.context_generation.detach(),
            post_vla_tokens=None
            if self.post_vla_tokens is None
            else self.post_vla_tokens.detach(),
            vla_context=None
            if self.vla_context is None
            else self.vla_context.detached(),
        )


class MultiLevelGatedFusion(nn.Module):
    """Exact local/context gated fusion from the endo_bbox_vla plan.

    ``L = Linear(LN(mid))`` and ``C = Dropout(Linear(LN(post)))``.
    For fresh token-level context, ``fused = L + sigmoid(MLP([L,C])) * C``;
    the final gate bias starts at -4.  Stale pooled semantics never enter the
    spatial tensor and are exposed separately with an age-decay gate.
    """

    VALID_MODES = frozenset(
        {
            "mid_only",
            "projected_baseline",
            "mid_projected",
            "true_post_vla",
            "post_only",
        }
    )

    def __init__(
        self,
        *,
        mid_dims: Sequence[int] = (1152,),
        projected_dim: int = 2048,
        post_vla_dim: int = 2048,
        hidden_dim: int = 256,
        feature_mode: FeatureMode = "true_post_vla",
        post_dropout: float = 0.2,
        semantic_decay_seconds: float = 1.0,
        max_context_age_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        dims = tuple((int(value) for value in mid_dims))
        if not dims or any((value <= 0 for value in dims)):
            raise ValueError("mid_dims must contain positive feature dimensions")
        if min(projected_dim, post_vla_dim, hidden_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if feature_mode not in self.VALID_MODES:
            raise ValueError(f"unsupported feature_mode {feature_mode!r}")
        if not 0 <= post_dropout < 1:
            raise ValueError("post_dropout must be in [0,1)")
        if semantic_decay_seconds <= 0 or max_context_age_seconds <= 0:
            raise ValueError("semantic age-decay scales must be positive")
        self.feature_mode: FeatureMode = feature_mode
        self.hidden_dim = int(hidden_dim)
        self.semantic_decay_seconds = float(semantic_decay_seconds)
        self.max_context_age_seconds = float(max_context_age_seconds)
        self.mid_projections = nn.ModuleList(
            (
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim))
                for dim in dims
            )
        )
        self.projected_projection = nn.Sequential(
            nn.LayerNorm(projected_dim), nn.Linear(projected_dim, hidden_dim)
        )
        self.post_projection = nn.Sequential(
            nn.LayerNorm(post_vla_dim), nn.Linear(post_vla_dim, hidden_dim)
        )
        self.post_dropout = nn.Dropout(post_dropout)
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        final_gate = self.fusion_gate_mlp[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, -4.0)

    @staticmethod
    def _reshape(tokens: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
        batch_size, _, hidden_dim = tokens.shape
        rows, columns = spatial_shape
        return tokens.transpose(1, 2).reshape(batch_size, hidden_dim, rows, columns)

    def _mid_local(self, packet: FeaturePacket) -> Tensor:
        if len(packet.mid_tokens) != len(self.mid_projections):
            raise ValueError(
                "FeaturePacket mid level count does not match fusion adapter"
            )
        projected = [
            adapter(tokens)
            for adapter, tokens in zip(
                self.mid_projections, packet.mid_tokens, strict=True
            )
        ]
        return torch.stack(projected, dim=0).mean(dim=0)

    def _require_context(self, packet: FeaturePacket) -> None:
        if packet.vla_context is None:
            raise RuntimeError(
                f"{self.feature_mode} requires an explicit PooledVLAContext provider"
            )

    def _semantic(
        self, packet: FeaturePacket, reference: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_size = packet.batch_size
        if packet.vla_context is None:
            return (
                reference.new_zeros(batch_size, self.hidden_dim),
                reference.new_zeros(batch_size),
            )
        context = packet.vla_context
        semantic = self.post_projection(
            context.pooled_context.to(dtype=reference.dtype)
        )
        age = packet.context_age.to(device=reference.device, dtype=reference.dtype)
        gate = torch.exp(-age.clamp_min(0) / self.semantic_decay_seconds)
        gate = gate * (age <= self.max_context_age_seconds).to(dtype=gate.dtype)
        return (semantic, gate)

    def forward(self, packet: FeaturePacket) -> AdaptedFeaturePyramid:
        mode = self.feature_mode
        batch_size, num_tokens = (packet.batch_size, packet.num_tokens)
        zeros_gate = packet.projected_tokens.new_zeros(batch_size, num_tokens, 1)
        available = torch.ones(
            batch_size, device=packet.projected_tokens.device, dtype=torch.bool
        )
        if mode in {"mid_only", "mid_projected", "true_post_vla"}:
            local = self._mid_local(packet)
        elif mode == "projected_baseline":
            local = self.projected_projection(packet.projected_tokens)
        else:
            local = packet.projected_tokens.new_zeros(
                batch_size, num_tokens, self.hidden_dim
            )
        if mode == "mid_projected":
            context_tokens = self.post_dropout(
                self.projected_projection(packet.projected_tokens).to(dtype=local.dtype)
            )
            fusion_gate = torch.sigmoid(
                self.fusion_gate_mlp(torch.cat((local, context_tokens), -1))
            )
            fused = local + fusion_gate * context_tokens
        elif mode == "true_post_vla":
            self._require_context(packet)
            context = packet.vla_context
            assert context is not None
            if bool(context.fresh.any()) and packet.post_vla_tokens is None:
                raise RuntimeError(
                    "fresh true_post_vla requires same-forward spatial post_vla_tokens"
                )
            if packet.post_vla_tokens is None:
                fused, fusion_gate = (local, zeros_gate)
            else:
                context_tokens = self.post_dropout(
                    self.post_projection(packet.post_vla_tokens).to(dtype=local.dtype)
                )
                fusion_gate = torch.sigmoid(
                    self.fusion_gate_mlp(torch.cat((local, context_tokens), -1))
                )
                fused = local + fusion_gate * context_tokens
        elif mode == "post_only":
            self._require_context(packet)
            context = packet.vla_context
            assert context is not None
            if bool(context.fresh.any()) and packet.post_vla_tokens is None:
                raise RuntimeError(
                    "fresh post_only requires same-forward spatial post_vla_tokens"
                )
            if packet.post_vla_tokens is None:
                fused, fusion_gate = (local, zeros_gate)
                available = torch.zeros_like(available)
            else:
                fused = self.post_dropout(self.post_projection(packet.post_vla_tokens))
                fusion_gate = torch.ones_like(zeros_gate)
        else:
            fused, fusion_gate = (local, zeros_gate)
        if mode in {"true_post_vla", "post_only"}:
            semantic, semantic_gate = self._semantic(packet, fused)
        else:
            semantic = fused.new_zeros(batch_size, self.hidden_dim)
            semantic_gate = fused.new_zeros(batch_size)
        level = self._reshape(fused, packet.spatial_shape)
        mask = packet.padding_mask.reshape(batch_size, *packet.spatial_shape)
        return AdaptedFeaturePyramid(
            levels=(level,),
            masks=(mask,),
            fused_tokens=fused,
            semantic_context=semantic,
            fusion_gate=fusion_gate,
            semantic_gate=semantic_gate,
            spatial_feature_available=available,
        )


class GridSampleDeformableCrossAttention(nn.Module):
    """Small differentiable sparse cross-attention implemented with grid_sample."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_levels <= 0 or num_points <= 0:
            raise ValueError("num_levels and num_points must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.head_dim = hidden_dim // num_heads
        total = num_heads * num_levels * num_points
        self.sampling_offsets = nn.Linear(hidden_dim, total * 2)
        self.attention_weights = nn.Linear(hidden_dim, total)
        self.value_projections = nn.ModuleList(
            (
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
                for _ in range(num_levels)
            )
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True).clamp_min(
            1e-06
        )
        bias = directions[:, None, None, :].repeat(
            1, self.num_levels, self.num_points, 1
        )
        radii = torch.arange(1, self.num_points + 1, dtype=torch.float32)
        bias = bias * radii[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias.reshape(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        queries: Tensor,
        reference_points: Tensor,
        levels: tuple[Tensor, ...],
        masks: tuple[Tensor, ...],
    ) -> Tensor:
        if queries.ndim != 3:
            raise ValueError("queries must have shape [B,Q,D]")
        batch_size, num_queries, hidden_dim = queries.shape
        if hidden_dim != self.hidden_dim or tuple(reference_points.shape) != (
            batch_size,
            num_queries,
            2,
        ):
            raise ValueError("query/reference shapes do not match deformable attention")
        if len(levels) != self.num_levels or len(masks) != self.num_levels:
            raise ValueError(
                "feature level count does not match attention configuration"
            )
        offsets = self.sampling_offsets(queries).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(queries).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = weights.softmax(dim=-1).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )
        accumulated = queries.new_zeros(
            batch_size, num_queries, self.num_heads, self.head_dim
        )
        for level_index, (feature, mask, projection) in enumerate(
            zip(levels, masks, self.value_projections, strict=True)
        ):
            if feature.ndim != 4 or feature.shape[:2] != (batch_size, hidden_dim):
                raise ValueError("each feature level must be [B,D,H,W]")
            height, width = (int(feature.shape[-2]), int(feature.shape[-1]))
            if tuple(mask.shape) != (batch_size, height, width):
                raise ValueError("each feature mask must be [B,H,W]")
            value = projection(feature.masked_fill(mask[:, None], 0))
            value = value.reshape(
                batch_size * self.num_heads, self.head_dim, height, width
            )
            normalizer = queries.new_tensor((width, height))
            level_offsets = torch.tanh(offsets[:, :, :, level_index])
            level_offsets = level_offsets * self.num_points / normalizer
            locations = reference_points[:, :, None, None, :] + level_offsets
            grid = locations.permute(0, 2, 1, 3, 4).reshape(
                batch_size * self.num_heads, num_queries, self.num_points, 2
            )
            sampled = F.grid_sample(
                value,
                grid.mul(2).sub(1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(
                batch_size, self.num_heads, self.head_dim, num_queries, self.num_points
            ).permute(0, 3, 1, 4, 2)
            level_weights = weights[:, :, :, level_index].unsqueeze(-1)
            accumulated = accumulated + (sampled * level_weights).sum(dim=3)
        return self.dropout(
            self.output_projection(
                accumulated.reshape(batch_size, num_queries, hidden_dim)
            )
        )


def _temporal_deformable_inverse_sigmoid(values: Tensor, eps: float = 1e-05) -> Tensor:
    values = values.clamp(min=eps, max=1 - eps)
    return torch.log(values / (1 - values))


class _DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        num_points: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = GridSampleDeformableCrossAttention(
            hidden_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.reference_delta = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.reference_delta.weight)
        nn.init.zeros_(self.reference_delta.bias)

    def forward(
        self,
        queries: Tensor,
        reference_points: Tensor,
        levels: tuple[Tensor, ...],
        masks: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(queries)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + attended
        queries = queries + self.cross_attention(
            self.norm2(queries), reference_points, levels, masks
        )
        queries = queries + self.feedforward(self.norm3(queries))
        reference_points = torch.sigmoid(
            _temporal_deformable_inverse_sigmoid(reference_points)
            + self.reference_delta(queries)
        )
        return (queries, reference_points)


class _DenseDecoderLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, *, num_heads: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.reference_delta = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.reference_delta.weight)
        nn.init.zeros_(self.reference_delta.bias)

    def forward(
        self,
        queries: Tensor,
        reference_points: Tensor,
        levels: tuple[Tensor, ...],
        masks: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(queries)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + attended
        memory = torch.cat(
            [level.flatten(2).transpose(1, 2) for level in levels], dim=1
        )
        padding_mask = torch.cat([mask.flatten(1) for mask in masks], dim=1)
        attended, _ = self.cross_attention(
            self.norm2(queries),
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        queries = queries + self.feedforward(self.norm3(queries))
        reference_points = torch.sigmoid(
            _temporal_deformable_inverse_sigmoid(reference_points)
            + self.reference_delta(queries)
        )
        return (queries, reference_points)


class TemporalPolypHead(nn.Module):
    """Configurable endo_bbox_vla head with explicit temporal input/output state."""

    def __init__(
        self,
        *,
        mid_dims: Sequence[int] = (1152,),
        projected_dim: int = 2048,
        post_vla_dim: int = 2048,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_mode: FeatureMode = "true_post_vla",
        decoder_type: DecoderType = "deformable",
        num_queries: int = 2,
        use_previous_box_reference: bool = True,
        use_previous_query: bool = True,
        use_validity_gate: bool = True,
        stale_memory_training: bool = True,
        memory_confidence_threshold: float = 0.5,
        memory_gate_temperature: float = 10.0,
        memory_time_decay_seconds: float = 1.0,
        max_time_gap_seconds: float = 1.0,
        max_frame_gap: int = 30,
        semantic_decay_seconds: float = 1.0,
        max_context_age_seconds: float = 2.0,
        feature_adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("endo_bbox_vla is fixed to exactly two decoder layers")
        if decoder_type not in {"dense", "deformable"}:
            raise ValueError("decoder_type must be dense or deformable")
        if num_queries not in (1, 2):
            raise ValueError("num_queries must be 1 or 2")
        if num_queries == 1 and any(
            (use_previous_box_reference, use_previous_query, use_validity_gate)
        ):
            raise ValueError(
                "one-query ablation must disable all temporal-memory flags"
            )
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if dim_feedforward <= 0 or num_points <= 0:
            raise ValueError("feedforward and sampling-point counts must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= memory_confidence_threshold <= 1:
            raise ValueError("memory_confidence_threshold must be in [0,1]")
        if (
            min(
                memory_gate_temperature,
                memory_time_decay_seconds,
                max_time_gap_seconds,
                max_frame_gap,
            )
            <= 0
        ):
            raise ValueError("temporal gate scales and gaps must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.decoder_type: DecoderType = decoder_type
        self.use_previous_box_reference = bool(use_previous_box_reference)
        self.use_previous_query = bool(use_previous_query)
        self.use_validity_gate = bool(use_validity_gate)
        self.stale_memory_training = bool(stale_memory_training)
        self.memory_confidence_threshold = float(memory_confidence_threshold)
        self.memory_gate_temperature = float(memory_gate_temperature)
        self.memory_time_decay_seconds = float(memory_time_decay_seconds)
        self.max_time_gap_seconds = float(max_time_gap_seconds)
        self.max_frame_gap = int(max_frame_gap)
        if feature_adapter is not None and (not isinstance(feature_adapter, nn.Module)):
            raise TypeError("feature_adapter must be a torch.nn.Module")
        self.adapter = (
            MultiLevelGatedFusion(
                mid_dims=mid_dims,
                projected_dim=projected_dim,
                post_vla_dim=post_vla_dim,
                hidden_dim=hidden_dim,
                feature_mode=feature_mode,
                post_dropout=0.2,
                semantic_decay_seconds=semantic_decay_seconds,
                max_context_age_seconds=max_context_age_seconds,
            )
            if feature_adapter is None
            else feature_adapter
        )
        self.spatial_position = detr_head_MLP(2, hidden_dim, hidden_dim, num_layers=2)
        self.coarse_heatmap = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.local_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)
        self.memory_projection = nn.Linear(hidden_dim, hidden_dim)
        self.time_projection = detr_head_MLP(2, hidden_dim, hidden_dim, num_layers=2)
        if decoder_type == "deformable":

            def layer_factory() -> nn.Module:
                return _DeformableDecoderLayer(
                    hidden_dim,
                    num_heads=num_heads,
                    num_points=num_points,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
        else:

            def layer_factory() -> nn.Module:
                return _DenseDecoderLayer(
                    hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )

        self.decoder = nn.ModuleList((layer_factory() for _ in range(num_layers)))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.box_head = detr_head_MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.motion_head = detr_head_MLP(hidden_dim, hidden_dim, 4, num_layers=2)
        self.context_box_residual = detr_head_MLP(
            hidden_dim, hidden_dim, 4, num_layers=2
        )
        self.context_reference_residual = detr_head_MLP(
            hidden_dim, hidden_dim, 2, num_layers=2
        )
        nn.init.normal_(self.query_embeddings.weight, std=0.02)
        final_box = self.box_head.layers[-1]
        final_context_box = self.context_box_residual.layers[-1]
        final_context_ref = self.context_reference_residual.layers[-1]
        assert isinstance(final_box, nn.Linear)
        assert isinstance(final_context_box, nn.Linear)
        assert isinstance(final_context_ref, nn.Linear)
        with torch.no_grad():
            final_box.bias[2:].fill_(-2.1972246)
        nn.init.zeros_(final_context_box.weight)
        nn.init.zeros_(final_context_box.bias)
        nn.init.zeros_(final_context_ref.weight)
        nn.init.zeros_(final_context_ref.bias)

    @property
    def feature_mode(self) -> str:
        mode = getattr(self.adapter, "feature_mode", None)
        if not isinstance(mode, str) or not mode:
            raise AttributeError(
                "feature_adapter must expose a non-empty string feature_mode"
            )
        return mode

    @property
    def reset_propagated_query(self) -> Tensor:
        """Generic prior-query state used only for stale clip-start training.

        Runtime ``TemporalQueryState.query`` stores a final-normalized decoder
        query.  A synthetic previous state must therefore use the same feature
        domain.  Reusing the learned propagation-slot embedding avoids adding
        a stale-only parameter (and keeps checkpoint/parameter-count contracts
        unchanged) while allowing gradients through the normal query pathway.
        """
        if self.num_queries != 2 or not self.use_previous_query:
            raise RuntimeError(
                "reset_propagated_query requires the two-query previous-query pathway"
            )
        return self.final_norm(self.query_embeddings.weight[1])

    @property
    def ablation_metadata(self) -> dict[str, Any]:
        return {
            "feature_mode": self.feature_mode,
            "decoder_type": self.decoder_type,
            "num_queries": self.num_queries,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "use_previous_box_reference": self.use_previous_box_reference,
            "use_previous_query": self.use_previous_query,
            "use_validity_gate": self.use_validity_gate,
            "stale_memory_training": self.stale_memory_training,
        }

    @staticmethod
    def _masked_spatial_mean(feature: Tensor, mask: Tensor) -> Tensor:
        valid = (~mask).to(dtype=feature.dtype)[:, None]
        return (feature * valid).sum(dim=(2, 3)) / valid.sum(dim=(2, 3)).clamp_min(1)

    def _positioned_levels(self, pyramid: AdaptedFeaturePyramid) -> tuple[Tensor, ...]:
        result = []
        for level in pyramid.levels:
            batch_size, _, height, width = level.shape
            ys = (
                torch.arange(height, device=level.device, dtype=level.dtype) + 0.5
            ) / height
            xs = (
                torch.arange(width, device=level.device, dtype=level.dtype) + 0.5
            ) / width
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(
                1, height * width, 2
            )
            position = self.spatial_position(coordinates).expand(batch_size, -1, -1)
            result.append(level + position.transpose(1, 2).reshape_as(level))
        return tuple(result)

    def _coarse_reference(self, feature: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, height, width = feature.shape
        if bool(mask.flatten(1).all(dim=1).any()):
            raise ValueError(
                "every feature packet must contain at least one valid token"
            )
        heatmap = self.coarse_heatmap(feature).squeeze(1)
        masked = heatmap.masked_fill(mask, -20.0)
        probabilities = masked.flatten(1).softmax(dim=-1)
        ys = (
            torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5
        ) / height
        xs = (
            torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5
        ) / width
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(
            1, height * width, 2
        )
        reference = (probabilities.unsqueeze(-1) * coordinates).sum(dim=1)
        return (reference, masked)

    @staticmethod
    def _reset_mask(
        reset: bool | Tensor, batch_size: int, device: torch.device
    ) -> Tensor:
        value = torch.as_tensor(reset, device=device, dtype=torch.bool)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if tuple(value.shape) != (batch_size,):
            raise ValueError("reset must be bool or bool tensor [B]")
        return value

    def _memory_gate(
        self,
        packet: TemporalFeaturePacket,
        previous_state: TemporalQueryState | None,
        reset: bool | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.query_embeddings.weight.device
        dtype = self.query_embeddings.weight.dtype
        batch_size = packet.batch_size
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        if previous_state is None or not (
            self.use_previous_box_reference or self.use_previous_query
        ):
            return (zeros, zeros, zeros)
        if (
            previous_state.batch_size != batch_size
            or previous_state.query.shape[1] != self.hidden_dim
        ):
            raise ValueError("previous state is incompatible with current detector")
        if any(
            (
                tensor.device != device
                for tensor in (
                    previous_state.query,
                    previous_state.reference_points,
                    previous_state.reference_box,
                    previous_state.confidence,
                    previous_state.validity,
                    previous_state.age,
                    previous_state.generation,
                    previous_state.frame_ids,
                    previous_state.timestamps,
                )
            )
        ):
            raise ValueError("previous state and detector must share a device")
        frame_ids = packet.frame_ids.to(device=device)
        timestamps = packet.timestamps.to(device=device, dtype=dtype)
        generation = packet.detector_generation.to(device=device)
        frame_delta = frame_ids - previous_state.frame_ids
        time_delta = timestamps - previous_state.timestamps.to(dtype=dtype)
        causal = (frame_delta > 0) & (time_delta >= 0)
        within_gap = (frame_delta <= self.max_frame_gap) & (
            time_delta <= self.max_time_gap_seconds
        )
        same_generation = generation == previous_state.generation
        allowed = (
            causal
            & within_gap
            & same_generation
            & ~self._reset_mask(reset, batch_size, device)
        )
        if self.use_validity_gate:
            confidence = torch.sigmoid(
                (
                    previous_state.confidence.to(dtype=dtype)
                    - self.memory_confidence_threshold
                )
                * self.memory_gate_temperature
            )
            confidence = confidence * previous_state.validity.to(dtype=dtype)
        else:
            confidence = torch.ones_like(zeros)
        decay = torch.exp(-time_delta.clamp_min(0) / self.memory_time_decay_seconds)
        gate = confidence * decay * allowed.to(dtype=dtype)
        return (gate, time_delta.clamp_min(0), frame_delta.clamp_min(0).to(dtype=dtype))

    def _predict_layer(
        self,
        queries: Tensor,
        references: Tensor,
        semantic_context: Tensor,
        semantic_gate: Tensor,
    ) -> DecoderLayerOutput:
        normalized = self.final_norm(queries)
        presence = self.presence_head(normalized).squeeze(-1)
        validity = self.validity_head(normalized).squeeze(-1)
        local_raw = self.box_head(normalized)
        contextual_raw = self.context_box_residual(semantic_context)[:, None]
        raw = local_raw + semantic_gate[:, None, None] * contextual_raw
        centers = torch.sigmoid(
            _temporal_deformable_inverse_sigmoid(references) + raw[..., :2]
        )
        sizes = raw[..., 2:].sigmoid()
        boxes = torch.cat((centers, sizes), dim=-1)
        motion = torch.tanh(self.motion_head(normalized))
        return DecoderLayerOutput(
            presence_logits=presence,
            validity_logits=validity,
            boxes_cxcywh=boxes,
            motion_residual=motion,
            reference_points=references,
        )

    def forward(
        self,
        feature_packet: TemporalFeaturePacket,
        previous_state: TemporalQueryState | None = None,
        reset: bool | Tensor = False,
        *,
        detach_next_state: bool = False,
    ) -> tuple[TemporalDetectorOutput, TemporalQueryState]:
        pyramid = self.adapter(feature_packet)
        if not isinstance(pyramid, AdaptedFeaturePyramid):
            raise TypeError("feature_adapter must return AdaptedFeaturePyramid")
        levels = self._positioned_levels(pyramid)
        mask = pyramid.masks[0]
        coarse_reference, heatmap_logits = self._coarse_reference(levels[0], mask)
        local_context = self.local_context_projection(
            self._masked_spatial_mean(levels[0], mask)
        )
        semantic_residual = self.semantic_query_projection(pyramid.semantic_context)
        context = local_context + pyramid.semantic_gate[:, None] * semantic_residual
        batch_size = feature_packet.batch_size
        queries = (
            self.query_embeddings.weight[None].expand(batch_size, -1, -1)
            + context[:, None]
        )
        references = coarse_reference[:, None].expand(-1, self.num_queries, -1).clone()
        gate, time_delta, frame_delta = self._memory_gate(
            feature_packet, previous_state, reset
        )
        if self.num_queries == 2 and previous_state is not None:
            if self.use_previous_query:
                time_features = torch.stack(
                    (
                        time_delta / self.max_time_gap_seconds,
                        frame_delta / self.max_frame_gap,
                    ),
                    dim=-1,
                )
                memory_query = (
                    self.memory_projection(previous_state.query.to(dtype=queries.dtype))
                    + self.time_projection(time_features)
                    + context
                )
                propagated_query = (
                    gate[:, None] * memory_query + (1 - gate[:, None]) * queries[:, 1]
                )
                queries = torch.cat((queries[:, :1], propagated_query[:, None]), dim=1)
            if self.use_previous_box_reference:
                previous_reference = previous_state.reference_box[:, :2].to(
                    dtype=references.dtype
                )
                propagated_reference = (
                    gate[:, None] * previous_reference
                    + (1 - gate[:, None]) * references[:, 1]
                )
                references = torch.cat(
                    (references[:, :1], propagated_reference[:, None]), dim=1
                )
        layer_outputs: list[DecoderLayerOutput] = []
        for layer in self.decoder:
            queries, references = layer(queries, references, levels, pyramid.masks)
            contextual_reference = self.context_reference_residual(
                pyramid.semantic_context
            )
            references = torch.sigmoid(
                _temporal_deformable_inverse_sigmoid(references)
                + pyramid.semantic_gate[:, None, None] * contextual_reference[:, None]
            )
            layer_outputs.append(
                self._predict_layer(
                    queries, references, pyramid.semantic_context, pyramid.semantic_gate
                )
            )
        final = layer_outputs[-1]
        normalized_queries = self.final_norm(queries)
        selected = final.presence_logits.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=selected.device)
        confidence = final.presence_logits.sigmoid()[batch_indices, selected]
        validity = final.validity_logits.sigmoid()[batch_indices, selected] >= 0.5
        if previous_state is None:
            age = confidence.new_zeros(batch_size)
        else:
            age = torch.where(
                gate > 0,
                previous_state.age.to(dtype=confidence.dtype) + time_delta,
                torch.zeros_like(confidence),
            )
        next_state = TemporalQueryState(
            query=normalized_queries[batch_indices, selected],
            reference_points=final.reference_points[batch_indices, selected],
            reference_box=final.boxes_cxcywh[batch_indices, selected],
            confidence=confidence,
            validity=validity,
            age=age,
            generation=feature_packet.detector_generation.to(device=confidence.device),
            frame_ids=feature_packet.frame_ids.to(device=confidence.device),
            timestamps=feature_packet.timestamps.to(
                device=confidence.device, dtype=confidence.dtype
            ),
        )
        if detach_next_state:
            next_state = next_state.detached()
        outputs = TemporalDetectorOutput(
            presence_logits=final.presence_logits,
            validity_logits=final.validity_logits,
            boxes_cxcywh=final.boxes_cxcywh,
            motion_residual=final.motion_residual,
            queries=normalized_queries,
            reference_points=final.reference_points,
            coarse_reference_points=coarse_reference,
            memory_gate=gate,
            fusion_gate=pyramid.fusion_gate,
            semantic_gate=pyramid.semantic_gate,
            coarse_heatmap_logits=heatmap_logits,
            spatial_feature_available=pyramid.spatial_feature_available,
            decoder_layer_outputs=tuple(layer_outputs),
        )
        return (outputs, next_state)


class TemporalDeformablePolypHead(TemporalPolypHead):
    def __init__(self, **kwargs: Any) -> None:
        if "decoder_type" in kwargs and kwargs["decoder_type"] != "deformable":
            raise ValueError(
                "TemporalDeformablePolypHead requires decoder_type=deformable"
            )
        kwargs["decoder_type"] = "deformable"
        super().__init__(**kwargs)


GrootTemporalFeatureMode = Literal["mid_only", "dit_conditioned"]


class GrootTemporalFeatureAdapter(nn.Module):
    """One spatial visual level with optional pooled DiT semantic conditioning."""

    VALID_MODES = frozenset({"mid_only", "dit_conditioned"})

    def __init__(
        self,
        *,
        feature_mode: GrootTemporalFeatureMode = "dit_conditioned",
        mid_dim: int = GROOT_N17_MID_DIM,
        dit_conditioning_dim: int = GROOT_N17_PROJECTED_DIM,
        hidden_dim: int = 256,
        conditioning_dropout: float = 0.2,
        semantic_decay_seconds: float = 1.0,
        max_context_age_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        if feature_mode not in self.VALID_MODES:
            raise ValueError(
                f"unsupported GR00T temporal feature_mode {feature_mode!r}"
            )
        if min(mid_dim, dit_conditioning_dim, hidden_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if not 0 <= conditioning_dropout < 1:
            raise ValueError("conditioning_dropout must be in [0,1)")
        if semantic_decay_seconds <= 0 or max_context_age_seconds <= 0:
            raise ValueError("semantic age-decay scales must be positive")
        self.feature_mode: GrootTemporalFeatureMode = feature_mode
        self.hidden_dim = int(hidden_dim)
        self.semantic_decay_seconds = float(semantic_decay_seconds)
        self.max_context_age_seconds = float(max_context_age_seconds)
        self.mid_projection = nn.Sequential(
            nn.LayerNorm(mid_dim), nn.Linear(mid_dim, hidden_dim)
        )
        self.conditioning_projection = nn.Sequential(
            nn.LayerNorm(dit_conditioning_dim),
            nn.Linear(dit_conditioning_dim, hidden_dim),
        )
        self.conditioning_dropout = nn.Dropout(conditioning_dropout)
        self.semantic_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        final_gate = self.semantic_gate_mlp[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, -4.0)

    @staticmethod
    def _masked_mean(tokens: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(self, packet: GrootTemporalFeaturePacket) -> AdaptedFeaturePyramid:
        local = self.mid_projection(packet.mid_tokens)
        batch_size, num_tokens, _ = local.shape
        mid_mask = packet.mid_padding_mask
        local_pool = self._masked_mean(local, mid_mask)
        semantic = local.new_zeros(batch_size, self.hidden_dim)
        semantic_gate = local.new_zeros(batch_size)
        if (
            self.feature_mode == "dit_conditioned"
            and packet.dit_conditioning_tokens is not None
        ):
            conditioning_mask = packet.dit_conditioning_padding_mask
            assert conditioning_mask is not None
            conditioning = self.conditioning_projection(
                packet.dit_conditioning_tokens.to(dtype=local.dtype)
            )
            conditioning = self.conditioning_dropout(conditioning)
            semantic = self._masked_mean(conditioning, conditioning_mask)
            learned_gate = torch.sigmoid(
                self.semantic_gate_mlp(
                    torch.cat((local_pool, semantic), dim=-1)
                ).squeeze(-1)
            )
            age = packet.context_age.to(device=local.device, dtype=local.dtype)
            available = packet.conditioning_available.to(dtype=local.dtype)
            age_decay = torch.exp(-age.clamp_min(0) / self.semantic_decay_seconds)
            age_decay = age_decay * (age <= self.max_context_age_seconds).to(
                dtype=local.dtype
            )
            semantic_gate = learned_gate * age_decay * available
        rows, columns = packet.mid_spatial_shape
        level = local.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, rows, columns
        )
        mask = mid_mask.reshape(batch_size, rows, columns)
        dense_fusion_gate = local.new_zeros(batch_size, num_tokens, 1)
        return AdaptedFeaturePyramid(
            levels=(level,),
            masks=(mask,),
            fused_tokens=local,
            semantic_context=semantic,
            fusion_gate=dense_fusion_gate,
            semantic_gate=semantic_gate,
            spatial_feature_available=torch.ones(
                batch_size, dtype=torch.bool, device=local.device
            ),
        )


def build_groot_temporal_head(
    *,
    feature_mode: GrootTemporalFeatureMode = "dit_conditioned",
    mid_dim: int = GROOT_N17_MID_DIM,
    dit_conditioning_dim: int = GROOT_N17_PROJECTED_DIM,
    hidden_dim: int = 256,
    conditioning_dropout: float = 0.2,
    semantic_decay_seconds: float = 1.0,
    max_context_age_seconds: float = 2.0,
    **head_kwargs: Any,
) -> TemporalDeformablePolypHead:
    """Build the controlled GR00T temporal arm on the shared endo_bbox_vla core.

    The injected adapter owns the GR00T-specific feature contract.  The head
    retains the same two-query temporal state, deformable decoder, and loss
    interface used by the Pi0.5 study without pretending that the 16x16 visual
    grid and 8x8 DiT-conditioning grid are token-aligned.
    """
    if "feature_adapter" in head_kwargs:
        raise ValueError("build_groot_temporal_head owns feature_adapter")
    if "feature_mode" in head_kwargs:
        raise ValueError("pass GR00T feature_mode as the explicit factory argument")
    adapter = GrootTemporalFeatureAdapter(
        feature_mode=feature_mode,
        mid_dim=mid_dim,
        dit_conditioning_dim=dit_conditioning_dim,
        hidden_dim=hidden_dim,
        conditioning_dropout=conditioning_dropout,
        semantic_decay_seconds=semantic_decay_seconds,
        max_context_age_seconds=max_context_age_seconds,
    )
    return TemporalDeformablePolypHead(
        hidden_dim=hidden_dim, feature_adapter=adapter, **head_kwargs
    )


def _build_head(
    config: ExperimentConfig, variant: TrainingVariant, device: torch.device
) -> SharedDetectorHead:
    from endo_bbox_vla.config import _variant_config

    head = config.section("head")
    backbone = config.section("backbone")
    settings = _variant_config(config, variant)
    kind = str(settings.get("head_kind", "location_first"))
    default_queries = 1 if kind == "one_query" else int(head["location_queries"])
    return SharedDetectorHead(
        input_dim=int(backbone["feature_dim"]),
        hidden_dim=int(head["hidden_dim"]),
        grid_size=tuple(backbone["grid_size"]),
        kind=kind,
        num_queries=int(settings.get("num_queries", default_queries)),
        decoder_layers=int(head["decoder_layers"]),
        attention_heads=int(head["attention_heads"]),
        feedforward_dim=int(head["feedforward_dim"]),
        dropout=float(head["dropout"]),
    ).to(device)


def select_single_prediction(outputs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Return the highest-presence score and its matching box per sample."""
    logits = outputs["presence_logits"]
    boxes = outputs["boxes_cxcywh"]
    indices = logits.argmax(dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return (logits[rows, indices].sigmoid(), boxes[rows, indices])


class Learned2DPositionEncoding(nn.Module):
    """Learned row/column embeddings for a fixed spatial token grid."""

    def __init__(self, hidden_dim: int, grid_size: tuple[int, int]) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for row/column position encoding")
        rows, columns = grid_size
        self.row_embedding = nn.Embedding(rows, hidden_dim // 2)
        self.column_embedding = nn.Embedding(columns, hidden_dim // 2)
        nn.init.uniform_(self.row_embedding.weight)
        nn.init.uniform_(self.column_embedding.weight)

    def forward(self) -> Tensor:
        rows = self.row_embedding.weight
        columns = self.column_embedding.weight
        row_grid = rows[:, None, :].expand(-1, columns.shape[0], -1)
        column_grid = columns[None, :, :].expand(rows.shape[0], -1, -1)
        return torch.cat((column_grid, row_grid), dim=-1).flatten(0, 1).unsqueeze(0)


def build_letterbox_padding_mask(
    batch_size: int,
    *,
    grid_size: tuple[int, int] = (16, 16),
    padded_rows: tuple[int, int] = (2, 2),
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a DETR key-padding mask for centered 4:3 -> square letterboxing.

    ``True`` denotes tokens that attention must ignore.  For Pi0.5's 224 x 224
    input and 14-pixel patches, the 28-pixel top/bottom padding occupies two
    rows at each edge of the 16 x 16 grid.
    """
    rows, columns = grid_size
    top, bottom = padded_rows
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rows <= 0 or columns <= 0:
        raise ValueError("grid_size values must be positive")
    if top < 0 or bottom < 0 or top + bottom >= rows:
        raise ValueError("padded_rows must leave at least one unmasked grid row")
    mask = torch.zeros((batch_size, rows, columns), dtype=torch.bool, device=device)
    if top:
        mask[:, :top, :] = True
    if bottom:
        mask[:, rows - bottom :, :] = True
    return mask.flatten(1)


class OneQueryDetrHead(nn.Module):
    """Decode one presence score and one normalized ``cxcywh`` box."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        grid_size: tuple[int, int] = (16, 16),
        padded_rows: tuple[int, int] = (2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.grid_size = grid_size
        self.padded_rows = padded_rows
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = Learned2DPositionEncoding(hidden_dim, grid_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.query_embedding = nn.Embedding(1, hidden_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.box_head = detr_head_MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    @property
    def num_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def default_padding_mask(
        self, batch_size: int, device: torch.device | str
    ) -> Tensor:
        return build_letterbox_padding_mask(
            batch_size,
            grid_size=self.grid_size,
            padded_rows=self.padded_rows,
            device=device,
        )

    def forward(
        self, projected_tokens: Tensor, padding_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        if projected_tokens.ndim != 3:
            raise ValueError(
                f"Expected BNC projected tokens, got {tuple(projected_tokens.shape)}"
            )
        batch_size, num_tokens, _ = projected_tokens.shape
        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} spatial tokens, got {num_tokens}"
            )
        memory = self.input_proj(projected_tokens)
        positions = self.position_encoding().to(
            device=memory.device, dtype=memory.dtype
        )
        memory = memory + positions
        if padding_mask is None:
            padding_mask = self.default_padding_mask(batch_size, memory.device)
        elif padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0).expand(batch_size, -1)
        expected_mask_shape = (batch_size, num_tokens)
        if tuple(padding_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"Expected padding mask shape {expected_mask_shape}, got {tuple(padding_mask.shape)}"
            )
        padding_mask = padding_mask.to(device=memory.device, dtype=torch.bool)
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(query, memory, memory_key_padding_mask=padding_mask)[
            :, 0
        ]
        return {
            "presence_logits": self.presence_head(decoded).squeeze(-1),
            "boxes_cxcywh": self.box_head(decoded).sigmoid(),
        }


class TemporalDensePolypHead(TemporalPolypHead):
    def __init__(self, **kwargs: Any) -> None:
        if "decoder_type" in kwargs and kwargs["decoder_type"] != "dense":
            raise ValueError("TemporalDensePolypHead requires decoder_type=dense")
        kwargs["decoder_type"] = "dense"
        super().__init__(**kwargs)


def build_endo_bbox_vla_head(
    *, decoder_type: DecoderType = "deformable", **kwargs: Any
) -> TemporalPolypHead:
    """Factory used by experiment configs; all ablation flags remain explicit."""
    cls = (
        TemporalDeformablePolypHead
        if decoder_type == "deformable"
        else TemporalDensePolypHead
    )
    if decoder_type not in {"dense", "deformable"}:
        raise ValueError("decoder_type must be dense or deformable")
    return cls(**kwargs)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        (
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def endo_bbox_vla_parameter_report(**common_kwargs: Any) -> dict[str, int]:
    """Instantiate fair dense/deformable variants and report trainable counts."""
    common_kwargs.pop("decoder_type", None)
    dense = build_endo_bbox_vla_head(decoder_type="dense", **common_kwargs)
    deformable = build_endo_bbox_vla_head(decoder_type="deformable", **common_kwargs)
    dense_count = trainable_parameter_count(dense)
    deformable_count = trainable_parameter_count(deformable)
    return {
        "dense_trainable": dense_count,
        "deformable_trainable": deformable_count,
        "deformable_minus_dense": deformable_count - dense_count,
    }


def build_projected_one_query_baseline(
    *,
    projected_dim: int = 2048,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
) -> OneQueryDetrHead:
    """Existing shared one-query head on frozen five-task projected tokens."""
    return OneQueryDetrHead(
        input_dim=projected_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


class ProjectedOneQueryBaselineAdapter(nn.Module):
    """Feed five-task pre-LM projected tokens to the historical shared head.

    No temporal query, heatmap, validity, motion, or post-VLA path is present.
    The wrapper itself has no trainable parameters beyond the exact contained
    :class:`OneQueryDetrHead`.
    """

    def __init__(
        self, head: OneQueryDetrHead | None = None, **head_kwargs: Any
    ) -> None:
        super().__init__()
        if head is not None and head_kwargs:
            raise ValueError("provide either an existing head or head kwargs, not both")
        self.head = head if head is not None else OneQueryDetrHead(**head_kwargs)

    @property
    def ablation_metadata(self) -> dict[str, Any]:
        return {
            "feature_mode": "projected_baseline",
            "decoder_type": "dense",
            "num_queries": 1,
            "num_layers": len(self.head.decoder.layers),
            "hidden_dim": self.head.query_embedding.embedding_dim,
            "use_previous_box_reference": False,
            "use_previous_query": False,
            "use_validity_gate": False,
            "stale_memory_training": False,
            "legacy_one_query_contract": True,
        }

    def forward(
        self,
        feature_packet: FeaturePacket,
        previous_state: TemporalQueryState | None = None,
        reset: bool | Tensor = False,
        **_: Any,
    ) -> tuple[dict[str, Tensor], None]:
        if previous_state is not None:
            raise ValueError("projected one-query baseline has no temporal state")
        reset_value = torch.as_tensor(reset, dtype=torch.bool)
        if reset_value.ndim > 1:
            raise ValueError("reset must be scalar or [B]")
        outputs = self.head(
            feature_packet.projected_tokens, feature_packet.padding_mask
        )
        return (outputs, None)


def build_projected_one_query_candidate_adapter(
    *,
    projected_dim: int = 2048,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
) -> ProjectedOneQueryBaselineAdapter:
    if num_layers != 3:
        raise ValueError(
            "the registered projected-token baseline uses exactly three layers"
        )
    return ProjectedOneQueryBaselineAdapter(
        input_dim=projected_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
