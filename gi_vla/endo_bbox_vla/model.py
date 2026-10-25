"""Read-only frozen GR00T feature extraction and detector assembly."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from endo_bbox_vla.network import (
    GROOT_CAM_LOW_KEY,
    GROOT_N17_MID_DIM,
    GROOT_N17_PROJECTED_DIM,
    GROOT_N17_SPATIAL_MERGE_SIZE,
)

if TYPE_CHECKING:
    from endo_bbox_vla.network import (
        GrootCheckpointIdentity,
        GrootFeaturePacket,
        GrootTemporalFeaturePacket,
        _CapturedBackboneInput,
        _CapturedTemporalFeatures,
        _CapturedVisual,
        _groot_temporal_Result,
        _Result,
    )

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


class FrozenGrootN17MultiLevelExtractor(nn.Module):
    """Call-scoped block-11/merger capture for the pinned five-task policy."""

    def __init__(
        self,
        policy: nn.Module,
        *,
        camera_order: Sequence[str],
        camera_key: str = GROOT_CAM_LOW_KEY,
        identity: GrootCheckpointIdentity | None = None,
        expected_mid_dim: int = GROOT_N17_MID_DIM,
        expected_projected_dim: int = GROOT_N17_PROJECTED_DIM,
        expected_merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE,
    ) -> None:
        from endo_bbox_vla.network import GrootCheckpointIdentity, _camera_view_index

        super().__init__()
        if not isinstance(policy, nn.Module):
            raise TypeError("policy must be a torch.nn.Module")
        order, camera_view_index = _camera_view_index(camera_order, camera_key)
        if min(expected_mid_dim, expected_projected_dim, expected_merge_size) <= 0:
            raise ValueError("expected dimensions and merge size must be positive")
        pinned_identity = identity or GrootCheckpointIdentity()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval()
        object.__setattr__(self, "_frozen_policy", policy)
        object.__setattr__(self, "_capture_lock", threading.RLock())
        self.policy_camera_order = order
        self.camera_key = str(camera_key)
        self.camera_view_index = camera_view_index
        self.identity = pinned_identity
        self.expected_mid_dim = int(expected_mid_dim)
        self.expected_projected_dim = int(expected_projected_dim)
        self.expected_merge_size = int(expected_merge_size)
        self.train(False)

    @property
    def policy(self) -> nn.Module:
        return object.__getattribute__(self, "_frozen_policy")

    @property
    def token_provenance(self) -> dict[str, Any]:
        from endo_bbox_vla.network import GROOT_N17_MID_BLOCK_INDEX

        return {
            "feature_family": "groot_n17_qwen3_vl",
            "base_checkpoint": self.identity.repo_id,
            "base_revision": self.identity.revision,
            "mid_source": "qwen3_vl_visual_block_output_pre_merger",
            "mid_layer_indexing": "zero_based",
            "mid_layer_indices": [GROOT_N17_MID_BLOCK_INDEX],
            "projected_source": "qwen3_vl_visual_merger_output_pre_language_model",
            "policy_camera_order": list(self.policy_camera_order),
            "camera_key": self.camera_key,
            "cam_low_view_index": self.camera_view_index,
            "spatial_merge_size": self.expected_merge_size,
            "policy_frozen": True,
        }

    def _visual(self) -> nn.Module:
        from endo_bbox_vla.network import GROOT_N17_MID_BLOCK_INDEX

        try:
            visual = self.policy._groot_model.backbone.visual
        except AttributeError as error:
            raise AttributeError(
                "policy does not expose _groot_model.backbone.visual from the local GR00T N1.7 implementation"
            ) from error
        blocks = getattr(visual, "blocks", None)
        if (
            not isinstance(blocks, nn.ModuleList)
            or len(blocks) <= GROOT_N17_MID_BLOCK_INDEX
        ):
            raise AttributeError(
                "Qwen3-VL visual.blocks does not contain zero-based block 11"
            )
        if not isinstance(getattr(visual, "merger", None), nn.Module):
            raise AttributeError("Qwen3-VL visual.merger is unavailable")
        actual_merge_size = getattr(visual, "spatial_merge_size", None)
        if actual_merge_size is None:
            actual_merge_size = getattr(
                getattr(visual, "config", None), "spatial_merge_size", None
            )
        if int(actual_merge_size or -1) != self.expected_merge_size:
            raise ValueError(
                f"loaded Qwen3-VL spatial merge size differs from the pinned extractor contract: {actual_merge_size!r} != {self.expected_merge_size}"
            )
        return visual

    def _freeze_policy(self) -> None:
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _require_tensor(output: Any, *, source: str) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError(
                f"{source} hook expected a Tensor output, got {type(output).__name__}"
            )
        return output

    def _packet(
        self,
        *,
        captured: _CapturedVisual,
        image_grid_thw: Tensor,
        batch_size: int,
        timestep_index: int,
    ) -> GrootFeaturePacket:
        from endo_bbox_vla.network import (
            GrootFeaturePacket,
            plan_groot_camera_token_spans,
            unshuffle_qwen_raw_tokens,
        )

        if len(captured.mid) != 1 or len(captured.projected) != 1:
            raise RuntimeError(
                f"one GR00T capture call must traverse the image visual tower exactly once; observed block11={len(captured.mid)}, merger={len(captured.projected)}"
            )
        raw = captured.mid[0]
        projected = captured.projected[0]
        if raw.ndim != 2 or raw.shape[-1] != self.expected_mid_dim:
            raise RuntimeError(
                f"unexpected block-11 shape {tuple(raw.shape)}; expected [sum_raw,{self.expected_mid_dim}]"
            )
        if projected.ndim != 2 or projected.shape[-1] != self.expected_projected_dim:
            raise RuntimeError(
                f"unexpected merger shape {tuple(projected.shape)}; expected [sum_merged,{self.expected_projected_dim}]"
            )
        spans = plan_groot_camera_token_spans(
            image_grid_thw,
            camera_order=self.policy_camera_order,
            camera_key=self.camera_key,
            batch_size=batch_size,
            timestep_index=timestep_index,
            merge_size=self.expected_merge_size,
        )
        grid_rows = image_grid_thw.detach().to(device="cpu", dtype=torch.int64)
        expected_raw_total = int(grid_rows.prod(dim=-1).sum().item())
        expected_projected_total = int(
            (
                grid_rows[:, 0]
                * (grid_rows[:, 1] // self.expected_merge_size)
                * (grid_rows[:, 2] // self.expected_merge_size)
            )
            .sum()
            .item()
        )
        if (
            raw.shape[0] != expected_raw_total
            or projected.shape[0] != expected_projected_total
        ):
            raise RuntimeError(
                f"captured visual token totals disagree with image_grid_thw: raw {raw.shape[0]} != {expected_raw_total}, projected {projected.shape[0]} != {expected_projected_total}"
            )
        selected_grids = tuple((span.grid_thw for span in spans))
        if len(set(selected_grids)) != 1:
            raise ValueError(
                f"selected cam_low grids differ across the batch and cannot be stacked for the dense heads: {selected_grids}"
            )
        mid_rows: list[Tensor] = []
        projected_rows: list[Tensor] = []
        for span in spans:
            mid_rows.append(
                unshuffle_qwen_raw_tokens(
                    raw[span.raw_start : span.raw_stop],
                    grid_thw=span.grid_thw,
                    merge_size=self.expected_merge_size,
                )
            )
            projected_rows.append(
                projected[span.projected_start : span.projected_stop].contiguous()
            )
        mid_tokens = torch.stack(mid_rows, dim=0).detach()
        projected_tokens = torch.stack(projected_rows, dim=0).detach()
        grid_t, grid_h, grid_w = selected_grids[0]
        if grid_t != 1:
            raise ValueError("detection packet requires still-image grid t=1")
        mid_spatial_shape = (grid_h, grid_w)
        projected_spatial_shape = (
            grid_h // self.expected_merge_size,
            grid_w // self.expected_merge_size,
        )
        mid_padding_mask = torch.zeros(
            (batch_size, mid_tokens.shape[1]),
            dtype=torch.bool,
            device=mid_tokens.device,
        )
        projected_padding_mask = torch.zeros(
            (batch_size, projected_tokens.shape[1]),
            dtype=torch.bool,
            device=projected_tokens.device,
        )
        return GrootFeaturePacket(
            mid_tokens=mid_tokens,
            projected_tokens=projected_tokens,
            mid_padding_mask=mid_padding_mask,
            projected_padding_mask=projected_padding_mask,
            mid_spatial_shape=mid_spatial_shape,
            projected_spatial_shape=projected_spatial_shape,
            selected_grid_thw=selected_grids,
            spans=spans,
            policy_camera_order=self.policy_camera_order,
            camera_key=self.camera_key,
            camera_view_index=self.camera_view_index,
            timestep_index=spans[0].timestep_index,
            identity=self.identity,
            merge_size=self.expected_merge_size,
        )

    def capture_action_call(
        self,
        call: Callable[[], _Result],
        *,
        image_grid_thw: Tensor,
        batch_size: int,
        timestep_index: int = -1,
    ) -> tuple[_Result, GrootFeaturePacket]:
        """Run ``call`` once and return its unchanged result plus cam-low features."""
        from endo_bbox_vla.network import GROOT_N17_MID_BLOCK_INDEX, _CapturedVisual

        if not callable(call):
            raise TypeError("call must be callable")
        visual = self._visual()
        block = visual.blocks[GROOT_N17_MID_BLOCK_INDEX]
        merger = visual.merger
        captured = _CapturedVisual(mid=[], projected=[])

        def capture_mid(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            captured.mid.append(
                self._require_tensor(output, source="visual block 11").detach()
            )
            return None

        def capture_projected(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            captured.projected.append(
                self._require_tensor(output, source="visual merger").detach()
            )
            return None

        handles: list[Any] = []
        lock = object.__getattribute__(self, "_capture_lock")
        with lock:
            self._freeze_policy()
            try:
                handles.append(block.register_forward_hook(capture_mid))
                handles.append(merger.register_forward_hook(capture_projected))
                with torch.inference_mode():
                    result = call()
            finally:
                for handle in reversed(handles):
                    handle.remove()
        packet = self._packet(
            captured=captured,
            image_grid_thw=image_grid_thw,
            batch_size=batch_size,
            timestep_index=timestep_index,
        )
        return (result, packet)


class FrozenGrootN17TemporalExtractor(nn.Module):
    """Call-scoped exact block-11 visual and VL-conditioning capture for a frozen GR00T policy.

    ``capture_policy_action_call`` is the policy/runtime API.  It derives token
    metadata from the actual ``_groot_model.backbone`` invocation and is
    fail-closed to one eager B=1 forward.

    Hooks are process-global state on the policy modules.  Extractors sharing a
    policy also share :attr:`policy_capture_lock`; every direct call into that
    policy must either use a capture method or hold this lock for its entire
    duration.  Calling the policy concurrently without this contract is not a
    supported runtime configuration.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        camera_order: Sequence[str],
        camera_key: str = GROOT_CAM_LOW_KEY,
        identity: GrootCheckpointIdentity | None = None,
        expected_mid_dim: int = GROOT_N17_MID_DIM,
        expected_dit_conditioning_dim: int = GROOT_N17_PROJECTED_DIM,
        expected_merge_size: int = GROOT_N17_SPATIAL_MERGE_SIZE,
        image_token_id: int | None = None,
    ) -> None:
        from endo_bbox_vla.network import (
            GrootCheckpointIdentity,
            _normalise_camera_order,
            _policy_capture_lock,
        )

        super().__init__()
        if not isinstance(policy, nn.Module):
            raise TypeError("policy must be a torch.nn.Module")
        order, camera_view_index = _normalise_camera_order(camera_order, camera_key)
        if (
            min(expected_mid_dim, expected_dit_conditioning_dim, expected_merge_size)
            <= 0
        ):
            raise ValueError("expected dimensions and merge size must be positive")
        if image_token_id is not None and image_token_id < 0:
            raise ValueError("image_token_id must be non-negative")
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval()
        object.__setattr__(self, "_frozen_policy", policy)
        object.__setattr__(self, "_capture_lock", _policy_capture_lock(policy))
        self.policy_camera_order = order
        self.camera_key = str(camera_key)
        self.camera_view_index = camera_view_index
        self.identity = identity or GrootCheckpointIdentity()
        self.expected_mid_dim = int(expected_mid_dim)
        self.expected_dit_conditioning_dim = int(expected_dit_conditioning_dim)
        self.expected_merge_size = int(expected_merge_size)
        self.configured_image_token_id = image_token_id
        self.train(False)

    @property
    def policy(self) -> nn.Module:
        return object.__getattribute__(self, "_frozen_policy")

    @property
    def policy_capture_lock(self) -> Any:
        """Shared re-entrant lock required around every call into this policy.

        Both capture methods acquire it automatically.  Runtime code that must
        issue an uncaptured policy call can use ``with
        extractor.policy_capture_lock: ...`` so a transient hook scope can
        never overlap that call.
        """
        return object.__getattribute__(self, "_capture_lock")

    @property
    def token_provenance(self) -> dict[str, Any]:
        from endo_bbox_vla.network import GROOT_N17_MID_BLOCK_INDEX

        return {
            "feature_family": "groot_n17_temporal_qwen3_vl",
            "base_checkpoint": self.identity.repo_id,
            "base_revision": self.identity.revision,
            "mid_source": "qwen3_vl_visual_block_11_pre_merger",
            "mid_layer_indexing": "zero_based",
            "mid_layer_indices": [GROOT_N17_MID_BLOCK_INDEX],
            "dit_conditioning_source": "action_head_vlln_then_vl_self_attention_output",
            "dit_conditioning_is_exact_diffusion_encoder_hidden_states": True,
            "policy_camera_order": list(self.policy_camera_order),
            "camera_key": self.camera_key,
            "cam_low_view_index": self.camera_view_index,
            "spatial_merge_size": self.expected_merge_size,
            "batch_size": 1,
            "policy_frozen": True,
        }

    def _modules_for_capture(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        from endo_bbox_vla.network import GROOT_N17_MID_BLOCK_INDEX

        try:
            root = self.policy._groot_model
            backbone = root.backbone
            visual = backbone.visual
            action_head = root.action_head
            vl_self_attention = action_head.vl_self_attention
        except AttributeError as error:
            raise AttributeError(
                "policy must expose _groot_model.backbone.visual and _groot_model.action_head.vl_self_attention"
            ) from error
        blocks = getattr(visual, "blocks", None)
        if (
            not isinstance(blocks, nn.ModuleList)
            or len(blocks) <= GROOT_N17_MID_BLOCK_INDEX
        ):
            raise AttributeError(
                "Qwen3-VL visual.blocks does not contain zero-based block 11"
            )
        if not isinstance(vl_self_attention, nn.Module):
            raise AttributeError("GR00T action_head.vl_self_attention is unavailable")
        actual_merge_size = getattr(visual, "spatial_merge_size", None)
        if actual_merge_size is None:
            actual_merge_size = getattr(
                getattr(visual, "config", None), "spatial_merge_size", None
            )
        if int(actual_merge_size or -1) != self.expected_merge_size:
            raise ValueError(
                f"loaded Qwen3-VL spatial merge size differs from the pinned extractor contract: {actual_merge_size!r} != {self.expected_merge_size}"
            )
        if not isinstance(backbone, nn.Module):
            raise AttributeError("GR00T backbone is not a torch.nn.Module")
        return (backbone, visual, blocks[GROOT_N17_MID_BLOCK_INDEX], vl_self_attention)

    def _image_token_id(self) -> int:
        if self.configured_image_token_id is not None:
            return int(self.configured_image_token_id)
        try:
            config = self.policy._groot_model.backbone.model.config
            token_id = config.image_token_id
        except AttributeError as error:
            raise AttributeError(
                "cannot resolve Qwen image_token_id from policy._groot_model.backbone.model.config"
            ) from error
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"invalid Qwen image_token_id {token_id!r}")
        return token_id

    def _freeze_policy(self) -> None:
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _looks_compiled(value: object) -> bool:
        """Best-effort detection for torch.compile/OptimizedModule wrappers."""
        if value is None:
            return False
        return any(
            (
                hasattr(value, marker)
                for marker in ("_torchdynamo_orig_callable", "_orig_mod", "dynamo_ctx")
            )
        )

    def _assert_eager_capture(self, call: Callable[[], object]) -> None:
        """Reject known graph-capture wrappers before installing eager hooks."""
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            raise RuntimeError("GR00T temporal hooks cannot run inside torch.compile")
        root = getattr(self.policy, "_groot_model", None)
        backbone = getattr(root, "backbone", None)
        candidates = {
            "capture call": call,
            "policy": self.policy,
            "policy.predict_action_chunk": getattr(
                self.policy, "predict_action_chunk", None
            ),
            "policy._groot_model": root,
            "policy._groot_model.backbone": backbone,
        }
        compiled = [
            name for name, value in candidates.items() if self._looks_compiled(value)
        ]
        if compiled:
            raise RuntimeError(
                f"GR00T temporal feature capture requires eager execution; disable torch.compile for {', '.join(compiled)}"
            )

    @staticmethod
    def _capture_backbone_input(
        inputs: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Resolve and snapshot the BatchFeature passed to backbone.forward."""
        if len(inputs) == 1 and (not kwargs):
            vl_input = inputs[0]
        elif not inputs and set(kwargs) == {"vl_input"}:
            vl_input = kwargs["vl_input"]
        else:
            raise RuntimeError(
                "safe GR00T capture expected backbone.forward(vl_input) exactly once"
            )
        if not isinstance(vl_input, Mapping):
            raise TypeError(
                "GR00T backbone input must be a BatchFeature/mapping for safe capture"
            )
        captured: list[Tensor] = []
        for name in ("input_ids", "image_grid_thw", "attention_mask"):
            value = vl_input.get(name)
            if not isinstance(value, Tensor):
                raise TypeError(f"GR00T backbone {name} must be a torch.Tensor")
            captured.append(value.detach().clone())
        input_ids, image_grid_thw, attention_mask = captured
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "safe GR00T temporal capture is fail-closed to backbone input_ids [1,S]"
            )
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(
            input_ids.shape
        ):
            raise ValueError(
                "backbone attention_mask must have the same [1,S] shape as input_ids"
            )
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError("backbone image_grid_thw must have shape [num_images,3]")
        return (input_ids, image_grid_thw, attention_mask)

    @staticmethod
    def _hook_tensor(output: Any, *, source: str) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError(
                f"{source} hook expected a Tensor output, got {type(output).__name__}"
            )
        return output

    @staticmethod
    def _batch_vector(
        value: Tensor | int | float | None,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        default: int | float,
    ) -> Tensor:
        if value is None:
            result = torch.tensor([default], device=device, dtype=dtype)
        elif isinstance(value, Tensor):
            result = value.detach().to(device=device, dtype=dtype)
            if result.ndim == 0:
                result = result.reshape(1)
        else:
            result = torch.tensor([value], device=device, dtype=dtype)
        if tuple(result.shape) != (1,):
            raise ValueError(
                f"{name} must be scalar or shape [1] for fail-closed B=1 capture"
            )
        return result

    def _packet(
        self,
        *,
        captured: _CapturedTemporalFeatures,
        input_ids: Tensor,
        image_grid_thw: Tensor,
        attention_mask: Tensor | None,
        timestep_index: int,
        frame_ids: Tensor | int | None,
        timestamps: Tensor | float | None,
        prompt_id: str,
        detector_generation: Tensor | int | None,
        context_generation: Tensor | int | None,
    ) -> GrootTemporalFeaturePacket:
        from endo_bbox_vla.network import (
            GrootTemporalFeaturePacket,
            _require_integer,
            plan_groot_camera_token_spans,
            unshuffle_qwen_raw_tokens,
        )

        if len(captured.mid) != 1 or len(captured.dit_conditioning) != 1:
            raise RuntimeError(
                f"one GR00T temporal action call must traverse visual block 11 and action_head.vl_self_attention exactly once; observed block11={len(captured.mid)}, vl_self_attention={len(captured.dit_conditioning)}"
            )
        raw = captured.mid[0]
        conditioning_sequence = captured.dit_conditioning[0]
        if raw.ndim != 2 or raw.shape[-1] != self.expected_mid_dim:
            raise RuntimeError(
                f"unexpected block-11 shape {tuple(raw.shape)}; expected [sum_raw,{self.expected_mid_dim}]"
            )
        if (
            conditioning_sequence.ndim != 3
            or conditioning_sequence.shape[0] != 1
            or conditioning_sequence.shape[-1] != self.expected_dit_conditioning_dim
        ):
            raise RuntimeError(
                f"unexpected DiT conditioning sequence shape {tuple(conditioning_sequence.shape)}; expected [1,S,{self.expected_dit_conditioning_dim}]"
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "input_ids must have shape [1,S] for fail-closed B=1 capture"
            )
        _require_integer("input_ids", input_ids)
        if input_ids.shape[1] != conditioning_sequence.shape[1]:
            raise RuntimeError(
                f"input_ids and exact DiT conditioning sequence lengths differ: {input_ids.shape[1]} != {conditioning_sequence.shape[1]}"
            )
        spans = plan_groot_camera_token_spans(
            image_grid_thw,
            camera_order=self.policy_camera_order,
            camera_key=self.camera_key,
            batch_size=1,
            timestep_index=timestep_index,
            merge_size=self.expected_merge_size,
        )
        span = spans[0]
        grid_rows = image_grid_thw.detach().to(device="cpu", dtype=torch.int64)
        expected_raw_total = int(grid_rows.prod(dim=-1).sum().item())
        expected_conditioning_image_tokens = int(
            (
                grid_rows[:, 0]
                * (grid_rows[:, 1] // self.expected_merge_size)
                * (grid_rows[:, 2] // self.expected_merge_size)
            )
            .sum()
            .item()
        )
        if raw.shape[0] != expected_raw_total:
            raise RuntimeError(
                f"captured visual token total disagrees with image_grid_thw: {raw.shape[0]} != {expected_raw_total}"
            )
        image_mask = input_ids == self._image_token_id()
        image_positions = image_mask[0].nonzero(as_tuple=False).flatten()
        if image_positions.numel() != expected_conditioning_image_tokens:
            raise RuntimeError(
                f"input_ids image-token count disagrees with image_grid_thw merged geometry: {image_positions.numel()} != {expected_conditioning_image_tokens}"
            )
        if span.projected_stop > image_positions.numel():
            raise RuntimeError(
                "cam-low projected span exceeds the Qwen image-token positions"
            )
        selected_positions = image_positions[span.projected_start : span.projected_stop]
        if selected_positions.numel() != span.projected_num_tokens:
            raise RuntimeError(
                "failed to resolve the complete cam-low DiT conditioning span"
            )
        device = conditioning_sequence.device
        selected_positions = selected_positions.to(device=device)
        mid_tokens = (
            unshuffle_qwen_raw_tokens(
                raw[span.raw_start : span.raw_stop],
                grid_thw=span.grid_thw,
                merge_size=self.expected_merge_size,
            )
            .unsqueeze(0)
            .clone()
            .detach()
        )
        dit_tokens = (
            conditioning_sequence[0]
            .index_select(0, selected_positions)
            .unsqueeze(0)
            .clone()
            .detach()
        )
        grid_t, grid_h, grid_w = span.grid_thw
        if grid_t != 1:
            raise ValueError(
                "the temporal detector requires a still-image cam-low grid t=1"
            )
        mid_spatial_shape = (grid_h, grid_w)
        dit_spatial_shape = (
            grid_h // self.expected_merge_size,
            grid_w // self.expected_merge_size,
        )
        mid_mask = torch.zeros(
            (1, mid_tokens.shape[1]), dtype=torch.bool, device=device
        )
        if attention_mask is None:
            dit_mask = torch.zeros(
                (1, dit_tokens.shape[1]), dtype=torch.bool, device=device
            )
        else:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(
                input_ids.shape
            ):
                raise ValueError(
                    "attention_mask must have the same [1,S] shape as input_ids"
                )
            attention = attention_mask.to(device=device)
            if attention.dtype == torch.bool:
                valid_attention = attention
            elif attention.dtype == torch.uint8 or not attention.is_floating_point():
                valid_attention = attention != 0
            else:
                raise TypeError("attention_mask must use bool or integer dtype")
            dit_mask = ~valid_attention[0].index_select(
                0, selected_positions
            ).unsqueeze(0)
        if bool(dit_mask.all()):
            raise RuntimeError(
                "all selected cam-low DiT conditioning positions are padded"
            )
        current_frame_ids = self._batch_vector(
            frame_ids, name="frame_ids", device=device, dtype=torch.int64, default=0
        )
        current_timestamps = self._batch_vector(
            timestamps,
            name="timestamps",
            device=device,
            dtype=torch.float32,
            default=0.0,
        )
        current_generation = self._batch_vector(
            detector_generation,
            name="detector_generation",
            device=device,
            dtype=torch.int64,
            default=0,
        )
        dit_generation = self._batch_vector(
            context_generation,
            name="context_generation",
            device=device,
            dtype=torch.int64,
            default=0,
        )
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("prompt_id must be a non-empty string")
        return GrootTemporalFeaturePacket(
            mid_tokens=mid_tokens,
            mid_padding_mask=mid_mask,
            mid_spatial_shape=mid_spatial_shape,
            frame_ids=current_frame_ids,
            timestamps=current_timestamps,
            prompt_id=(prompt_id,),
            detector_generation=current_generation,
            dit_conditioning_tokens=dit_tokens,
            dit_conditioning_padding_mask=dit_mask,
            dit_conditioning_spatial_shape=dit_spatial_shape,
            dit_sequence_positions=selected_positions.unsqueeze(0).detach(),
            dit_source_frame_ids=current_frame_ids.clone(),
            dit_source_timestamps=current_timestamps.clone(),
            dit_prompt_id=(prompt_id,),
            dit_context_generation=dit_generation,
            selected_grid_thw=(span.grid_thw,),
            spans=spans,
            policy_camera_order=self.policy_camera_order,
            camera_key=self.camera_key,
            camera_view_index=self.camera_view_index,
            timestep_index=span.timestep_index,
            identity=self.identity,
            merge_size=self.expected_merge_size,
            mid_feature_dim=self.expected_mid_dim,
            dit_conditioning_dim=self.expected_dit_conditioning_dim,
        )

    def capture_policy_action_call(
        self,
        call: Callable[[], _groot_temporal_Result],
        *,
        batch_size: int = 1,
        timestep_index: int = -1,
        frame_ids: Tensor | int | None = None,
        timestamps: Tensor | float | None = None,
        prompt_id: str = "unspecified",
        detector_generation: Tensor | int | None = None,
        context_generation: Tensor | int | None = None,
    ) -> tuple[_groot_temporal_Result, GrootTemporalFeaturePacket]:
        """Capture one eager policy action call using its actual backbone input.

        A forward-pre-hook on ``policy._groot_model.backbone`` snapshots the
        exact ``input_ids``, ``image_grid_thw``, and ``attention_mask`` consumed
        by that call.  No externally supplied token metadata is accepted.  The
        backbone, visual block 11, and DiT-conditioning hooks must each belong
        to one and the same B=1 forward, otherwise the call fails closed.

        The caller must disable ``torch.compile`` and honor
        :attr:`policy_capture_lock` for every call into the shared policy.
        This method acquires that lock automatically.
        """
        from endo_bbox_vla.network import _CapturedTemporalFeatures

        if not callable(call):
            raise TypeError("call must be callable")
        if batch_size != 1:
            raise ValueError("GR00T temporal extraction is fail-closed to batch_size=1")
        self._assert_eager_capture(call)
        backbone, _visual, block, vl_self_attention = self._modules_for_capture()
        captured = _CapturedTemporalFeatures(mid=[], dit_conditioning=[])
        backbone_inputs: list[_CapturedBackboneInput] = []
        completed_invocations: list[int] = []
        mid_invocations: list[int] = []
        dit_invocations: list[int] = []
        active_invocation: list[int | None] = [None]

        def capture_backbone_pre(
            _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            from endo_bbox_vla.network import _CapturedBackboneInput

            if backbone_inputs or active_invocation[0] is not None:
                raise RuntimeError(
                    "safe GR00T action capture requires exactly one backbone forward"
                )
            input_ids, image_grid_thw, attention_mask = self._capture_backbone_input(
                inputs, kwargs
            )
            invocation_id = 1
            active_invocation[0] = invocation_id
            backbone_inputs.append(
                _CapturedBackboneInput(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    invocation_id=invocation_id,
                )
            )
            return None

        def finish_backbone(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            invocation_id = active_invocation[0]
            if invocation_id is None:
                raise RuntimeError("GR00T backbone post-hook has no matching pre-hook")
            if mid_invocations != [invocation_id]:
                raise RuntimeError(
                    "visual block 11 was not captured exactly once inside the same backbone forward"
                )
            completed_invocations.append(invocation_id)
            active_invocation[0] = None
            return None

        def capture_mid(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            invocation_id = active_invocation[0]
            if invocation_id is None:
                raise RuntimeError(
                    "visual block 11 ran outside the captured GR00T backbone forward"
                )
            if captured.mid:
                raise RuntimeError(
                    "visual block 11 must run exactly once per action capture"
                )
            captured.mid.append(
                self._hook_tensor(output, source="visual block 11").detach()
            )
            mid_invocations.append(invocation_id)
            return None

        def capture_dit(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            if active_invocation[0] is not None or completed_invocations != [1]:
                raise RuntimeError(
                    "DiT conditioning did not follow the same completed GR00T backbone forward"
                )
            if captured.dit_conditioning:
                raise RuntimeError(
                    "action_head.vl_self_attention must run exactly once per action capture"
                )
            captured.dit_conditioning.append(
                self._hook_tensor(
                    output, source="action_head.vl_self_attention"
                ).detach()
            )
            dit_invocations.append(completed_invocations[0])
            return None

        handles: list[Any] = []
        with self.policy_capture_lock:
            self._freeze_policy()
            try:
                handles.append(
                    backbone.register_forward_pre_hook(
                        capture_backbone_pre, with_kwargs=True
                    )
                )
                handles.append(
                    backbone.register_forward_hook(finish_backbone, with_kwargs=True)
                )
                handles.append(block.register_forward_hook(capture_mid))
                handles.append(vl_self_attention.register_forward_hook(capture_dit))
                with torch.no_grad():
                    result = call()
            finally:
                for handle in reversed(handles):
                    handle.remove()
        if (
            len(backbone_inputs) != 1
            or completed_invocations != [1]
            or mid_invocations != [1]
            or (dit_invocations != [1])
            or (active_invocation[0] is not None)
        ):
            raise RuntimeError(
                "one safe GR00T action call must contain exactly one same-forward backbone/block11/vl_self_attention capture"
            )
        consumed = backbone_inputs[0]
        return (
            result,
            self._packet(
                captured=captured,
                input_ids=consumed.input_ids,
                image_grid_thw=consumed.image_grid_thw,
                attention_mask=consumed.attention_mask,
                timestep_index=timestep_index,
                frame_ids=frame_ids,
                timestamps=timestamps,
                prompt_id=prompt_id,
                detector_generation=detector_generation,
                context_generation=context_generation,
            ),
        )

    def capture_action_call(
        self,
        call: Callable[[], _groot_temporal_Result],
        *,
        input_ids: Tensor,
        image_grid_thw: Tensor,
        attention_mask: Tensor | None = None,
        batch_size: int = 1,
        timestep_index: int = -1,
        frame_ids: Tensor | int | None = None,
        timestamps: Tensor | float | None = None,
        prompt_id: str = "unspecified",
        detector_generation: Tensor | int | None = None,
        context_generation: Tensor | int | None = None,
    ) -> tuple[_groot_temporal_Result, GrootTemporalFeaturePacket]:
        """Low-level hook capture with caller-supplied token metadata.

        This API cannot prove that ``input_ids``, ``image_grid_thw``, and
        ``attention_mask`` are the tensors consumed by ``call``.  The caller is
        solely responsible for that identity.  It remains available for
        backbone-only capture and hook-cleanup tests; policy/action runtime code
        must use :meth:`capture_policy_action_call` instead.
        """
        from endo_bbox_vla.network import _CapturedTemporalFeatures

        if not callable(call):
            raise TypeError("call must be callable")
        if batch_size != 1:
            raise ValueError("GR00T temporal extraction is fail-closed to batch_size=1")
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError(
                "input_ids must have shape [1,S] for fail-closed B=1 capture"
            )
        if not isinstance(image_grid_thw, Tensor):
            raise TypeError("image_grid_thw must be a torch.Tensor")
        self._assert_eager_capture(call)
        _backbone, _visual, block, vl_self_attention = self._modules_for_capture()
        captured = _CapturedTemporalFeatures(mid=[], dit_conditioning=[])

        def capture_mid(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            captured.mid.append(
                self._hook_tensor(output, source="visual block 11").detach()
            )
            return None

        def capture_dit(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            captured.dit_conditioning.append(
                self._hook_tensor(
                    output, source="action_head.vl_self_attention"
                ).detach()
            )
            return None

        handles: list[Any] = []
        lock = self.policy_capture_lock
        with lock:
            self._freeze_policy()
            try:
                handles.append(block.register_forward_hook(capture_mid))
                handles.append(vl_self_attention.register_forward_hook(capture_dit))
                with torch.no_grad():
                    result = call()
            finally:
                for handle in reversed(handles):
                    handle.remove()
        packet = self._packet(
            captured=captured,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            timestep_index=timestep_index,
            frame_ids=frame_ids,
            timestamps=timestamps,
            prompt_id=prompt_id,
            detector_generation=detector_generation,
            context_generation=context_generation,
        )
        return (result, packet)


def build_detector(model: str = "ours", **kwargs):
    """Build Ours or a named temporal comparison, without downloading a backbone."""
    from endo_bbox_vla.config import paper_model
    from endo_bbox_vla.network import SharedDetectorHead, build_groot_temporal_head

    identity = paper_model(model)
    if model == "ours":
        return SharedDetectorHead(**kwargs)
    if identity.feature_mode is not None:
        return build_groot_temporal_head(feature_mode=identity.feature_mode, **kwargs)
    raise ValueError(
        f"{identity.label} is built through comparisons, not this shared-head factory"
    )
