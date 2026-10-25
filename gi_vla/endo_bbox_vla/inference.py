"""Same-forward action/detection and chronological detector memory."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from endo_bbox_vla.config import SharedDetectorSpec
    from endo_bbox_vla.model import FrozenGrootN17MultiLevelExtractor

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class KOfNDecision:
    detected: bool
    hits: int
    observations: int
    ready: bool
    active: bool
    fired: bool


@dataclass(frozen=True, slots=True)
class PolypPrediction:
    frame_id: int
    timestamp: float
    score: float
    threshold: float
    threshold_positive: bool
    bbox_valid: bool
    bbox_xyxy: tuple[float, float, float, float] | None
    event: KOfNDecision | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy"] = None if self.bbox_xyxy is None else list(self.bbox_xyxy)
        return payload


@dataclass(frozen=True, slots=True)
class ActionDetectionResult:
    action_chunk: Tensor | None
    detector: PolypPrediction
    mode: str


class KOfNEventRule:
    """Rising-edge gate matching the validation rule sweep exactly.

    Unlike the historical endo_bbox_vla gate, this rule stays inactive until a
    complete ``window_size`` history exists. That detail is required to match
    the 30 Hz validation analysis at episode/reset boundaries.
    """

    def __init__(
        self,
        *,
        hits_required: int = 12,
        window_size: int = 15,
        require_full_window: bool = True,
    ) -> None:
        if hits_required <= 0 or window_size <= 0 or hits_required > window_size:
            raise ValueError("event rule requires 0 < hits_required <= window_size")
        self.hits_required = int(hits_required)
        self.window_size = int(window_size)
        self.require_full_window = bool(require_full_window)
        self._window: deque[bool] = deque(maxlen=self.window_size)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._window.clear()
        self._active = False

    def update(self, detected: bool) -> KOfNDecision:
        self._window.append(bool(detected))
        hits = sum(self._window)
        ready = len(self._window) == self.window_size or not self.require_full_window
        active = ready and hits >= self.hits_required
        fired = active and (not self._active)
        self._active = active
        return KOfNDecision(
            detected=bool(detected),
            hits=hits,
            observations=len(self._window),
            ready=ready,
            active=active,
            fired=fired,
        )


def _install_local_processor(spec: SharedDetectorSpec, processor_path: Path) -> None:
    from endo_bbox_vla.utils import sha256_file

    expected_hashes = dict(spec.processor["files_sha256"])
    actual_hashes = {
        filename: sha256_file(processor_path / filename) for filename in expected_hashes
    }
    if actual_hashes != expected_hashes:
        raise ValueError("bundled GR00T processor files failed SHA-256 verification")
    from lerobot.policies.groot import processor_groot

    original = processor_groot._build_n1_7_processor
    expected_repo = str(spec.processor["upstream_repo"])

    def build_local_processor(model_name: str = expected_repo) -> Any:
        if model_name != expected_repo:
            raise ValueError(f"unexpected GR00T processor repository: {model_name!r}")
        return original(str(processor_path))

    processor_groot._build_n1_7_processor = build_local_processor


def _normalized_cxcywh_to_original_xyxy(
    box: Tensor,
) -> tuple[float, float, float, float] | None:
    from endo_bbox_vla.data import GROOT_N17_GEOMETRY

    value = box.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if value.numel() != 4 or not bool(torch.isfinite(value).all()):
        return None
    center, size = (value[:2], value[2:])
    xyxy = torch.cat((center - size / 2, center + size / 2)).clamp(0, 1)
    if bool((xyxy[2:] <= xyxy[:2]).any()):
        return None
    width, height = GROOT_N17_GEOMETRY.output_size
    scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype)
    original = GROOT_N17_GEOMETRY.inverse_xyxy((xyxy * scale).unsqueeze(0))[0]
    if not bool(torch.isfinite(original).all()) or bool(
        (original[2:] <= original[:2]).any()
    ):
        return None
    return tuple((float(item) for item in original))


def _metadata_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, Tensor):
            copied[key] = value.detach().clone()
        elif isinstance(value, np.ndarray) or hasattr(value, "copy"):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


class SharedDetectorRuntime:
    """Load one frozen GR00T policy and attach the validation-selected block-11 visual head.

    ``predict_action_and_detector`` captures block-11 visual tokens from the exact GR00T
    action forward. ``predict_detector`` performs a fresh backbone-only forward
    and is intended for shadow detection between action replans. Detector output
    never enters the action model.
    """

    def __init__(
        self,
        *,
        spec: SharedDetectorSpec,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        extractor: FrozenGrootN17MultiLevelExtractor,
        head: torch.nn.Module,
        device: torch.device,
        base_snapshot: Path,
        checkpoint: Path,
    ) -> None:
        self.spec = spec
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.extractor = extractor
        self.head = head
        self.device = device
        self.base_snapshot = base_snapshot
        self.checkpoint = checkpoint
        rule = spec.event_rule
        self.event_rule = KOfNEventRule(
            hits_required=int(rule["hits_required"]),
            window_size=int(rule["window_size"]),
            require_full_window=bool(rule["require_full_window"]),
        )
        self._camera_order_validated = False
        self._last_frame_id: int | None = None

    @classmethod
    def load(
        cls,
        config: str | Path,
        *,
        device: str = "cuda:0",
        checkpoint: str | Path | None = None,
        base_snapshot: str | Path | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_base_model_hash: bool = True,
    ) -> SharedDetectorRuntime:
        from endo_bbox_vla.config import load_online_spec
        from endo_bbox_vla.model import FrozenGrootN17MultiLevelExtractor
        from endo_bbox_vla.network import (
            OneQueryDetrHead,
            SpatiallyInitializedDetectorHead,
        )
        from endo_bbox_vla.utils import sha256_file

        spec = load_online_spec(config)
        target_device = torch.device(device)
        if target_device.type == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError(
                "CUDA is required for GR00T block-11 visual online evaluation"
            )
        if base_snapshot is None:
            from huggingface_hub import snapshot_download

            base_path = Path(
                snapshot_download(
                    repo_id=str(spec.base_policy["repo_id"]),
                    revision=str(spec.base_policy["revision"]),
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    token=token,
                    local_files_only=local_files_only,
                )
            ).resolve()
        else:
            base_path = Path(base_snapshot).resolve()
        model_path = base_path / str(spec.base_policy["model_filename"])
        if model_path.stat().st_size != int(spec.base_policy["model_bytes"]):
            raise ValueError(
                "GR00T model byte size differs from the pinned action checkpoint"
            )
        if (
            verify_base_model_hash
            and sha256_file(model_path) != spec.base_policy["model_sha256"]
        ):
            raise ValueError("GR00T action model SHA-256 mismatch")
        processor_path = spec.processor_path
        _install_local_processor(spec, processor_path)
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        policy = GrootPolicy.from_pretrained(
            base_path,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=True,
            strict=True,
        )
        policy.config.device = str(target_device)
        policy.to(target_device).eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        if not all((not parameter.requires_grad for parameter in policy.parameters())):
            raise ValueError("GR00T action policy is not fully frozen")
        if bool(getattr(policy.config, "use_relative_actions", False)) is not True:
            raise ValueError("loaded action policy lost its relative-action contract")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(base_path),
            preprocessor_overrides={"device_processor": {"device": str(target_device)}},
        )
        camera_order = tuple(
            (str(value) for value in spec.detector["policy_camera_order"])
        )
        extractor = (
            FrozenGrootN17MultiLevelExtractor(
                policy,
                camera_order=camera_order,
                camera_key=str(spec.detector["camera_key"]),
            )
            .to(target_device)
            .eval()
        )
        if extractor.state_dict():
            raise RuntimeError("GR00T extractor unexpectedly serialized policy weights")
        architecture = dict(spec.detector["architecture"])
        if architecture["type"] == "one_query_detr":
            head: torch.nn.Module = OneQueryDetrHead(
                input_dim=int(architecture["input_dim"]),
                hidden_dim=int(architecture["hidden_dim"]),
                num_layers=int(architecture["decoder_layers"]),
                num_heads=int(architecture["attention_heads"]),
                dim_feedforward=int(architecture["feedforward_dim"]),
                grid_size=tuple((int(value) for value in architecture["grid_size"])),
                padded_rows=tuple(
                    (int(value) for value in architecture["padded_rows"])
                ),
                dropout=float(architecture["dropout"]),
            ).to(target_device)
        elif architecture["type"] == "location_first_four_query":
            head = SpatiallyInitializedDetectorHead(
                input_dim=int(architecture["input_dim"]),
                hidden_dim=int(architecture["hidden_dim"]),
                grid_size=tuple((int(value) for value in architecture["grid_size"])),
                num_queries=int(architecture["location_queries"]),
                decoder_layers=int(architecture["decoder_layers"]),
                attention_heads=int(architecture["attention_heads"]),
                feedforward_dim=int(architecture["feedforward_dim"]),
                dropout=float(architecture["dropout"]),
            ).to(target_device)
        else:
            raise ValueError(
                f"unsupported block-11 visual online head: {architecture['type']!r}"
            )
        checkpoint_path = (
            spec.checkpoint_path if checkpoint is None else Path(checkpoint).resolve()
        )
        if sha256_file(checkpoint_path) != spec.detector["sha256"]:
            raise ValueError("block-11 visual detector checkpoint SHA-256 mismatch")
        payload = torch.load(
            checkpoint_path, map_location=target_device, weights_only=True
        )
        if not isinstance(payload, Mapping) or set(payload) != {"head", "metadata"}:
            raise TypeError("invalid block-11 visual detector checkpoint payload")
        metadata = dict(payload["metadata"])
        if _metadata_sha256(metadata) != spec.detector["metadata_sha256"]:
            raise ValueError("block-11 visual checkpoint metadata SHA-256 mismatch")
        if architecture["type"] == "one_query_detr":
            expected_metadata = {
                "kind": "groot_n17_detector_sidecar",
                "schema_version": 1,
                "experiment_id": spec.detector["experiment_id"],
                "feature_level": "e03",
                "backbone_checkpoint": spec.base_policy["repo_id"],
                "backbone_revision": spec.base_policy["revision"],
                "config_sha256": "eda6440e6198b0b5229945f2d061f6324eb701e15746c7e72d77ae9b2a22c538",
                "cache_manifest_sha256": spec.detector[
                    "training_cache_manifest_sha256"
                ],
                "seed": int(spec.detector["seed"]),
                "global_step": 4450,
                "termination_reason": "training_complete",
                "artifact_role": "production_final_epoch",
                "locked_test_used": False,
            }
        else:
            expected_metadata = {
                "kind": "shared_e03_location_first_detector_head",
                "schema_version": 1,
                "variant": spec.detector["variant"],
                "config_sha256": spec.detector["training_config_sha256"],
                "base_model_sha256": spec.base_policy["model_sha256"],
                "train_cache_manifest_sha256": spec.detector[
                    "training_cache_manifest_sha256"
                ],
                "validation_cache_manifest_sha256": spec.detector[
                    "validation_cache_manifest_sha256"
                ],
                "action_parity_sha256": spec.detector["action_parity_sha256"],
                "feature_source": spec.detector["feature_source"],
                "feature_capture": "same_action_forward",
                "visual_parameters_trainable": False,
                "serialized_parameters": "detector_head_only",
                "internal_test_used": False,
            }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"block-11 visual checkpoint provenance mismatch: {mismatches}"
            )
        head.load_state_dict(payload["head"], strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        return cls(
            spec=spec,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            extractor=extractor,
            head=head,
            device=target_device,
            base_snapshot=base_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        **load_kwargs: Any,
    ) -> SharedDetectorRuntime:
        from huggingface_hub import snapshot_download

        bundle = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                token=token,
                local_files_only=local_files_only,
            )
        )
        return cls.load(
            bundle / "deployment.yaml",
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            **load_kwargs,
        )

    def reset(self) -> None:
        self.policy.reset()
        self.event_rule.reset()
        self._last_frame_id = None

    def reset_policy(self) -> None:
        self.policy.reset()

    def reset_event_rule(self) -> None:
        self.event_rule.reset()

    def _prepare(
        self, observation: Mapping[str, Any], *, task: str, robot_type: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
        from lerobot.policies.utils import prepare_observation_for_inference

        prepared = prepare_observation_for_inference(
            _copy_observation(observation), self.device, task, robot_type
        )
        processed = self.preprocessor(prepared)
        if not isinstance(processed, Mapping):
            raise TypeError("serialized GR00T preprocessor returned a non-mapping")
        if not self._camera_order_validated:
            pack_steps = [
                step
                for step in self.preprocessor.steps
                if isinstance(step, GrootN17PackInputsStep)
            ]
            if len(pack_steps) != 1:
                raise RuntimeError(
                    "GR00T preprocessor must contain exactly one pack-input step"
                )
            observed_order = tuple(pack_steps[0]._ordered_image_keys(dict(prepared)))
            expected_order = tuple(
                (str(value) for value in self.spec.detector["policy_camera_order"])
            )
            if observed_order != expected_order:
                raise ValueError(
                    f"GR00T processor camera order changed: {observed_order} != {expected_order}"
                )
            self._camera_order_validated = True
        return (dict(prepared), dict(processed))

    def _backbone_inputs(self, processed: Mapping[str, Any]) -> dict[str, Any]:
        filtered = self.policy._filter_groot_inputs(
            dict(processed), include_action=False
        )
        inputs, _ = self.policy._groot_model.prepare_input(filtered)
        if "image_grid_thw" not in inputs:
            raise KeyError("GR00T backbone inputs lack image_grid_thw")
        return dict(inputs)

    def _prediction(
        self, packet: Any, *, frame_id: int, timestamp: float, update_event: bool
    ) -> PolypPrediction:
        from endo_bbox_vla.network import select_single_prediction

        mid_tokens, padding_mask = packet.head_inputs("mid")
        output = self.head(
            mid_tokens.detach().to(device=self.device, dtype=torch.float32),
            padding_mask.detach().to(device=self.device, dtype=torch.bool),
        )
        logits = output["presence_logits"]
        if logits.ndim == 1:
            score_tensor = logits[0].sigmoid()
            box_tensor = output["boxes_cxcywh"][0]
        elif logits.ndim == 2:
            scores, boxes = select_single_prediction(output)
            score_tensor = scores[0]
            box_tensor = boxes[0]
        else:
            raise ValueError(
                "block-11 visual online head returned an invalid presence tensor"
            )
        score = float(score_tensor.float().cpu())
        box = _normalized_cxcywh_to_original_xyxy(box_tensor)
        threshold = float(self.spec.detector["threshold"])
        threshold_positive = score >= threshold
        event = self.event_rule.update(threshold_positive) if update_event else None
        return PolypPrediction(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            score=score,
            threshold=threshold,
            threshold_positive=threshold_positive,
            bbox_valid=box is not None,
            bbox_xyxy=box,
            event=event,
        )

    def _validate_frame_identity(self, frame_id: int) -> None:
        frame = int(frame_id)
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise ValueError("online detector frame_id must increase strictly")
        self._last_frame_id = frame

    def _validate_actions(self, actions: Any) -> Tensor:
        if not isinstance(actions, Tensor):
            raise TypeError("GR00T postprocessor returned a non-tensor action chunk")
        expected_shape = (
            1,
            int(self.spec.base_policy["action_chunk_size"]),
            int(self.spec.base_policy["action_dim"]),
        )
        if tuple(actions.shape) != expected_shape or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError(
                f"GR00T action output must be finite {expected_shape}, got {tuple(actions.shape)}"
            )
        return actions

    def predict_action_chunk(
        self, observation: Mapping[str, Any], *, task: str, robot_type: str
    ) -> Tensor:
        """Return one full postprocessed action chunk without changing the event rule."""
        _, processed = self._prepare(observation, task=task, robot_type=robot_type)
        with torch.inference_mode():
            raw_actions = self.policy.predict_action_chunk(dict(processed))
            actions = self._validate_actions(self.postprocessor(raw_actions))
        return actions.detach().cpu()

    def predict_detector(
        self,
        observation: Mapping[str, Any],
        *,
        task: str,
        robot_type: str,
        frame_id: int,
        timestamp: float,
        update_event: bool = True,
    ) -> ActionDetectionResult:
        """Run a fresh backbone-only block-11 visual detector forward on one live observation."""
        self._validate_frame_identity(frame_id)
        _, processed = self._prepare(observation, task=task, robot_type=robot_type)
        inputs = self._backbone_inputs(processed)
        with torch.inference_mode():
            _, packet = self.extractor.capture_action_call(
                lambda: self.policy._groot_model.backbone(inputs),
                image_grid_thw=inputs["image_grid_thw"],
                batch_size=1,
                timestep_index=-1,
            )
            prediction = self._prediction(
                packet,
                frame_id=frame_id,
                timestamp=timestamp,
                update_event=update_event,
            )
        return ActionDetectionResult(None, prediction, "fresh_backbone_detector_only")

    def predict_action_and_detector(
        self,
        observation: Mapping[str, Any],
        *,
        task: str,
        robot_type: str,
        frame_id: int,
        timestamp: float,
        update_event: bool = True,
    ) -> ActionDetectionResult:
        """Return a full postprocessed GR00T action chunk and block-11 visual result from one forward."""
        self._validate_frame_identity(frame_id)
        _, processed = self._prepare(observation, task=task, robot_type=robot_type)
        inputs = self._backbone_inputs(processed)
        with torch.inference_mode():
            raw_actions, packet = self.extractor.capture_action_call(
                lambda: self.policy.predict_action_chunk(dict(processed)),
                image_grid_thw=inputs["image_grid_thw"],
                batch_size=1,
                timestep_index=-1,
            )
            actions = self._validate_actions(self.postprocessor(raw_actions))
            prediction = self._prediction(
                packet,
                frame_id=frame_id,
                timestamp=timestamp,
                update_event=update_event,
            )
        return ActionDetectionResult(
            action_chunk=actions.detach().cpu(),
            detector=prediction,
            mode="combined_action_forward_with_e03_capture",
        )


class TemporalDetectorRuntime:
    """Reuse one policy forward and carry detector memory between frames.

    The caller owns preprocessing, policy execution, and robot control. This
    object only observes the supplied action call and maintains detector state.
    """

    def __init__(self, extractor, head):
        import threading

        self.extractor = extractor
        self.head = head
        if isinstance(head, torch.nn.Module):
            self.head.eval()
        self.state = None
        self._lock = threading.Lock()

    def reset(self):
        """Clear detector memory at an episode or task boundary."""
        with self._lock:
            self.state = None

    @torch.inference_mode()
    def predict(self, action_call, *, reset=False, **frame_context):
        """Return the action and detection from the exact same policy forward."""
        with self._lock:
            action, packet = self.extractor.capture_policy_action_call(
                action_call, **frame_context
            )
            prediction, self.state = self.head(
                packet, previous_state=self.state, reset=reset, detach_next_state=True
            )
            return action, prediction
