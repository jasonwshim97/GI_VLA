# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Hugging Face LeRobot for the GI-VLA experiments.

import contextlib
import json
import logging
import math
import multiprocessing
import random
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs import RTCAttentionSchedule, parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
import lerobot.policies
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (
    Robot,
    RobotConfig,
    aloha_follower,
    make_robot_from_config,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.common.control_utils import predict_action
from lerobot.utils.keyboard_input import init_keyboard_listener, is_headless
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.common.aloha_dxl import ACTION_ORDER

from lerobot.teleoperators.knob_keyboard.knob_teleoperator import (
    KnobKeyboardConfig,
    KnobKeyboardTeleop,
)
from lerobot.robots.aloha_follower_with_knob.aloha_follower_with_knob import AlohaFollowerWithKnob

KNOB_ORDER = ("knob_ud.pos", "knob_lr.pos")
EXPECTED_STATE_NAMES_16D = list(ACTION_ORDER) + list(KNOB_ORDER)


def _cam_display_process(frame_queue: multiprocessing.Queue, cam_name: str):

    cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cam_name, 640, 480)

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
            if frame is None:
                break
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(cam_name, display)
            cv2.waitKey(1)
        except Exception:
            continue

    cv2.destroyAllWindows()


class CamDisplayProcess:

    def __init__(self, cam_name: str = "cam_low"):
        self._cam_name = cam_name
        self._queue = multiprocessing.Queue(maxsize=2)
        self._process = None

    def start(self):
        self._process = multiprocessing.Process(
            target=_cam_display_process,
            args=(self._queue, self._cam_name),
            daemon=True,
        )
        self._process.start()

    def send_frame(self, frame):
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put_nowait(frame)
        except Exception:
            pass

    def stop(self):
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()


def init_pedal_listener(events: dict) -> Any:

    if is_headless():
        return None

    try:
        from pynput import keyboard

        def on_press(key):
            try:
                k = key.char if hasattr(key, "char") else None
                if k == "1":
                    if events.get("in_reset_wait", False):
                        print("[Pedal 1] → Skip reset")
                        events["reset_skip"] = True
                    else:
                        print("[Pedal 1] → Success")
                        events["exit_early"] = True
                        events["episode_success"] = True
                elif k == "2":
                    print("[Pedal 2] → Fail")
                    events["exit_early"] = True
                    events["episode_success"] = False
                elif k == "3":
                    print("[Pedal 3] → Stop all")
                    events["stop_inference"] = True
                    events["exit_early"] = True
                elif k == "n":
                    if events.get("in_reset_wait", False):
                        print("[n] → Skip reset")
                        events["reset_skip"] = True
            except Exception as e:
                logging.debug(f"Pedal listener error: {e}")

            try:
                if key == keyboard.Key.right:
                    if events.get("in_reset_wait", False):
                        events["reset_skip"] = True
                    else:
                        events["exit_early"] = True
                        events["episode_success"] = True
                elif key == keyboard.Key.left:
                    events["exit_early"] = True
                    events["episode_success"] = False
                elif key == keyboard.Key.esc:
                    events["stop_inference"] = True
                    events["exit_early"] = True
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logging.info("Pedal/keyboard listener started")
        return listener

    except Exception as e:
        logging.warning(f"Pedal listener init failed: {e}")
        return None


@dataclass
class InferenceConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    n_episodes: int = 10
    episode_time_s: float = 60.0
    fps: int = 30
    single_task: str = "pick and insert endoscope"

    knob: KnobKeyboardConfig = field(default_factory=KnobKeyboardConfig)
    use_knob: bool = True

    output_dir: str = "./inference_results"
    save_video: bool = True
    save_dataset: bool = True
    save_action_trace: bool = False
    output_dataset_root: str | Path | None = None
    output_dataset_repo_id: str = ""
    dataset_video: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "auto"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None

    dataset_repo_id: str = ""
    dataset_root: str | Path | None = None

    play_sounds: bool = True
    display_data: bool = False
    display_camera: bool = False

    bbox_package_dir: str | Path | None = None
    bbox_deployment_config: str | Path | None = None
    bbox_detector_repo: str | None = None
    bbox_detector_revision: str | None = None
    bbox_base_policy_path: str | Path | None = None
    bbox_cache_dir: str | Path | None = None
    bbox_local_files_only: bool = False
    bbox_use_bf16: bool = True
    bbox_overlay_video: bool = True
    bbox_frame_cache_size: int = 8
    bbox_inference_hz: float = 3.0
    bbox_min_action_queue_remaining: int = 2

    molmo_async_replan: bool = False
    molmo_async_replan_trigger_remaining: int = 18
    molmo_async_replan_blend_steps: int = 3
    molmo_async_replan_timeout_s: float = 3.0
    molmo_async_replan_max_hold_steps: int = 6
    molmo_warmup: bool = False

    pi05_async_replan: bool = False
    pi05_async_replan_trigger_remaining: int = 25
    pi05_async_replan_blend_steps: int = 0
    pi05_async_replan_timeout_s: float = 2.0
    pi05_async_replan_max_hold_steps: int = 6
    pi05_rtc_execution_horizon: int = 15
    pi05_rtc_max_guidance_weight: float = 10.0
    pi05_rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR

    pi05_shared_full_chunk: bool = False

    sync_initial_pose: bool = True
    initial_pose_strategy: str = "same_task"
    initial_pose_episode_index: int | None = None
    initial_pose_average_task_a: str | None = None
    initial_pose_average_task_b: str | None = None
    initial_pose_wait_s: float = 3.0
    initial_pose_profile_velocity: int = 30
    initial_pose_profile_accel: int = 5
    initial_pose_profile_velocity_gripper: int = 30
    initial_pose_profile_accel_gripper: int = 5
    knob_zero_before_initial_pose: bool = False
    knob_zero_before_initial_pose_wait_s: float = 2.0
    knob_zero_after_episode: bool = False

    reset_time_s: float = 5.0
    reset_skip_enable_delay_s: float = 0.75

    enable_action_safety_limits: bool = True
    right_shoulder_min_deg: float = -116.0
    right_shoulder_max_deg: float = 45.0

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if not self.dataset_repo_id.strip():
            raise ValueError("--dataset_repo_id is required")
        if self.save_dataset and not self.output_dataset_repo_id.strip():
            raise ValueError(
                "--output_dataset_repo_id is required when --save_dataset=true")
        if self.bbox_inference_hz <= 0:
            raise ValueError("bbox_inference_hz must be positive")
        if self.bbox_min_action_queue_remaining < 0:
            raise ValueError(
                "bbox_min_action_queue_remaining cannot be negative")
        if self.molmo_async_replan_trigger_remaining <= 0:
            raise ValueError(
                "molmo_async_replan_trigger_remaining must be positive")
        if self.molmo_async_replan_blend_steps < 0:
            raise ValueError(
                "molmo_async_replan_blend_steps cannot be negative")
        if self.molmo_async_replan_timeout_s <= 0:
            raise ValueError("molmo_async_replan_timeout_s must be positive")
        if self.molmo_async_replan_max_hold_steps <= 0:
            raise ValueError(
                "molmo_async_replan_max_hold_steps must be positive")
        if self.molmo_warmup and not self.molmo_async_replan:
            raise ValueError(
                "molmo_warmup=true requires molmo_async_replan=true")
        if self.pi05_async_replan_trigger_remaining <= 0:
            raise ValueError(
                "pi05_async_replan_trigger_remaining must be positive")
        if self.pi05_async_replan_blend_steps < 0:
            raise ValueError(
                "pi05_async_replan_blend_steps cannot be negative")
        if self.pi05_async_replan_timeout_s <= 0:
            raise ValueError("pi05_async_replan_timeout_s must be positive")
        if self.pi05_async_replan_max_hold_steps <= 0:
            raise ValueError(
                "pi05_async_replan_max_hold_steps must be positive")
        if self.pi05_rtc_execution_horizon <= 0:
            raise ValueError("pi05_rtc_execution_horizon must be positive")
        if self.pi05_rtc_max_guidance_weight <= 0:
            raise ValueError("pi05_rtc_max_guidance_weight must be positive")
        if self.pi05_async_replan and self.pi05_async_replan_blend_steps != 0:
            raise ValueError(
                "Pi0.5 RTC already guides the overlapping prefix; "
                "use --pi05_async_replan_blend_steps=0 to avoid double blending"
            )
        if (self.pi05_async_replan and self.pi05_rtc_execution_horizon
                > self.pi05_async_replan_trigger_remaining):
            raise ValueError("pi05_rtc_execution_horizon cannot exceed "
                             "pi05_async_replan_trigger_remaining")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


ACTION_TRACE_FIELDS = (
    "frame_index",
    "loop_start_s",
    "frame_interval_s",
    "observation_s",
    "video_s",
    "policy_s",
    "action_send_s",
    "command_sent_s",
    "loop_compute_s",
    "sleep_requested_s",
    "loop_total_s",
    "replan_due",
    "queue_remaining",
    "lookahead_submitted",
    "lookahead_completed",
    "lookahead_pending",
    "lookahead_hold",
    "lookahead_pending_age_s",
    "lookahead_inference_s",
    "lookahead_wait_s",
    "lookahead_discarded_steps",
    "source_chunk_index",
    "source_action_index",
    "source_frame_index",
    "blend_alpha",
    "blend_old_chunk_index",
    "blend_old_action_index",
    "action_feedback_age_s",
    "observation_state",
    "predicted_action",
    "commanded_action",
)


def _new_action_trace() -> dict[str, list[Any]]:
    return {key: [] for key in ACTION_TRACE_FIELDS}


def _save_episode_action_trace(
    path: Path,
    trace: dict[str, list[Any]],
    *,
    state_names: list[str],
    action_names: list[str],
    task: str,
    fps: int,
    n_action_steps: int,
    success: bool | None,
    elapsed_s: float,
    molmo_async_replan: bool = False,
    molmo_async_trigger_remaining: int = -1,
    molmo_async_blend_steps: int = 0,
    molmo_async_timeout_s: float = math.nan,
    molmo_async_max_hold_steps: int = -1,
    molmo_warmup: bool = False,
    molmo_warmup_s: float = math.nan,
    molmo_planner_drain_s: float = 0.0,
    pi05_async_replan: bool = False,
    pi05_async_trigger_remaining: int = -1,
    pi05_async_blend_steps: int = 0,
    pi05_async_timeout_s: float = math.nan,
    pi05_async_max_hold_steps: int = -1,
    pi05_rtc_execution_horizon: int = -1,
    pi05_rtc_max_guidance_weight: float = math.nan,
    pi05_rtc_prefix_attention_schedule: str = "",
    episode_outcome: str | None = None,
    trace_complete: bool = True,
    episode_error_type: str = "",
    episode_error_message: str = "",
    episode_signaled_success: bool | None = None,
) -> None:

    n_frames = len(trace["frame_index"])
    for key in ACTION_TRACE_FIELDS:
        if len(trace[key]) != n_frames:
            raise ValueError(
                f"Action trace field length mismatch: {key} has {len(trace[key])}, expected {n_frames}."
            )

    def vector_array(key: str, width: int) -> np.ndarray:
        values = trace[key]
        if not values:
            return np.empty((0, width), dtype=np.float32)
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (n_frames, width):
            raise ValueError(
                f"Action trace {key} must have shape {(n_frames, width)}, got {array.shape}."
            )
        return array

    replan_due = np.asarray(trace["replan_due"], dtype=np.bool_)
    chunk_index = np.cumsum(replan_due.astype(np.int32)) - 1
    success_code = -1 if success is None else int(success)
    signaled_success_code = (-1 if episode_signaled_success is None else
                             int(episode_signaled_success))
    if episode_outcome is None:
        episode_outcome = ("success" if success is True else
                           "fail" if success is False else "timeout")

    np.savez_compressed(
        path,
        schema_version=np.asarray(5, dtype=np.int16),
        task=np.asarray(task, dtype=np.str_),
        target_fps=np.asarray(fps, dtype=np.int16),
        n_action_steps=np.asarray(n_action_steps, dtype=np.int16),
        episode_success_code=np.asarray(success_code, dtype=np.int8),
        episode_elapsed_s=np.asarray(elapsed_s, dtype=np.float64),
        molmo_async_replan=np.asarray(molmo_async_replan, dtype=np.bool_),
        molmo_async_trigger_remaining=np.asarray(molmo_async_trigger_remaining,
                                                 dtype=np.int16),
        molmo_async_blend_steps=np.asarray(molmo_async_blend_steps,
                                           dtype=np.int16),
        molmo_async_timeout_s=np.asarray(molmo_async_timeout_s,
                                         dtype=np.float64),
        molmo_async_max_hold_steps=np.asarray(molmo_async_max_hold_steps,
                                              dtype=np.int16),
        molmo_warmup=np.asarray(molmo_warmup, dtype=np.bool_),
        molmo_warmup_s=np.asarray(molmo_warmup_s, dtype=np.float64),
        molmo_planner_drain_s=np.asarray(molmo_planner_drain_s,
                                         dtype=np.float64),
        pi05_async_replan=np.asarray(pi05_async_replan, dtype=np.bool_),
        pi05_async_trigger_remaining=np.asarray(pi05_async_trigger_remaining,
                                                dtype=np.int16),
        pi05_async_blend_steps=np.asarray(pi05_async_blend_steps,
                                          dtype=np.int16),
        pi05_async_timeout_s=np.asarray(pi05_async_timeout_s,
                                        dtype=np.float64),
        pi05_async_max_hold_steps=np.asarray(pi05_async_max_hold_steps,
                                             dtype=np.int16),
        pi05_rtc_execution_horizon=np.asarray(pi05_rtc_execution_horizon,
                                              dtype=np.int16),
        pi05_rtc_max_guidance_weight=np.asarray(pi05_rtc_max_guidance_weight,
                                                dtype=np.float64),
        pi05_rtc_prefix_attention_schedule=np.asarray(
            pi05_rtc_prefix_attention_schedule, dtype=np.str_),
        episode_outcome=np.asarray(episode_outcome, dtype=np.str_),
        trace_complete=np.asarray(trace_complete, dtype=np.bool_),
        episode_error_type=np.asarray(episode_error_type, dtype=np.str_),
        episode_error_message=np.asarray(episode_error_message, dtype=np.str_),
        episode_signaled_success_code=np.asarray(signaled_success_code,
                                                 dtype=np.int8),
        state_names=np.asarray(state_names, dtype=np.str_),
        action_names=np.asarray(action_names, dtype=np.str_),
        frame_index=np.asarray(trace["frame_index"], dtype=np.int32),
        chunk_index=chunk_index.astype(np.int32, copy=False),
        loop_start_s=np.asarray(trace["loop_start_s"], dtype=np.float64),
        frame_interval_s=np.asarray(trace["frame_interval_s"],
                                    dtype=np.float64),
        observation_s=np.asarray(trace["observation_s"], dtype=np.float64),
        video_s=np.asarray(trace["video_s"], dtype=np.float64),
        policy_s=np.asarray(trace["policy_s"], dtype=np.float64),
        action_send_s=np.asarray(trace["action_send_s"], dtype=np.float64),
        command_sent_s=np.asarray(trace["command_sent_s"], dtype=np.float64),
        loop_compute_s=np.asarray(trace["loop_compute_s"], dtype=np.float64),
        sleep_requested_s=np.asarray(trace["sleep_requested_s"],
                                     dtype=np.float64),
        loop_total_s=np.asarray(trace["loop_total_s"], dtype=np.float64),
        replan_due=replan_due,
        queue_remaining=np.asarray(trace["queue_remaining"], dtype=np.int16),
        lookahead_submitted=np.asarray(trace["lookahead_submitted"],
                                       dtype=np.bool_),
        lookahead_completed=np.asarray(trace["lookahead_completed"],
                                       dtype=np.bool_),
        lookahead_pending=np.asarray(trace["lookahead_pending"],
                                     dtype=np.bool_),
        lookahead_hold=np.asarray(trace["lookahead_hold"], dtype=np.bool_),
        lookahead_pending_age_s=np.asarray(trace["lookahead_pending_age_s"],
                                           dtype=np.float64),
        lookahead_inference_s=np.asarray(trace["lookahead_inference_s"],
                                         dtype=np.float64),
        lookahead_wait_s=np.asarray(trace["lookahead_wait_s"],
                                    dtype=np.float64),
        lookahead_discarded_steps=np.asarray(
            trace["lookahead_discarded_steps"], dtype=np.int16),
        source_chunk_index=np.asarray(trace["source_chunk_index"],
                                      dtype=np.int32),
        source_action_index=np.asarray(trace["source_action_index"],
                                       dtype=np.int16),
        source_frame_index=np.asarray(trace["source_frame_index"],
                                      dtype=np.int32),
        blend_alpha=np.asarray(trace["blend_alpha"], dtype=np.float32),
        blend_old_chunk_index=np.asarray(trace["blend_old_chunk_index"],
                                         dtype=np.int32),
        blend_old_action_index=np.asarray(trace["blend_old_action_index"],
                                          dtype=np.int16),
        action_feedback_age_s=np.asarray(trace["action_feedback_age_s"],
                                         dtype=np.float64),
        observation_state=vector_array("observation_state", len(state_names)),
        predicted_action=vector_array("predicted_action", len(action_names)),
        commanded_action=vector_array("commanded_action", len(action_names)),
    )


def _requires_full_chunk_postprocessing(policy: PreTrainedPolicy) -> bool:

    policy_type = getattr(policy.config, "type", None)
    return policy_type == "molmoact2" or (policy_type == "groot" and bool(
        getattr(policy.config, "use_relative_actions", False)))


def _copy_observation_frame(observation: dict[str, Any]) -> dict[str, Any]:

    snapshot: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        elif isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().clone()
        else:
            snapshot[key] = value
    return snapshot


def _predict_action_chunk_pair(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    use_amp: bool,
    task: str,
    robot_type: str,
    *,
    predict_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if getattr(policy.config, "type", None) == "molmoact2" and policy.training:
        policy.eval()
    prepared_observation = prepare_observation_for_inference(
        dict(observation), device, task, robot_type)
    autocast_context = (torch.autocast(
        device_type=device.type) if device.type == "cuda" and use_amp else
                        contextlib.nullcontext())
    rtc_enabled = bool(
        callable(getattr(policy, "_rtc_enabled", None))
        and policy._rtc_enabled())

    grad_context = torch.no_grad() if rtc_enabled else torch.inference_mode()
    with grad_context, autocast_context:
        model_input = preprocessor(prepared_observation)
        policy_chunk = policy.predict_action_chunk(model_input,
                                                   **(predict_kwargs or {}))
        original_chunk = policy_chunk.detach().clone()
        absolute_chunk = postprocessor(policy_chunk)

    if not isinstance(original_chunk, torch.Tensor) or not isinstance(
            absolute_chunk, torch.Tensor):
        raise TypeError(
            "Full-chunk inference and postprocessing must return torch.Tensor values; "
            f"got original={type(original_chunk).__name__}, "
            f"decoded={type(absolute_chunk).__name__}.")
    if (original_chunk.ndim != 3 or absolute_chunk.ndim != 3
            or original_chunk.shape[0] != 1 or absolute_chunk.shape[0] != 1
            or original_chunk.shape != absolute_chunk.shape):
        raise ValueError(
            "Full-chunk inference expects matching original/decoded shapes "
            "(1, horizon, action_dim); "
            f"got original={tuple(original_chunk.shape)}, decoded={tuple(absolute_chunk.shape)}."
        )

    execution_horizon = min(
        int(absolute_chunk.shape[1]),
        int(getattr(policy.config, "n_action_steps", absolute_chunk.shape[1])),
    )
    if execution_horizon <= 0:
        raise ValueError("Policy returned an empty action chunk.")

    original_chunk = original_chunk[:, :execution_horizon]
    decoded_chunk = absolute_chunk[:, :execution_horizon]
    if not bool(torch.isfinite(original_chunk).all()) or not bool(
            torch.isfinite(decoded_chunk).all()):
        raise ValueError(
            "Policy returned non-finite values in an action chunk.")
    return original_chunk, decoded_chunk


def _predict_decoded_action_chunk(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    use_amp: bool,
    task: str,
    robot_type: str,
    *,
    predict_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:

    _, decoded_chunk = _predict_action_chunk_pair(
        observation=observation,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        predict_kwargs=predict_kwargs,
    )
    return decoded_chunk


@dataclass(frozen=True)
class _DecodedChunkResult:
    actions: torch.Tensor
    inference_s: float
    original_actions: torch.Tensor | None = None


class _MolmoChunkWorker:

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        device: torch.device,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        use_amp: bool,
        task: str,
        robot_type: str,
        expected_action_dim: int | None = None,
        planner_label: str = "MolmoAct2",
        rtc_config: RTCConfig | None = None,
    ) -> None:
        self.policy = policy
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.use_amp = use_amp
        self.task = task
        self.robot_type = robot_type
        self.expected_action_dim = expected_action_dim
        self.planner_label = planner_label
        self.rtc_config = rtc_config
        self._previous_rtc_config = getattr(policy.config, "rtc_config", None)
        self._rtc_installed = False
        if rtc_config is not None:
            if getattr(policy.config, "type", None) != "pi05":
                raise TypeError(
                    "Native RTC chunk guidance is currently enabled only for Pi0.5"
                )
            init_rtc_processor = getattr(policy, "init_rtc_processor", None)
            if not callable(init_rtc_processor):
                raise TypeError(
                    "Pi0.5 policy does not expose init_rtc_processor()")
            policy.config.rtc_config = rtc_config
            init_rtc_processor()
            self._rtc_installed = True
        thread_label = planner_label.lower().replace(".", "").replace(" ", "-")
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{thread_label}-chunk",
        )
        self._closed = False

    def _run_prediction(
        self,
        observation: dict[str, Any],
        predict_kwargs: dict[str, Any] | None = None,
    ) -> _DecodedChunkResult:
        started = time.perf_counter()
        worker_predict_kwargs = dict(predict_kwargs or {})
        previous_chunk = worker_predict_kwargs.get("prev_chunk_left_over")
        if isinstance(previous_chunk, torch.Tensor):
            worker_predict_kwargs["prev_chunk_left_over"] = previous_chunk.to(
                device=self.device,
                dtype=torch.float32,
            )
        original_actions, actions = _predict_action_chunk_pair(
            observation=observation,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.use_amp,
            task=self.task,
            robot_type=self.robot_type,
            predict_kwargs=worker_predict_kwargs,
        )
        original_actions = original_actions.detach().to(
            device="cpu", dtype=torch.float32).clone()
        actions = actions.detach().to(device="cpu",
                                      dtype=torch.float32).clone()
        if self.expected_action_dim is not None and int(
                actions.shape[2]) != self.expected_action_dim:
            raise RuntimeError(
                f"{self.planner_label} action dimension changed unexpectedly: "
                f"expected={self.expected_action_dim}, got={int(actions.shape[2])}."
            )
        return _DecodedChunkResult(
            actions=actions,
            inference_s=time.perf_counter() - started,
            original_actions=original_actions,
        )

    def submit(
        self,
        observation: dict[str, Any],
        *,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> Future[_DecodedChunkResult]:
        if self._closed:
            raise RuntimeError(f"{self.planner_label} chunk worker is closed.")
        snapshot = _copy_observation_frame(observation)
        kwargs_snapshot = dict(predict_kwargs or {})
        previous_chunk = kwargs_snapshot.get("prev_chunk_left_over")
        if isinstance(previous_chunk, torch.Tensor):
            kwargs_snapshot["prev_chunk_left_over"] = previous_chunk.detach(
            ).clone()
        return self._executor.submit(self._run_prediction, snapshot,
                                     kwargs_snapshot)

    def reset_episode(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.planner_label} chunk worker is closed.")

        def reset_on_worker() -> None:
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            self.policy.eval()

        self._executor.submit(reset_on_worker).result()

    def warmup(self, observation: dict[str, Any]) -> float:

        if self._closed:
            raise RuntimeError(f"{self.planner_label} chunk worker is closed.")
        snapshot = _copy_observation_frame(observation)

        def warmup_on_worker() -> float:
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_states = (torch.cuda.get_rng_state_all()
                               if self.device.type == "cuda"
                               and torch.cuda.is_available() else None)
            rollout_task_key = getattr(self.policy, "_rollout_task_key", None)
            rollout_index = getattr(self.policy, "_rollout_index_for_task",
                                    None)
            generator_device = self.device if self.device.type == "cuda" else torch.device(
                "cpu")
            warmup_generator = torch.Generator(device=generator_device)
            warmup_generator.manual_seed(0x4D4F4C4D4F)
            try:
                self.policy.eval()
                result = self._run_prediction(
                    snapshot,
                    predict_kwargs={"generator": warmup_generator},
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                return result.inference_s
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                if hasattr(self.policy, "_rollout_task_key"):
                    self.policy._rollout_task_key = rollout_task_key
                if hasattr(self.policy, "_rollout_index_for_task"):
                    self.policy._rollout_index_for_task = rollout_index
                self.policy.reset()
                self.preprocessor.reset()
                self.postprocessor.reset()
                self.policy.eval()

        return self._executor.submit(warmup_on_worker).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        finally:
            if self._rtc_installed:
                self.policy.config.rtc_config = self._previous_rtc_config
                self.policy.init_rtc_processor()
                self._rtc_installed = False


@dataclass(frozen=True)
class _QueuedAction:
    value: PolicyAction
    chunk_index: int
    model_action_index: int
    source_frame_index: int
    source_observation_t: float
    original_value: PolicyAction | None = None
    blend_alpha: float = 1.0
    blend_old_chunk_index: int = -1
    blend_old_action_index: int = -1


@dataclass(frozen=True)
class _PendingChunk:
    future: Future[_DecodedChunkResult]
    chunk_index: int
    source_frame_index: int
    source_observation_t: float
    start_pop_sequence: int
    submitted_t: float


@dataclass(frozen=True)
class _MergeStats:
    inference_s: float
    wait_s: float
    discarded_steps: int


@dataclass(frozen=True)
class _PlannerCloseResult:
    status: str
    error_type: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class _AsyncActionStep:
    action: PolicyAction
    replan_due: bool
    queue_remaining: int
    lookahead_submitted: bool
    lookahead_completed: bool
    lookahead_pending: bool
    lookahead_hold: bool
    lookahead_pending_age_s: float
    lookahead_inference_s: float
    lookahead_wait_s: float
    lookahead_discarded_steps: int
    source_chunk_index: int
    source_action_index: int
    source_frame_index: int
    source_observation_t: float
    blend_alpha: float
    blend_old_chunk_index: int
    blend_old_action_index: int


class _MolmoAsyncActionPlanner:

    def __init__(
        self,
        worker: _MolmoChunkWorker,
        *,
        execution_horizon: int,
        trigger_remaining: int,
        blend_steps: int,
        timeout_s: float,
        max_hold_steps: int = 6,
        planner_label: str = "MolmoAct2",
        control_period_s: float | None = None,
        rtc_execution_horizon: int | None = None,
    ) -> None:
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if not 0 < trigger_remaining < execution_horizon:
            raise ValueError(
                f"{planner_label} async trigger must be between 1 and "
                f"n_action_steps-1; got trigger={trigger_remaining}, horizon={execution_horizon}."
            )
        self.worker = worker
        self.execution_horizon = execution_horizon
        self.trigger_remaining = trigger_remaining
        self.blend_steps = blend_steps
        self.timeout_s = timeout_s
        if max_hold_steps <= 0:
            raise ValueError("max_hold_steps must be positive")
        self.max_hold_steps = max_hold_steps
        self.planner_label = planner_label
        if (control_period_s is None) != (rtc_execution_horizon is None):
            raise ValueError(
                "control_period_s and rtc_execution_horizon must be provided together"
            )
        if control_period_s is not None and control_period_s <= 0:
            raise ValueError("control_period_s must be positive")
        if rtc_execution_horizon is not None and rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if rtc_execution_horizon is not None and blend_steps != 0:
            raise ValueError(
                "RTC prefix guidance cannot be combined with external action blending"
            )
        self.control_period_s = control_period_s
        self.rtc_execution_horizon = rtc_execution_horizon
        self.rtc_enabled = rtc_execution_horizon is not None
        self._inference_durations: deque[float] = deque(maxlen=8)
        self._queue: deque[_QueuedAction] = deque()
        self._pending: _PendingChunk | None = None
        self._pop_sequence = 0
        self._next_chunk_index = 0
        self._last_executed_chunk_index = -1
        self._last_action: _QueuedAction | None = None
        self._consecutive_hold_steps = 0
        self._timeline_pause_started_t: float | None = None

    def start_episode(self) -> None:
        self.worker.reset_episode()

    def _submit(
        self,
        observation: dict[str, Any],
        *,
        source_frame_index: int,
        source_observation_t: float,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"{self.planner_label} look-ahead already has an in-flight chunk."
            )
        chunk_index = self._next_chunk_index
        self._next_chunk_index += 1
        predict_kwargs: dict[str, Any] | None = None
        if self.rtc_enabled and self._queue:
            original_prefix = [item.original_value for item in self._queue]
            if any(item is None for item in original_prefix):
                raise RuntimeError(
                    "Pi0.5 RTC requires the normalized original action prefix for every queued step"
                )
            prefix = torch.cat(
                [item for item in original_prefix if item is not None], dim=0)
            prefix = prefix[:self.rtc_execution_horizon]
            estimated_latency_s = (max(self._inference_durations)
                                   if self._inference_durations else 0.0)
            estimated_delay_steps = int(
                math.ceil(estimated_latency_s / self.control_period_s))
            predict_kwargs = {
                "inference_delay":
                estimated_delay_steps,
                "prev_chunk_left_over":
                prefix,
                "execution_horizon":
                min(self.rtc_execution_horizon, int(prefix.shape[0])),
            }
        future = (self.worker.submit(
            observation, predict_kwargs=predict_kwargs) if predict_kwargs
                  is not None else self.worker.submit(observation))
        self._pending = _PendingChunk(
            future=future,
            chunk_index=chunk_index,
            source_frame_index=source_frame_index,
            source_observation_t=source_observation_t,
            start_pop_sequence=self._pop_sequence,
            submitted_t=time.perf_counter(),
        )

    def _merge_pending(self, *, wait: bool) -> _MergeStats | None:
        pending = self._pending
        if pending is None or (not wait and not pending.future.done()):
            return None

        wait_started = time.perf_counter()
        pending_age_s = wait_started - pending.submitted_t
        if pending_age_s > self.timeout_s:
            self._pending = None

            if pending.future.done():
                pending.future.result()
            raise TimeoutError(
                f"{self.planner_label} look-ahead result exceeded its safety deadline: "
                f"age={pending_age_s:.3f}s, timeout={self.timeout_s:.3f}s.")
        try:
            timeout_s = (max(
                0.0, self.timeout_s -
                (time.perf_counter() - pending.submitted_t)) if wait else 0.0)
            result = pending.future.result(timeout=timeout_s)
        finally:
            self._pending = None
        wait_s = time.perf_counter() - wait_started if wait else 0.0
        self._inference_durations.append(float(result.inference_s))
        consumed_steps = self._pop_sequence - pending.start_pop_sequence
        horizon = int(result.actions.shape[1])
        if horizon != self.execution_horizon:
            raise RuntimeError(
                f"{self.planner_label} async chunk horizon changed unexpectedly: "
                f"expected={self.execution_horizon}, got={horizon}.")
        if consumed_steps >= horizon:
            logging.warning(
                f"Discarding stale {self.planner_label} look-ahead chunk: "
                f"consumed={consumed_steps}, horizon={horizon}.")
            return _MergeStats(
                inference_s=result.inference_s,
                wait_s=wait_s,
                discarded_steps=horizon,
            )

        aligned_actions = result.actions[:, consumed_steps:].unbind(dim=1)
        if self.rtc_enabled:
            if result.original_actions is None:
                raise RuntimeError(
                    "Pi0.5 RTC worker did not return normalized original actions"
                )
            if result.original_actions.shape != result.actions.shape:
                raise RuntimeError(
                    "Pi0.5 RTC original/decoded chunk shape mismatch: "
                    f"original={tuple(result.original_actions.shape)}, "
                    f"decoded={tuple(result.actions.shape)}")
            aligned_original_actions: list[PolicyAction | None] = list(
                result.original_actions[:, consumed_steps:].unbind(dim=1))
        else:
            aligned_original_actions = [None] * len(aligned_actions)
        old_tail = list(self._queue)
        new_queue = [
            _QueuedAction(
                value=action,
                chunk_index=pending.chunk_index,
                model_action_index=consumed_steps + offset,
                source_frame_index=pending.source_frame_index,
                source_observation_t=pending.source_observation_t,
                original_value=aligned_original_actions[offset],
            ) for offset, action in enumerate(aligned_actions)
        ]

        blend_count = min(self.blend_steps, len(old_tail), len(new_queue))
        for idx in range(blend_count):
            alpha = float(idx + 1) / float(blend_count + 1)
            blended_value = torch.lerp(old_tail[idx].value,
                                       new_queue[idx].value, alpha)
            new_queue[idx] = _QueuedAction(
                value=blended_value,
                chunk_index=new_queue[idx].chunk_index,
                model_action_index=new_queue[idx].model_action_index,
                source_frame_index=new_queue[idx].source_frame_index,
                source_observation_t=new_queue[idx].source_observation_t,
                original_value=new_queue[idx].original_value,
                blend_alpha=alpha,
                blend_old_chunk_index=old_tail[idx].chunk_index,
                blend_old_action_index=old_tail[idx].model_action_index,
            )

        self._queue.clear()
        self._queue.extend(new_queue)
        return _MergeStats(
            inference_s=result.inference_s,
            wait_s=wait_s,
            discarded_steps=consumed_steps,
        )

    def select_action(
        self,
        observation: dict[str, Any],
        *,
        frame_index: int,
        observation_t: float,
        consume: bool = True,
    ) -> _AsyncActionStep:
        now = time.perf_counter()
        if not consume:
            if self._timeline_pause_started_t is None:
                self._timeline_pause_started_t = now
            else:
                pause_delta_s = now - self._timeline_pause_started_t
                if self._pending is not None:
                    self._pending = replace(
                        self._pending,
                        submitted_t=self._pending.submitted_t + pause_delta_s,
                    )
                self._timeline_pause_started_t = now

            merge_stats = self._merge_pending(wait=False)
            if self._queue:
                queued_action = self._queue[0]
            elif self._last_action is not None:
                queued_action = self._last_action
            else:
                raise RuntimeError(
                    f"{self.planner_label} cannot pause before producing its first action."
                )
            pending_age_s = (time.perf_counter() - self._pending.submitted_t
                             if self._pending is not None else math.nan)
            return _AsyncActionStep(
                action=queued_action.value,
                replan_due=False,
                queue_remaining=len(self._queue),
                lookahead_submitted=False,
                lookahead_completed=merge_stats is not None,
                lookahead_pending=self._pending is not None,
                lookahead_hold=False,
                lookahead_pending_age_s=pending_age_s,
                lookahead_inference_s=(merge_stats.inference_s if merge_stats
                                       is not None else math.nan),
                lookahead_wait_s=(merge_stats.wait_s
                                  if merge_stats is not None else 0.0),
                lookahead_discarded_steps=(merge_stats.discarded_steps
                                           if merge_stats is not None else 0),
                source_chunk_index=queued_action.chunk_index,
                source_action_index=queued_action.model_action_index,
                source_frame_index=queued_action.source_frame_index,
                source_observation_t=queued_action.source_observation_t,
                blend_alpha=queued_action.blend_alpha,
                blend_old_chunk_index=queued_action.blend_old_chunk_index,
                blend_old_action_index=queued_action.blend_old_action_index,
            )

        if self._timeline_pause_started_t is not None:
            pause_s = now - self._timeline_pause_started_t
            if self._pending is not None:
                self._pending = replace(
                    self._pending,
                    submitted_t=self._pending.submitted_t + pause_s,
                )
            self._timeline_pause_started_t = None

        submitted = False
        completed = False
        inference_s = math.nan
        wait_s = 0.0
        discarded_steps = 0

        merge_stats = self._merge_pending(wait=False)
        if merge_stats is not None:
            completed = True
            inference_s = merge_stats.inference_s
            wait_s += merge_stats.wait_s
            discarded_steps = merge_stats.discarded_steps

        if not self._queue and self._last_action is None:
            if self._pending is None:
                self._submit(
                    observation,
                    source_frame_index=frame_index,
                    source_observation_t=observation_t,
                )
                submitted = True
            blocking_stats = self._merge_pending(wait=True)
            if blocking_stats is not None:
                completed = True
                inference_s = blocking_stats.inference_s
                wait_s += blocking_stats.wait_s
                discarded_steps = blocking_stats.discarded_steps

        if not self._queue:
            if self._last_action is None:
                raise RuntimeError(
                    f"{self.planner_label} async planner could not produce an initial action chunk."
                )
            if self._consecutive_hold_steps >= self.max_hold_steps:
                pending_age_s = (time.perf_counter() -
                                 self._pending.submitted_t
                                 if self._pending is not None else math.nan)
                raise TimeoutError(
                    f"{self.planner_label} look-ahead exceeded the consecutive HOLD budget: "
                    f"hold_steps={self._consecutive_hold_steps}, "
                    f"budget={self.max_hold_steps}, pending_age={pending_age_s:.3f}s."
                )
            if self._pending is None:
                self._submit(
                    observation,
                    source_frame_index=frame_index,
                    source_observation_t=observation_t,
                )
                submitted = True
            pending_age_s = time.perf_counter() - self._pending.submitted_t
            if pending_age_s > self.timeout_s:
                raise TimeoutError(
                    f"{self.planner_label} look-ahead inference exceeded its safety timeout: "
                    f"age={pending_age_s:.3f}s, timeout={self.timeout_s:.3f}s."
                )

            held_action = self._last_action
            self._pop_sequence += 1
            self._consecutive_hold_steps += 1
            return _AsyncActionStep(
                action=held_action.value,
                replan_due=False,
                queue_remaining=0,
                lookahead_submitted=submitted,
                lookahead_completed=completed,
                lookahead_pending=True,
                lookahead_hold=True,
                lookahead_pending_age_s=pending_age_s,
                lookahead_inference_s=inference_s,
                lookahead_wait_s=wait_s,
                lookahead_discarded_steps=discarded_steps,
                source_chunk_index=held_action.chunk_index,
                source_action_index=held_action.model_action_index,
                source_frame_index=held_action.source_frame_index,
                source_observation_t=held_action.source_observation_t,
                blend_alpha=held_action.blend_alpha,
                blend_old_chunk_index=held_action.blend_old_chunk_index,
                blend_old_action_index=held_action.blend_old_action_index,
            )

        if self._pending is None and len(
                self._queue) <= self.trigger_remaining:
            self._submit(
                observation,
                source_frame_index=frame_index,
                source_observation_t=observation_t,
            )
            submitted = True

        queued_action = self._queue.popleft()
        replan_due = queued_action.chunk_index != self._last_executed_chunk_index
        self._last_executed_chunk_index = queued_action.chunk_index
        self._last_action = queued_action
        self._pop_sequence += 1
        self._consecutive_hold_steps = 0
        pending_age_s = (time.perf_counter() - self._pending.submitted_t
                         if self._pending is not None else math.nan)
        return _AsyncActionStep(
            action=queued_action.value,
            replan_due=replan_due,
            queue_remaining=len(self._queue),
            lookahead_submitted=submitted,
            lookahead_completed=completed,
            lookahead_pending=self._pending is not None,
            lookahead_hold=False,
            lookahead_pending_age_s=pending_age_s,
            lookahead_inference_s=inference_s,
            lookahead_wait_s=wait_s,
            lookahead_discarded_steps=discarded_steps,
            source_chunk_index=queued_action.chunk_index,
            source_action_index=queued_action.model_action_index,
            source_frame_index=queued_action.source_frame_index,
            source_observation_t=queued_action.source_observation_t,
            blend_alpha=queued_action.blend_alpha,
            blend_old_chunk_index=queued_action.blend_old_chunk_index,
            blend_old_action_index=queued_action.blend_old_action_index,
        )

    def preview_action_dimension(
        self,
        action_index: int,
        *,
        max_steps: int | None = None,
    ) -> list[float]:

        if action_index < 0:
            raise ValueError("action_index cannot be negative")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")

        values: list[float] = []
        for queued_action in self._queue:
            vector = queued_action.value.detach().reshape(-1)
            if action_index >= int(vector.numel()):
                raise IndexError(
                    f"Action index {action_index} is outside queued action dimension "
                    f"{int(vector.numel())}.")
            value = float(vector[action_index].item())
            if not math.isfinite(value):
                raise ValueError(
                    "Queued action preview contains NaN or infinity")
            values.append(value)
            if max_steps is not None and len(values) >= max_steps:
                break
        return values

    def close_episode(self) -> _PlannerCloseResult:
        pending = self._pending
        self._pending = None
        try:
            if pending is not None:
                remaining_s = max(
                    0.0, self.timeout_s -
                    (time.perf_counter() - pending.submitted_t))
                pending.future.result(timeout=remaining_s)
            return _PlannerCloseResult(
                status="completed" if pending is not None else "idle")
        except TimeoutError as exc:
            return _PlannerCloseResult(
                status="timeout",
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                error_message=str(exc),
            )
        except BaseException as exc:
            return _PlannerCloseResult(
                status="error",
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                error_message=str(exc),
            )
        finally:
            self._queue.clear()
            self._last_action = None
            self._consecutive_hold_steps = 0
            self._timeline_pause_started_t = None
            self._inference_durations.clear()


def _close_active_molmo_planner(events: dict[str, Any]) -> _PlannerCloseResult:

    planner = events.pop("_active_molmo_planner", None)
    if planner is None:
        return _PlannerCloseResult(status="idle")

    drain_started = time.perf_counter()
    try:
        result = planner.close_episode()
    except BaseException as exc:
        result = _PlannerCloseResult(
            status="error",
            error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            error_message=str(exc),
        )
    finally:
        events["last_molmo_planner_drain_s"] = time.perf_counter(
        ) - drain_started

    events["last_molmo_planner_close"] = {
        "status": result.status,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    if result.status in {"timeout", "error"}:
        events["stop_inference"] = True
        logging.error(
            f"{getattr(planner, 'planner_label', 'Async')} planner cleanup failed after the "
            "episode; preserving the episode verdict "
            f"and stopping before another episode ({result.status}: {result.error_message})."
        )
    return result


def _close_owned_pi05_chunk_worker(events: dict[str, Any]) -> None:

    worker = events.pop("_owned_pi05_chunk_worker", None)
    if worker is not None:
        worker.close()


def _predict_full_chunk_and_pop(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    use_amp: bool,
    task: str,
    robot_type: str,
    action_queue: deque[PolicyAction],
    consume: bool = True,
    before_enqueue=None,
) -> tuple[PolicyAction, bool, int]:

    replan_due = len(action_queue) == 0
    if replan_due:
        absolute_chunk = _predict_decoded_action_chunk(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        if before_enqueue is not None:
            replacement = before_enqueue(absolute_chunk)
            if replacement is not None:
                expected_shape = tuple(absolute_chunk[:, 0].shape)
                if (not isinstance(replacement, torch.Tensor)
                        or tuple(replacement.shape) != expected_shape
                        or not bool(torch.isfinite(replacement).all())):
                    raise ValueError(
                        "before_enqueue must return None or a finite hold action with shape "
                        f"{expected_shape}; got {type(replacement).__name__} "
                        f"{getattr(replacement, 'shape', None)}")
                return replacement, True, 0
        action_queue.extend(absolute_chunk.unbind(dim=1))

    if consume:
        return action_queue.popleft(), replan_due, len(action_queue)
    return action_queue[0], replan_due, len(action_queue)


def _preview_action_queue_dimension(
    action_queue: deque[PolicyAction],
    action_index: int,
) -> list[float]:

    if action_index < 0:
        raise ValueError("action_index cannot be negative")
    values: list[float] = []
    for action in action_queue:
        vector = action.detach().reshape(-1)
        if action_index >= int(vector.numel()):
            raise IndexError(
                f"Action index {action_index} is outside queued action dimension "
                f"{int(vector.numel())}.")
        value = float(vector[action_index].item())
        if not math.isfinite(value):
            raise ValueError("Queued action preview contains NaN or infinity")
        values.append(value)
    return values


def _warmup_molmo_chunk_worker(
    *,
    worker: _MolmoChunkWorker,
    robot: Robot,
    robot_observation_processor: RobotProcessorPipeline,
    dataset_features: dict[str, Any],
) -> float:

    observation = robot.get_observation()
    observation_processed = robot_observation_processor(observation)
    observation_frame = build_dataset_frame(dataset_features,
                                            observation_processed,
                                            prefix=OBS_STR)
    try:
        return worker.warmup(observation_frame)
    finally:
        robot_observation_processor.reset()


@safe_stop_image_writer
def inference_loop(
    robot: Robot,
    events: dict,
    fps: int,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset_features: dict,
    cfg: InferenceConfig,
    single_task: str,
    control_time_s: float,
    cam_display: CamDisplayProcess | None = None,
    video_writer: cv2.VideoWriter | None = None,
    dataset: LeRobotDataset | None = None,
    knob_teleop=None,
    bbox_runtime=None,
    bbox_draw_overlay=None,
    bbox_threshold: float | None = None,
    action_trace: dict[str, list[Any]] | None = None,
    molmo_chunk_worker: _MolmoChunkWorker | None = None,
    stage_monitor=None,
    bbox_shared_feature_bridge=None,
    action_interlock=None,
    action_chunk_guard=None,
) -> bool | None:

    policy_type = getattr(policy.config, "type", None)
    use_molmo_async_replan = (policy_type == "molmoact2"
                              and cfg.molmo_async_replan)
    shared_detector_active = bbox_shared_feature_bridge is not None
    pi05_detector_stage_active = bbox_runtime is not None
    use_pi05_shared_full_chunk = (policy_type == "pi05"
                                  and cfg.pi05_shared_full_chunk
                                  and shared_detector_active)
    use_pi05_async_replan = (policy_type == "pi05" and cfg.pi05_async_replan
                             and not pi05_detector_stage_active)
    if policy_type == "pi05" and cfg.pi05_async_replan and shared_detector_active:
        logging.info(
            "Pi0.5 asynchronous look-ahead is paused for this shared-DETR stage."
        )
    use_full_chunk_postprocessing = (
        _requires_full_chunk_postprocessing(policy) or use_pi05_async_replan
        or use_pi05_shared_full_chunk)
    use_shared_feature_full_chunk = (shared_detector_active
                                     and use_full_chunk_postprocessing)
    if action_chunk_guard is not None and not use_full_chunk_postprocessing:
        raise ValueError(
            "action_chunk_guard requires local full-chunk postprocessing")
    async_planner: _MolmoAsyncActionPlanner | None = None
    if use_molmo_async_replan:
        if molmo_chunk_worker is None:
            raise RuntimeError(
                "Molmo async replanning requires a Molmo chunk worker.")
        async_planner = _MolmoAsyncActionPlanner(
            molmo_chunk_worker,
            execution_horizon=int(policy.config.n_action_steps),
            trigger_remaining=cfg.molmo_async_replan_trigger_remaining,
            blend_steps=cfg.molmo_async_replan_blend_steps,
            timeout_s=cfg.molmo_async_replan_timeout_s,
            max_hold_steps=cfg.molmo_async_replan_max_hold_steps,
            planner_label="MolmoAct2",
        )
        async_planner.start_episode()
        events["_active_molmo_planner"] = async_planner
        logging.info(
            "MolmoAct2 asynchronous look-ahead enabled: "
            f"horizon={policy.config.n_action_steps}, "
            f"trigger_remaining={cfg.molmo_async_replan_trigger_remaining}, "
            f"blend_steps={cfg.molmo_async_replan_blend_steps}, "
            f"timeout_s={cfg.molmo_async_replan_timeout_s}, "
            f"max_hold_steps={cfg.molmo_async_replan_max_hold_steps}.")
    elif use_pi05_async_replan:
        pi05_rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=cfg.pi05_rtc_prefix_attention_schedule,
            max_guidance_weight=cfg.pi05_rtc_max_guidance_weight,
            execution_horizon=cfg.pi05_rtc_execution_horizon,
        )
        pi05_chunk_worker = _MolmoChunkWorker(
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
            expected_action_dim=len(dataset_features[ACTION]["names"]),
            planner_label="Pi0.5",
            rtc_config=pi05_rtc_config,
        )
        events["_owned_pi05_chunk_worker"] = pi05_chunk_worker
        async_planner = _MolmoAsyncActionPlanner(
            pi05_chunk_worker,
            execution_horizon=int(policy.config.n_action_steps),
            trigger_remaining=cfg.pi05_async_replan_trigger_remaining,
            blend_steps=cfg.pi05_async_replan_blend_steps,
            timeout_s=cfg.pi05_async_replan_timeout_s,
            max_hold_steps=cfg.pi05_async_replan_max_hold_steps,
            planner_label="Pi0.5",
            control_period_s=1.0 / float(fps),
            rtc_execution_horizon=cfg.pi05_rtc_execution_horizon,
        )
        async_planner.start_episode()
        events["_active_molmo_planner"] = async_planner
        logging.info(
            "Pi0.5 native RTC asynchronous look-ahead enabled: "
            f"horizon={policy.config.n_action_steps}, "
            f"trigger_remaining={cfg.pi05_async_replan_trigger_remaining}, "
            f"rtc_execution_horizon={cfg.pi05_rtc_execution_horizon}, "
            f"rtc_max_guidance_weight={cfg.pi05_rtc_max_guidance_weight}, "
            f"rtc_schedule={cfg.pi05_rtc_prefix_attention_schedule.value}, "
            f"timeout_s={cfg.pi05_async_replan_timeout_s}, "
            f"max_hold_steps={cfg.pi05_async_replan_max_hold_steps}.")
    else:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    if use_pi05_async_replan:
        events["stage_scheduler_mode"] = "pi05_rtc_async"
    elif use_shared_feature_full_chunk:
        events["stage_scheduler_mode"] = (
            "pi05_shared_full_chunk" if policy_type == "pi05" else
            f"{policy_type}_shared_feature_full_chunk")
        logging.info(
            "%s shared-detector full-chunk queue enabled: preprocess, infer, "
            "capture detector features, and postprocess only when the local queue is empty.",
            policy_type,
        )
    elif use_full_chunk_postprocessing:
        events["stage_scheduler_mode"] = "local_full_chunk"
    else:
        events["stage_scheduler_mode"] = "policy_internal_queue"

    decoded_action_queue: deque[PolicyAction] = deque()
    if use_full_chunk_postprocessing:
        if policy_type == "molmoact2":
            if not use_molmo_async_replan:
                logging.info(
                    "MolmoAct2 synchronous local action-chunk queue enabled: preprocess, predict, "
                    "and postprocess only when the queue is empty.")
        elif policy_type == "groot":
            logging.info(
                "GR00T relative-action rollout enabled: predict and postprocess the full action chunk, "
                "then execute decoded absolute actions from a local queue.")

    events["exit_early"] = False
    events["episode_success"] = None
    events["last_molmo_planner_drain_s"] = 0.0
    events["last_molmo_planner_close"] = {
        "status": "idle",
        "error_type": "",
        "error_message": "",
    }

    timestamp = 0.0
    start_t = time.perf_counter()
    next_bbox_submit_t = start_t
    frame_id = 0
    previous_loop_start: float | None = None
    sync_chunk_index = -1
    sync_action_index = -1
    sync_source_frame_index = -1
    sync_source_observation_t = math.nan

    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        loop_start_s = loop_start - start_t
        frame_interval_s = (math.nan if previous_loop_start is None else
                            loop_start - previous_loop_start)
        previous_loop_start = loop_start

        if events["exit_early"]:
            break

        observation_start = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_captured_t = time.perf_counter()
        observation_s = time.perf_counter() - observation_start

        video_start = time.perf_counter()
        if cam_display is not None or video_writer is not None:
            cam_order = [
                "cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"
            ]
            frames = {}
            for cam_name in cam_order:
                if cam_name in obs:
                    frame_rgb = obs[cam_name]
                    if (cam_name == "cam_low" and bbox_runtime is not None
                            and bbox_draw_overlay is not None
                            and bbox_threshold is not None):
                        try:
                            synchronized = bbox_runtime.latest_synchronized_frame(
                            )
                            frame_rgb = bbox_draw_overlay(
                                synchronized.frame,
                                synchronized.prediction,
                                threshold=bbox_threshold,
                                pending=synchronized.pending,
                                color_order="rgb",
                            )
                        except Exception as exc:
                            if not events.get("bbox_overlay_error_logged",
                                              False):
                                logging.warning(
                                    f"BBox overlay failed; raw cam_low will be saved: {exc}"
                                )
                                events["bbox_overlay_error_logged"] = True
                    frames[cam_name] = cv2.resize(
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (640, 480))
                else:
                    frames[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

            grid_top = np.hstack([frames["cam_high"], frames["cam_low"]])
            grid_bot = np.hstack(
                [frames["cam_left_wrist"], frames["cam_right_wrist"]])
            grid = np.vstack([grid_top, grid_bot])

            if cam_display is not None:
                cam_display.send_frame(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
            if video_writer is not None:
                video_writer.write(grid)
        video_s = time.perf_counter() - video_start

        observation_frame = build_dataset_frame(dataset_features,
                                                obs_processed,
                                                prefix=OBS_STR)
        policy_start = time.perf_counter()
        lookahead_submitted = False
        lookahead_completed = False
        lookahead_pending = False
        lookahead_hold = False
        lookahead_pending_age_s = math.nan
        lookahead_inference_s = math.nan
        lookahead_wait_s = 0.0
        lookahead_discarded_steps = 0
        action_source_chunk_index = -1
        action_source_action_index = -1
        action_source_frame_index = -1
        action_source_observation_t = math.nan
        action_blend_alpha = 1.0
        action_blend_old_chunk_index = -1
        action_blend_old_action_index = -1
        pause_policy_timeline = bool(
            action_interlock is not None
            and getattr(action_interlock, "pause_policy_timeline", False))
        if async_planner is not None:
            async_step = async_planner.select_action(
                observation_frame,
                frame_index=frame_id,
                observation_t=observation_captured_t,
                consume=not pause_policy_timeline,
            )
            action_values = async_step.action
            action_replan_due = async_step.replan_due
            action_queue_remaining = async_step.queue_remaining
            lookahead_submitted = async_step.lookahead_submitted
            lookahead_completed = async_step.lookahead_completed
            lookahead_pending = async_step.lookahead_pending
            lookahead_hold = async_step.lookahead_hold
            lookahead_pending_age_s = async_step.lookahead_pending_age_s
            lookahead_inference_s = async_step.lookahead_inference_s
            lookahead_wait_s = async_step.lookahead_wait_s
            lookahead_discarded_steps = async_step.lookahead_discarded_steps
            action_source_chunk_index = async_step.source_chunk_index
            action_source_action_index = async_step.source_action_index
            action_source_frame_index = async_step.source_frame_index
            action_source_observation_t = async_step.source_observation_t
            action_blend_alpha = async_step.blend_alpha
            action_blend_old_chunk_index = async_step.blend_old_chunk_index
            action_blend_old_action_index = async_step.blend_old_action_index
            if lookahead_hold:
                now = time.perf_counter()
                if now - float(events.get("last_molmo_hold_log_t", 0.0)) > 1.0:
                    logging.warning(
                        f"{async_planner.planner_label} look-ahead queue underrun: "
                        "holding the last position target "
                        f"while inference is pending ({lookahead_pending_age_s:.3f}s)."
                    )
                    events["last_molmo_hold_log_t"] = now
        elif use_full_chunk_postprocessing:
            shared_capture_state = None
            shared_capture = contextlib.nullcontext(None)
            if (use_shared_feature_full_chunk and bbox_runtime is not None
                    and len(decoded_action_queue) == 0):
                shared_capture = bbox_shared_feature_bridge.capture_action(
                    observation=observation_frame,
                    frame_id=frame_id,
                    timestamp=time.time(),
                    runtime=bbox_runtime,
                )

            def review_before_enqueue(absolute_chunk: torch.Tensor):
                if action_chunk_guard is None:
                    return None
                first_action_values = absolute_chunk[:, 0]
                first_action = make_robot_action(first_action_values,
                                                 dataset_features)
                guard_result = action_chunk_guard({
                    "raw_observation":
                    obs,
                    "observation_frame":
                    observation_frame,
                    "action":
                    first_action,
                    "frame_id":
                    frame_id,
                    "elapsed_s":
                    time.perf_counter() - start_t,
                    "monotonic_s":
                    time.perf_counter(),
                })
                if guard_result is None:
                    return None
                if not isinstance(guard_result, dict):
                    raise TypeError(
                        "action_chunk_guard must return None or a dict")
                replacement_action = guard_result.get("action")
                if replacement_action is None:
                    raise ValueError(
                        "action_chunk_guard rejection requires a hold action")
                action_names = list(dataset_features[ACTION]["names"])
                missing = [
                    name for name in action_names
                    if name not in replacement_action
                ]
                if missing:
                    raise KeyError(
                        f"action_chunk_guard hold action is missing fields: {missing}"
                    )
                hold_values = first_action_values.detach().clone()
                for index, name in enumerate(action_names):
                    hold_values[..., index] = float(replacement_action[name])
                if guard_result.get("stop", False):
                    events["stage_stop_reason"] = str(
                        guard_result.get("reason", "action_chunk_guard"))
                    events["episode_success"] = guard_result.get(
                        "success", False)
                    events["exit_early"] = True
                return hold_values

            with shared_capture as shared_capture_state:
                action_values, action_replan_due, action_queue_remaining = (
                    _predict_full_chunk_and_pop(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                        action_queue=decoded_action_queue,
                        consume=not pause_policy_timeline,
                        before_enqueue=review_before_enqueue,
                    ))
            if shared_capture_state is not None and shared_capture_state.submitted:
                next_bbox_submit_t = time.perf_counter(
                ) + 1.0 / cfg.bbox_inference_hz
            if action_replan_due:
                sync_chunk_index += 1
                sync_action_index = 0
                sync_source_frame_index = frame_id
                sync_source_observation_t = observation_captured_t
            elif not pause_policy_timeline:
                sync_action_index += 1
            action_source_chunk_index = sync_chunk_index
            action_source_action_index = sync_action_index
            action_source_frame_index = sync_source_frame_index
            action_source_observation_t = sync_source_observation_t
        else:
            if pause_policy_timeline:
                raise RuntimeError(
                    "Action timeline pause requires an async planner or local full-chunk queue."
                )
            policy_action_queue = getattr(policy, "_action_queue", None)
            action_replan_due = policy_action_queue is not None and len(
                policy_action_queue) == 0
            shared_capture = contextlib.nullcontext(None)
            if (bbox_shared_feature_bridge is not None
                    and bbox_runtime is not None and action_replan_due):
                shared_capture = bbox_shared_feature_bridge.capture_action(
                    observation=observation_frame,
                    frame_id=frame_id,
                    timestamp=time.time(),
                    runtime=bbox_runtime,
                )
            with shared_capture as shared_capture_state:
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
            if shared_capture_state is not None and shared_capture_state.submitted:
                next_bbox_submit_t = time.perf_counter(
                ) + 1.0 / cfg.bbox_inference_hz
            policy_action_queue = getattr(policy, "_action_queue", None)
            action_queue_remaining = (len(policy_action_queue)
                                      if policy_action_queue is not None else
                                      None)
        policy_s = time.perf_counter() - policy_start

        act_processed: RobotAction = make_robot_action(action_values,
                                                       dataset_features)
        act_processed = _apply_action_safety_limits(act_processed, cfg, events)
        if action_interlock is not None:
            future_gripper_commands: list[float] = []
            if async_planner is not None:
                action_names = dataset_features[ACTION].get("names")
                if action_names and "right_gripper.pos" in action_names:
                    future_gripper_commands = async_planner.preview_action_dimension(
                        action_names.index("right_gripper.pos"))
            elif use_full_chunk_postprocessing:
                action_names = dataset_features[ACTION].get("names")
                if action_names and "right_gripper.pos" in action_names:
                    future_gripper_commands = _preview_action_queue_dimension(
                        decoded_action_queue,
                        action_names.index("right_gripper.pos"),
                    )
            interlock_result = action_interlock({
                "raw_observation":
                obs,
                "action":
                act_processed,
                "frame_id":
                frame_id,
                "elapsed_s":
                time.perf_counter() - start_t,
                "monotonic_s":
                time.perf_counter(),
                "future_gripper_commands":
                future_gripper_commands,
            })
            if interlock_result is not None:
                if not isinstance(interlock_result, dict):
                    raise TypeError(
                        "action_interlock must return None or a dict")
                replacement_action = interlock_result.get("action")
                if replacement_action is not None:
                    act_processed = replacement_action
                if interlock_result.get("stop", False):
                    events["stage_stop_reason"] = str(
                        interlock_result.get("reason", "action_interlock"))
                    events["episode_success"] = interlock_result.get(
                        "success", False)
                    events["exit_early"] = True
        robot_action_to_send = robot_action_processor((act_processed, obs))

        trace_observation_state = None
        trace_predicted_action = None
        trace_commanded_action = None
        if action_trace is not None:
            trace_observation_state = np.asarray(observation_frame[OBS_STATE],
                                                 dtype=np.float32).copy()
            trace_predicted_action = (action_values.detach().to(
                device="cpu", dtype=torch.float32).reshape(-1).numpy().copy())

        action_send_start = time.perf_counter()
        sent_robot_action = robot.send_action(robot_action_to_send)

        if knob_teleop is not None and knob_teleop.is_connected:
            knob_action = {
                k: act_processed[k]
                for k in ("knob_ud.pos", "knob_lr.pos") if k in act_processed
            }
            if knob_action:
                knob_teleop.send_action(knob_action)

        action_send_s = time.perf_counter() - action_send_start
        command_sent_s = time.perf_counter() - start_t
        if action_trace is not None:
            actual_command = dict(act_processed)
            if sent_robot_action is not None:
                actual_command.update(sent_robot_action)
            action_names = dataset_features[ACTION]["names"]
            trace_commanded_action = np.asarray(
                [actual_command[name] for name in action_names],
                dtype=np.float32)
        action_feedback_age_s = (time.perf_counter() -
                                 action_source_observation_t
                                 if math.isfinite(action_source_observation_t)
                                 else math.nan)

        if bbox_runtime is not None:
            try:
                now = time.perf_counter()
                queue_remaining = action_queue_remaining
                queue_has_margin = (queue_remaining is None or queue_remaining
                                    >= cfg.bbox_min_action_queue_remaining)
                if (not action_replan_due and queue_has_margin
                        and not bbox_runtime.bbox_pending
                        and now >= next_bbox_submit_t):
                    bbox_runtime.submit_bbox(
                        observation_frame,
                        frame_id=frame_id,
                        timestamp=time.time(),
                    )
                    next_bbox_submit_t = now + 1.0 / cfg.bbox_inference_hz
            except Exception as exc:
                if not events.get("bbox_error_logged", False):
                    logging.warning(
                        f"BBox sidecar submit failed; overlay disabled for this episode: {exc}"
                    )
                    events["bbox_error_logged"] = True
                bbox_runtime = None

        if stage_monitor is not None and not events["exit_early"]:
            bbox_snapshot = None
            if bbox_runtime is not None:
                try:
                    bbox_snapshot = bbox_runtime.latest_synchronized_frame()
                except RuntimeError:

                    bbox_snapshot = None
            monitor_result = stage_monitor({
                "raw_observation":
                obs,
                "observation_frame":
                observation_frame,
                "action":
                act_processed,
                "frame_id":
                frame_id,
                "elapsed_s":
                time.perf_counter() - start_t,
                "bbox":
                bbox_snapshot,
                "action_interlock_active":
                bool(action_interlock is not None and getattr(
                    action_interlock, "pause_policy_timeline", False)),
            })
            if monitor_result is not None:
                if not isinstance(monitor_result, dict):
                    raise TypeError("stage_monitor must return None or a dict")
                if monitor_result.get("stop", False):
                    events["stage_stop_reason"] = str(
                        monitor_result.get("reason", "stage_monitor"))
                    events["episode_success"] = monitor_result.get(
                        "success", True)
                    events["exit_early"] = True

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features,
                                               act_processed,
                                               prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        dt_s = time.perf_counter() - loop_start
        sleep_s = 1 / fps - dt_s
        if sleep_s < 0:
            logging.warning(
                f"Inference loop running slower ({1/dt_s:.1f} Hz) than target {fps} Hz"
            )
        sleep_requested_s = max(sleep_s, 0.0)
        precise_sleep(sleep_requested_s)
        loop_total_s = time.perf_counter() - loop_start

        if action_trace is not None:
            action_trace["frame_index"].append(frame_id)
            action_trace["loop_start_s"].append(loop_start_s)
            action_trace["frame_interval_s"].append(frame_interval_s)
            action_trace["observation_s"].append(observation_s)
            action_trace["video_s"].append(video_s)
            action_trace["policy_s"].append(policy_s)
            action_trace["action_send_s"].append(action_send_s)
            action_trace["command_sent_s"].append(command_sent_s)
            action_trace["loop_compute_s"].append(dt_s)
            action_trace["sleep_requested_s"].append(sleep_requested_s)
            action_trace["loop_total_s"].append(loop_total_s)
            action_trace["replan_due"].append(action_replan_due)
            action_trace["queue_remaining"].append(
                -1 if action_queue_remaining is
                None else action_queue_remaining)
            action_trace["lookahead_submitted"].append(lookahead_submitted)
            action_trace["lookahead_completed"].append(lookahead_completed)
            action_trace["lookahead_pending"].append(lookahead_pending)
            action_trace["lookahead_hold"].append(lookahead_hold)
            action_trace["lookahead_pending_age_s"].append(
                lookahead_pending_age_s)
            action_trace["lookahead_inference_s"].append(lookahead_inference_s)
            action_trace["lookahead_wait_s"].append(lookahead_wait_s)
            action_trace["lookahead_discarded_steps"].append(
                lookahead_discarded_steps)
            action_trace["source_chunk_index"].append(
                action_source_chunk_index)
            action_trace["source_action_index"].append(
                action_source_action_index)
            action_trace["source_frame_index"].append(
                action_source_frame_index)
            action_trace["blend_alpha"].append(action_blend_alpha)
            action_trace["blend_old_chunk_index"].append(
                action_blend_old_chunk_index)
            action_trace["blend_old_action_index"].append(
                action_blend_old_action_index)
            action_trace["action_feedback_age_s"].append(action_feedback_age_s)
            action_trace["observation_state"].append(trace_observation_state)
            action_trace["predicted_action"].append(trace_predicted_action)
            action_trace["commanded_action"].append(trace_commanded_action)

        timestamp = time.perf_counter() - start_t
        frame_id += 1

    episode_result = events.get("episode_success", None)
    if async_planner is not None:
        _close_active_molmo_planner(events)
    _close_owned_pi05_chunk_worker(events)
    return episode_result


def _task_strings(value) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_task_strings(item))
        return out
    return [str(value)]


def _normalize_task(value: str) -> str:
    return " ".join(str(value).strip().split())


POLYP_TASK_PROMPT = "Move the endoscope to center the polyp."
POLYP_INITIAL_POSE_CLUSTERS = (
    (111, 123),
    (302, 314),
    (315, 330),
    (331, 339),
    (340, 347),
    (348, 358),
    (359, 367),
    (368, 379),
    (380, 388),
)
INSPECT_TASK_PROMPT = "Move the endoscope around to inspect the stomach."
INSPECT_INITIAL_POSE_CLUSTERS = (
    (72, 110),
    (439, 499),
)
TASK_INITIAL_POSE_CLUSTERS = {
    _normalize_task(POLYP_TASK_PROMPT): POLYP_INITIAL_POSE_CLUSTERS,
    _normalize_task(INSPECT_TASK_PROMPT): INSPECT_INITIAL_POSE_CLUSTERS,
}
CLUSTER_MEAN_STRATEGIES = {"polyp_cluster_mean", "task_cluster_mean"}


def _episode_range(start: int, end: int) -> list[int]:
    return list(range(int(start), int(end) + 1))


def _format_episode_range(episodes: list[int]) -> str:
    if not episodes:
        return "empty"
    if episodes == list(range(episodes[0], episodes[-1] + 1)):
        return f"{episodes[0]}-{episodes[-1]}"
    return ",".join(str(ep) for ep in episodes)


def _find_task_episode_candidates(ds_root: Path, task: str) -> list[int]:
    wanted = _normalize_task(task)
    matching_episodes = []
    for episodes_path in sorted(ds_root.glob("meta/episodes/**/*.parquet")):
        try:
            episodes = pd.read_parquet(episodes_path,
                                       columns=["episode_index", "tasks"])
        except Exception:
            continue
        for _, row in episodes.iterrows():
            tasks = {
                _normalize_task(item)
                for item in _task_strings(row["tasks"])
            }
            if wanted in tasks:
                matching_episodes.append(int(row["episode_index"]))
    return sorted(set(matching_episodes))


def _available_dataset_tasks(ds_root: Path) -> list[str]:
    tasks = set()
    for episodes_path in sorted(ds_root.glob("meta/episodes/**/*.parquet")):
        try:
            episodes = pd.read_parquet(episodes_path, columns=["tasks"])
        except Exception:
            continue
        for value in episodes["tasks"]:
            for item in _task_strings(value):
                tasks.add(_normalize_task(item))
    return sorted(tasks)


def _find_initial_pose_episodes(ds_root: Path, single_task: str,
                                explicit_episode: int | None) -> list[int]:
    if explicit_episode is not None:
        return [int(explicit_episode)]

    matching_episodes = _find_task_episode_candidates(ds_root, single_task)

    if matching_episodes:
        return matching_episodes

    available_tasks = _available_dataset_tasks(ds_root)
    raise ValueError("single_task와 일치하는 초기 자세 episode를 찾지 못했습니다. "
                     f"single_task={single_task!r}. "
                     f"available_tasks={available_tasks}. "
                     "학습/수집에 사용한 prompt와 정확히 같은 문자열을 사용하거나, "
                     "--initial_pose_episode_index를 명시하세요.")


def _build_nonrepeating_episode_schedule(candidates: list[int],
                                         num_trials: int) -> list[int]:

    if num_trials < 0:
        raise ValueError(f"num_trials must be non-negative, got {num_trials}.")

    unique_candidates = list(
        dict.fromkeys(int(episode) for episode in candidates))
    if num_trials > 0 and not unique_candidates:
        raise ValueError(
            "Cannot build an initial-pose schedule without episode candidates."
        )

    schedule: list[int] = []
    while len(schedule) < num_trials:
        cycle = unique_candidates.copy()
        random.shuffle(cycle)

        if schedule and len(cycle) > 1 and cycle[0] == schedule[-1]:
            cycle[0], cycle[1] = cycle[1], cycle[0]

        schedule.extend(cycle)

    return schedule[:num_trials]


def _find_required_task_initial_pose_episodes(ds_root: Path, task: str,
                                              label: str) -> list[int]:
    matching_episodes = _find_task_episode_candidates(ds_root, task)
    if matching_episodes:
        return matching_episodes

    available_tasks = _available_dataset_tasks(ds_root)
    raise ValueError(f"{label} task와 일치하는 초기 자세 episode를 찾지 못했습니다. "
                     f"{label}={task!r}. "
                     f"available_tasks={available_tasks}. "
                     "학습/수집에 사용한 prompt와 정확히 같은 문자열을 사용하세요.")


def _load_initial_state(ds_root: Path, episode_index: int):
    for parquet_path in sorted(ds_root.glob("data/**/*.parquet")):
        df = pd.read_parquet(
            parquet_path,
            columns=["episode_index", "frame_index", "observation.state"])
        ep_df = df[df["episode_index"] == episode_index]
        if not ep_df.empty:
            row = ep_df.sort_values("frame_index").iloc[0]
            return row["observation.state"], parquet_path, int(
                row["frame_index"])
    raise FileNotFoundError(
        f"episode_index={episode_index} 초기 frame을 {ds_root}/data 에서 찾을 수 없음")


def _load_first_initial_states(
        ds_root: Path, episode_indices: list[int],
        label: str) -> tuple[dict[int, np.ndarray], list[int]]:
    wanted = {int(ep) for ep in episode_indices}
    first_states: dict[int, np.ndarray] = {}

    for parquet_path in sorted(ds_root.glob("data/**/*.parquet")):
        df = pd.read_parquet(
            parquet_path,
            columns=["episode_index", "frame_index", "observation.state"])
        df = df[df["episode_index"].isin(wanted)]
        if df.empty:
            continue

        first_rows = df.sort_values(["episode_index",
                                     "frame_index"]).groupby("episode_index",
                                                             sort=True).head(1)
        for _, row in first_rows.iterrows():
            ep = int(row["episode_index"])
            if ep in first_states:
                continue
            first_states[ep] = _as_16d_initial_state(
                row["observation.state"],
                f"{label}: {parquet_path} episode={ep} frame={int(row['frame_index'])}",
            )

    missing = sorted(wanted - set(first_states))
    if missing:
        raise FileNotFoundError(
            f"{label}: 초기 frame을 찾지 못한 episode가 있습니다: {missing}")

    return first_states, sorted(first_states)


def _load_mean_initial_state(ds_root: Path, episode_indices: list[int],
                             label: str) -> tuple[np.ndarray, list[int]]:
    first_states, used_episodes = _load_first_initial_states(
        ds_root, episode_indices, label)
    stacked = np.stack([first_states[ep] for ep in used_episodes], axis=0)
    return stacked.mean(axis=0).astype(np.float32), used_episodes


def _load_task_cluster_mean_initial_states(
        ds_root: Path, single_task: str) -> list[dict[str, Any]]:
    normalized_task = _normalize_task(single_task)
    cluster_ranges = TASK_INITIAL_POSE_CLUSTERS.get(normalized_task)
    if cluster_ranges is None:
        raise ValueError(
            "Cluster mean initial pose is not configured for this task. "
            f"single_task={single_task!r}, "
            f"supported_tasks={sorted(TASK_INITIAL_POSE_CLUSTERS)}.")

    task_candidates = set(
        _find_initial_pose_episodes(ds_root, single_task, None))
    clusters = []
    for start, end in cluster_ranges:
        episodes = _episode_range(start, end)
        missing_from_task = [
            ep for ep in episodes if ep not in task_candidates
        ]
        if missing_from_task:
            raise ValueError(
                f"Task cluster {start}-{end} contains episodes not matching task {single_task!r}: "
                f"{missing_from_task}")
        mean_state, used_episodes = _load_mean_initial_state(
            ds_root,
            episodes,
            f"task_cluster_mean task={single_task!r} episodes={start}-{end}",
        )
        clusters.append({
            "range": f"{start}-{end}",
            "episodes": used_episodes,
            "mean_state": mean_state,
        })

    covered = {ep for cluster in clusters for ep in cluster["episodes"]}
    missing_task_episodes = sorted(task_candidates - covered)
    extra = sorted(covered - task_candidates)
    if missing_task_episodes or extra:
        raise ValueError(
            "Cluster definitions do not exactly cover the task episodes. "
            f"task={single_task!r}, missing={missing_task_episodes}, extra={extra}"
        )

    return clusters


def _load_polyp_cluster_mean_initial_states(
        ds_root: Path, single_task: str) -> list[dict[str, Any]]:

    return _load_task_cluster_mean_initial_states(ds_root, single_task)


def _as_16d_initial_state(state, source: str) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if len(values) != 16:
        raise ValueError(
            f"Expected 16D initial observation.state, got {len(values)}D from {source}. "
            "The with-knob inference path requires ALOHA pos14 + knob2.")
    if not np.isfinite(values).all():
        raise ValueError(
            f"Initial observation.state has NaN/Inf values from {source}.")
    return values


def _split_initial_state(first_state: np.ndarray):
    return (
        [float(v) for v in first_state[:6]],
        [float(v) for v in first_state[7:13]],
        float(first_state[6]),
        float(first_state[13]),
        [float(first_state[14]),
         float(first_state[15])],
    )


def _apply_initial_arm_pose(
    robot: Robot,
    left_joints: list[float],
    right_joints: list[float],
    left_gripper: float | None = None,
    right_gripper: float | None = None,
) -> None:
    robot.left.set_joint_positions_rad(left_joints)
    robot.right.set_joint_positions_rad(right_joints)
    if left_gripper is not None and right_gripper is not None:
        robot._set_gripper_norm(float(left_gripper), float(right_gripper))


def _apply_initial_knob_pose(knob_teleop,
                             knob_values: list[float] | None) -> None:
    if knob_teleop is None or knob_values is None:
        return
    if not knob_teleop.is_connected:
        return
    knob_teleop.send_action({
        "knob_ud.pos": float(knob_values[0]),
        "knob_lr.pos": float(knob_values[1]),
    })


def _reset_knob_to_zero(knob_teleop, wait_s: float, label: str) -> None:
    if knob_teleop is None or not knob_teleop.is_connected:
        return
    knob_teleop.send_action({
        "knob_ud.pos": 0.0,
        "knob_lr.pos": 0.0,
    })
    logging.info(f"{label}: knob 0,0 reset 명령 전송. {wait_s}s 대기 중...")
    time.sleep(max(wait_s, 0.0))
    logging.info(f"{label}: knob 0,0 reset 완료")


def _set_arm_profiles(
    robot: Robot,
    velocity: int,
    accel: int,
    velocity_gripper: int,
    accel_gripper: int,
) -> None:
    robot.left.set_profile(velocity, accel, velocity_gripper, accel_gripper)
    robot.right.set_profile(velocity, accel, velocity_gripper, accel_gripper)


def _reset_to_initial_pose(
    robot: Robot,
    left_joints: list[float],
    right_joints: list[float],
    left_gripper: float,
    right_gripper: float,
    knob_teleop,
    knob_values: list[float] | None,
    wait_s: float,
    reset_velocity: int,
    reset_accel: int,
    reset_velocity_gripper: int,
    reset_accel_gripper: int,
    label: str,
) -> None:
    run_velocity = robot.config.run_profile_velocity
    run_accel = robot.config.run_profile_accel
    run_velocity_gripper = robot.config.run_profile_velocity_gripper
    run_accel_gripper = robot.config.run_profile_accel_gripper

    _set_arm_profiles(robot, reset_velocity, reset_accel,
                      reset_velocity_gripper, reset_accel_gripper)
    try:
        _apply_initial_arm_pose(robot, left_joints, right_joints, left_gripper,
                                right_gripper)
        _apply_initial_knob_pose(knob_teleop, knob_values)
        logging.info(
            f"{label}: 초기 자세 reset 명령 전송 "
            f"(profile={reset_velocity}/{reset_accel}, gripper={reset_velocity_gripper}/{reset_accel_gripper}). "
            f"{wait_s}s 대기 중...")
        time.sleep(max(wait_s, 0.0))
        logging.info(f"{label}: 초기 자세 reset 완료")
    finally:
        _set_arm_profiles(robot, run_velocity, run_accel, run_velocity_gripper,
                          run_accel_gripper)
        logging.info(
            f"{label}: inference run profile 복구 "
            f"(profile={run_velocity}/{run_accel}, gripper={run_velocity_gripper}/{run_accel_gripper})"
        )


def _validate_16d_state_features(ds_meta: LeRobotDatasetMetadata,
                                 use_knob: bool) -> None:
    if not use_knob:
        raise ValueError(
            "run_inference_aloha_with_knob.py expects --use_knob=true so action and "
            "observation.state are both 16D. Use run_inference_aloha.py for 14D non-knob models."
        )

    state_ft = ds_meta.features.get("observation.state")
    if state_ft is None:
        raise ValueError("Dataset metadata has no observation.state feature.")

    names = list(state_ft.get("names", []))
    shape = state_ft.get("shape", None)
    dim = int(shape[0]) if isinstance(shape,
                                      (list, tuple)) and shape else len(names)

    expected_names = EXPECTED_STATE_NAMES_16D
    expected_dim = len(expected_names)

    if dim != expected_dim or names != expected_names:
        raise ValueError(
            "Dataset/model state schema mismatch. "
            f"Expected observation.state {expected_dim}D names={expected_names}, "
            f"but dataset has {dim}D names={names}. "
            "Collect/train with 16D state = ALOHA pos14 + knob2 and run with --use_knob=true."
        )


def _apply_action_safety_limits(action: RobotAction, cfg: InferenceConfig,
                                events: dict) -> RobotAction:
    if not cfg.enable_action_safety_limits:
        return action

    key = "right_shoulder.pos"
    if key not in action:
        return action

    value_deg = math.degrees(float(action[key]))
    clipped_deg = min(max(value_deg, cfg.right_shoulder_min_deg),
                      cfg.right_shoulder_max_deg)
    if clipped_deg == value_deg:
        return action

    clipped_action = dict(action)
    clipped_action[key] = math.radians(clipped_deg)

    now = time.perf_counter()
    if now - float(events.get("last_safety_log_t", 0.0)) > 1.0:
        logging.warning(
            f"Action safety clamp: {key} {value_deg:.1f}° -> {clipped_deg:.1f}° "
            f"(limit {cfg.right_shoulder_min_deg:.1f}° ~ {cfg.right_shoulder_max_deg:.1f}°)"
        )
        events["last_safety_log_t"] = now

    return clipped_action


def _bbox_enabled(cfg: InferenceConfig) -> bool:
    return cfg.bbox_deployment_config is not None or cfg.bbox_detector_repo is not None


def _add_bbox_package_path(cfg: InferenceConfig) -> None:
    candidates: list[Path] = []
    if cfg.bbox_package_dir is not None:
        candidates.append(Path(cfg.bbox_package_dir).expanduser().resolve())
    if cfg.bbox_deployment_config is not None:
        deployment_parent = Path(
            cfg.bbox_deployment_config).expanduser().resolve().parent
        if (deployment_parent / "bbox_vla").is_dir():
            candidates.append(deployment_parent)

    for candidate in candidates:
        if not candidate.exists():
            continue
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _existing_local_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.exists() else None


def _load_bbox_deployment_for_eval(cfg: InferenceConfig):
    _add_bbox_package_path(cfg)
    try:
        from bbox_vla.deployment import load_deployment, resolve_hub_deployment_config
        from bbox_vla.live_overlay import draw_polyp_overlay
        from bbox_vla.runtime import Pi05BboxRuntime
    except Exception as exc:
        raise ImportError(
            "BBox sidecar를 사용하려면 Task4_Polyp_BBox_Pi05_DETR 경로가 PYTHONPATH에 있거나 "
            "--bbox_package_dir=<path-to-bbox-package> 를 지정해야 합니다."
        ) from exc

    config_path = cfg.bbox_deployment_config
    if config_path is None:
        config_path = resolve_hub_deployment_config(
            cfg.bbox_detector_repo,
            revision=cfg.bbox_detector_revision,
            cache_dir=cfg.bbox_cache_dir,
            local_files_only=cfg.bbox_local_files_only,
        )

    base_policy_path = _existing_local_path(cfg.bbox_base_policy_path)
    if base_policy_path is None and cfg.policy is not None:
        base_policy_path = _existing_local_path(
            getattr(cfg.policy, "pretrained_path", None))

    deployment = load_deployment(
        config_path,
        device=cfg.policy.device,
        base_policy_path=base_policy_path,
        n_action_steps=cfg.policy.n_action_steps,
        detector_repo=cfg.bbox_detector_repo,
        detector_revision=cfg.bbox_detector_revision,
        cache_dir=cfg.bbox_cache_dir,
        local_files_only=cfg.bbox_local_files_only,
        use_bf16=cfg.bbox_use_bf16,
    )
    policy = deployment.policy_bundle.policy
    actual_chunk_size = int(policy.config.chunk_size)
    actual_n_action_steps = int(policy.config.n_action_steps)
    if actual_chunk_size != int(cfg.policy.chunk_size):
        raise RuntimeError(
            "BBox policy chunk_size mismatch: "
            f"requested={cfg.policy.chunk_size}, loaded={actual_chunk_size}")
    if actual_n_action_steps != int(cfg.policy.n_action_steps):
        raise RuntimeError(
            "BBox policy n_action_steps mismatch: "
            f"requested={cfg.policy.n_action_steps}, loaded={actual_n_action_steps}"
        )
    if deployment.detector.vision_adapter.policy is not policy:
        raise RuntimeError(
            "BBox detector is not sharing the loaded Pi0.5 policy instance")
    runtime = Pi05BboxRuntime(
        None,
        deployment.predictor,
        camera_key=deployment.spec.camera_key,
        camera_aliases=deployment.spec.camera_aliases,
        frame_cache_size=cfg.bbox_frame_cache_size,
    )
    logging.info(
        "BBox sidecar enabled: "
        f"deployment={config_path}, detector={deployment.checkpoint_path}, "
        f"threshold={deployment.spec.confidence_threshold:.7f}, "
        f"base_policy_path={base_policy_path if base_policy_path is not None else deployment.spec.base_policy_repo}"
    )
    logging.info(
        "BBox action parity: "
        f"chunk_size={actual_chunk_size}, n_action_steps={actual_n_action_steps}, "
        "shared_pi05=True, "
        f"bbox_target_hz={cfg.bbox_inference_hz:.2f}, "
        f"bbox_min_action_queue_remaining={cfg.bbox_min_action_queue_remaining}"
    )
    return deployment, runtime, draw_polyp_overlay


def _wait_reset_or_skip(events: dict,
                        reset_time_s: float,
                        skip_enable_delay_s: float = 0.0) -> str:
    events["reset_skip"] = False
    events["in_reset_wait"] = True
    start_t = time.perf_counter()
    deadline = time.perf_counter() + max(reset_time_s, 0.0)
    try:
        while time.perf_counter() < deadline:
            if events.get("stop_inference", False):
                return "stopped"
            if time.perf_counter() - start_t < max(skip_enable_delay_s, 0.0):
                events["reset_skip"] = False
                precise_sleep(0.05)
                continue
            if events.get("reset_skip", False):
                return "skipped"
            precise_sleep(0.05)
    finally:
        events["in_reset_wait"] = False
        events["reset_skip"] = False
    return "completed"


@parser.wrap()
def run_inference(cfg: InferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if (cfg.policy is not None
            and getattr(cfg.policy, "type", None) == "molmoact2"
            and cfg.molmo_async_replan):
        horizon = int(cfg.policy.n_action_steps)
        trigger = cfg.molmo_async_replan_trigger_remaining
        if not 0 < trigger < horizon:
            raise ValueError(
                "Molmo async trigger must be between 1 and n_action_steps-1; "
                f"got trigger={trigger}, n_action_steps={horizon}.")
    if (cfg.policy is not None and getattr(cfg.policy, "type", None) == "pi05"
            and cfg.pi05_async_replan):
        horizon = int(cfg.policy.n_action_steps)
        trigger = cfg.pi05_async_replan_trigger_remaining
        if not 0 < trigger < horizon:
            raise ValueError(
                "Pi0.5 async trigger must be between 1 and n_action_steps-1; "
                f"got trigger={trigger}, n_action_steps={horizon}.")
        if cfg.pi05_rtc_execution_horizon > horizon:
            raise ValueError(
                "Pi0.5 RTC execution horizon cannot exceed n_action_steps; "
                f"got rtc_horizon={cfg.pi05_rtc_execution_horizon}, "
                f"n_action_steps={horizon}.")

    output_dir = Path(cfg.output_dir)
    output_dataset_root = (Path(cfg.output_dataset_root)
                           if cfg.output_dataset_root is not None else
                           output_dir / "lerobot_dataset")
    if cfg.save_dataset and output_dataset_root.resolve(
    ) == output_dir.resolve():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    robot = make_robot_from_config(cfg.robot)

    knob_teleop = None
    if cfg.use_knob:
        knob_teleop = KnobKeyboardTeleop(cfg.knob)

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
    )
    _validate_16d_state_features(ds_meta, cfg.use_knob)
    logging.info("Validated observation.state schema: 16 dim")

    _, robot_action_processor, robot_observation_processor = make_default_processors(
    )

    obs_features = create_initial_features(
        observation=robot.observation_features)
    action_features = dict(robot.action_features)
    if knob_teleop is not None:
        action_features.update(knob_teleop.action_features)

    robot_obs_pipeline_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=obs_features,
        use_videos=cfg.dataset_video,
    )

    teleop_action_processor_tmp, _, _ = make_default_processors()
    robot_action_pipeline_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor_tmp,
        initial_features=create_initial_features(action=action_features),
        use_videos=False,
    )
    dataset_features = combine_feature_dicts(
        robot_obs_pipeline_features,
        robot_action_pipeline_features,
    )

    output_dataset = None
    if cfg.save_dataset:
        if output_dataset_root.exists():
            raise FileExistsError(
                f"Output dataset root already exists: {output_dataset_root}. "
                "Use a new --output_dataset_root or delete the existing directory first."
            )
        output_dataset_root.parent.mkdir(parents=True, exist_ok=True)
        output_dataset = LeRobotDataset.create(
            cfg.output_dataset_repo_id,
            cfg.fps,
            root=output_dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset_video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera *
            len(getattr(robot, "cameras", {})),
            batch_encoding_size=cfg.video_encoding_batch_size,
            vcodec=cfg.vcodec,
            streaming_encoding=cfg.streaming_encoding,
            encoder_queue_maxsize=cfg.encoder_queue_maxsize,
            encoder_threads=cfg.encoder_threads,
        )
        logging.info(
            f"Inference LeRobot dataset will be saved to: {output_dataset_root}"
        )

    bbox_deployment = None
    bbox_runtime = None
    bbox_draw_overlay = None
    if _bbox_enabled(cfg):
        if cfg.policy is None:
            raise ValueError(
                "--policy.path is required when bbox sidecar inference is enabled."
            )
        bbox_deployment, bbox_runtime, bbox_draw_overlay = _load_bbox_deployment_for_eval(
            cfg)
        policy = bbox_deployment.policy_bundle.policy
        preprocessor = bbox_deployment.policy_bundle.preprocessor
        postprocessor = bbox_deployment.policy_bundle.postprocessor
    else:
        policy = make_policy(cfg.policy, ds_meta=ds_meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(ds_meta.stats, {}),
            preprocessor_overrides={
                "device_processor": {
                    "device": cfg.policy.device
                },
            },
        )

    initial_pose_ds_root = None
    initial_pose_candidates: list[int] = []
    initial_pose_average_candidates: tuple[list[int], list[int]] | None = None
    initial_pose_mean_state: np.ndarray | None = None
    initial_pose_mean_episodes: list[int] = []
    initial_pose_task_clusters: list[dict[str, Any]] = []
    initial_pose_trial_schedule: list[int] = []
    initial_pose_strategy = _normalize_task(cfg.initial_pose_strategy).lower()
    has_episode_reset_pose = False
    if cfg.sync_initial_pose:
        initial_pose_ds_root = Path(
            cfg.dataset_root) if cfg.dataset_root else (Path.home() /
                                                        ".cache" /
                                                        "huggingface" /
                                                        "lerobot" /
                                                        cfg.dataset_repo_id)
        parquet_files = sorted(initial_pose_ds_root.glob("data/**/*.parquet"))
        if parquet_files:
            if initial_pose_strategy in {"same_task", "task_random", "task"}:
                initial_pose_source_task = cfg.single_task
                initial_pose_candidates = _find_initial_pose_episodes(
                    initial_pose_ds_root, initial_pose_source_task,
                    cfg.initial_pose_episode_index)
                has_episode_reset_pose = bool(initial_pose_candidates)
                if cfg.initial_pose_episode_index is not None:
                    logging.info(
                        f"초기 자세 후보: explicit episode={initial_pose_candidates[0]}"
                    )
                else:
                    initial_pose_trial_schedule = _build_nonrepeating_episode_schedule(
                        initial_pose_candidates, cfg.n_episodes)
                    logging.info(
                        f"초기 자세 후보: strategy={initial_pose_strategy}, "
                        f"task='{initial_pose_source_task}' matching episodes="
                        f"{initial_pose_candidates} ({len(initial_pose_candidates)}개)"
                    )
                    logging.info(
                        "초기 자세 선택 순서: 후보를 비복원 추출하며, 모든 후보를 소진한 뒤에만 "
                        f"새 cycle을 시작합니다. trials={cfg.n_episodes}, "
                        f"unique_candidates={len(initial_pose_candidates)}")
            elif initial_pose_strategy in {
                    "same_task_mean", "task_mean", "same_task_average_all"
            }:
                if cfg.initial_pose_episode_index is not None:
                    raise ValueError(
                        "--initial_pose_strategy=same_task_mean 에서는 "
                        "--initial_pose_episode_index를 함께 사용할 수 없습니다.")
                initial_pose_source_task = cfg.single_task
                initial_pose_candidates = _find_initial_pose_episodes(
                    initial_pose_ds_root, initial_pose_source_task, None)
                initial_pose_mean_state, initial_pose_mean_episodes = _load_mean_initial_state(
                    initial_pose_ds_root,
                    initial_pose_candidates,
                    f"same_task_mean task='{initial_pose_source_task}'",
                )
                has_episode_reset_pose = True
                logging.info(
                    f"초기 자세 후보: strategy={initial_pose_strategy}, "
                    f"task='{initial_pose_source_task}', averaged episodes="
                    f"{initial_pose_mean_episodes} ({len(initial_pose_mean_episodes)}개)"
                )
            elif initial_pose_strategy in CLUSTER_MEAN_STRATEGIES:
                if cfg.initial_pose_episode_index is not None:
                    raise ValueError(
                        f"--initial_pose_strategy={initial_pose_strategy} 에서는 "
                        "--initial_pose_episode_index를 함께 사용할 수 없습니다.")
                initial_pose_task_clusters = _load_task_cluster_mean_initial_states(
                    initial_pose_ds_root, cfg.single_task)
                has_episode_reset_pose = True
                logging.info(
                    f"초기 자세 후보: strategy={initial_pose_strategy}, "
                    f"task='{cfg.single_task}', clusters=" + ", ".join(
                        f"{cluster['range']}({len(cluster['episodes'])}개)"
                        for cluster in initial_pose_task_clusters))
            elif initial_pose_strategy == "task_average":
                if cfg.initial_pose_episode_index is not None:
                    raise ValueError(
                        "--initial_pose_strategy=task_average 에서는 "
                        "--initial_pose_episode_index를 함께 사용할 수 없습니다.")
                if not cfg.initial_pose_average_task_a or not cfg.initial_pose_average_task_b:
                    raise ValueError(
                        "--initial_pose_strategy=task_average 사용 시 "
                        "--initial_pose_average_task_a 와 --initial_pose_average_task_b 를 모두 지정하세요."
                    )
                task_a_candidates = _find_required_task_initial_pose_episodes(
                    initial_pose_ds_root, cfg.initial_pose_average_task_a,
                    "initial_pose_average_task_a")
                task_b_candidates = _find_required_task_initial_pose_episodes(
                    initial_pose_ds_root, cfg.initial_pose_average_task_b,
                    "initial_pose_average_task_b")
                initial_pose_average_candidates = (task_a_candidates,
                                                   task_b_candidates)
                has_episode_reset_pose = True
                logging.info(
                    f"초기 자세 후보: strategy=task_average, "
                    f"task_a='{cfg.initial_pose_average_task_a}' episodes={len(task_a_candidates)}개, "
                    f"task_b='{cfg.initial_pose_average_task_b}' episodes={len(task_b_candidates)}개"
                )
            else:
                raise ValueError(
                    f"Unsupported initial_pose_strategy={cfg.initial_pose_strategy!r}. "
                    "Use 'same_task', 'same_task_mean', 'task_cluster_mean', "
                    "'polyp_cluster_mean', or 'task_average'.")
        else:
            logging.warning(
                f"초기 자세 동기화 실패: {initial_pose_ds_root} 에서 parquet 파일을 찾을 수 없음")

    cam_display = None
    if cfg.display_camera and not is_headless():
        cam_display = CamDisplayProcess(cam_name="cam_low")
        cam_display.start()

    events = {
        "exit_early": False,
        "episode_success": None,
        "stop_inference": False,
        "in_reset_wait": False,
        "reset_skip": False,
        "last_safety_log_t": 0.0,
    }
    _, _ = init_keyboard_listener()
    pedal_listener = init_pedal_listener(events)

    results = []
    molmo_chunk_worker: _MolmoChunkWorker | None = None
    molmo_warmup_s: float | None = None

    try:

        if knob_teleop is not None:
            knob_teleop.connect()
            logging.info(f"Knob connected: {cfg.knob.device_port}")
            robot._knob = knob_teleop
        robot.connect()

        if getattr(policy.config, "type",
                   None) == "molmoact2" and (cfg.molmo_async_replan
                                             or cfg.molmo_warmup):
            molmo_chunk_worker = _MolmoChunkWorker(
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=cfg.single_task,
                robot_type=robot.robot_type,
                expected_action_dim=len(dataset_features[ACTION]["names"]),
            )
            if cfg.molmo_warmup:
                logging.info(
                    "Warming up MolmoAct2 before robot policy motion; no action will be sent."
                )
                molmo_warmup_s = _warmup_molmo_chunk_worker(
                    worker=molmo_chunk_worker,
                    robot=robot,
                    robot_observation_processor=robot_observation_processor,
                    dataset_features=dataset_features,
                )
                logging.info(
                    f"MolmoAct2 warm-up complete in {molmo_warmup_s:.3f}s; output discarded."
                )

        encoding_context = (VideoEncodingManager(output_dataset)
                            if output_dataset is not None else
                            contextlib.nullcontext())
        with encoding_context:
            for ep_idx in range(cfg.n_episodes):
                if events["stop_inference"]:
                    break

                log_say(f"Episode {ep_idx + 1} / {cfg.n_episodes}",
                        cfg.play_sounds)
                logging.info(
                    f"\n{'='*60}\nEpisode {ep_idx+1}/{cfg.n_episodes}\n{'='*60}"
                )

                initial_episode = None
                initial_frame = None
                initial_parquet = None
                left_joints = None
                right_joints = None
                left_gripper = None
                right_gripper = None
                knob_initial = None
                if has_episode_reset_pose:
                    if initial_pose_strategy in {
                            "same_task_mean", "task_mean",
                            "same_task_average_all"
                    }:
                        if initial_pose_mean_state is None:
                            raise RuntimeError(
                                "initial_pose_mean_state is not initialized.")
                        first_state = initial_pose_mean_state.copy()
                        initial_episode = {
                            "strategy": initial_pose_strategy,
                            "source_task": cfg.single_task,
                            "averaged_episodes": initial_pose_mean_episodes,
                        }
                        initial_frame = "first_frame_mean"
                        initial_parquet = "all_task_matching_data_parquets"
                        logging.info(
                            f"초기 자세 mean source: task='{cfg.single_task}', "
                            f"episodes={initial_pose_mean_episodes} ({len(initial_pose_mean_episodes)}개)"
                        )
                    elif initial_pose_strategy in CLUSTER_MEAN_STRATEGIES:
                        if not initial_pose_task_clusters:
                            raise RuntimeError(
                                "initial_pose_task_clusters is not initialized."
                            )
                        cluster = random.choice(initial_pose_task_clusters)
                        first_state = cluster["mean_state"].copy()
                        initial_episode = {
                            "strategy": initial_pose_strategy,
                            "source_task": cfg.single_task,
                            "cluster_range": cluster["range"],
                            "averaged_episodes": cluster["episodes"],
                        }
                        initial_frame = "cluster_first_frame_mean"
                        initial_parquet = "all_task_cluster_data_parquets"
                        logging.info(
                            f"초기 자세 task cluster mean source: task='{cfg.single_task}', "
                            f"range={cluster['range']}, "
                            f"episodes={cluster['episodes']} ({len(cluster['episodes'])}개)"
                        )
                    elif initial_pose_strategy == "task_average":
                        if initial_pose_average_candidates is None:
                            raise RuntimeError(
                                "initial_pose_average_candidates is not initialized."
                            )
                        task_a_candidates, task_b_candidates = initial_pose_average_candidates
                        initial_episode_a = random.choice(task_a_candidates)
                        initial_episode_b = random.choice(task_b_candidates)
                        state_a, parquet_a, frame_a = _load_initial_state(
                            initial_pose_ds_root, initial_episode_a)
                        state_b, parquet_b, frame_b = _load_initial_state(
                            initial_pose_ds_root, initial_episode_b)
                        state_a = _as_16d_initial_state(
                            state_a,
                            f"{parquet_a} episode={initial_episode_a}")
                        state_b = _as_16d_initial_state(
                            state_b,
                            f"{parquet_b} episode={initial_episode_b}")
                        first_state = ((state_a + state_b) / 2.0).astype(
                            np.float32)
                        initial_episode = {
                            "strategy": "task_average",
                            "task_a_episode": initial_episode_a,
                            "task_b_episode": initial_episode_b,
                        }
                        initial_frame = {
                            "task_a_frame": frame_a,
                            "task_b_frame": frame_b
                        }
                        initial_parquet = {
                            "task_a_file": str(parquet_a),
                            "task_b_file": str(parquet_b)
                        }
                        logging.info(
                            "초기 자세 average source: "
                            f"task_a_episode={initial_episode_a}, frame={frame_a}, file={parquet_a}; "
                            f"task_b_episode={initial_episode_b}, frame={frame_b}, file={parquet_b}"
                        )
                    else:
                        if cfg.initial_pose_episode_index is not None:
                            initial_episode = initial_pose_candidates[0]
                        else:
                            if len(initial_pose_trial_schedule
                                   ) != cfg.n_episodes:
                                raise RuntimeError(
                                    "initial_pose_trial_schedule is not initialized."
                                )
                            initial_episode = initial_pose_trial_schedule[
                                ep_idx]
                        first_state_raw, initial_parquet, initial_frame = _load_initial_state(
                            initial_pose_ds_root, initial_episode)
                        first_state = _as_16d_initial_state(
                            first_state_raw,
                            f"{initial_parquet} episode={initial_episode}")
                        logging.info(
                            f"초기 자세 random source: episode={initial_episode}, frame={initial_frame}, "
                            f"file={initial_parquet}")

                    left_joints, right_joints, left_gripper, right_gripper, knob_initial = _split_initial_state(
                        first_state)
                    logging.info(
                        f"초기 자세 동기화: left={[f'{math.degrees(v):.1f}°' for v in left_joints]}, "
                        f"left_gripper={left_gripper:.3f}, "
                        f"right={[f'{math.degrees(v):.1f}°' for v in right_joints]}, "
                        f"right_gripper={right_gripper:.3f}, "
                        f"knob={[f'{math.degrees(v):.1f}°' for v in knob_initial]}"
                    )
                    if cfg.knob_zero_before_initial_pose:
                        _reset_knob_to_zero(
                            knob_teleop,
                            cfg.knob_zero_before_initial_pose_wait_s,
                            f"Episode {ep_idx+1} 시작 전",
                        )

                    _reset_to_initial_pose(
                        robot,
                        left_joints,
                        right_joints,
                        left_gripper,
                        right_gripper,
                        knob_teleop,
                        knob_initial,
                        cfg.initial_pose_wait_s,
                        cfg.initial_pose_profile_velocity,
                        cfg.initial_pose_profile_accel,
                        cfg.initial_pose_profile_velocity_gripper,
                        cfg.initial_pose_profile_accel_gripper,
                        f"Episode {ep_idx+1} 시작 전",
                    )

                video_writer = None
                if cfg.save_video:
                    video_path = output_dir / f"episode_{ep_idx+1:04d}.mp4"

                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cfg.fps,
                        (1280, 960),
                    )

                start_time = time.time()
                action_trace = _new_action_trace(
                ) if cfg.save_action_trace else None
                success = None
                action_trace_path = None
                saved_dataset_episode = None
                episode_error: BaseException | None = None
                artifact_errors: list[dict[str, str]] = []

                try:
                    success = inference_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.fps,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dataset_features=dataset_features,
                        cfg=cfg,
                        single_task=cfg.single_task,
                        control_time_s=cfg.episode_time_s,
                        cam_display=cam_display,
                        video_writer=video_writer,
                        dataset=output_dataset,
                        knob_teleop=knob_teleop,
                        bbox_runtime=bbox_runtime
                        if cfg.bbox_overlay_video else None,
                        bbox_draw_overlay=bbox_draw_overlay,
                        bbox_threshold=(
                            bbox_deployment.spec.confidence_threshold
                            if bbox_deployment is not None else None),
                        action_trace=action_trace,
                        molmo_chunk_worker=molmo_chunk_worker,
                    )
                except BaseException as exc:
                    episode_error = exc
                    raise
                finally:

                    if events.get("_active_molmo_planner") is not None:
                        _close_active_molmo_planner(events)
                    _close_owned_pi05_chunk_worker(events)

                    elapsed = time.time() - start_time
                    signaled_success = events.get("episode_success")
                    episode_outcome = (
                        "error"
                        if episode_error is not None else "success" if success
                        is True else "fail" if success is False else "timeout")
                    episode_error_type = (
                        f"{type(episode_error).__module__}.{type(episode_error).__qualname__}"
                        if episode_error is not None else "")
                    episode_error_message = str(
                        episode_error) if episode_error is not None else ""

                    if video_writer is not None:
                        try:
                            video_writer.release()
                        except Exception as exc:
                            artifact_errors.append({
                                "stage": "video_release",
                                "type":
                                f"{type(exc).__module__}.{type(exc).__qualname__}",
                                "message": str(exc),
                            })
                            logging.exception(
                                "Failed to release episode video writer.")

                    if (episode_error is None and output_dataset is not None
                            and output_dataset.has_pending_frames()):
                        try:
                            saved_dataset_episode = output_dataset.num_episodes
                            output_dataset.save_episode()
                        except Exception as exc:
                            saved_dataset_episode = None
                            artifact_errors.append({
                                "stage": "dataset_save",
                                "type":
                                f"{type(exc).__module__}.{type(exc).__qualname__}",
                                "message": str(exc),
                            })
                            logging.exception(
                                "Failed to save inference dataset episode.")

                    if action_trace is not None:
                        trace_path = output_dir / f"episode_{ep_idx+1:04d}_action_trace.npz"
                        try:
                            _save_episode_action_trace(
                                trace_path,
                                action_trace,
                                state_names=list(
                                    dataset_features[OBS_STATE]["names"]),
                                action_names=list(
                                    dataset_features[ACTION]["names"]),
                                task=cfg.single_task,
                                fps=cfg.fps,
                                n_action_steps=int(
                                    getattr(policy.config, "n_action_steps",
                                            1)),
                                success=success
                                if episode_error is None else None,
                                elapsed_s=elapsed,
                                molmo_async_replan=cfg.molmo_async_replan,
                                molmo_async_trigger_remaining=cfg.
                                molmo_async_replan_trigger_remaining,
                                molmo_async_blend_steps=cfg.
                                molmo_async_replan_blend_steps,
                                molmo_async_timeout_s=cfg.
                                molmo_async_replan_timeout_s,
                                molmo_async_max_hold_steps=cfg.
                                molmo_async_replan_max_hold_steps,
                                molmo_warmup=cfg.molmo_warmup,
                                molmo_warmup_s=(molmo_warmup_s
                                                if molmo_warmup_s is not None
                                                else math.nan),
                                molmo_planner_drain_s=float(
                                    events.get("last_molmo_planner_drain_s",
                                               0.0)),
                                pi05_async_replan=cfg.pi05_async_replan,
                                pi05_async_trigger_remaining=(
                                    cfg.pi05_async_replan_trigger_remaining),
                                pi05_async_blend_steps=cfg.
                                pi05_async_replan_blend_steps,
                                pi05_async_timeout_s=cfg.
                                pi05_async_replan_timeout_s,
                                pi05_async_max_hold_steps=cfg.
                                pi05_async_replan_max_hold_steps,
                                pi05_rtc_execution_horizon=cfg.
                                pi05_rtc_execution_horizon,
                                pi05_rtc_max_guidance_weight=(
                                    cfg.pi05_rtc_max_guidance_weight),
                                pi05_rtc_prefix_attention_schedule=(
                                    cfg.pi05_rtc_prefix_attention_schedule.
                                    value),
                                episode_outcome=episode_outcome,
                                trace_complete=episode_error is None,
                                episode_error_type=episode_error_type,
                                episode_error_message=episode_error_message,
                                episode_signaled_success=signaled_success,
                            )
                            action_trace_path = str(trace_path)
                            logging.info(
                                f"Action trace saved: {trace_path} "
                                f"({len(action_trace['frame_index'])} frames)")
                        except Exception as exc:
                            artifact_errors.append({
                                "stage": "action_trace_save",
                                "type":
                                f"{type(exc).__module__}.{type(exc).__qualname__}",
                                "message": str(exc),
                            })
                            logging.exception(
                                "Failed to save episode action trace.")

                    planner_close = dict(
                        events.get("last_molmo_planner_close", {}))
                    result_entry = {
                        "episode":
                        ep_idx + 1,
                        "outcome":
                        episode_outcome,
                        "success":
                        success if episode_error is None else None,
                        "signaled_success":
                        signaled_success,
                        "error": (None if episode_error is None else {
                            "type": episode_error_type,
                            "message": episode_error_message,
                            "stage": "inference_loop",
                        }),
                        "artifact_errors":
                        artifact_errors,
                        "elapsed_s":
                        round(elapsed, 2),
                        "video":
                        (str(output_dir / f"episode_{ep_idx+1:04d}.mp4")
                         if cfg.save_video else None),
                        "action_trace":
                        action_trace_path,
                        "trace_frames": (len(action_trace["frame_index"])
                                         if action_trace is not None else 0),
                        "molmo_planner_drain_s":
                        round(
                            float(events.get("last_molmo_planner_drain_s",
                                             0.0)), 4),
                        "molmo_planner_close":
                        planner_close,
                        "dataset_episode":
                        saved_dataset_episode,
                        "initial_pose_episode":
                        initial_episode,
                        "initial_pose_frame":
                        initial_frame,
                    }
                    results.append(result_entry)

                    logging.info(
                        f"Episode {ep_idx+1} -> {episode_outcome.upper()}  ({elapsed:.1f}s)"
                    )

                if not events["stop_inference"] and has_episode_reset_pose:
                    end_reset_knob = [
                        0.0, 0.0
                    ] if cfg.knob_zero_after_episode else knob_initial
                    if cfg.knob_zero_after_episode:
                        logging.info(
                            f"Episode {ep_idx+1} 종료 후: arm/gripper는 trial 초기자세로, "
                            "knob는 0,0으로 reset합니다.")
                    _reset_to_initial_pose(
                        robot,
                        left_joints,
                        right_joints,
                        left_gripper,
                        right_gripper,
                        knob_teleop,
                        end_reset_knob,
                        cfg.initial_pose_wait_s,
                        cfg.initial_pose_profile_velocity,
                        cfg.initial_pose_profile_accel,
                        cfg.initial_pose_profile_velocity_gripper,
                        cfg.initial_pose_profile_accel_gripper,
                        f"Episode {ep_idx+1} 종료 후",
                    )

                if not events["stop_inference"] and ep_idx < cfg.n_episodes - 1:
                    log_say("Reset the environment", cfg.play_sounds)
                    logging.info(f"환경 리셋 대기 {cfg.reset_time_s}s...")
                    reset_status = _wait_reset_or_skip(
                        events,
                        cfg.reset_time_s,
                        cfg.reset_skip_enable_delay_s,
                    )
                    if reset_status == "skipped":
                        logging.info("Reset wait skipped")
                    elif reset_status == "stopped":
                        logging.info("Reset wait stopped")

    finally:
        primary_error = sys.exc_info()[1]

        n_success = sum(1 for r in results if r["outcome"] == "success")
        n_fail = sum(1 for r in results if r["outcome"] == "fail")
        n_timeout = sum(1 for r in results if r["outcome"] == "timeout")
        n_error = sum(1 for r in results if r["outcome"] == "error")
        n_total = len(results)

        summary = {
            "model":
            str(cfg.policy.pretrained_path) if cfg.policy else "unknown",
            "task": cfg.single_task,
            "total_episodes": n_total,
            "success": n_success,
            "fail": n_fail,
            "timeout": n_timeout,
            "error": n_error,
            "success_rate":
            round(n_success / n_total, 4) if n_total > 0 else 0.0,
            "lerobot_dataset":
            str(output_dataset_root) if cfg.save_dataset else None,
            "molmo_scheduler": {
                "async_replan": cfg.molmo_async_replan,
                "trigger_remaining": cfg.molmo_async_replan_trigger_remaining,
                "blend_steps": cfg.molmo_async_replan_blend_steps,
                "timeout_s": cfg.molmo_async_replan_timeout_s,
                "max_hold_steps": cfg.molmo_async_replan_max_hold_steps,
                "warmup": cfg.molmo_warmup,
                "warmup_s": molmo_warmup_s,
            },
            "pi05_rtc_scheduler": {
                "enabled":
                cfg.pi05_async_replan,
                "trigger_remaining":
                cfg.pi05_async_replan_trigger_remaining,
                "execution_horizon":
                cfg.pi05_rtc_execution_horizon,
                "max_guidance_weight":
                cfg.pi05_rtc_max_guidance_weight,
                "prefix_attention_schedule":
                (cfg.pi05_rtc_prefix_attention_schedule.value),
                "external_blend_steps":
                cfg.pi05_async_replan_blend_steps,
                "timeout_s":
                cfg.pi05_async_replan_timeout_s,
                "max_hold_steps":
                cfg.pi05_async_replan_max_hold_steps,
            },
            "episodes": results,
        }

        summary_path = output_dir / "results_summary.json"
        summary_write_error: Exception | None = None
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logging.info("\n" + "=" * 60)
            logging.info("  Inference complete")
            logging.info(f"  Model : {summary['model']}")
            logging.info(f"  Task  : {summary['task']}")
            logging.info(f"  Total : {n_total} episodes")
            logging.info(
                f"  Success Rate: {summary['success_rate']*100:.1f}%  "
                f"({n_success} success / {n_fail} fail / {n_timeout} timeout / "
                f"{n_error} error)")
            logging.info(f"  Results saved → {summary_path}")
            logging.info("=" * 60)
        except Exception as exc:
            summary_write_error = exc
            logging.exception(
                f"Failed to save inference summary: {summary_path}")
        finally:

            cleanup_actions = [
                ("pedal listener", lambda: pedal_listener.stop()
                 if pedal_listener is not None else None),
                ("robot", lambda: robot.disconnect()
                 if robot.is_connected else None),
                (
                    "knob",
                    lambda: knob_teleop.disconnect() if knob_teleop is not None
                    and knob_teleop.is_connected else None,
                ),
                ("bbox runtime", lambda: bbox_runtime.close()
                 if bbox_runtime is not None else None),
                ("camera display", lambda: cam_display.stop()
                 if cam_display is not None else None),
            ]
            for resource_name, cleanup_action in cleanup_actions:
                try:
                    cleanup_action()
                except Exception:
                    logging.exception(f"Failed to clean up {resource_name}.")

            try:
                _close_owned_pi05_chunk_worker(events)
            except Exception:
                logging.exception("Failed to close Pi0.5 chunk worker.")

            if molmo_chunk_worker is not None:
                try:
                    molmo_chunk_worker.close()
                except Exception:
                    logging.exception("Failed to close Molmo chunk worker.")

            try:
                log_say("Inference complete", cfg.play_sounds)
            except Exception:
                logging.exception("Failed to play inference completion sound.")

        if summary_write_error is not None and primary_error is None:
            raise summary_write_error

    return summary


def main():
    multiprocessing.set_start_method("spawn", force=True)
    register_third_party_plugins()
    run_inference()


if __name__ == "__main__":
    main()
