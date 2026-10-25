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

import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
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
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    aloha_master,
    make_teleoperator_from_config,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot.teleoperators.knob_keyboard.knob_teleoperator import (
    KnobKeyboardConfig,
    KnobKeyboardTeleop,
)
from lerobot.robots.aloha_follower_with_knob.aloha_follower_with_knob import AlohaFollowerWithKnob


def _cam_display_process(frame_queue: multiprocessing.Queue, cam_name: str):

    import cv2

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
        logging.info(
            f"CamDisplayProcess: {self._cam_name} 실시간 표시 시작 (별도 프로세스)")

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
        logging.info("CamDisplayProcess: 표시 종료")


def init_pedal_listener(events: dict) -> Any:

    if is_headless():
        return None

    try:
        from pynput import keyboard

        def on_press(key):
            try:
                k = key.char if hasattr(key, 'char') else None
                if k == '1':
                    print("페달 1 눌림 → 에피소드 저장")
                    events["exit_early"] = True
                elif k == '2':
                    print("페달 2 눌림 → 재녹화")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif k == '3':
                    print("페달 3 눌림 → 전체 종료")
                    events["stop_recording"] = True
                    events["exit_early"] = True
            except Exception as e:
                logging.debug(f"Pedal listener error: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logging.info("페달 리스너 시작 (페달1=저장, 페달2=재녹화, 페달3=종료)")
        return listener

    except Exception as e:
        logging.warning(f"페달 리스너 초기화 실패: {e}")
        return None


@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = True
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "auto"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    record_data: bool = True

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError(
                "You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    display_cam_low: bool = False
    play_sounds: bool = True
    resume: bool = False

    knob: KnobKeyboardConfig = field(default_factory=KnobKeyboardConfig)
    use_knob: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError(
                "Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction,
                                                          RobotObservation],
                                                    RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction,
                                                         RobotObservation],
                                                   RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation,
                                                        RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]
    | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    knob_teleop: KnobKeyboardTeleop | None = None,
    cam_display: CamDisplayProcess | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})."
        )

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features,
                                                    obs_processed,
                                                    prefix=OBS_STR)

        if cam_display is not None and "cam_low" in obs:
            cam_display.send_frame(obs["cam_low"])

        if policy is not None and preprocessor is not None and postprocessor is not None:
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
            act_processed_policy: RobotAction = make_robot_action(
                action_values, dataset.features)

        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            if knob_teleop is not None and knob_teleop.is_connected:
                act.update(knob_teleop.get_action())

            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
            )
            continue

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor(
                (act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor(
                (act_processed_teleop, obs))

        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features,
                                               action_values,
                                               prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed,
                           action=action_values,
                           compress_images=display_compressed_images)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = 1 / fps - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                "Dataset frames might be dropped and robot control might be unstable. "
                "Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording",
                   ip=cfg.display_ip,
                   port=cfg.display_port)
    display_compressed_images = (True if (
        cfg.display_data and cfg.display_ip is not None
        and cfg.display_port is not None) else cfg.display_compressed_images)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(
        cfg.teleop) if cfg.teleop is not None else None

    knob_teleop = None
    if cfg.use_knob:
        knob_teleop = KnobKeyboardTeleop(cfg.knob)
        knob_teleop.connect()
        logging.info(f"Knob connected: {cfg.knob.device_port}")
        robot._knob = knob_teleop
        logging.info("Knob teleop injected into robot.")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors(
    )

    obs_features = create_initial_features(
        observation=robot.observation_features)

    action_features = dict(robot.action_features)
    if knob_teleop is not None:
        action_features.update(knob_teleop.action_features)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=obs_features,
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    ds_meta = None
    listener = None
    pedal_listener = None
    cam_display = None

    try:
        if cfg.dataset.record_data and cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot,
                                                     cfg.dataset.fps,
                                                     dataset_features)
            ds_meta = dataset.meta

        elif cfg.dataset.record_data:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.
                num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            ds_meta = dataset.meta

        policy = None if cfg.policy is None else make_policy(cfg.policy,
                                                             ds_meta=ds_meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(ds_meta.stats,
                                           cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {
                        "device": cfg.policy.device
                    },
                    "rename_observations_processor": {
                        "rename_map": cfg.dataset.rename_map
                    },
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        if cfg.display_cam_low and not is_headless():
            cam_display = CamDisplayProcess(cam_name="cam_low")
            cam_display.start()

        listener, events = init_keyboard_listener()
        pedal_listener = init_pedal_listener(events)

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it "
                "for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 "
                "# --dataset.vcodec=auto. More info in the documentation: "
                "https://huggingface.co/docs/lerobot/streaming_video_encoding")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events[
                    "stop_recording"]:
                if dataset is not None:
                    log_say(f"Recording episode {dataset.num_episodes}",
                            cfg.play_sounds)
                else:
                    log_say(f"Running rollout {recorded_episodes + 1}",
                            cfg.play_sounds)

                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    knob_teleop=knob_teleop,
                    cam_display=cam_display,
                )

                should_rerecord = events["rerecord_episode"]
                if should_rerecord:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    if dataset is not None:
                        dataset.clear_episode_buffer()

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1)
                        or should_rerecord):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        knob_teleop=knob_teleop,
                        cam_display=cam_display,
                    )

                if should_rerecord:
                    continue

                if dataset is not None:
                    dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if cam_display is not None:
            cam_display.stop()

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if knob_teleop is not None and knob_teleop.is_connected:
            knob_teleop.disconnect()
            logging.info("Knob disconnected.")

        if not is_headless() and listener:
            listener.stop()

        if pedal_listener is not None:
            pedal_listener.stop()
            logging.info("페달 리스너 종료.")

        if cfg.dataset.push_to_hub and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags,
                                private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    multiprocessing.set_start_method("spawn", force=True)
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
