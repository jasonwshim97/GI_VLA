# GI-VLA

This repository provides the high-level data acquisition and policy inference
scripts used in the GI-VLA experiments.

## Released Files

### `record_aloha_with_knob.py`

Implements the high-level demonstration recording pipeline for an ALOHA system
with two additional knob dimensions. When `--use_knob=true`, the script records
robot observations, camera observations, task descriptions, and 16-dimensional
actions composed of the 14-dimensional ALOHA action and two knob values.

This script is based on Hugging Face LeRobot v0.5.1.

### `run_inference_aloha_with_knob.py`

Implements the high-level policy inference and evaluation pipeline for the same
16-dimensional ALOHA and knob action space. It loads a policy checkpoint and a
LeRobot dataset from paths supplied at runtime, executes episodic inference,
and can save videos, action traces, and evaluation datasets.

This script is based on Hugging Face LeRobot v0.6.1.

## Code Scope

This repository releases only the high-level data acquisition and inference
logic used in our experiments. It does not include the following
hardware-specific components:

- Low-level ALOHA hardware drivers
- Knob motor drivers
- Dynamixel communication code
- Robot and camera calibration files
- Camera serial numbers and device paths
- Complete lab-specific hardware configurations and comprehensive safety systems

The released scripts are therefore not a complete standalone software package.
Running them on a physical robot requires a compatible LeRobot hardware backend
and appropriate hardware configuration.

The inference script retains experiment-specific high-level defaults, including
initial-pose behavior, motion-profile values, task-specific initial-pose
groupings, and a right-shoulder action clamp. These defaults must be reviewed,
adapted, or removed before the script is used with another robot or dataset.

## Software Versions

The two scripts were developed in different LeRobot environments:

| Script | LeRobot version |
| --- | --- |
| `record_aloha_with_knob.py` | v0.5.1 |
| `run_inference_aloha_with_knob.py` | v0.6.1 |

Use separate environments for recording and inference unless the recording
script has been explicitly ported and validated against LeRobot v0.6.1.

## Models and Datasets

Model weights and datasets are not included in this repository. Their locations
must be supplied through command-line arguments.

The recording script accepts the destination dataset repository and local
storage path through:

```text
--dataset.repo_id=<DATASET_REPOSITORY_ID>
--dataset.root=<LOCAL_DATASET_PATH>
```

The inference script accepts the policy and source dataset through:

```text
--policy.path=<MODEL_REPOSITORY_ID_OR_LOCAL_PATH>
--dataset_repo_id=<DATASET_REPOSITORY_ID>
--dataset_root=<LOCAL_DATASET_PATH>
```

If inference results are saved as a LeRobot dataset, an output repository ID
must also be provided:

```text
--save_dataset=true
--output_dataset_repo_id=<OUTPUT_DATASET_REPOSITORY_ID>
```

## Example Usage

The following commands are templates. Install the required LeRobot version or
set `PYTHONPATH` to a compatible LeRobot source tree that also contains the
required hardware backend. Replace all example paths and repository IDs with
values appropriate for your system.

### Data acquisition

```bash
export PYTHONPATH=/path/to/lerobot-v0.5.1/src

python record_aloha_with_knob.py \
  --robot.type=aloha_follower_with_knob \
  --robot.left_arm_config_path=/path/to/left_follower.yaml \
  --robot.right_arm_config_path=/path/to/right_follower.yaml \
  --teleop.type=aloha_master \
  --teleop.left_arm_config_path=/path/to/left_master.yaml \
  --teleop.right_arm_config_path=/path/to/right_master.yaml \
  --dataset.repo_id=YOUR_HF_ORG/YOUR_DATASET \
  --dataset.root=/path/to/local/dataset \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.single_task="YOUR_TASK_DESCRIPTION" \
  --knob.device_port=/path/to/knob-device \
  --use_knob=true
```

### Policy inference

```bash
export PYTHONPATH=/path/to/lerobot-v0.6.1/src

python run_inference_aloha_with_knob.py \
  --robot.type=aloha_follower_with_knob \
  --robot.left_arm_config_path=/path/to/left_follower.yaml \
  --robot.right_arm_config_path=/path/to/right_follower.yaml \
  --policy.path=YOUR_HF_ORG/YOUR_MODEL \
  --dataset_repo_id=YOUR_HF_ORG/YOUR_DATASET \
  --dataset_root=/path/to/local/dataset \
  --single_task="YOUR_TASK_DESCRIPTION" \
  --knob.device_port=/path/to/knob-device \
  --use_knob=true \
  --n_episodes=10 \
  --episode_time_s=300 \
  --fps=30 \
  --save_video=true \
  --save_dataset=false \
  --output_dir=./inference_results
```

## Safety

These scripts can command a physical robot. With `--sync_initial_pose=true`,
which is the inference default, the robot can move to an initial pose derived
from the source dataset before policy execution. Before execution:

- Validate the observation and action ordering.
- Confirm all joint, gripper, and knob limits for the target hardware.
- Test policy outputs without enabling robot motion.
- Keep an emergency stop available.
- Operate the robot only under qualified supervision.
- Do not disable action safety limits without an independent safety review.

The built-in action clamp covers only the configured right-shoulder joint. It
is not a comprehensive robot safety system and does not replace hardware limits,
collision avoidance, workspace validation, or an emergency stop.

The authors are not responsible for damage or injury resulting from improper
hardware configuration or unsafe operation.

## Acknowledgements

These files contain modified code derived from
[Hugging Face LeRobot](https://github.com/huggingface/lerobot) and are
distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the
license terms and third-party notices.

## Citation

If you use this code, please cite the corresponding GI-VLA paper. The complete
bibliographic entry should be added here when the paper metadata is finalized.
