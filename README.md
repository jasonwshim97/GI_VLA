# GI-VLA

## 1. Overview

GI-VLA provides the polyp detection code and ALOHA demonstration recording and policy inference scripts used in our endoscopic robotics research.

| Directory | Contents |
| --- | --- |
| [`gi_vla/`](gi_vla/) | Polyp detection using shared GR00T visual features, temporal extensions, comparison models, and annotation tools |
| [`aloha/`](aloha/) | Demonstration recording and policy inference with the ALOHA dual-arm robot and endoscope knobs |

```text
GI_VLA/
├── gi_vla/
│   ├── endo_bbox_vla/     # Polyp detection models, training, evaluation, and inference
│   ├── comparisons/       # Comparison models
│   ├── annotation/        # Annotation preparation and review tools
│   ├── configs/           # Model and experiment configurations
│   ├── pyproject.toml
│   └── README.md
├── aloha/
│   ├── record_aloha_with_knob.py
│   ├── run_inference_aloha_with_knob.py
│   └── README.md
└── LICENSE
```

## 2. gi_vla: Polyp Detection

### Model Overview

Ours reuses the visual features computed by a frozen GR00T N1.7 policy during action generation. It extracts the `16 × 16 × 1024` features corresponding to the endoscopic camera, `cam_low`, from the output of Qwen3-VL visual block 11. A detection head then predicts polyp presence probabilities and bounding boxes.

The detection head initializes four spatial queries from a dense center heatmap and refines them through three Transformer decoder layers. GR00T remains frozen, and only approximately 3.56M detection parameters are trained. Sharing features from the same action forward eliminates an additional vision-encoder pass for detection. Detection outputs are not fed back into the VLA action pathway.

### Code Organization

| File or directory | Responsibility |
| --- | --- |
| `endo_bbox_vla/config.py` | Paper model identifiers, configuration contracts, and external path resolution |
| `endo_bbox_vla/data.py` | Datasets, manifests, preprocessing, sampling, and clip construction |
| `endo_bbox_vla/cache.py` | Frozen feature-cache storage, loading, and integrity checks |
| `endo_bbox_vla/network.py` | Trainable detection heads and temporal query structures |
| `endo_bbox_vla/model.py` | Frozen backbone feature extraction and model construction |
| `endo_bbox_vla/loss.py` | Static and temporal detection losses |
| `endo_bbox_vla/train.py` | Training, continued training, and checkpoint resume |
| `endo_bbox_vla/metrics.py` | IoU, AP, recall, false-positive metrics, and threshold selection |
| `endo_bbox_vla/evaluate.py` | Validation selection and evaluation with fixed thresholds |
| `endo_bbox_vla/inference.py` | Same-forward action and detection inference, temporal state, and event rules |
| `endo_bbox_vla/utils.py` | Shared I/O, hashing, and random-seed utilities |
| `comparisons/` | EndoVLA-inspired SFT and standalone detector comparisons |
| `annotation/` | Annotation preparation and web-based review tools |

Use `endo_bbox_vla.model.build_detector("ours")` to construct the detection head programmatically. Model weights, policy preprocessing, and the physical robot environment must be provided separately.

### Installation

Clone LeRobot from its [original GitHub repository](https://github.com/huggingface/lerobot) into sibling directories alongside `GI_VLA/`. Use v0.6.1 for polyp detection and policy inference, and v0.5.1 for ALOHA demonstration recording.

```text
workspace/
├── GI_VLA/
├── lerobot_v061/    # Polyp detection and policy inference
└── lerobot_v051/    # ALOHA demonstration recording
```

```bash
# Run from the GI_VLA repository root
git clone --branch v0.6.1 https://github.com/huggingface/lerobot.git ../lerobot_v061
git clone --branch v0.5.1 https://github.com/huggingface/lerobot.git ../lerobot_v051
```

This layout is an example. If you clone LeRobot elsewhere, update the installation paths below and the `PYTHONPATH` values in the ALOHA examples accordingly. The ALOHA knob and hardware backends must be prepared separately.

Create a Python 3.12 environment and install the packages from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e "../lerobot_v061[groot]"
python -m pip install -e ./gi_vla
```

Actual GR00T execution requires a compatible LeRobot/Transformers environment and saved policy processors. EndoVLA-inspired SFT was performed in a separate Transformers 4.49 environment. These installation commands do not download backbone weights or research datasets.

### Training and Evaluation Examples

After installation, run the following examples from `gi_vla/`:

```bash
cd gi_vla

# First training stage for the final model
python -m endo_bbox_vla.train static \
  --config configs/ours.yaml --stage spatial_queries --device cuda

# Hard-positive continued training from the first-stage checkpoint
python -m endo_bbox_vla.train static \
  --config configs/ours.yaml --stage hard_positive --device cuda
```

Both commands require the external feature caches and manifests. Temporal training additionally takes the completed full training plan, frame manifest, feature cache, and output path. Inspect the available arguments with:

```bash
python -m endo_bbox_vla.train temporal --help
python -m endo_bbox_vla.evaluate --help
python -m comparisons.detectors --help
python -m comparisons.ar_train --help
python -m annotation.prepare --help
python -m annotation.review --help
```

## 3. aloha: Demonstration Recording and Policy Inference

`aloha` contains high-level scripts for demonstration recording and policy execution with the ALOHA dual-arm robot. With `--use_knob=true`, the scripts use a **16-dimensional action space** comprising the 14-dimensional ALOHA action and two endoscope knob dimensions.

### Scripts and Software Environments

| File | Main functionality | Base LeRobot version |
| --- | --- | --- |
| [`record_aloha_with_knob.py`](aloha/record_aloha_with_knob.py) | Records robot and camera observations, task descriptions, and demonstration actions | v0.5.1 |
| [`run_inference_aloha_with_knob.py`](aloha/run_inference_aloha_with_knob.py) | Loads a policy and dataset, runs episodic inference, and saves videos, action traces, and evaluation datasets | v0.6.1 |

The two scripts were developed in different LeRobot environments. Use separate environments for recording and inference until the recording script has been ported to and validated against v0.6.1.

### Hardware Configuration

The following hardware-specific components are not included in the released scripts:

- Low-level ALOHA hardware drivers
- Knob motor drivers and Dynamixel communication code
- Robot and camera calibration files
- Camera serial numbers and device paths
- Complete lab-specific hardware configurations and comprehensive safety systems

Running on physical hardware requires a compatible LeRobot hardware backend that provides these components. The inference script's initial poses, motion profiles, task-specific initial-pose groupings, and right-shoulder action limits must also be reviewed for the target robot and dataset.

### Demonstration Recording Example

The following template starts from the repository root. Replace the LeRobot path, device configuration files, and repository IDs with values for your environment.

```bash
cd aloha
export PYTHONPATH="$(pwd)/../../lerobot_v051/src"

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

`--dataset.repo_id` and `--dataset.root` specify the destination dataset repository ID and local storage path, respectively.

### Policy Inference Example

Start this example separately from the repository root. In the inference LeRobot environment, specify the policy checkpoint and reference dataset paths.

```bash
cd aloha
export PYTHONPATH="$(pwd)/../../lerobot_v061/src"

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

`--policy.path` accepts a model repository ID or local checkpoint path. `--dataset_repo_id` and `--dataset_root` specify the reference dataset repository ID and local path.

To save inference results as a LeRobot dataset, provide both arguments below:

```text
--save_dataset=true
--output_dataset_repo_id=YOUR_HF_ORG/YOUR_OUTPUT_DATASET
```

### Physical Robot Execution

These scripts can move a physical robot. With the inference default `--sync_initial_pose=true`, the robot may move to an initial pose derived from the reference dataset before policy execution.

- Verify observation and action dimensions and ordering.
- Check the joint, gripper, and knob limits of the target hardware.
- Inspect policy outputs before enabling robot motion.
- Keep an emergency stop available and operate under qualified supervision.
- Do not disable action safety limits without an independent safety review.
