# endo_bbox_vla

Consolidated code for the completed endoscopic polyp detection study. **Ours** is
the final static shared detector. The temporal visual-memory and VL-conditioned
models are completed comparison extensions. All reuse frozen GR00T N1.7 features.
This package contains code and immutable configurations only.

## Paper model names

| Paper model or comparison | Public identifier | Configuration |
| --- | --- | --- |
| **Ours** — final static shared detector | `ours` | `configs/ours.yaml` |
| Temporal — visual memory | `temporal_visual_memory` | Full temporal plan and external cache |
| Temporal — VL-conditioned | `temporal_vl_conditioned` | Full temporal plan and external cache |
| EndoVLA-inspired bbox-only SFT | `endovla_bbox_sft` | `configs/comparisons/endovla_bbox_sft.yaml` |
| EndoVLA-inspired bbox+action SFT | `endovla_bbox_action_sft` | `configs/comparisons/endovla_bbox_action_sft.yaml` |
| Faster R-CNN R50-FPN / DETR-R50 | `faster_rcnn` / `detr_r50` | `configs/standalone_detectors.yaml` |

See [paper-to-code mapping](#paper-to-code-mapping) for architecture APIs, training
stages, ablations and the historical IDs in existing checkpoints. The registry
is available as `config.PAPER_MODELS` and `config.paper_model("ours")`.

## Layout

| File | Responsibility |
| --- | --- |
| `endo_bbox_vla/config.py` | Paper model registry, configuration contracts and external paths |
| `endo_bbox_vla/data.py` | Datasets, manifests, transforms, sampling, clip plans |
| `endo_bbox_vla/cache.py` | Frozen feature caches and identity validation |
| `endo_bbox_vla/network.py` | Trainable detection heads and tensor packet types |
| `endo_bbox_vla/model.py` | Frozen backbone feature extraction and model assembly |
| `endo_bbox_vla/loss.py` | Detection and temporal training objectives |
| `endo_bbox_vla/train.py` | Static/temporal training loops and resume state |
| `endo_bbox_vla/metrics.py` | IoU/AP/recall, threshold selection, frame metrics |
| `endo_bbox_vla/evaluate.py` | Validation selection and frozen-threshold scoring |
| `endo_bbox_vla/inference.py` | Same-forward inference, memory and event rules |
| `endo_bbox_vla/utils.py` | I/O, hashing, seeds and common conversions |
| `annotation/prepare.py`, `review.py` | Annotation preparation and web review |
| `comparisons/` | Completed AR-Qwen SFT and standalone detector comparisons |

The package name is `endo_bbox_vla`; the historical nested `bbox_vla.detr.*`
Python namespace and compatibility wrapper files are retired in this release.
Configuration bytes, checkpoint tensor keys, and serialized experiment schema
identifiers retain their recorded identities.

## Setup

Use Python 3.12, from this directory:

```bash
python3.12 -m venv /tmp/endo-bbox-vla-env
/tmp/endo-bbox-vla-env/bin/python -m pip install -e .
```

Real GR00T execution requires its compatible external LeRobot/Transformers
environment and saved policy processors. AR-Qwen SFT used a separate
Transformers 4.49 environment. Backbone dependencies and weights must be
supplied separately.

## Main entry points

```bash
python -m endo_bbox_vla.train --help
python -m endo_bbox_vla.evaluate --help
python -m comparisons.detectors --help
python -m comparisons.ar_train --help
python -m annotation.prepare --help
python -m annotation.review --help
```

Static training reads the frozen feature caches bound by the selected config:

```bash
python -m endo_bbox_vla.train static \
  --config configs/ours.yaml --stage spatial_queries --device cuda

python -m endo_bbox_vla.train static \
  --config configs/ours.yaml --stage hard_positive --device cuda
```

Ours uses spatial-query training followed by hard-positive continuation. The
first stage must already exist before the second; the stages are not two
different paper models. The recorded config retains the original stage keys.
The full temporal trainer takes the original full plan, frame manifest and
feature cache through `temporal --model temporal_visual_memory` or
`temporal --model temporal_vl_conditioned`. `train temporal --help` lists inputs.
These commands require external assets.

Saved predictions can be scored without loading a VLA:

```bash
python -m endo_bbox_vla.evaluate predictions \
  --input /path/to/predictions.jsonl --threshold 0.1787109375 \
  --output /path/to/metrics.json
```

The supplied threshold must come from validation. No internal-test threshold
selection is performed by this command.

For programmatic use, `model.build_detector("ours")` constructs the static head;
the two temporal identifiers above construct their respective heads.
`FrozenGrootN17MultiLevelExtractor` and `FrozenGrootN17TemporalExtractor` provide
read-only backbone capture. `inference.SharedDetectorRuntime` covers the static
bundle, and `TemporalDetectorRuntime` carries temporal state around a supplied
policy action call. The caller owns preprocessing and robot control.

## Scope and external assets

Datasets, annotations, model weights, caches, experiment outputs and server backups
are excluded. Recorded paths in YAML are resolved by `config.resolve_path()`;
`ENDO_BBOX_VLA_DATA_ROOT` can point to an external asset tree for recorded
`bbox_vla/...` paths. Config filenames use descriptive public names; immutable
YAML contents, serialized provenance fields and hashes are unchanged. Historical
config references are translated by the resolver, rather than editing the
recorded experiment payloads. See [the mapping](#paper-to-code-mapping).

Older Pi0.5 pipelines, unexecuted GRPO, historical report/acceptance frameworks,
and server job orchestration are outside this public core. The complete original
research source remains preserved separately. This release does not claim a
completed robot online workflow result.

Local pre-release validation passed 267 CPU regression cases; 30 cases requiring
excluded original assets were skipped. Static/temporal outputs and gradients,
and metrics replayed from 18,876 saved predictions, matched the preserved
implementation. The local test suite and internal documentation are not
distributed in this repository.


## Paper-to-code mapping

The completed-study handoff identifies **Ours** as the final static shared
detector. This release uses paper-facing model identifiers and descriptive API
names. Temporal labels below describe the completed comparison arms; they do
not promote a temporal checkpoint to the paper's final Ours.

### Models and training stages

| Paper model or role | Public identifier | Historical record | Code entry point |
| --- | --- | --- | --- |
| **Ours** — static shared detector | `ours` | Shared E03-C; `e03_shared_location_c` | `model.build_detector("ours")`; `network.SharedDetectorHead`; `configs/ours.yaml` |
| Temporal — visual memory | `temporal_visual_memory` | T1; `T1_e03_temporal_full` | `model.build_detector("temporal_visual_memory")`; `train temporal --model temporal_visual_memory` |
| Temporal — VL-conditioned | `temporal_vl_conditioned` | T2; `T2_e03_dit_temporal_full` | `model.build_detector("temporal_vl_conditioned")`; `train temporal --model temporal_vl_conditioned` |
| EndoVLA-inspired bbox-only SFT | `endovla_bbox_sft` | `qwen2vl_7b_ar_e03_1cam_bbox_only_sft_v1` | `comparisons/ar_*.py`; `configs/comparisons/endovla_bbox_sft.yaml` |
| EndoVLA-inspired bbox+action SFT | `endovla_bbox_action_sft` | `qwen2vl_7b_ar_e03_4cam_cont16_sft_v1` | `comparisons/ar_*.py`; `configs/comparisons/endovla_bbox_action_sft.yaml` |
| Faster R-CNN R50-FPN | `faster_rcnn` | `faster_rcnn` | `comparisons.detectors`; `configs/standalone_detectors.yaml` |
| DETR-R50 | `detr_r50` | `detr_r50` | `comparisons.detectors`; `configs/standalone_detectors.yaml` |

The EndoVLA-inspired methods are adaptations, not exact reproductions. Their
serialized action targets are auxiliary text supervision, not robot commands.

Ours has two optimization stages using the same architecture:

| Public training stage | Recorded key | Meaning |
| --- | --- | --- |
| `spatial_queries` | `b` / Shared E03-B | 50 epochs of spatial-query detector training |
| `hard_positive` | `c` / Shared E03-C | 20 epochs of size/location-balanced hard-positive continuation from the first stage |

Only the validation-selected continuation checkpoint is the final Ours. Its
recorded SHA-256 is
`8458c303733d5640f30fc3debb9a4c2616b49bd0eecf8f38a40f909931ce1227` and its
validation-selected threshold is `0.1787109375`. No weights are distributed here.

### Features and public APIs

`visual` / `mid` means the detached `cam_low` features from Qwen3-VL visual
block 11 before the spatial merger: 256 tokens of dimension 1024. This was
historically named `e03`. `projected` means the post-merger features previously
named `e02`. These are feature sources, not distinct final model names.

The static detector uses dense-center-based spatial query initialization,
followed by three decoder layers. `SpatiallyInitializedDetectorHead` fixes
this head configuration, while `SharedDetectorHead` also supports controlled
query ablations. The VLA remains frozen and shares the action forward.

| Public API | Responsibility | Previous name |
| --- | --- | --- |
| `network.SharedDetectorHead` | Static shared head and query ablations | `SharedE03DetectorHead` |
| `network.SpatiallyInitializedDetectorHead` | Spatially initialized production head | `SharedE03LocationFirstHead` |
| `network.SpatialQueryHead` | Configurable spatial-query architecture | `E03HRDetectorHead` |
| `loss.build_shared_detection_criterion` | Shared detector loss | `build_shared_e03_criterion` |
| `inference.SharedDetectorRuntime` | Same-forward static action and detection | `GrootE03OnlineRuntime` |
| `config.SharedDetectorSpec` | Validated static runtime bundle | `GrootE03OnlineSpec` |
| `cache.LoadedVisualFeatures` | Frozen visual feature payload | `LoadedE03Features` |
| `cache.GrootTemporalCache.load_visual` / `put_visual` | Visual cache I/O | `load_e03` / `put_e03` |

The temporal visual-memory model uses recurrent query state and current
block-11 visual features. The VL-conditioned arm additionally uses pooled
vision-language self-attention features. Internal serialized feature modes
remain `mid_only` and `dit_conditioned`, respectively.

### Other configurations

| Public config | Role | Historical config basename |
| --- | --- | --- |
| `ours.yaml` | Final static model, both training stages | `e03_shared_location_v1.yaml` |
| `ours_runtime.yaml` | Shared-model runtime specification and provisional event rule | `groot_e03_shared_online_eval_v1.yaml` |
| `baseline_single_query_runtime.yaml` | Original one-query runtime, separate from Ours | `groot_e03_online_eval_v1.yaml` |
| `ablation_components.yaml` | Controlled query/initialization/continuation ablation | `e03_shared_ablation_v1.yaml` |
| `ablation_query_count.yaml` | One-query continuation follow-up | `e03_shared_query_count_ablation_v1.yaml` |
| `standalone_detectors.yaml` | Faster R-CNN and DETR comparison | `e03_separate_detector_efficiency_v1.yaml` |
| `high_resolution_upper_bound.yaml` | Separate high-resolution detector experiment | `e03_hr_abcd_v1.yaml` |
| `feature_comparison.yaml` / `feature_comparison_qc.yaml` | Original projected-versus-block-11 study | `g_e02_e03_scenario_static_v1.yaml` / `g_e02_e03_qc.yaml` |
| `temporal_smoke.yaml` | Temporal development smoke protocol, separate from full study | `g_temporal_smoke.yaml` |
| `comparisons/endovla_bbox_action_sft_low_memory.yaml` | Lower-memory four-camera SFT configuration | `qwen2vl_7b_ar_e03_4cam_cont16_sft_v1_76g_fallback.yaml` |
| `comparisons/endovla_scenario_sft.yaml` / `endovla_polyp_sft.yaml` | Supporting autoregressive SFT configurations | `qwen2vl_7b_ar_e03_scenario_static_v1.yaml` / `qwen2vl_7b_ar_polyp.yaml` |

All paths in this table are relative to `configs/`. The controlled ablation
`full` run and the final Ours checkpoint remain different saved runs.

### Recorded artifact compatibility

The old Python API names and source-file provenance comments are removed from
the public implementation. A small recorded-path map in `config.py` translates
references inside original YAML files to their descriptive public filenames.

YAML payloads remain byte-identical so existing config hashes, checkpoints,
cache identities and experiment receipts remain valid. Therefore recorded
fields such as `study_id`, `kind`, `experiment_id`, `feature_level`, stage keys,
cache filenames, and external asset paths still contain historical identifiers.
They are serialized provenance, not the public naming convention.

In Python, `GrootTemporalCacheSchema` uses `visual_shape` and
`visual_source_hook`. `to_dict()` writes the original schema, and
`from_dict()` reads it without changing its SHA-256. Checkpoint tensor keys are
unchanged. Runtime bundle identity checks continue to distinguish the original
one-query baseline from the final shared detector and its provisional rule.
