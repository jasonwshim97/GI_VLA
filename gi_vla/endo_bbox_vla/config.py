"""Configuration contracts and external asset path resolution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from endo_bbox_vla.train import TrainingVariant

from pathlib import Path as _ProjectPath

PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]


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
        from endo_bbox_vla.utils import _resolve_repo_path

        return _resolve_repo_path(self.section(section)[key])


def _variant_config(
    config: ExperimentConfig, variant: TrainingVariant
) -> dict[str, Any]:
    variants = config.section("training").get("variants")
    if not isinstance(variants, Mapping) or variant not in variants:
        raise KeyError(f"unknown shared detector variant: {variant!r}")
    value = variants[variant]
    if not isinstance(value, Mapping):
        raise TypeError(f"shared detector variant {variant!r} must be a mapping")
    return dict(value)


PROMPT_FRACTIONS = {"actual": 0.5, "counterfactual": 0.25, "neutral": 0.25}

CANONICAL_TASK_PROMPTS = (
    "Move the endoscope forward into the proximal stomach.",
    "Move the endoscope forward to the pylorus.",
    "Move the endoscope around to inspect the stomach.",
    "Move the endoscope to center the polyp.",
    "Pull the endoscope back and remove it through the mouth.",
)

NEUTRAL_PROMPT = "Locate any visible polyp in the endoscopic image."

TASK1_SOURCE_DATASET = "ENDO_task1_Withdraw_400episode"

TASK4_SOURCE_DATASET = "Task4_Polyp_1to8"


def _variant_names(config: ExperimentConfig) -> tuple[TrainingVariant, ...]:
    variants = config.section("training").get("variants")
    if not isinstance(variants, Mapping):
        raise TypeError("shared detector training variants must be a mapping")
    return tuple((str(name) for name in variants))


contracts_TASK1_SOURCE_DATASET = "ENDO_task1_Withdraw_400episode"

contracts_TASK4_SOURCE_DATASET = "Task4_Polyp_1to8"


CLIP_LENGTH = 4


contracts_CANONICAL_TASK_PROMPTS = (
    "Move the endoscope forward into the proximal stomach.",
    "Move the endoscope forward to the pylorus.",
    "Move the endoscope around to inspect the stomach.",
    "Move the endoscope to center the polyp.",
    "Pull the endoscope back and remove it through the mouth.",
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class SharedDetectorSpec:
    path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def base_policy(self) -> dict[str, Any]:
        return _mapping(self.raw["base_policy"], "base_policy")

    @property
    def processor(self) -> dict[str, Any]:
        return _mapping(self.raw["processor"], "processor")

    @property
    def detector(self) -> dict[str, Any]:
        return _mapping(self.raw["detector"], "detector")

    @property
    def event_rule(self) -> dict[str, Any]:
        return _mapping(self.raw["event_rule"], "event_rule")

    @property
    def runtime(self) -> dict[str, Any]:
        return _mapping(self.raw["runtime"], "runtime")

    @property
    def checkpoint_path(self) -> Path:
        return self.path.parent / str(self.detector["filename"])

    @property
    def processor_path(self) -> Path:
        return self.path.parent / str(self.processor["directory"])


def _digest(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        (character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _git_commit(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 40 or any(
        (character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git commit")
    return result


def _validate(spec: SharedDetectorSpec) -> None:
    # Recorded IDs below are serialized provenance, not public model names.
    if spec.raw.get("schema_version") != 1:
        raise ValueError("unsupported online deployment schema")
    if spec.raw.get("kind") != "groot_n17_e03_polyp_online_eval_bundle":
        raise ValueError("unexpected online deployment kind")
    base = spec.base_policy
    if (
        base.get("repo_id")
        != "Jason-Doldol-Research/task1_5_100_groot_n17_relact_005000"
    ):
        raise ValueError("online bundle requires the frozen five-task GR00T checkpoint")
    if base.get("revision") != "efbfbb0b4d13bf482734ead3d77d1ad2ceceac1b":
        raise ValueError("online bundle requires the pinned GR00T revision")
    _digest(base.get("model_sha256"), "base_policy.model_sha256")
    _git_commit(base.get("lerobot_commit"), "base_policy.lerobot_commit")
    if int(base.get("model_bytes", -1)) != 12576215296:
        raise ValueError("base policy byte-size contract changed")
    if (int(base.get("action_chunk_size", -1)), int(base.get("action_dim", -1))) != (
        40,
        16,
    ):
        raise ValueError("GR00T action shape contract must be [B,40,16]")
    if base.get("use_relative_actions") is not True:
        raise ValueError("the pinned GR00T action checkpoint uses relative actions")
    processor = spec.processor
    processor_hashes = _mapping(processor.get("files_sha256"), "processor.files_sha256")
    expected_processor_files = {
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
    if set(processor_hashes) != expected_processor_files:
        raise ValueError("processor file inventory changed")
    for filename, digest in processor_hashes.items():
        _digest(digest, f"processor.files_sha256.{filename}")
    detector = spec.detector
    experiment_id = detector.get("experiment_id")
    if experiment_id not in {
        "g_e03_mid_static_dense",
        "e03_shared_location_b",
        "e03_shared_location_c",
    }:
        raise ValueError(
            "online detector must be a validation-selected shared detector model"
        )
    if detector.get("feature_level") != "e03":
        raise ValueError("online detector must use the block-11 visual feature level")
    if detector.get("camera_key") != "observation.images.cam_low":
        raise ValueError("online detector must use cam_low")
    if tuple(detector.get("policy_camera_order", ())) != (
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        raise ValueError("GR00T policy camera order changed")
    _digest(detector.get("sha256"), "detector.sha256")
    _digest(detector.get("metadata_sha256"), "detector.metadata_sha256")
    _digest(
        detector.get("training_manifest_sha256"), "detector.training_manifest_sha256"
    )
    _digest(
        detector.get("training_cache_manifest_sha256"),
        "detector.training_cache_manifest_sha256",
    )
    threshold = float(detector.get("threshold", -1))
    if experiment_id == "g_e03_mid_static_dense":
        if threshold != 0.054931640625:
            raise ValueError("block-11 visual validation-selected threshold changed")
    elif not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "shared detector validation-selected threshold must be in [0,1]"
        )
    architecture = _mapping(detector.get("architecture"), "detector.architecture")
    one_query_architecture = {
        "type": "one_query_detr",
        "input_dim": 1024,
        "hidden_dim": 256,
        "decoder_layers": 3,
        "attention_heads": 8,
        "feedforward_dim": 1024,
        "grid_size": [16, 16],
        "padded_rows": [0, 0],
        "dropout": 0.1,
    }
    location_first_architecture = {
        "type": "location_first_four_query",
        "input_dim": 1024,
        "hidden_dim": 256,
        "decoder_layers": 3,
        "attention_heads": 8,
        "feedforward_dim": 1024,
        "grid_size": [16, 16],
        "location_queries": 4,
        "dropout": 0.1,
    }
    expected_architecture = (
        one_query_architecture
        if experiment_id == "g_e03_mid_static_dense"
        else location_first_architecture
    )
    if architecture != expected_architecture:
        raise ValueError("block-11 visual detector architecture changed")
    if experiment_id != "g_e03_mid_static_dense":
        shared_required_sha = (
            "training_config_sha256",
            "validation_cache_manifest_sha256",
            "action_parity_sha256",
        )
        for name in shared_required_sha:
            _digest(detector.get(name), f"detector.{name}")
        if detector.get("variant") not in {"b", "c"}:
            raise ValueError("shared detector detector variant must be b or c")
    rule = spec.event_rule
    common_rule = {
        "window_size": 15,
        "require_full_window": True,
        "cadence_hz": 30.0,
        "bbox_iou_stability": None,
        "internal_test_used_for_selection": False,
    }
    if experiment_id == "g_e03_mid_static_dense":
        expected_rule = {"name": "kofn_12_15", "hits_required": 12, **common_rule}
    elif rule.get("status") == "requires_online_revalidation_for_new_head":
        expected_rule = {"name": "kofn_12_15", "hits_required": 12, **common_rule}
    else:
        expected_rule = {"name": "kofn_10_15", "hits_required": 10, **common_rule}
    mismatches = {
        key: {"expected": value, "actual": rule.get(key)}
        for key, value in expected_rule.items()
        if rule.get(key) != value
    }
    if mismatches:
        raise ValueError(f"provisional event-rule contract changed: {mismatches}")
    if experiment_id == "g_e03_mid_static_dense":
        if rule.get("status") != "provisional_validation_selected":
            raise ValueError("event rule must remain explicitly provisional")
        if int(rule.get("validation_false_trigger_episodes", -1)) != 2:
            raise ValueError("event-rule validation summary changed")
    else:
        status = rule.get("status")
        if status == "requires_online_revalidation_for_new_head":
            if rule.get("validation_false_trigger_episodes") is not None:
                raise ValueError(
                    "old-head false-trigger results cannot be assigned to the new head"
                )
        elif status == "provisional_validation_selected":
            expected_summary = {
                "selection_split": "Task1 validation episodes 150-169",
                "validation_negative_episodes": 20,
                "validation_frames": 20226,
                "validation_false_trigger_episodes": 7,
                "validation_false_bursts": 20,
                "task4_diagnostic_episodes": 5,
                "task4_diagnostic_triggered": 5,
                "task4_diagnostic_p95_delay_seconds": 0.4666666666666667,
                "selection_analysis_sha256": "80fd41001dc257e0a5815685f62c5e4abb6b8d3a4eb84b5ad0b883d9b1c90ca9",
            }
            summary_mismatches = {
                key: {"expected": value, "actual": rule.get(key)}
                for key, value in expected_summary.items()
                if rule.get(key) != value
            }
            if summary_mismatches:
                raise ValueError(
                    f"shared event-rule validation summary changed: {summary_mismatches}"
                )
            _digest(
                rule.get("selection_analysis_sha256"),
                "event_rule.selection_analysis_sha256",
            )
        else:
            raise ValueError("shared detector event-rule status changed")
    if spec.runtime.get("clinical_use") is not False:
        raise ValueError("research online bundle must prohibit clinical use")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_online_spec(path: str | Path) -> SharedDetectorSpec:
    source = Path(path).resolve()
    payload = source.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise TypeError("online deployment config root must be a mapping")
    spec = SharedDetectorSpec(path=source, sha256=_sha256(payload), raw=dict(raw))
    _validate(spec)
    return spec


STATE_KEY = "observation.state"

TASK_KEY = "task"

CAMERA_KEYS = (
    "observation.images.cam_low",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

STATE_DIM = 16


# Public configuration/path API.
import os as _path_os
import sys as _path_sys

CONFIG_ROOT = PROJECT_ROOT / "configs"
if not CONFIG_ROOT.is_dir():
    CONFIG_ROOT = Path(_path_sys.prefix) / "share" / "endo_bbox_vla" / "configs"


@dataclass(frozen=True, slots=True)
class PaperModel:
    """Public model identity and its link to the completed research records."""

    key: str
    label: str
    role: str
    recorded_ids: tuple[str, ...]
    config: str | None = None
    feature_mode: str | None = None


PAPER_MODELS = {
    "ours": PaperModel(
        "ours",
        "Ours",
        "Final static shared detector",
        ("e03_shared_location_c", "Shared E03-C"),
        "ours.yaml",
    ),
    "temporal_visual_memory": PaperModel(
        "temporal_visual_memory",
        "Temporal (visual memory)",
        "Completed temporal extension using block-11 visual features",
        ("T1_e03_temporal_full", "T1"),
        feature_mode="mid_only",
    ),
    "temporal_vl_conditioned": PaperModel(
        "temporal_vl_conditioned",
        "Temporal (VL-conditioned)",
        "Completed temporal comparison with vision-language conditioning",
        ("T2_e03_dit_temporal_full", "T2"),
        feature_mode="dit_conditioned",
    ),
    "endovla_bbox_sft": PaperModel(
        "endovla_bbox_sft",
        "EndoVLA-inspired bbox-only SFT",
        "Single-camera autoregressive comparison",
        ("qwen2vl_7b_ar_e03_1cam_bbox_only_sft_v1",),
        "comparisons/endovla_bbox_sft.yaml",
    ),
    "endovla_bbox_action_sft": PaperModel(
        "endovla_bbox_action_sft",
        "EndoVLA-inspired bbox+action SFT",
        "Four-camera autoregressive comparison with continuous-action targets",
        ("qwen2vl_7b_ar_e03_4cam_cont16_sft_v1",),
        "comparisons/endovla_bbox_action_sft.yaml",
    ),
    "faster_rcnn": PaperModel(
        "faster_rcnn",
        "Faster R-CNN R50-FPN",
        "Standalone detector comparison",
        ("faster_rcnn",),
        "standalone_detectors.yaml",
    ),
    "detr_r50": PaperModel(
        "detr_r50",
        "DETR-R50",
        "Standalone detector comparison",
        ("detr_r50",),
        "standalone_detectors.yaml",
    ),
}

TRAINING_STAGES = {
    "spatial_queries": "b",
    "hard_positive": "c",
    "single_query": "single_query",
    "learned_queries": "learned_4query",
    "spatial_queries_ablation": "location_first",
    "full_ablation": "full",
    "single_query_full_ablation": "full_k1",
}

# Only immutable serialized records use these historical configuration paths.
RECORDED_CONFIG_PATHS = {
    "groot_cross_backbone/e03_shared_location_v1.yaml": "ours.yaml",
    "groot_cross_backbone/e03_shared_ablation_v1.yaml": "ablation_components.yaml",
    "groot_cross_backbone/e03_shared_query_count_ablation_v1.yaml": "ablation_query_count.yaml",
    "groot_cross_backbone/e03_separate_detector_efficiency_v1.yaml": "standalone_detectors.yaml",
    "groot_cross_backbone/e03_hr_abcd_v1.yaml": "high_resolution_upper_bound.yaml",
    "groot_cross_backbone/g_e02_e03_scenario_static_v1.yaml": "feature_comparison.yaml",
    "groot_cross_backbone/g_e02_e03_qc.yaml": "feature_comparison_qc.yaml",
    "groot_cross_backbone/g_temporal_smoke.yaml": "temporal_smoke.yaml",
    "groot_e03_online_eval_v1.yaml": "baseline_single_query_runtime.yaml",
    "groot_e03_shared_online_eval_v1.yaml": "ours_runtime.yaml",
    "ar_qwen/qwen2vl_7b_ar_e03_1cam_bbox_only_sft_v1.yaml": "comparisons/endovla_bbox_sft.yaml",
    "ar_qwen/qwen2vl_7b_ar_e03_4cam_cont16_sft_v1.yaml": "comparisons/endovla_bbox_action_sft.yaml",
    "ar_qwen/qwen2vl_7b_ar_e03_4cam_cont16_sft_v1_76g_fallback.yaml": "comparisons/endovla_bbox_action_sft_low_memory.yaml",
    "ar_qwen/qwen2vl_7b_ar_e03_scenario_static_v1.yaml": "comparisons/endovla_scenario_sft.yaml",
    "ar_qwen/qwen2vl_7b_ar_polyp.yaml": "comparisons/endovla_polyp_sft.yaml",
}


def paper_model(key: str) -> PaperModel:
    """Look up a paper-facing identity without loading data or weights."""
    try:
        return PAPER_MODELS[key]
    except KeyError as error:
        raise ValueError(
            f"Unknown paper model {key!r}; choose from {tuple(PAPER_MODELS)}"
        ) from error


def resolve_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve recorded paths without editing their hash-bound config payloads."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    text = path.as_posix()
    for prefix, subdir in (
        ("bbox_vla/detr/configs/", ""),
        ("bbox_vla/ar_qwen/configs/", "ar_qwen/"),
        ("configs/", ""),
    ):
        if text.startswith(prefix):
            relative = subdir + text[len(prefix) :]
            return (
                CONFIG_ROOT / RECORDED_CONFIG_PATHS.get(relative, relative)
            ).resolve()
    data_root = _path_os.environ.get("ENDO_BBOX_VLA_DATA_ROOT")
    if data_root and text.startswith("bbox_vla/"):
        return (Path(data_root) / text[len("bbox_vla/") :]).resolve()
    return (Path(root) / path if root is not None else PROJECT_ROOT / path).resolve()
