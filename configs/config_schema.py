"""
Configuration schema using Python dataclasses.

Mirrors configs/config.yaml exactly.  Every field has a type annotation so
that mistakes (e.g. passing a string where a float is expected) are caught at
import time, not mid-training.

Usage
-----
    from configs.config_schema import load_config
    cfg = load_config("configs/config.yaml")
    print(cfg.training.learning_rate)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Leaf-level dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str = "ASD_Multimodal_Detection"
    version: str = "1.0.0"
    description: str = ""
    random_seed: int = 42
    device: str = "auto"          # auto: cuda → mps → cpu
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PathsConfig:
    root: str = "."          # repo root (OS-agnostic; was a hardcoded Windows path)
    data_raw_mri: str = "datasets/raw/mri"
    data_raw_genetics: str = "datasets/raw/genetics"
    data_processed_mri: str = "datasets/processed/mri"
    data_processed_genetics: str = "datasets/processed/genetics"
    splits: str = "datasets/splits"
    saved_models: str = "saved_models"
    results: str = "results"
    figures: str = "results/figures"
    tables: str = "results/tables"
    reports: str = "results/reports"
    paper_figures: str = "paper/figures"
    logs: str = "results/logs"

    def abs(self, relative_key: str) -> Path:
        """Return absolute Path for a relative sub-path attribute."""
        return Path(self.root) / getattr(self, relative_key)


@dataclass
class DatasetConfig:
    name: str = "ABIDE_I"
    mri_modality: str = "func_preproc"
    genetic_type: str = "gene_expression"
    abide_pipeline: str = "cpac"
    abide_strategy: str = "filt_global"
    target_column: str = "DX_GROUP"
    class_map: Dict[str, int] = field(default_factory=lambda: {"ASD": 1, "TC": 0})
    test_size: float = 0.15
    val_size: float = 0.15
    stratify: bool = True


@dataclass
class AugmentationConfig:
    enabled: bool = True
    flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    noise_std: float = 0.01
    elastic_deform: bool = False


@dataclass
class MRIPreprocessingConfig:
    target_shape: List[int] = field(default_factory=lambda: [96, 96, 96])
    target_voxel_size: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    intensity_norm: str = "z_score"
    apply_brain_mask: bool = True
    apply_bias_correction: bool = True
    smooth_fwhm: float = 6.0
    quality_threshold: float = 0.8
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class DimReductionConfig:
    method: str = "pca"
    n_components: int = 256
    explained_variance_threshold: float = 0.95


@dataclass
class GeneticsPreprocessingConfig:
    missing_threshold: float = 0.2
    variance_threshold: float = 0.01
    normalization: str = "robust"
    selection_method: str = "combined"
    n_top_features: int = 1000
    batch_correction: str = "combat"
    dimensionality_reduction: DimReductionConfig = field(default_factory=DimReductionConfig)


@dataclass
class Swin3DConfig:
    embed_dim: int = 96
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: List[int] = field(default_factory=lambda: [7, 7, 7])
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2


@dataclass
class DenseNet3DConfig:
    growth_rate: int = 32
    block_config: List[int] = field(default_factory=lambda: [6, 12, 24, 16])
    num_init_features: int = 64
    bn_size: int = 4
    drop_rate: float = 0.2


@dataclass
class MRIModelConfig:
    backbone: str = "medicalnet_resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    feature_dim: int = 512
    dropout: float = 0.3
    swin3d: Swin3DConfig = field(default_factory=Swin3DConfig)
    densenet3d: DenseNet3DConfig = field(default_factory=DenseNet3DConfig)


@dataclass
class GeneticsModelConfig:
    architecture: str = "transformer"
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    feature_dim: int = 256
    dropout: float = 0.3
    num_heads: int = 8
    num_layers: int = 4


@dataclass
class GatedFusionConfig:
    gate_activation: str = "sigmoid"


@dataclass
class CrossAttentionConfig:
    num_layers: int = 2
    ffn_dim: int = 1024


@dataclass
class FusionConfig:
    method: str = "cross_attention"
    mri_feature_dim: int = 512
    genetics_feature_dim: int = 256
    fusion_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.3
    num_classes: int = 2
    gated: GatedFusionConfig = field(default_factory=GatedFusionConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 15
    monitor: str = "val_auc"
    mode: str = "max"
    min_delta: float = 1e-4


@dataclass
class CrossValidationConfig:
    enabled: bool = True
    n_folds: int = 5
    stratified: bool = True
    group_by_site: bool = True


@dataclass
class LossConfig:
    type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    save_top_k: int = 3
    checkpoint_metric: str = "val_auc"


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "specificity",
        "f1", "roc_auc", "pr_auc", "balanced_accuracy", "mcc"
    ])
    bootstrap_iterations: int = 1000
    significance_level: float = 0.05
    plot_roc: bool = True
    plot_pr: bool = True
    plot_confusion: bool = True
    plot_calibration: bool = True


@dataclass
class ExplainabilityConfig:
    methods: List[str] = field(default_factory=lambda: [
        "gradcam", "gradcam_plus_plus", "integrated_gradients", "shap", "attention_rollout"
    ])
    gradcam_layer: str = "auto"
    shap_samples: int = 100
    ig_steps: int = 50
    save_heatmaps: bool = True
    visualize_top_k_genes: int = 30


@dataclass
class HParamTuningConfig:
    enabled: bool = False
    method: str = "optuna"
    n_trials: int = 50
    timeout_hours: float = 12.0
    metric: str = "val_auc"
    direction: str = "maximize"
    pruner: str = "hyperband"


@dataclass
class MLflowConfig:
    enabled: bool = True
    experiment_name: str = "ASD_Multimodal"
    tracking_uri: str = "mlruns"


@dataclass
class WandBConfig:
    enabled: bool = False
    project: str = "asd-multimodal"
    entity: Optional[str] = None


@dataclass
class TrackingConfig:
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    log_to_file: bool = True
    log_dir: str = "results/logs"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mri_preprocessing: MRIPreprocessingConfig = field(default_factory=MRIPreprocessingConfig)
    genetics_preprocessing: GeneticsPreprocessingConfig = field(default_factory=GeneticsPreprocessingConfig)
    mri_model: MRIModelConfig = field(default_factory=MRIModelConfig)
    genetics_model: GeneticsModelConfig = field(default_factory=GeneticsModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    hparam_tuning: HParamTuningConfig = field(default_factory=HParamTuningConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a nested dict into the target dataclass."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue  # ignore unknown keys gracefully

        annotation = cls.__dataclass_fields__[key].type
        # Resolve string annotations (from __future__ annotations)
        if isinstance(annotation, str):
            import typing
            annotation = eval(annotation, {**vars(typing), **globals()})

        origin = getattr(annotation, "__origin__", None)

        if origin is None and isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
            # Nested dataclass
            kwargs[key] = _dict_to_dataclass(annotation, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Config:
    """
    Load and validate configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    Config
        Fully validated Config dataclass instance.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required fields are missing or have wrong types.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = _dict_to_dataclass(Config, raw)

    # Ensure all output directories exist
    root = Path(cfg.paths.root)
    for attr_name in cfg.paths.__dataclass_fields__:
        if attr_name == "root":
            continue
        rel = getattr(cfg.paths, attr_name)
        (root / rel).mkdir(parents=True, exist_ok=True)

    return cfg


def override_config(cfg: Config, overrides: dict) -> Config:
    """
    Apply a flat dict of dot-separated overrides to a Config object.

    Example
    -------
        override_config(cfg, {"training.learning_rate": 5e-5, "fusion.method": "gated"})
    """
    import copy
    cfg = copy.deepcopy(cfg)

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg
