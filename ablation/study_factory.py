"""
Pre-built ablation study factories for the ASD detection framework.

Each factory returns an ``AblationStudy`` ready to be passed to
``AblationRunner.run_study()``.  The override keys match the config
field names used in ``config.yaml`` / ``Config`` dataclass.

Usage
-----
::

    study = build_fusion_ablation(base_cfg)
    runner = AblationRunner(train_fn=trainer.run_cv, save_dir="results/ablation")
    results = runner.run_study(study)
    analyzer = AblationAnalyzer(results)
    print(analyzer.markdown_table())
"""

from __future__ import annotations

from ablation.ablation_config import AblationDimension, AblationStudy


# ---------------------------------------------------------------------------
# Individual dimension definitions
# ---------------------------------------------------------------------------

def _fusion_dimension() -> AblationDimension:
    """Compare all 5 fusion strategies."""
    return AblationDimension(
        name="fusion",
        variants={
            "cross_attention": {"fusion.architecture": "cross_attention"},
            "gated":           {"fusion.architecture": "gated"},
            "late":            {"fusion.architecture": "late"},
            "intermediate":    {"fusion.architecture": "intermediate"},
            "dynamic":         {"fusion.architecture": "dynamic"},
        },
        default="cross_attention",
        description="Multimodal fusion strategy",
    )


def _mri_backbone_dimension() -> AblationDimension:
    """Compare 5 MRI backbone architectures."""
    return AblationDimension(
        name="backbone",
        variants={
            "resnet10":    {"model.mri.architecture": "resnet10"},
            "resnet50":    {"model.mri.architecture": "resnet50"},
            "densenet121": {"model.mri.architecture": "densenet121"},
            "swin3d":      {"model.mri.architecture": "swin3d"},
            "convnext3d":  {"model.mri.architecture": "convnext3d"},
        },
        default="resnet10",
        description="3D MRI feature extractor backbone",
    )


def _genetics_dimension() -> AblationDimension:
    """Compare 4 genetics encoder architectures."""
    return AblationDimension(
        name="genetics",
        variants={
            "transformer": {"model.genetics.architecture": "transformer"},
            "tabnet":      {"model.genetics.architecture": "tabnet"},
            "gnn":         {"model.genetics.architecture": "gnn"},
            "mlp":         {"model.genetics.architecture": "mlp"},
        },
        default="transformer",
        description="Genetics feature encoder architecture",
    )


def _modality_dimension() -> AblationDimension:
    """Ablate individual modalities."""
    return AblationDimension(
        name="modality",
        variants={
            "multimodal":    {"model.modality": "multimodal"},
            "mri_only":      {"model.modality": "mri_only"},
            "genetics_only": {"model.modality": "genetics_only"},
        },
        default="multimodal",
        description="Input modality combination",
    )


def _loss_dimension() -> AblationDimension:
    """Compare loss functions."""
    return AblationDimension(
        name="loss",
        variants={
            "focal":        {"training.loss": "focal"},
            "cross_entropy": {"training.loss": "cross_entropy"},
            "balanced":     {"training.loss": "balanced"},
        },
        default="focal",
        description="Training loss function",
    )


def _ema_dimension() -> AblationDimension:
    """EMA enabled vs disabled."""
    return AblationDimension(
        name="ema",
        variants={
            "enabled":  {"training.use_ema": True},
            "disabled": {"training.use_ema": False},
        },
        default="enabled",
        description="Exponential Moving Average (EMA)",
    )


# ---------------------------------------------------------------------------
# Study factories
# ---------------------------------------------------------------------------

def build_fusion_ablation(base_config, mode: str = "ofat") -> AblationStudy:
    """
    Ablation study for multimodal fusion strategy.

    Compares cross-attention (proposed) vs gated, late, intermediate, dynamic.
    This is the primary ablation for the paper's main contribution.
    """
    return AblationStudy(
        name="fusion_ablation",
        base_config=base_config,
        dimensions=[_fusion_dimension()],
        mode=mode,
    )


def build_backbone_ablation(base_config, mode: str = "ofat") -> AblationStudy:
    """Ablation study for MRI backbone architecture choice."""
    return AblationStudy(
        name="backbone_ablation",
        base_config=base_config,
        dimensions=[_mri_backbone_dimension()],
        mode=mode,
    )


def build_genetics_ablation(base_config, mode: str = "ofat") -> AblationStudy:
    """Ablation study for genetics encoder architecture choice."""
    return AblationStudy(
        name="genetics_ablation",
        base_config=base_config,
        dimensions=[_genetics_dimension()],
        mode=mode,
    )


def build_modality_ablation(base_config, mode: str = "ofat") -> AblationStudy:
    """
    Modality ablation: multimodal vs MRI-only vs genetics-only.

    This is essential for demonstrating that the multimodal approach
    outperforms unimodal baselines.
    """
    return AblationStudy(
        name="modality_ablation",
        base_config=base_config,
        dimensions=[_modality_dimension()],
        mode=mode,
    )


def build_full_ablation(base_config) -> AblationStudy:
    """
    Comprehensive OFAT ablation across all six dimensions.

    Total variants = 1 baseline + 4 fusion + 4 backbone + 3 genetics
                   + 2 modality + 2 loss + 1 ema = 17 variants.
    """
    return AblationStudy(
        name="full_ofat_ablation",
        base_config=base_config,
        dimensions=[
            _fusion_dimension(),
            _mri_backbone_dimension(),
            _genetics_dimension(),
            _modality_dimension(),
            _loss_dimension(),
            _ema_dimension(),
        ],
        mode="ofat",
    )


def build_fusion_backbone_factorial(base_config) -> AblationStudy:
    """
    Factorial study: all combinations of fusion × backbone (5×5 = 25 variants).

    Useful for analysing interaction effects between fusion strategy and
    the MRI representation quality.
    """
    return AblationStudy(
        name="fusion_backbone_factorial",
        base_config=base_config,
        dimensions=[_fusion_dimension(), _mri_backbone_dimension()],
        mode="factorial",
    )
