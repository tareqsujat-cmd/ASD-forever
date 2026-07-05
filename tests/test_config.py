"""
Unit tests for the configuration loading system.

These tests do NOT require GPU or any medical imaging libraries.
Run with:  pytest tests/test_config.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_schema import Config, load_config, override_config


SAMPLE_YAML = """
project:
  name: "TEST_PROJECT"
  random_seed: 123
  device: "cpu"
  mixed_precision: false

paths:
  root: "{tmpdir}"
  data_raw_mri: "datasets/raw/mri"
  data_raw_genetics: "datasets/raw/genetics"
  data_processed_mri: "datasets/processed/mri"
  data_processed_genetics: "datasets/processed/genetics"
  splits: "datasets/splits"
  saved_models: "saved_models"
  results: "results"
  figures: "results/figures"
  tables: "results/tables"
  reports: "results/reports"
  paper_figures: "paper/figures"
  logs: "results/logs"

training:
  learning_rate: 5.0e-5
  batch_size: 4

fusion:
  method: "gated"
"""


@pytest.fixture
def tmp_config(tmp_path):
    """Write a minimal valid config YAML to a temp file."""
    content = SAMPLE_YAML.format(tmpdir=str(tmp_path).replace("\\", "/"))
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path, tmp_path


class TestConfigLoading:
    def test_load_returns_config_instance(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        assert isinstance(cfg, Config)

    def test_project_fields_parsed(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        assert cfg.project.name == "TEST_PROJECT"
        assert cfg.project.random_seed == 123
        assert cfg.project.device == "cpu"
        assert cfg.project.mixed_precision is False

    def test_nested_training_config(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        assert cfg.training.learning_rate == pytest.approx(5e-5)
        assert cfg.training.batch_size == 4

    def test_fusion_method_override(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        assert cfg.fusion.method == "gated"

    def test_defaults_filled_for_missing_keys(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        # These keys are not in SAMPLE_YAML; must use defaults
        assert cfg.training.max_epochs == 100
        assert cfg.training.optimizer == "adamw"
        assert cfg.evaluation.bootstrap_iterations == 1000

    def test_output_dirs_created(self, tmp_config):
        cfg_path, tmp_path = tmp_config
        cfg = load_config(cfg_path)
        assert (tmp_path / "results" / "figures").exists()
        assert (tmp_path / "datasets" / "splits").exists()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestOverrideConfig:
    def test_single_override(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        cfg2 = override_config(cfg, {"training.learning_rate": 1e-3})
        assert cfg2.training.learning_rate == pytest.approx(1e-3)
        # Original unchanged (deep copy)
        assert cfg.training.learning_rate == pytest.approx(5e-5)

    def test_multiple_overrides(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        cfg2 = override_config(cfg, {
            "fusion.method": "cross_attention",
            "training.batch_size": 16,
            "project.random_seed": 999,
        })
        assert cfg2.fusion.method == "cross_attention"
        assert cfg2.training.batch_size == 16
        assert cfg2.project.random_seed == 999

    def test_deep_copy_isolation(self, tmp_config):
        cfg_path, _ = tmp_config
        cfg = load_config(cfg_path)
        cfg2 = override_config(cfg, {"training.optimizer": "sgd"})
        assert cfg.training.optimizer == "adamw"
        assert cfg2.training.optimizer == "sgd"


class TestReproducibility:
    def test_seed_everything_runs(self):
        from utilities.reproducibility import seed_everything
        seed_everything(42)  # Should not raise

    def test_derive_seed_deterministic(self):
        from utilities.reproducibility import derive_seed
        s1 = derive_seed("fold_1", base_seed=42)
        s2 = derive_seed("fold_1", base_seed=42)
        s3 = derive_seed("fold_2", base_seed=42)
        assert s1 == s2
        assert s1 != s3

    def test_derive_seed_range(self):
        from utilities.reproducibility import derive_seed
        for scope in ["a", "b", "c", "train", "augmentation"]:
            s = derive_seed(scope, 42)
            assert 0 <= s < 2**32

    def test_scoped_seed_restores_state(self):
        import random
        import numpy as np
        from utilities.reproducibility import ScopedSeed
        random.seed(999)
        before = random.random()
        random.seed(999)  # Reset
        with ScopedSeed("test", 42):
            _ = random.random()  # Consume inside scope
        after = random.random()  # Should match 'before'
        assert abs(before - after) < 1e-10
