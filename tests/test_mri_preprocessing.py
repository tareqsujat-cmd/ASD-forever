"""
Unit tests for the MRI preprocessing module.

These tests use synthetic numpy arrays — no actual MRI data or GPU required.
Run with:  pytest tests/test_mri_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_volume():
    """96^3 float32 MRI-like volume with a bright 'brain' region."""
    vol = np.zeros((96, 96, 96), dtype=np.float32)
    vol[20:76, 20:76, 20:76] = np.random.default_rng(0).normal(1000, 100, (56, 56, 56))
    vol = np.clip(vol, 0, None)
    return vol


@pytest.fixture
def brain_mask():
    """Binary mask matching the bright region in synthetic_volume."""
    mask = np.zeros((96, 96, 96), dtype=np.float32)
    mask[20:76, 20:76, 20:76] = 1.0
    return mask


@pytest.fixture
def identity_affine():
    """Identity affine with 2mm isotropic voxels."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return affine


# ---------------------------------------------------------------------------
# NIfTI utils
# ---------------------------------------------------------------------------

class TestNiftiUtils:
    def test_save_and_load_roundtrip(self, synthetic_volume, identity_affine, tmp_path):
        from preprocessing.mri.nifti_utils import save_nifti, load_nifti
        path = tmp_path / "test.nii.gz"
        save_nifti(synthetic_volume, identity_affine, path)
        loaded, loaded_affine = load_nifti(path)
        assert loaded.shape == synthetic_volume.shape
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, synthetic_volume, rtol=1e-5)

    def test_get_voxel_size(self, identity_affine):
        from preprocessing.mri.nifti_utils import get_voxel_size
        vox = get_voxel_size(identity_affine)
        np.testing.assert_allclose(vox, [2.0, 2.0, 2.0], atol=1e-6)

    def test_pad_or_crop_pads_when_smaller(self):
        from preprocessing.mri.nifti_utils import pad_or_crop_to_shape
        vol = np.ones((64, 64, 64), dtype=np.float32)
        out = pad_or_crop_to_shape(vol, (96, 96, 96))
        assert out.shape == (96, 96, 96)

    def test_pad_or_crop_crops_when_larger(self):
        from preprocessing.mri.nifti_utils import pad_or_crop_to_shape
        vol = np.ones((120, 120, 120), dtype=np.float32)
        out = pad_or_crop_to_shape(vol, (96, 96, 96))
        assert out.shape == (96, 96, 96)

    def test_pad_or_crop_exact_shape_unchanged(self):
        from preprocessing.mri.nifti_utils import pad_or_crop_to_shape
        vol = np.ones((96, 96, 96), dtype=np.float32)
        out = pad_or_crop_to_shape(vol, (96, 96, 96))
        assert out.shape == (96, 96, 96)
        np.testing.assert_array_equal(out, vol)

    def test_apply_brain_mask_zeros_background(self, synthetic_volume, brain_mask):
        from preprocessing.mri.nifti_utils import apply_brain_mask
        masked = apply_brain_mask(synthetic_volume, brain_mask)
        # Background should be zero
        assert masked[0, 0, 0] == 0.0
        # Inside brain should be unchanged
        np.testing.assert_allclose(
            masked[20:76, 20:76, 20:76],
            synthetic_volume[20:76, 20:76, 20:76],
        )

    def test_crop_to_nonzero_finds_bounding_box(self, brain_mask):
        from preprocessing.mri.nifti_utils import crop_to_nonzero
        cropped, slices = crop_to_nonzero(brain_mask, margin=0)
        # Should be close to 56^3
        assert all(s > 30 for s in cropped.shape)

    def test_compute_brain_volume_sanity(self, brain_mask):
        from preprocessing.mri.nifti_utils import compute_brain_volume
        vol = compute_brain_volume(brain_mask, voxel_size_mm=2.0)
        # 56^3 voxels at 2mm = 56^3 * 0.008 cm³ ≈ 1404 cm³
        assert 50 < vol < 5000  # wide range for sanity check

    def test_nifti_load_missing_file_raises(self):
        from preprocessing.mri.nifti_utils import load_nifti
        with pytest.raises(FileNotFoundError):
            load_nifti("/nonexistent/path.nii.gz")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestIntensityNormalization:
    def test_zscore_zero_mean_unit_std_on_brain(self, synthetic_volume, brain_mask):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="z_score", site_aware=False)
        norm.fit([synthetic_volume], [brain_mask])
        result = norm.transform(synthetic_volume, brain_mask)
        brain_vals = result[brain_mask > 0]
        assert abs(brain_vals.mean()) < 0.1
        assert abs(brain_vals.std() - 1.0) < 0.1

    def test_minmax_range_01(self, synthetic_volume, brain_mask):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="min_max", site_aware=False)
        norm.fit([synthetic_volume], [brain_mask])
        result = norm.transform(synthetic_volume, brain_mask)
        brain_vals = result[brain_mask > 0]
        assert brain_vals.min() >= -0.01
        assert brain_vals.max() <= 1.01

    def test_percentile_clipping(self, synthetic_volume, brain_mask):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="percentile", site_aware=False)
        norm.fit([synthetic_volume], [brain_mask])
        result = norm.transform(synthetic_volume, brain_mask)
        brain_vals = result[brain_mask > 0]
        assert brain_vals.min() >= -0.01
        assert brain_vals.max() <= 1.01

    def test_transform_before_fit_raises(self, synthetic_volume):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="z_score")
        with pytest.raises(RuntimeError):
            norm.transform(synthetic_volume)

    def test_background_stays_zero_after_normalization(self, synthetic_volume, brain_mask):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="z_score", site_aware=False)
        norm.fit([synthetic_volume], [brain_mask])
        result = norm.transform(synthetic_volume, brain_mask)
        # Background voxels (outside mask) must remain zero
        bg = result[brain_mask == 0]
        assert np.all(bg == 0.0)

    def test_site_aware_fit_stores_per_site_stats(self, synthetic_volume, brain_mask):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="z_score", site_aware=True)
        norm.fit(
            [synthetic_volume, synthetic_volume],
            [brain_mask, brain_mask],
            site_ids=["NYU", "UCLA"],
        )
        assert "NYU" in norm._site_stats
        assert "UCLA" in norm._site_stats

    def test_save_load_roundtrip(self, synthetic_volume, brain_mask, tmp_path):
        from preprocessing.mri.normalization import IntensityNormalizer
        norm = IntensityNormalizer(method="z_score", site_aware=False)
        norm.fit([synthetic_volume], [brain_mask])
        path = tmp_path / "norm.pkl"
        norm.save(path)
        loaded = IntensityNormalizer.load(path)
        r1 = norm.transform(synthetic_volume, brain_mask)
        r2 = loaded.transform(synthetic_volume, brain_mask)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Quality Control
# ---------------------------------------------------------------------------

class TestMRIQualityControl:
    def test_high_quality_scan_passes(self, synthetic_volume, brain_mask):
        from preprocessing.mri.quality_control import MRIQualityChecker
        # Relax thresholds so synthetic volume passes
        checker = MRIQualityChecker(
            snr_threshold=0.1,
            cnr_threshold=0.0,
            fber_threshold=0.0,
            efc_threshold=1.0,
            brain_volume_min_cm3=1.0,
            brain_volume_max_cm3=1e9,
        )
        report = checker.evaluate(synthetic_volume, brain_mask, "sub_001")
        assert report.passed

    def test_zero_volume_fails(self, brain_mask):
        from preprocessing.mri.quality_control import MRIQualityChecker
        checker = MRIQualityChecker(snr_threshold=1.0)
        zero_vol = np.zeros((96, 96, 96), dtype=np.float32)
        report = checker.evaluate(zero_vol, brain_mask, "sub_bad")
        assert not report.passed

    def test_qc_report_to_dict(self, synthetic_volume, brain_mask):
        from preprocessing.mri.quality_control import MRIQualityChecker
        checker = MRIQualityChecker(
            snr_threshold=0.0, cnr_threshold=0.0, fber_threshold=0.0,
            efc_threshold=1.0, brain_volume_min_cm3=1.0, brain_volume_max_cm3=1e9,
        )
        report = checker.evaluate(synthetic_volume, brain_mask, "sub_001")
        d = report.to_dict()
        assert "subject_id" in d
        assert "snr" in d
        assert "passed" in d


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestMRIAugmentation:
    def test_augmentation_preserves_shape(self, synthetic_volume):
        from preprocessing.mri.augmentation import MRIAugmentor
        aug = MRIAugmentor(enabled=True, seed=42)
        out = aug(synthetic_volume)
        assert out.shape == synthetic_volume.shape

    def test_flip_changes_values(self, synthetic_volume):
        from preprocessing.mri.augmentation import MRIAugmentor
        aug = MRIAugmentor(enabled=True, flip_prob=1.0, rotation_degrees=0,
                           noise_std=0, cutout_prob=0, seed=0)
        out = aug(synthetic_volume)
        assert not np.allclose(out, synthetic_volume)

    def test_disabled_augmentation_returns_copy(self, synthetic_volume):
        from preprocessing.mri.augmentation import MRIAugmentor
        aug = MRIAugmentor(enabled=False)
        out = aug(synthetic_volume)
        np.testing.assert_array_equal(out, synthetic_volume)

    def test_augmentation_deterministic_with_seed(self, synthetic_volume):
        from preprocessing.mri.augmentation import MRIAugmentor
        aug1 = MRIAugmentor(enabled=True, seed=123)
        aug2 = MRIAugmentor(enabled=True, seed=123)
        out1 = aug1(synthetic_volume)
        out2 = aug2(synthetic_volume)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestMRIDataset:
    def _make_dataset(self, tmp_path, n=4):
        """Create minimal MRIDataset with synthetic volumes saved to disk."""
        from preprocessing.mri.nifti_utils import save_nifti
        from preprocessing.mri.dataset import MRIDataset
        import pandas as pd

        rows = []
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        for i in range(n):
            sid = f"sub_{i:04d}"
            lbl = i % 2
            vol = np.random.default_rng(i).normal(500, 100, (96, 96, 96)).astype(np.float32)
            vol = np.clip(vol, 0, None)
            out = tmp_path / f"{sid}_preprocessed.nii.gz"
            save_nifti(vol, affine, out)
            rows.append({"subject_id": sid, "label": lbl, "site_name": "TestSite"})

        df = pd.DataFrame(rows)
        ds = MRIDataset(df, processed_dir=tmp_path, target_shape=(96, 96, 96))
        return ds

    def test_dataset_len(self, tmp_path):
        ds = self._make_dataset(tmp_path, n=4)
        assert len(ds) == 4

    def test_dataset_getitem_shapes(self, tmp_path):
        import torch
        ds = self._make_dataset(tmp_path, n=4)
        item = ds[0]
        assert item["image"].shape == (1, 96, 96, 96)
        assert item["label"].dtype == torch.long
        assert isinstance(item["subject"], str)

    def test_dataset_all_labels_valid(self, tmp_path):
        ds = self._make_dataset(tmp_path, n=4)
        for i in range(len(ds)):
            assert ds[i]["label"].item() in (0, 1)

    def test_weighted_sampler_weights_shape(self, tmp_path):
        import torch
        ds = self._make_dataset(tmp_path, n=4)
        weights = ds.get_class_weights()
        assert weights.shape == (4,)
        assert weights.min() > 0
