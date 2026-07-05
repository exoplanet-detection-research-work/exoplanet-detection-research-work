"""Unit tests for the representation layer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from exodet.data.base import LightCurve
from exodet.exceptions import DataError, NotFittedError, PipelineError
from exodet.representation import (
    AUGMENTERS,
    FEATURE_SCALERS,
    PHASE_FOLDERS,
    PHYSICS_EXTRACTORS,
    SPLITTERS,
    VIEW_BUILDERS,
    AugmentationPipeline,
    FeatureScaler,
    GaussianNoiseAugmenter,
    PhaseFolder,
    PhysicsFeatureExtractor,
    RepresentationCache,
    RepresentationDataset,
    StarLevelSplitter,
    fold_phase,
    sample_fingerprint,
)
from exodet.representation.containers import DatasetSample, FeatureVector, View
from exodet.representation.folding import PhaseFolder
from exodet.representation.splitting import (
    CandidateLevelSplitter,
    assert_no_group_leakage,
)
from exodet.representation.views import (
    GlobalViewGenerator,
    LocalViewGenerator,
    _fill_empty_bins,
    bin_folded_curve,
)
from exodet.tce.candidate import TransitCandidate
from tests.conftest import make_synthetic_tess_curve
from tests.test_tce import make_candidate


class TestRegistration:
    def test_registries_populated(self) -> None:
        assert "standard" in PHASE_FOLDERS
        assert "global" in VIEW_BUILDERS
        assert "local" in VIEW_BUILDERS
        assert "standard" in PHYSICS_EXTRACTORS
        assert "standard" in FEATURE_SCALERS
        assert "star" in SPLITTERS
        assert "gaussian_noise" in AUGMENTERS


class TestFoldPhase:
    def test_transit_at_zero(self) -> None:
        period = 2.0
        epoch = 1.0
        times = np.array([epoch, epoch + period, epoch + 0.5 * period])
        phases = fold_phase(times, period, epoch)
        assert phases[0] == pytest.approx(0.0)
        assert phases[1] == pytest.approx(0.0)
        assert phases[2] == pytest.approx(-0.5)  # half period → ±0.5 in centered convention


class TestPhaseFolder:
    def test_folds_and_aligns(self) -> None:
        curve = make_synthetic_tess_curve(defects=False, n_per_sector=800, seed=0)
        candidate = make_candidate(
            period_days=1.3,
            epoch_days=0.5,
            duration_days=0.1,
            target_id=curve.target_id,
        )
        folded = PhaseFolder().fold(curve, candidate)
        assert len(folded) > 100
        assert folded.phase.min() >= -0.5
        assert folded.phase.max() <= 0.5
        assert np.all(np.diff(folded.phase) >= 0)
        assert "phase_fold" in folded.history[-1]

    def test_deduplicates_phases(self) -> None:
        curve = LightCurve(
            target_id="DUP",
            time=np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
            flux=np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        )
        candidate = make_candidate(
            candidate_id="DUP-01", target_id="DUP", period_days=4.0, epoch_days=0.0
        )
        folded = PhaseFolder(deduplicate=True).fold(curve, candidate)
        assert folded.meta["n_duplicates_merged"] >= 1
        assert len(folded) == 4

    def test_invalid_period_raises(self) -> None:
        curve = make_synthetic_tess_curve(defects=False, n_per_sector=50)
        with pytest.raises(PipelineError, match="invalid period"):
            PhaseFolder().fold(curve, make_candidate(period_days=-1.0))

    def test_too_few_points_raises(self) -> None:
        curve = LightCurve(
            target_id="TINY", time=np.arange(3, dtype=float), flux=np.ones(3)
        )
        with pytest.raises(PipelineError, match="at least 5"):
            PhaseFolder().fold(curve, make_candidate(target_id="TINY"))


class TestBinning:
    def test_median_binning(self) -> None:
        phase = np.array([-0.4, -0.35, 0.0, 0.05, 0.4])
        flux = np.array([1.0, 1.1, 0.5, 0.6, 1.0])
        edges = np.linspace(-0.5, 0.5, 6)
        values, counts = bin_folded_curve(phase, flux, edges, statistic="median")
        assert counts.sum() == 5
        assert np.count_nonzero(counts) == 3  # three occupied bins

    def test_weighted_mean(self) -> None:
        phase = np.array([0.0, 0.0, 0.1])
        flux = np.array([1.0, 3.0, 2.0])
        err = np.array([0.1, 0.1, 1.0])
        edges = np.linspace(-0.5, 0.5, 11)
        values, counts = bin_folded_curve(
            phase, flux, edges, statistic="weighted_mean", flux_err=err
        )
        assert counts.sum() == 3
        assert counts.max() >= 2

    def test_fill_empty_bins_linear(self) -> None:
        centers = np.linspace(-0.5, 0.5, 5)
        values = np.array([1.0, np.nan, np.nan, 0.5, 1.0])
        filled = _fill_empty_bins(values, centers, "linear")
        assert np.isfinite(filled).all()
        assert filled[2] == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_fill_empty_bins_nearest(self) -> None:
        centers = np.linspace(-0.5, 0.5, 5)
        values = np.array([1.0, np.nan, np.nan, 0.5, 1.0])
        filled = _fill_empty_bins(values, centers, "nearest")
        assert np.isfinite(filled).all()


class TestViewGenerators:
    @pytest.fixture()
    def folded(self):
        curve = make_synthetic_tess_curve(defects=False, n_per_sector=1200, seed=1)
        candidate = make_candidate(
            period_days=1.3,
            epoch_days=0.3,
            duration_days=0.1,
            target_id=curve.target_id,
        )
        return PhaseFolder().fold(curve, candidate)

    def test_global_view_shape(self, folded) -> None:
        view = GlobalViewGenerator(n_bins=201).generate(folded)
        assert view.n_bins == 201
        assert view.kind == "global"
        assert len(view.values) == 201
        assert view.empty_fraction < 0.5

    def test_local_view_shape(self, folded) -> None:
        view = LocalViewGenerator(n_bins=81).generate(folded)
        assert view.n_bins == 81
        assert view.kind == "local"
        assert view.bin_centers[0] >= -0.5
        assert view.bin_centers[-1] <= 0.5

    def test_malformed_view_rejected(self, folded) -> None:
        gen = GlobalViewGenerator(n_bins=2001, max_empty_fraction=0.01)
        with pytest.raises(DataError, match="rejected"):
            gen.generate(folded)


class TestPhysicsFeatures:
    @pytest.fixture()
    def views(self):
        curve = make_synthetic_tess_curve(defects=False, n_per_sector=1200, seed=2)
        candidate = make_candidate(
            period_days=1.3,
            epoch_days=0.3,
            duration_days=0.1,
            target_id=curve.target_id,
            meta={"depth_odd": 0.008, "depth_even": 0.007},
        )
        folded = PhaseFolder().fold(curve, candidate)
        global_view = GlobalViewGenerator(n_bins=201, normalization="none").generate(
            folded
        )
        local_view = LocalViewGenerator(n_bins=81, normalization="none").generate(
            folded
        )
        return candidate, folded, global_view, local_view

    def test_extracts_named_features(self, views) -> None:
        candidate, folded, global_view, local_view = views
        fv = PhysicsFeatureExtractor().extract(
            candidate, folded, global_view, local_view
        )
        assert len(fv) >= 20
        assert "period_days" in fv.names
        assert "snr" in fv.names
        assert "global_rms" in fv.names
        assert fv.values.shape == (len(fv.names),)

    def test_group_selection(self, views) -> None:
        candidate, folded, global_view, local_view = views
        fv = PhysicsFeatureExtractor(groups=("ephemeris",)).extract(
            candidate, folded, global_view, local_view
        )
        assert set(fv.names) == {
            "period_days",
            "epoch_days",
            "duration_days",
            "depth",
            "n_transits",
            "duty_cycle",
        }


class TestFeatureScaler:
    def test_fit_transform_inverse(self) -> None:
        names = ("a", "b")
        matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = FeatureScaler(method="standard")
        scaler.fit(matrix, names)
        scaled = scaler.transform(matrix, names)
        restored = scaler.inverse_transform(scaled, names)
        np.testing.assert_allclose(restored, matrix, rtol=1e-10)

    def test_log_transform(self) -> None:
        names = ("x",)
        matrix = np.array([[9.0], [99.0]])
        scaler = FeatureScaler(method="minmax", log_features=("x",))
        scaler.fit(matrix, names)
        scaled = scaler.transform(np.array([[9.0]]), names)
        restored = scaler.inverse_transform(scaled, names)
        assert restored[0, 0] == pytest.approx(9.0, rel=1e-6)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        names = ("f",)
        matrix = np.array([[1.0], [2.0], [3.0]])
        scaler = FeatureScaler(method="robust")
        scaler.fit(matrix, names)
        path = scaler.save(tmp_path / "scaler.json")
        loaded = FeatureScaler.load(path)
        np.testing.assert_array_equal(
            loaded.transform(matrix, names), scaler.transform(matrix, names)
        )

    def test_unfitted_raises(self) -> None:
        with pytest.raises(NotFittedError):
            FeatureScaler().transform(np.array([[1.0]]), ("a",))


class TestSplitting:
    def _samples(self, n_stars: int = 4, per_star: int = 2) -> list[DatasetSample]:
        samples = []
        for star in range(n_stars):
            for j in range(per_star):
                samples.append(
                    DatasetSample(
                        sample_id=f"TIC_{star}-{j:02d}_v1",
                        target_id=f"TIC {star}",
                        candidate=make_candidate(
                            candidate_id=f"TIC_{star}-{j:02d}",
                            target_id=f"TIC {star}",
                        ),
                        global_view=np.zeros(10),
                        local_view=np.zeros(5),
                        feature_names=("f",),
                        features=np.array([float(star)]),
                        label=star % 2,
                    )
                )
        return samples

    def test_star_split_no_leakage(self) -> None:
        splits = StarLevelSplitter(
            validation_fraction=0.25, test_fraction=0.25, seed=0
        ).split(self._samples())
        assert_no_group_leakage(splits, key=lambda s: s.target_id)
        assert len(splits) == 8

    def test_candidate_split_blocks_leakage_by_default(self) -> None:
        with pytest.raises(DataError, match="leakage"):
            CandidateLevelSplitter(seed=0).split(self._samples())

    def test_candidate_split_with_opt_in(self) -> None:
        splits = CandidateLevelSplitter(
            seed=0, allow_star_leakage=True
        ).split(self._samples())
        assert len(splits) == 8


class TestCache:
    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        cache = RepresentationCache(tmp_path, compress=True)
        arrays = {
            "global_view": np.arange(10, dtype=float),
            "local_view": np.arange(5, dtype=float),
            "features": np.array([1.0, 2.0]),
            "feature_names": np.array(["a", "b"]),
        }
        fp = "abc123"
        cache.put(fp, arrays)
        loaded = cache.get(fp)
        assert loaded is not None
        np.testing.assert_array_equal(loaded["global_view"], arrays["global_view"])

    def test_fingerprint_changes_with_config(self) -> None:
        curve = make_synthetic_tess_curve(defects=False, n_per_sector=100)
        candidate = make_candidate(target_id=curve.target_id)
        sig_a = {"folding": {"n_bins": 10}}
        sig_b = {"folding": {"n_bins": 11}}
        assert sample_fingerprint(curve, candidate, sig_a, "v1") != sample_fingerprint(
            curve, candidate, sig_b, "v1"
        )

    def test_mmap_requires_uncompressed(self) -> None:
        with pytest.raises(PipelineError, match="Memory mapping"):
            RepresentationCache("/tmp/x", compress=True, mmap=True)


class TestAugmentation:
    def _sample(self) -> DatasetSample:
        return DatasetSample(
            sample_id="S-01_v1",
            target_id="TIC 1",
            candidate=make_candidate(),
            global_view=np.linspace(0.0, -1.0, 50),
            local_view=np.linspace(0.0, -1.0, 20),
            feature_names=("f",),
            features=np.array([1.0]),
            meta={"local_window_phase": 0.2},
        )

    def test_gaussian_noise_changes_views(self) -> None:
        rng = np.random.default_rng(0)
        aug = GaussianNoiseAugmenter(sigma_fraction=0.5).apply(self._sample(), rng)
        assert not np.allclose(aug.global_view, self._sample().global_view)

    def test_augmentation_pipeline(self) -> None:
        pipeline = AugmentationPipeline([GaussianNoiseAugmenter()], seed=0)
        copies = pipeline.augment([self._sample()], copies=2)
        assert len(copies) == 2
        assert copies[0].sample_id.endswith("_aug1")


class TestDatasetContainers:
    def test_feature_vector_mismatch_raises(self) -> None:
        with pytest.raises(DataError, match="misaligned"):
            FeatureVector(names=("a",), values=np.array([1.0, 2.0]))

    def test_dataset_numpy_export(self) -> None:
        sample = DatasetSample(
            sample_id="S_v1",
            target_id="TIC 1",
            candidate=make_candidate(),
            global_view=np.zeros(3),
            local_view=np.zeros(2),
            feature_names=("f",),
            features=np.array([1.0]),
            label=1,
        )
        ds = RepresentationDataset([sample])
        arrays = ds.to_numpy()
        assert arrays["global_view"].shape == (1, 3)
        assert arrays["labels"][0] == 1

    def test_dataset_pandas_export(self) -> None:
        sample = DatasetSample(
            sample_id="S_v1",
            target_id="TIC 1",
            candidate=make_candidate(),
            global_view=np.zeros(3),
            local_view=np.zeros(2),
            feature_names=("f",),
            features=np.array([1.0]),
        )
        df = RepresentationDataset([sample]).to_pandas()
        assert "f" in df.columns
        assert len(df) == 1

    def test_dataset_save_load(self, tmp_path: Path) -> None:
        sample = DatasetSample(
            sample_id="S_v1",
            target_id="TIC 1",
            candidate=make_candidate(),
            global_view=np.array([0.1, 0.2]),
            local_view=np.array([0.3]),
            feature_names=("f",),
            features=np.array([1.0]),
            label=0,
        )
        path = tmp_path / "ds.npz"
        RepresentationDataset([sample], version="v1").save(path)
        loaded = RepresentationDataset.load(path)
        assert len(loaded) == 1
        assert loaded.samples[0].sample_id == "S_v1"
        np.testing.assert_array_equal(loaded.samples[0].global_view, sample.global_view)
