"""Unit tests for dataset builder module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset_builder import SignalDatasetBuilder
from src.data.dataset_io import DatasetIO


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'project': {
            'name': 'Test Project',
            'random_seed': 42,
            'test_random_seed': 123
        },
        'data': {
            'sampling_rate': 1000,
            'time_range': [0, 10],
            'frequencies': [1, 3, 5, 7],
            'samples_per_frequency': {
                'train': 100,  # Small number for fast tests
                'test': 50
            },
            'amplitude_range': [0.5, 2.0],
            'phase_range': [0, 2 * np.pi],
            'noise': {
                'type': 'gaussian',
                'std': 0.1
            }
        },
        'paths': {
            'data_dir': 'data',
            'processed_data_dir': 'data/processed'
        }
    }


@pytest.fixture
def dataset_builder(sample_config):
    """Create dataset builder instance."""
    return SignalDatasetBuilder(sample_config)


class TestSignalDatasetBuilder:
    """Tests for SignalDatasetBuilder class."""

    def test_initialization(self, sample_config):
        """Test SignalDatasetBuilder initialization."""
        builder = SignalDatasetBuilder(sample_config)
        assert builder.config == sample_config
        assert builder.frequencies == [1, 3, 5, 7]
        assert builder.sampling_rate == 1000

    def test_create_condition_vector(self, dataset_builder):
        """Test condition vector creation."""
        # Test for each frequency index
        for idx in range(4):
            condition = dataset_builder._create_condition_vector(idx)

            # Should be one-hot encoded
            assert len(condition) == 4
            assert condition[idx] == 1.0
            assert np.sum(condition) == 1.0

            # Check type
            assert condition.dtype == np.float32

    def test_create_condition_vector_invalid_idx(self, dataset_builder):
        """Test condition vector with invalid index."""
        with pytest.raises(ValueError):
            dataset_builder._create_condition_vector(-1)

        with pytest.raises(ValueError):
            dataset_builder._create_condition_vector(4)

    def test_generate_sample_structure(self, dataset_builder):
        """Test that generated sample has correct structure."""
        from src.data.parameter_sampler import ParameterSampler

        sampler = ParameterSampler(dataset_builder.config, seed=42)
        sample = dataset_builder.generate_sample(
            target_frequency_idx=0,
            parameter_sampler=sampler
        )

        # Check keys
        required_keys = {'mixed_signal', 'target_signal', 'condition_vector', 'metadata'}
        assert set(sample.keys()) == required_keys

        # Check shapes
        assert sample['mixed_signal'].shape == (10000,)
        assert sample['target_signal'].shape == (10000,)
        assert sample['condition_vector'].shape == (4,)

        # Check types
        assert sample['mixed_signal'].dtype == np.float32
        assert sample['target_signal'].dtype == np.float32
        assert sample['condition_vector'].dtype == np.float32

        # Check metadata
        assert 'target_frequency' in sample['metadata']
        assert 'target_amplitude' in sample['metadata']
        assert 'target_phase' in sample['metadata']

    def test_generate_sample_correct_target(self, dataset_builder):
        """Test that generated sample has correct target frequency."""
        from src.data.parameter_sampler import ParameterSampler

        sampler = ParameterSampler(dataset_builder.config, seed=42)

        for target_idx in range(4):
            sample = dataset_builder.generate_sample(
                target_frequency_idx=target_idx,
                parameter_sampler=sampler
            )

            # Check that metadata has correct frequency
            expected_freq = dataset_builder.frequencies[target_idx]
            assert sample['metadata']['target_frequency'] == expected_freq

            # Check condition vector
            assert sample['condition_vector'][target_idx] == 1.0

    def test_generate_sample_reproducibility(self, dataset_builder):
        """Test that same seed produces same sample."""
        from src.data.parameter_sampler import ParameterSampler

        # Set numpy seed for reproducibility of noise
        np.random.seed(42)
        sampler1 = ParameterSampler(dataset_builder.config, seed=42)
        sample1 = dataset_builder.generate_sample(0, sampler1)

        # Reset seed
        np.random.seed(42)
        sampler2 = ParameterSampler(dataset_builder.config, seed=42)
        sample2 = dataset_builder.generate_sample(0, sampler2)

        # Should be identical
        assert np.allclose(sample1['mixed_signal'], sample2['mixed_signal'])
        assert np.allclose(sample1['target_signal'], sample2['target_signal'])

    def test_generate_dataset_structure(self, dataset_builder):
        """Test that generated dataset has correct structure."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        # Check keys
        required_keys = {
            'mixed_signals', 'target_signals', 'condition_vectors',
            'metadata', 'split', 'config'
        }
        assert set(dataset.keys()) == required_keys

        # Check split
        assert dataset['split'] == 'train'

        # Check config
        assert dataset['config'] == dataset_builder.config

    def test_generate_dataset_size(self, dataset_builder):
        """Test that dataset has correct number of samples."""
        # Train dataset
        train_dataset = dataset_builder.generate_dataset(
            split='train',
            show_progress=False
        )

        # Should have 100 samples per frequency * 4 frequencies = 400 samples
        expected_size = 400
        assert len(train_dataset['mixed_signals']) == expected_size
        assert len(train_dataset['target_signals']) == expected_size
        assert len(train_dataset['condition_vectors']) == expected_size
        assert len(train_dataset['metadata']) == expected_size

        # Test dataset
        test_dataset = dataset_builder.generate_dataset(
            split='test',
            show_progress=False
        )

        # Should have 50 samples per frequency * 4 frequencies = 200 samples
        expected_size = 200
        assert len(test_dataset['mixed_signals']) == expected_size

    def test_generate_dataset_shapes(self, dataset_builder):
        """Test that dataset arrays have correct shapes."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        n_samples = 400  # 100 per frequency * 4
        n_timesteps = 10000
        n_frequencies = 4

        assert dataset['mixed_signals'].shape == (n_samples, n_timesteps)
        assert dataset['target_signals'].shape == (n_samples, n_timesteps)
        assert dataset['condition_vectors'].shape == (n_samples, n_frequencies)

    def test_generate_dataset_balance(self, dataset_builder):
        """Test that dataset is balanced across frequencies."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        # Count samples per frequency
        freq_counts = {freq: 0 for freq in dataset_builder.frequencies}

        for meta in dataset['metadata']:
            freq_counts[meta['target_frequency']] += 1

        # All frequencies should have equal samples
        expected_count = 100
        for freq, count in freq_counts.items():
            assert count == expected_count

    def test_generate_dataset_different_seeds(self, dataset_builder):
        """Test that train and test datasets use different seeds."""
        train_dataset = dataset_builder.generate_dataset(
            split='train',
            show_progress=False
        )

        test_dataset = dataset_builder.generate_dataset(
            split='test',
            show_progress=False
        )

        # Datasets should be different
        assert not np.allclose(
            train_dataset['mixed_signals'][:10],
            test_dataset['mixed_signals'][:10]
        )

    def test_generate_dataset_invalid_split(self, dataset_builder):
        """Test dataset generation with invalid split."""
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            dataset_builder.generate_dataset(split='invalid')

    def test_save_and_load_dataset(self, dataset_builder):
        """Test saving and loading dataset."""
        # Generate small dataset
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_dataset.h5'

            dataset_builder.save_dataset(dataset, filepath)

            # Check file exists
            assert filepath.exists()

            # Load dataset
            loaded_dataset = dataset_builder.load_dataset(filepath)

            # Check that loaded data matches original
            assert np.allclose(
                dataset['mixed_signals'],
                loaded_dataset['mixed_signals']
            )
            assert np.allclose(
                dataset['target_signals'],
                loaded_dataset['target_signals']
            )
            assert np.allclose(
                dataset['condition_vectors'],
                loaded_dataset['condition_vectors']
            )
            assert dataset['split'] == loaded_dataset['split']
            assert len(dataset['metadata']) == len(loaded_dataset['metadata'])


class TestDatasetIO:
    """Tests for DatasetIO class."""

    def test_save_hdf5(self, dataset_builder):
        """Test saving dataset to HDF5."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            DatasetIO.save_hdf5(dataset, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_load_hdf5(self, dataset_builder):
        """Test loading dataset from HDF5."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            DatasetIO.save_hdf5(dataset, filepath)
            loaded = DatasetIO.load_hdf5(filepath)

            # Verify all data matches
            assert np.allclose(dataset['mixed_signals'], loaded['mixed_signals'])
            assert np.allclose(dataset['target_signals'], loaded['target_signals'])
            assert dataset['split'] == loaded['split']

    def test_get_dataset_info(self, dataset_builder):
        """Test getting dataset info without loading all data."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            DatasetIO.save_hdf5(dataset, filepath)
            info = DatasetIO.get_dataset_info(filepath)

            assert info['split'] == 'train'
            assert info['n_samples'] == 400
            assert info['n_timesteps'] == 10000
            assert info['n_frequencies'] == 4
            assert info['file_size_mb'] > 0

    def test_save_invalid_dataset(self):
        """Test saving invalid dataset."""
        invalid_dataset = {'mixed_signals': np.array([1, 2, 3])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            with pytest.raises(ValueError, match="Dataset missing required keys"):
                DatasetIO.save_hdf5(invalid_dataset, filepath)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DatasetIO.load_hdf5(Path('nonexistent.h5'))

    def test_dataset_compression(self, dataset_builder):
        """Test that HDF5 compression works."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            DatasetIO.save_hdf5(dataset, filepath)

            # Calculate uncompressed size
            uncompressed_size = (
                dataset['mixed_signals'].nbytes +
                dataset['target_signals'].nbytes +
                dataset['condition_vectors'].nbytes
            )

            # File should be smaller than uncompressed size
            file_size = filepath.stat().st_size

            # With compression, file should be significantly smaller
            # (at least some compression should occur)
            assert file_size < uncompressed_size


class TestDatasetProperties:
    """Tests for dataset properties and quality."""

    def test_dataset_value_ranges(self, dataset_builder):
        """Test that dataset values are in reasonable ranges."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        # Mixed signals should have reasonable amplitude
        # (4 sinusoids with max amplitude 2.0, averaged, plus noise)
        max_expected = 3.0  # Conservative upper bound
        assert np.max(np.abs(dataset['mixed_signals'])) < max_expected

        # Target signals should be within amplitude range
        # Each target is a pure sinusoid
        max_target = np.max(np.abs(dataset['target_signals']))
        assert max_target <= 2.0  # Max amplitude in config

        # Condition vectors should be binary
        assert np.all((dataset['condition_vectors'] == 0) |
                     (dataset['condition_vectors'] == 1))

    def test_condition_vectors_one_hot(self, dataset_builder):
        """Test that all condition vectors are one-hot encoded."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        for condition in dataset['condition_vectors']:
            # Sum should be 1
            assert np.isclose(np.sum(condition), 1.0)

            # Should have exactly one 1 and three 0s
            assert np.sum(condition == 1.0) == 1
            assert np.sum(condition == 0.0) == 3

    def test_metadata_consistency(self, dataset_builder):
        """Test that metadata is consistent with data."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        for i, meta in enumerate(dataset['metadata']):
            # Check that condition vector matches target frequency
            target_freq = meta['target_frequency']
            freq_idx = dataset_builder.frequencies.index(target_freq)

            condition = dataset['condition_vectors'][i]
            assert condition[freq_idx] == 1.0

            # Check that amplitude is in valid range
            assert 0.5 <= meta['target_amplitude'] <= 2.0

            # Check that phase is in valid range
            assert 0 <= meta['target_phase'] <= 2 * np.pi

    def test_no_nan_or_inf(self, dataset_builder):
        """Test that dataset contains no NaN or Inf values."""
        dataset = dataset_builder.generate_dataset(split='train', show_progress=False)

        assert not np.any(np.isnan(dataset['mixed_signals']))
        assert not np.any(np.isinf(dataset['mixed_signals']))
        assert not np.any(np.isnan(dataset['target_signals']))
        assert not np.any(np.isinf(dataset['target_signals']))
        assert not np.any(np.isnan(dataset['condition_vectors']))
        assert not np.any(np.isinf(dataset['condition_vectors']))
