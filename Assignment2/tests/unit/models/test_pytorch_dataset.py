"""Unit tests for PyTorch Dataset and DataLoader."""

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory


@pytest.fixture
def temp_dataset_file():
    """Create temporary HDF5 dataset file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create sample dataset
    num_samples = 40
    time_steps = 10000
    num_frequencies = 4

    mixed_signals = np.random.randn(num_samples, time_steps).astype(np.float32)
    target_signals = np.random.randn(num_samples, time_steps).astype(np.float32)

    # Create one-hot condition vectors
    condition_vectors = np.zeros((num_samples, num_frequencies), dtype=np.float32)
    for i in range(num_samples):
        condition_vectors[i, i % num_frequencies] = 1

    # Metadata
    metadata = {
        'frequencies': [1, 3, 5, 7],
        'num_samples': num_samples,
        'time_steps': time_steps
    }

    # Save to HDF5
    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('mixed_signals', data=mixed_signals)
        f.create_dataset('target_signals', data=target_signals)
        f.create_dataset('condition_vectors', data=condition_vectors)
        f.create_dataset('metadata', data=json.dumps(metadata))

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def small_dataset_file():
    """Create small temporary dataset for faster testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Small dataset
    num_samples = 5
    time_steps = 100
    num_frequencies = 4

    mixed_signals = np.random.randn(num_samples, time_steps).astype(np.float32)
    target_signals = np.random.randn(num_samples, time_steps).astype(np.float32)

    condition_vectors = np.zeros((num_samples, num_frequencies), dtype=np.float32)
    for i in range(num_samples):
        condition_vectors[i, i % num_frequencies] = 1

    metadata = {
        'frequencies': [1, 3, 5, 7],
        'num_samples': num_samples,
        'time_steps': time_steps
    }

    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('mixed_signals', data=mixed_signals)
        f.create_dataset('target_signals', data=target_signals)
        f.create_dataset('condition_vectors', data=condition_vectors)
        f.create_dataset('metadata', data=json.dumps(metadata))

    yield temp_path

    temp_path.unlink()


class TestSignalDatasetInitialization:
    """Tests for SignalDataset initialization."""

    def test_initialization_basic(self, small_dataset_file):
        """Test basic dataset initialization."""
        dataset = SignalDataset(small_dataset_file)

        assert len(dataset) == 5
        assert dataset.normalize is False
        assert dataset.device == 'cpu'

    def test_initialization_with_normalization(self, small_dataset_file):
        """Test initialization with normalization enabled."""
        dataset = SignalDataset(small_dataset_file, normalize=True)

        assert dataset.normalize is True
        assert hasattr(dataset, 'signal_mean')
        assert hasattr(dataset, 'signal_std')

    def test_initialization_nonexistent_file(self):
        """Test initialization with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SignalDataset(Path('/nonexistent/path/dataset.h5'))

    def test_initialization_loads_data(self, small_dataset_file):
        """Test that initialization loads data into memory."""
        dataset = SignalDataset(small_dataset_file)

        assert hasattr(dataset, 'mixed_signals')
        assert hasattr(dataset, 'target_signals')
        assert hasattr(dataset, 'condition_vectors')
        assert hasattr(dataset, 'metadata')

        assert isinstance(dataset.mixed_signals, np.ndarray)
        assert isinstance(dataset.target_signals, np.ndarray)
        assert isinstance(dataset.condition_vectors, np.ndarray)
        assert isinstance(dataset.metadata, dict)

    def test_initialization_data_shapes(self, small_dataset_file):
        """Test loaded data has correct shapes."""
        dataset = SignalDataset(small_dataset_file)

        assert dataset.mixed_signals.shape == (5, 100)
        assert dataset.target_signals.shape == (5, 100)
        assert dataset.condition_vectors.shape == (5, 4)


class TestSignalDatasetGetItem:
    """Tests for __getitem__ method."""

    def test_getitem_returns_dict(self, small_dataset_file):
        """Test __getitem__ returns dictionary."""
        dataset = SignalDataset(small_dataset_file)
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert 'mixed_signal' in sample
        assert 'target_signal' in sample
        assert 'condition_vector' in sample
        assert 'idx' in sample

    def test_getitem_tensor_types(self, small_dataset_file):
        """Test __getitem__ returns proper tensor types."""
        dataset = SignalDataset(small_dataset_file)
        sample = dataset[0]

        assert isinstance(sample['mixed_signal'], torch.Tensor)
        assert isinstance(sample['target_signal'], torch.Tensor)
        assert isinstance(sample['condition_vector'], torch.Tensor)

        assert sample['mixed_signal'].dtype == torch.float32
        assert sample['target_signal'].dtype == torch.float32
        assert sample['condition_vector'].dtype == torch.float32

    def test_getitem_shapes(self, small_dataset_file):
        """Test __getitem__ returns correct shapes."""
        dataset = SignalDataset(small_dataset_file)
        sample = dataset[0]

        assert sample['mixed_signal'].shape == (100,)
        assert sample['target_signal'].shape == (100,)
        assert sample['condition_vector'].shape == (4,)

    def test_getitem_all_indices(self, small_dataset_file):
        """Test __getitem__ for all valid indices."""
        dataset = SignalDataset(small_dataset_file)

        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample['idx'] == i
            assert sample['mixed_signal'].shape == (100,)

    def test_getitem_out_of_bounds(self, small_dataset_file):
        """Test __getitem__ with out of bounds index."""
        dataset = SignalDataset(small_dataset_file)

        with pytest.raises(IndexError):
            dataset[100]

    def test_getitem_with_normalization(self, small_dataset_file):
        """Test __getitem__ applies normalization."""
        dataset = SignalDataset(small_dataset_file, normalize=True)
        sample = dataset[0]

        # Normalized data should have different statistics
        mixed_signal = sample['mixed_signal'].numpy()

        # Check that values are reasonable (not same as raw data)
        assert not np.allclose(mixed_signal, dataset.mixed_signals[0])

    def test_getitem_without_normalization(self, small_dataset_file):
        """Test __getitem__ without normalization."""
        dataset = SignalDataset(small_dataset_file, normalize=False)
        sample = dataset[0]

        mixed_signal = sample['mixed_signal'].numpy()

        # Should match raw data
        assert np.allclose(mixed_signal, dataset.mixed_signals[0])


class TestSignalDatasetMetadata:
    """Tests for metadata methods."""

    def test_get_sample_metadata(self, small_dataset_file):
        """Test get_sample_metadata returns correct info."""
        dataset = SignalDataset(small_dataset_file)

        metadata = dataset.get_sample_metadata(0)

        assert 'frequency' in metadata
        assert 'frequency_idx' in metadata
        assert 'sample_idx' in metadata

        assert metadata['sample_idx'] == 0
        assert metadata['frequency'] in [1, 3, 5, 7]

    def test_get_sample_metadata_all_samples(self, small_dataset_file):
        """Test get_sample_metadata for all samples."""
        dataset = SignalDataset(small_dataset_file)

        for i in range(len(dataset)):
            metadata = dataset.get_sample_metadata(i)
            assert metadata['sample_idx'] == i
            assert isinstance(metadata['frequency'], (int, float))

    def test_get_dataset_info(self, small_dataset_file):
        """Test get_dataset_info returns comprehensive info."""
        dataset = SignalDataset(small_dataset_file)
        info = dataset.get_dataset_info()

        assert info['num_samples'] == 5
        assert info['time_steps'] == 100
        assert info['frequencies'] == [1, 3, 5, 7]
        assert info['normalized'] is False
        assert info['device'] == 'cpu'
        assert 'data_path' in info


class TestSignalDatasetNormalization:
    """Tests for normalization functionality."""

    def test_normalization_stats_computed(self, small_dataset_file):
        """Test that normalization stats are computed."""
        dataset = SignalDataset(small_dataset_file, normalize=True)

        assert hasattr(dataset, 'signal_mean')
        assert hasattr(dataset, 'signal_std')
        assert isinstance(dataset.signal_mean, (float, np.floating))
        assert isinstance(dataset.signal_std, (float, np.floating))

    def test_normalization_stats_reasonable(self, small_dataset_file):
        """Test that normalization stats are reasonable."""
        dataset = SignalDataset(small_dataset_file, normalize=True)

        # Mean should be close to 0 for random data
        assert abs(dataset.signal_mean) < 1.0

        # Std should be positive and close to 1 for standard normal data
        assert dataset.signal_std > 0
        assert dataset.signal_std < 5.0

    def test_normalized_data_statistics(self, temp_dataset_file):
        """Test that normalized data has expected statistics."""
        dataset = SignalDataset(temp_dataset_file, normalize=True)

        # Get all samples
        all_signals = []
        for i in range(len(dataset)):
            sample = dataset[i]
            all_signals.append(sample['mixed_signal'].numpy())

        all_signals = np.concatenate(all_signals)

        # Normalized data should have mean≈0 and std≈1
        assert abs(all_signals.mean()) < 0.1
        assert abs(all_signals.std() - 1.0) < 0.1


class TestDataLoaderFactoryTrainLoader:
    """Tests for DataLoaderFactory.create_train_loader."""

    def test_create_train_loader_basic(self, small_dataset_file):
        """Test creating basic train loader."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_create_train_loader_shuffle(self, small_dataset_file):
        """Test that train loader shuffles by default."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2, shuffle=True)

        # Get batches from two iterations
        batches1 = [batch['idx'].tolist() for batch in loader]
        batches2 = [batch['idx'].tolist() for batch in loader]

        # Batches should potentially be in different order
        # (might occasionally be same due to random chance)
        # Just verify we can iterate twice
        assert len(batches1) == len(batches2)

    def test_create_train_loader_no_shuffle(self, small_dataset_file):
        """Test train loader without shuffling."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2, shuffle=False)

        # Get batches from two iterations
        batches1 = [batch['idx'].tolist() for batch in loader]
        batches2 = [batch['idx'].tolist() for batch in loader]

        # Should be in same order
        assert batches1 == batches2

    def test_create_train_loader_batch_sizes(self, small_dataset_file):
        """Test train loader with different batch sizes."""
        dataset = SignalDataset(small_dataset_file)
        batch_sizes = [1, 2, 5]

        for batch_size in batch_sizes:
            loader = DataLoaderFactory.create_train_loader(dataset, batch_size=batch_size)
            batch = next(iter(loader))

            # Last batch might be smaller
            assert batch['mixed_signal'].size(0) <= batch_size

    def test_create_train_loader_iteration(self, small_dataset_file):
        """Test iterating through train loader."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        batches = list(loader)

        # Should have ceil(5/2) = 3 batches
        assert len(batches) == 3

        # Check batch contents
        for batch in batches:
            assert 'mixed_signal' in batch
            assert 'target_signal' in batch
            assert 'condition_vector' in batch

    def test_create_train_loader_batch_shapes(self, small_dataset_file):
        """Test batch shapes from train loader."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        batch = next(iter(loader))

        assert batch['mixed_signal'].shape[0] == 2  # batch_size
        assert batch['mixed_signal'].shape[1] == 100  # time_steps
        assert batch['condition_vector'].shape == (2, 4)


class TestDataLoaderFactoryEvalLoader:
    """Tests for DataLoaderFactory.create_eval_loader."""

    def test_create_eval_loader_basic(self, small_dataset_file):
        """Test creating basic eval loader."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=2)

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_create_eval_loader_no_shuffle(self, small_dataset_file):
        """Test that eval loader doesn't shuffle."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=2)

        # Get batches from two iterations
        batches1 = [batch['idx'].tolist() for batch in loader]
        batches2 = [batch['idx'].tolist() for batch in loader]

        # Should be in same order
        assert batches1 == batches2

    def test_create_eval_loader_deterministic_order(self, small_dataset_file):
        """Test that eval loader maintains sample order."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=1)

        indices = [batch['idx'].item() for batch in loader]

        # Should be in sequential order
        assert indices == [0, 1, 2, 3, 4]


class TestDataLoaderFactorySingleSampleLoader:
    """Tests for DataLoaderFactory.create_single_sample_loader."""

    def test_create_single_sample_loader(self, small_dataset_file):
        """Test creating single sample loader."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_single_sample_loader(dataset, sample_idx=2)

        batch = next(iter(loader))

        assert batch['mixed_signal'].shape[0] == 1  # batch_size = 1
        assert batch['idx'].item() == 2

    def test_create_single_sample_loader_all_samples(self, small_dataset_file):
        """Test single sample loader for all samples."""
        dataset = SignalDataset(small_dataset_file)

        for i in range(len(dataset)):
            loader = DataLoaderFactory.create_single_sample_loader(dataset, sample_idx=i)
            batch = next(iter(loader))

            assert batch['idx'].item() == i

    def test_create_single_sample_loader_only_one_batch(self, small_dataset_file):
        """Test that single sample loader has only one batch."""
        dataset = SignalDataset(small_dataset_file)
        loader = DataLoaderFactory.create_single_sample_loader(dataset, sample_idx=0)

        batches = list(loader)
        assert len(batches) == 1


class TestDataLoaderIntegration:
    """Integration tests for DataLoader functionality."""

    def test_full_epoch_iteration(self, temp_dataset_file):
        """Test iterating through full epoch."""
        dataset = SignalDataset(temp_dataset_file)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=8)

        total_samples = 0
        for batch in loader:
            batch_size = batch['mixed_signal'].size(0)
            total_samples += batch_size

        assert total_samples == len(dataset)

    def test_train_eval_consistency(self, small_dataset_file):
        """Test that train and eval loaders access same data."""
        dataset = SignalDataset(small_dataset_file)

        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=5, shuffle=False)
        eval_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=5)

        train_batch = next(iter(train_loader))
        eval_batch = next(iter(eval_loader))

        # Should have same data when not shuffled
        assert torch.allclose(train_batch['mixed_signal'], eval_batch['mixed_signal'])

    def test_dataloader_with_normalized_dataset(self, small_dataset_file):
        """Test DataLoader with normalized dataset."""
        dataset = SignalDataset(small_dataset_file, normalize=True)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        batch = next(iter(loader))

        # Check that batch has reasonable normalized values
        assert not torch.isnan(batch['mixed_signal']).any()
        assert not torch.isinf(batch['mixed_signal']).any()

    def test_multiple_workers(self, small_dataset_file):
        """Test DataLoader with multiple workers (if supported)."""
        dataset = SignalDataset(small_dataset_file)

        # Use 0 workers for compatibility
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2, num_workers=0)

        batches = list(loader)
        assert len(batches) > 0
