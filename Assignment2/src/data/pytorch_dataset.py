"""
PyTorch Dataset and DataLoader classes for signal extraction.

This module provides PyTorch-compatible dataset classes that load and prepare
data for LSTM training and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SignalDataset(Dataset):
    """
    PyTorch Dataset for signal extraction.

    Loads data from HDF5 files generated in Phase 1 and provides samples
    in a format suitable for PyTorch DataLoader.

    Each sample contains:
        - mixed_signal: Noisy mixed signal S(t)
        - target_signal: Pure sinusoid at target frequency
        - condition_vector: One-hot [C1, C2, C3, C4]
        - metadata: Additional information (frequency, amplitude, etc.)

    Example:
        >>> dataset = SignalDataset('data/processed/train_dataset.h5')
        >>> print(len(dataset))  # 40 (or actual number of samples)
        >>> sample = dataset[0]
        >>> print(sample['mixed_signal'].shape)  # torch.Size([10000])
    """

    def __init__(
        self,
        data_path: Path,
        normalize: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize dataset from HDF5 file.

        Args:
            data_path: Path to HDF5 dataset file
            normalize: Whether to normalize signals (default: False)
            device: Device to load tensors to ('cpu' or 'cuda')

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data file is invalid
        """
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.device = device

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        # Load data into memory for faster access
        self._load_data()

        # Compute normalization statistics if needed
        if normalize:
            self._compute_normalization_stats()

        logger.info(
            f"Loaded dataset: {len(self)} samples from {data_path}"
        )

    def _load_data(self):
        """Load dataset from HDF5 file into memory."""
        with h5py.File(self.data_path, 'r') as f:
            # Load arrays
            self.mixed_signals = np.array(f['mixed_signals'])
            self.target_signals = np.array(f['target_signals'])
            self.condition_vectors = np.array(f['condition_vectors'])

            # Load metadata (stored as JSON string)
            import json
            metadata_json = f['metadata'][()]
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
            self.metadata = json.loads(metadata_json)

        logger.debug(
            f"Loaded data shapes: "
            f"mixed_signals={self.mixed_signals.shape}, "
            f"target_signals={self.target_signals.shape}, "
            f"condition_vectors={self.condition_vectors.shape}"
        )

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        # Compute statistics over all mixed signals
        self.signal_mean = self.mixed_signals.mean()
        self.signal_std = self.mixed_signals.std()

        logger.debug(
            f"Normalization stats: mean={self.signal_mean:.6f}, "
            f"std={self.signal_std:.6f}"
        )

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            Number of samples
        """
        return len(self.mixed_signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'mixed_signal': Tensor of shape (time_steps,)
                - 'target_signal': Tensor of shape (time_steps,)
                - 'condition_vector': Tensor of shape (4,)
                - 'metadata': Dict with sample information

        Example:
            >>> sample = dataset[0]
            >>> print(sample.keys())
            dict_keys(['mixed_signal', 'target_signal', 'condition_vector', 'metadata'])
        """
        # Get arrays
        mixed_signal = self.mixed_signals[idx]
        target_signal = self.target_signals[idx]
        condition_vector = self.condition_vectors[idx]

        # Normalize if requested
        if self.normalize:
            mixed_signal = (mixed_signal - self.signal_mean) / self.signal_std
            # Note: target_signal typically not normalized as it's the ground truth

        # Convert to tensors
        sample = {
            'mixed_signal': torch.from_numpy(mixed_signal).float(),
            'target_signal': torch.from_numpy(target_signal).float(),
            'condition_vector': torch.from_numpy(condition_vector).float(),
            'idx': idx
        }

        return sample

    def get_sample_metadata(self, idx: int) -> Dict:
        """
        Get metadata for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata

        Example:
            >>> metadata = dataset.get_sample_metadata(0)
            >>> print(metadata.get('frequency'))
        """
        # Extract frequency from condition vector
        condition_idx = int(np.argmax(self.condition_vectors[idx]))

        # Handle both dict and list metadata formats
        if isinstance(self.metadata, dict):
            frequency = self.metadata['frequencies'][condition_idx]
        elif isinstance(self.metadata, list):
            frequency = self.metadata[condition_idx]  # Metadata is frequencies list
        else:
            frequency = None

        return {
            'frequency': frequency,
            'frequency_idx': condition_idx,
            'sample_idx': idx
        }

    def get_dataset_info(self) -> Dict:
        """
        Get dataset information.

        Returns:
            Dictionary with dataset statistics

        Example:
            >>> info = dataset.get_dataset_info()
            >>> print(info['num_samples'])
        """
        # Handle both dict and list metadata formats
        if isinstance(self.metadata, dict):
            frequencies = self.metadata.get('frequencies', [])
        elif isinstance(self.metadata, list):
            frequencies = self.metadata  # Metadata is the frequencies list
        else:
            frequencies = []

        return {
            'num_samples': len(self),
            'time_steps': self.mixed_signals.shape[1],
            'frequencies': frequencies,
            'normalized': self.normalize,
            'device': self.device,
            'data_path': str(self.data_path)
        }


class DataLoaderFactory:
    """
    Factory for creating DataLoaders with proper settings.

    Provides convenient methods to create training and evaluation DataLoaders
    with appropriate configurations.

    Example:
        >>> factory = DataLoaderFactory()
        >>> train_dataset = SignalDataset('data/processed/train_dataset.h5')
        >>> train_loader = factory.create_train_loader(train_dataset, batch_size=32)
        >>> for batch in train_loader:
        ...     print(batch['mixed_signal'].shape)
        ...     break
    """

    @staticmethod
    def create_train_loader(
        dataset: SignalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create DataLoader for training.

        Args:
            dataset: SignalDataset instance
            batch_size: Number of samples per batch (default: 32)
            shuffle: Whether to shuffle data (default: True)
            num_workers: Number of worker processes (default: 0 for single-process)
            pin_memory: Whether to pin memory for faster GPU transfer (default: True)

        Returns:
            DataLoader instance configured for training

        Example:
            >>> train_loader = DataLoaderFactory.create_train_loader(
            ...     dataset, batch_size=64, shuffle=True
            ... )
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False  # Keep all samples
        )

        logger.info(
            f"Created train DataLoader: batch_size={batch_size}, "
            f"shuffle={shuffle}, num_workers={num_workers}"
        )

        return loader

    @staticmethod
    def create_eval_loader(
        dataset: SignalDataset,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create DataLoader for evaluation/testing.

        Evaluation loader has shuffle=False to maintain sample order.

        Args:
            dataset: SignalDataset instance
            batch_size: Number of samples per batch (default: 32)
            num_workers: Number of worker processes (default: 0)
            pin_memory: Whether to pin memory (default: True)

        Returns:
            DataLoader instance configured for evaluation

        Example:
            >>> eval_loader = DataLoaderFactory.create_eval_loader(dataset)
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False
        )

        logger.info(
            f"Created eval DataLoader: batch_size={batch_size}, "
            f"num_workers={num_workers}"
        )

        return loader

    @staticmethod
    def create_single_sample_loader(
        dataset: SignalDataset,
        sample_idx: int
    ) -> DataLoader:
        """
        Create DataLoader for single sample (for visualization/debugging).

        Args:
            dataset: SignalDataset instance
            sample_idx: Index of sample to load

        Returns:
            DataLoader with single sample

        Example:
            >>> loader = DataLoaderFactory.create_single_sample_loader(dataset, 0)
            >>> sample = next(iter(loader))
        """
        # Create subset with single sample
        from torch.utils.data import Subset
        subset = Subset(dataset, [sample_idx])

        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        logger.debug(f"Created single-sample DataLoader for index {sample_idx}")

        return loader
