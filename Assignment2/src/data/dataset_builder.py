"""Dataset builder for generating training and test datasets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .signal_generator import MixedSignalGenerator
from .parameter_sampler import ParameterSampler
from .dataset_io import DatasetIO


logger = logging.getLogger(__name__)


class SignalDatasetBuilder:
    """Build complete training and test datasets."""

    def __init__(self, config: Dict):
        """
        Initialize dataset builder with configuration.

        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.frequencies = config['data']['frequencies']
        self.samples_per_frequency = config['data']['samples_per_frequency']
        self.sampling_rate = config['data']['sampling_rate']
        self.time_range = config['data']['time_range']

        # Initialize signal generator
        self.signal_generator = MixedSignalGenerator(config)

        logger.info("Initialized SignalDatasetBuilder")

    def _create_condition_vector(self, target_frequency_idx: int) -> np.ndarray:
        """
        Create one-hot encoded condition vector.

        Args:
            target_frequency_idx: Index of target frequency (0-3)

        Returns:
            One-hot encoded vector [C1, C2, C3, C4] of shape (4,)

        Raises:
            ValueError: If target_frequency_idx is out of range

        Examples:
            >>> builder._create_condition_vector(0)
            array([1., 0., 0., 0.])
            >>> builder._create_condition_vector(2)
            array([0., 0., 1., 0.])
        """
        if not 0 <= target_frequency_idx < len(self.frequencies):
            raise ValueError(
                f"target_frequency_idx must be in [0, {len(self.frequencies)-1}], "
                f"got {target_frequency_idx}"
            )

        condition_vector = np.zeros(len(self.frequencies), dtype=np.float32)
        condition_vector[target_frequency_idx] = 1.0
        return condition_vector

    def generate_sample(
        self,
        target_frequency_idx: int,
        parameter_sampler: ParameterSampler
    ) -> Dict[str, np.ndarray]:
        """
        Generate single sample for training/testing.

        Args:
            target_frequency_idx: Index of target frequency (0-3)
            parameter_sampler: Parameter sampler for random parameters

        Returns:
            Dictionary with:
                - 'mixed_signal': S(t) shape (10000,)
                - 'target_signal': Pure component shape (10000,)
                - 'condition_vector': One-hot [C1,C2,C3,C4] shape (4,)
                - 'metadata': Dict with frequency, amplitude, phase, etc.

        Raises:
            ValueError: If target_frequency_idx is invalid
        """
        if not 0 <= target_frequency_idx < len(self.frequencies):
            raise ValueError(
                f"target_frequency_idx must be in [0, {len(self.frequencies)-1}], "
                f"got {target_frequency_idx}"
            )

        # Sample random parameters for all frequencies
        amplitudes, phases = parameter_sampler.sample_parameters(len(self.frequencies))

        # Generate mixed signal and all components
        mixed_signal, components = self.signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=True
        )

        # Extract target component
        target_signal = components[target_frequency_idx]

        # Create condition vector
        condition_vector = self._create_condition_vector(target_frequency_idx)

        # Create metadata
        metadata = {
            'target_frequency_idx': target_frequency_idx,
            'target_frequency': self.frequencies[target_frequency_idx],
            'amplitudes': amplitudes,
            'phases': phases,
            'target_amplitude': amplitudes[target_frequency_idx],
            'target_phase': phases[target_frequency_idx],
            'sampling_rate': self.sampling_rate,
            'duration': self.time_range[1] - self.time_range[0],
            'n_samples': len(mixed_signal)
        }

        sample = {
            'mixed_signal': mixed_signal.astype(np.float32),
            'target_signal': target_signal.astype(np.float32),
            'condition_vector': condition_vector,
            'metadata': metadata
        }

        return sample

    def generate_dataset(
        self,
        split: str = 'train',
        show_progress: bool = True
    ) -> Dict:
        """
        Generate complete dataset (40,000 samples).

        The dataset is balanced across all frequencies:
        - 10,000 samples per frequency
        - Total: 40,000 samples

        Args:
            split: 'train' or 'test'
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with structure:
                {
                    'mixed_signals': np.ndarray of shape (40000, 10000),
                    'target_signals': np.ndarray of shape (40000, 10000),
                    'condition_vectors': np.ndarray of shape (40000, 4),
                    'metadata': List of 40000 metadata dicts,
                    'split': str ('train' or 'test'),
                    'config': Dict (copy of configuration)
                }

        Raises:
            ValueError: If split is not 'train' or 'test'
        """
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        # Get random seed based on split
        if split == 'train':
            seed = self.config['project']['random_seed']
        else:
            seed = self.config['project'].get('test_random_seed', 123)

        logger.info(f"Generating {split} dataset with seed={seed}")

        # Set numpy random seed for reproducibility of noise generation
        np.random.seed(seed)

        # Initialize parameter sampler with appropriate seed
        parameter_sampler = ParameterSampler(self.config, seed=seed)

        # Calculate total samples
        samples_per_freq = self.samples_per_frequency[split]
        total_samples = samples_per_freq * len(self.frequencies)

        # Pre-allocate arrays for efficiency
        n_timesteps = int(self.sampling_rate * (self.time_range[1] - self.time_range[0]))

        mixed_signals = np.zeros((total_samples, n_timesteps), dtype=np.float32)
        target_signals = np.zeros((total_samples, n_timesteps), dtype=np.float32)
        condition_vectors = np.zeros((total_samples, len(self.frequencies)), dtype=np.float32)
        metadata_list = []

        # Generate samples for each frequency
        sample_idx = 0

        # Create progress bar
        pbar = tqdm(
            total=total_samples,
            desc=f"Generating {split} dataset",
            disable=not show_progress
        )

        for freq_idx in range(len(self.frequencies)):
            for _ in range(samples_per_freq):
                # Generate sample
                sample = self.generate_sample(freq_idx, parameter_sampler)

                # Store in arrays
                mixed_signals[sample_idx] = sample['mixed_signal']
                target_signals[sample_idx] = sample['target_signal']
                condition_vectors[sample_idx] = sample['condition_vector']
                metadata_list.append(sample['metadata'])

                sample_idx += 1
                pbar.update(1)

        pbar.close()

        # Create dataset dictionary
        dataset = {
            'mixed_signals': mixed_signals,
            'target_signals': target_signals,
            'condition_vectors': condition_vectors,
            'metadata': metadata_list,
            'split': split,
            'config': self.config.copy()
        }

        logger.info(
            f"Generated {split} dataset: {total_samples} samples, "
            f"shape={mixed_signals.shape}"
        )

        return dataset

    def save_dataset(self, dataset: Dict, filepath: Path):
        """
        Save dataset to HDF5 format.

        Args:
            dataset: Dataset dictionary from generate_dataset()
            filepath: Output file path (should end with .h5 or .hdf5)
        """
        DatasetIO.save_hdf5(dataset, filepath)
        logger.info(f"Saved dataset to {filepath}")

    def load_dataset(self, filepath: Path) -> Dict:
        """
        Load dataset from HDF5 format.

        Args:
            filepath: Path to dataset file

        Returns:
            Dataset dictionary
        """
        dataset = DatasetIO.load_hdf5(filepath)
        logger.info(f"Loaded dataset from {filepath}")
        return dataset
