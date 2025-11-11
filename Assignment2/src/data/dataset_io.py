"""Dataset I/O operations for saving and loading datasets."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np


logger = logging.getLogger(__name__)


class DatasetIO:
    """Handle dataset input/output operations."""

    @staticmethod
    def save_hdf5(dataset: Dict, filepath: Path):
        """
        Save dataset in HDF5 format with compression.

        Args:
            filepath: Output file path
            dataset: Dataset dictionary with structure:
                - 'mixed_signals': np.ndarray
                - 'target_signals': np.ndarray
                - 'condition_vectors': np.ndarray
                - 'metadata': List[Dict]
                - 'split': str
                - 'config': Dict

        Raises:
            ValueError: If dataset structure is invalid
            IOError: If file cannot be written
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Validate dataset structure
        required_keys = ['mixed_signals', 'target_signals', 'condition_vectors',
                         'metadata', 'split', 'config']
        missing_keys = set(required_keys) - set(dataset.keys())
        if missing_keys:
            raise ValueError(f"Dataset missing required keys: {missing_keys}")

        try:
            with h5py.File(filepath, 'w') as f:
                # Save array data with compression
                f.create_dataset(
                    'mixed_signals',
                    data=dataset['mixed_signals'],
                    compression='gzip',
                    compression_opts=4
                )

                f.create_dataset(
                    'target_signals',
                    data=dataset['target_signals'],
                    compression='gzip',
                    compression_opts=4
                )

                f.create_dataset(
                    'condition_vectors',
                    data=dataset['condition_vectors'],
                    compression='gzip',
                    compression_opts=4
                )

                # Save metadata as JSON string
                metadata_json = json.dumps(dataset['metadata'])
                f.create_dataset(
                    'metadata',
                    data=metadata_json,
                    dtype=h5py.string_dtype('utf-8')
                )

                # Save config as JSON string
                config_json = json.dumps(dataset['config'])
                f.create_dataset(
                    'config',
                    data=config_json,
                    dtype=h5py.string_dtype('utf-8')
                )

                # Save split as attribute
                f.attrs['split'] = dataset['split']

                # Save summary statistics
                f.attrs['n_samples'] = len(dataset['mixed_signals'])
                f.attrs['n_timesteps'] = dataset['mixed_signals'].shape[1]
                f.attrs['n_frequencies'] = dataset['condition_vectors'].shape[1]

            logger.info(
                f"Saved dataset to {filepath}: "
                f"{len(dataset['mixed_signals'])} samples, "
                f"size={filepath.stat().st_size / 1024 / 1024:.2f} MB"
            )

        except Exception as e:
            logger.error(f"Error saving dataset to {filepath}: {e}")
            raise IOError(f"Failed to save dataset: {e}")

    @staticmethod
    def load_hdf5(filepath: Path) -> Dict:
        """
        Load dataset from HDF5.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dataset dictionary with same structure as save_hdf5()

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            with h5py.File(filepath, 'r') as f:
                # Load array data
                mixed_signals = f['mixed_signals'][:]
                target_signals = f['target_signals'][:]
                condition_vectors = f['condition_vectors'][:]

                # Load metadata from JSON
                metadata_json = f['metadata'][()]
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                metadata = json.loads(metadata_json)

                # Load config from JSON
                config_json = f['config'][()]
                if isinstance(config_json, bytes):
                    config_json = config_json.decode('utf-8')
                config = json.loads(config_json)

                # Load split
                split = f.attrs['split']

            dataset = {
                'mixed_signals': mixed_signals,
                'target_signals': target_signals,
                'condition_vectors': condition_vectors,
                'metadata': metadata,
                'config': config,
                'split': split
            }

            logger.info(
                f"Loaded dataset from {filepath}: "
                f"{len(mixed_signals)} samples"
            )

            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            raise IOError(f"Failed to load dataset: {e}")

    @staticmethod
    def get_dataset_info(filepath: Path) -> Dict[str, Any]:
        """
        Get metadata about saved dataset without loading all data.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dictionary with dataset information:
                - split: str
                - n_samples: int
                - n_timesteps: int
                - n_frequencies: int
                - file_size_mb: float

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            with h5py.File(filepath, 'r') as f:
                info = {
                    'split': f.attrs['split'],
                    'n_samples': f.attrs['n_samples'],
                    'n_timesteps': f.attrs['n_timesteps'],
                    'n_frequencies': f.attrs['n_frequencies'],
                    'file_size_mb': filepath.stat().st_size / 1024 / 1024
                }

            return info

        except Exception as e:
            logger.error(f"Error reading dataset info from {filepath}: {e}")
            raise IOError(f"Failed to read dataset info: {e}")
