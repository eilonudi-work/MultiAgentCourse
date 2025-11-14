"""Configuration loader for the LSTM Signal Extraction project."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    @staticmethod
    def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'data.sampling_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = {'data': {'sampling_rate': 1000}}
            >>> ConfigLoader.get_nested(config, 'data.sampling_rate')
            1000
        """
        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration has required fields.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = [
            'project.name',
            'project.random_seed',
            'data.frequencies',
            'data.time_range',
            'data.sampling_rate',
            'data.samples_per_frequency',
            'data.amplitude_range',
            'data.phase_range',
            'data.noise.std',
            'paths.data_dir',
        ]

        for field in required_fields:
            value = ConfigLoader.get_nested(config, field)
            if value is None:
                raise ValueError(f"Required configuration field missing: {field}")

        # Validate data types and ranges
        if not isinstance(config['data']['frequencies'], list):
            raise ValueError("data.frequencies must be a list")

        if len(config['data']['frequencies']) != 4:
            raise ValueError("data.frequencies must contain exactly 4 frequencies")

        if config['data']['sampling_rate'] <= 0:
            raise ValueError("data.sampling_rate must be positive")

        if config['data']['noise']['std'] < 0:
            raise ValueError("data.noise.std must be non-negative")

        logger.info("Configuration validation passed")
        return True
