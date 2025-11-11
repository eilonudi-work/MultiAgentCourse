"""Parameter sampling module for random signal generation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class ParameterSampler:
    """Sample random amplitudes and phases for signal generation."""

    def __init__(self, config: Dict, seed: Optional[int] = None):
        """
        Initialize parameter sampler with configuration and random seed.

        Args:
            config: Configuration dictionary with:
                - data.amplitude_range: List[float] ([min, max])
                - data.phase_range: List[float] ([min, max] in radians)
            seed: Random seed for reproducibility (optional)

        Raises:
            ValueError: If configuration is invalid
        """
        self.amplitude_range = config['data']['amplitude_range']
        self.phase_range = config['data']['phase_range']

        # Validate ranges
        if len(self.amplitude_range) != 2:
            raise ValueError(
                f"Amplitude range must have 2 elements, got {len(self.amplitude_range)}"
            )

        if self.amplitude_range[0] >= self.amplitude_range[1]:
            raise ValueError(
                f"Amplitude range invalid: min={self.amplitude_range[0]} >= "
                f"max={self.amplitude_range[1]}"
            )

        if len(self.phase_range) != 2:
            raise ValueError(
                f"Phase range must have 2 elements, got {len(self.phase_range)}"
            )

        if self.phase_range[0] >= self.phase_range[1]:
            raise ValueError(
                f"Phase range invalid: min={self.phase_range[0]} >= "
                f"max={self.phase_range[1]}"
            )

        # Initialize random number generator with seed
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        logger.debug(
            f"Initialized ParameterSampler with seed={seed}, "
            f"amplitude_range={self.amplitude_range}, phase_range={self.phase_range}"
        )

    def sample_amplitude(self) -> float:
        """
        Sample amplitude from Uniform(0.5, 2.0).

        Returns:
            Random amplitude value

        Examples:
            >>> sampler = ParameterSampler(config, seed=42)
            >>> amp = sampler.sample_amplitude()
            >>> 0.5 <= amp <= 2.0
            True
        """
        amplitude = self.rng.uniform(
            low=self.amplitude_range[0],
            high=self.amplitude_range[1]
        )

        logger.debug(f"Sampled amplitude: {amplitude:.4f}")
        return amplitude

    def sample_phase(self) -> float:
        """
        Sample phase from Uniform(0, 2π).

        Returns:
            Random phase value in radians

        Examples:
            >>> sampler = ParameterSampler(config, seed=42)
            >>> phase = sampler.sample_phase()
            >>> 0 <= phase <= 2*np.pi
            True
        """
        phase = self.rng.uniform(
            low=self.phase_range[0],
            high=self.phase_range[1]
        )

        logger.debug(f"Sampled phase: {phase:.4f} rad")
        return phase

    def sample_parameters(self, n_frequencies: int) -> Tuple[List[float], List[float]]:
        """
        Sample all parameters for mixed signal generation.

        Args:
            n_frequencies: Number of frequency components (typically 4)

        Returns:
            Tuple containing:
                - amplitudes: List of sampled amplitudes
                - phases: List of sampled phases in radians

        Raises:
            ValueError: If n_frequencies is not positive

        Examples:
            >>> sampler = ParameterSampler(config, seed=42)
            >>> amps, phases = sampler.sample_parameters(4)
            >>> len(amps), len(phases)
            (4, 4)
        """
        if n_frequencies <= 0:
            raise ValueError(f"n_frequencies must be positive, got {n_frequencies}")

        amplitudes = [self.sample_amplitude() for _ in range(n_frequencies)]
        phases = [self.sample_phase() for _ in range(n_frequencies)]

        logger.debug(
            f"Sampled {n_frequencies} parameter sets: "
            f"amplitudes={[f'{a:.3f}' for a in amplitudes]}, "
            f"phases={[f'{p:.3f}' for p in phases]}"
        )

        return amplitudes, phases

    def reset_seed(self, seed: Optional[int] = None):
        """
        Reset the random number generator with a new seed.

        Args:
            seed: New random seed (uses original seed if None)
        """
        if seed is None:
            seed = self.seed

        self.rng = np.random.RandomState(seed)
        logger.debug(f"Reset random seed to {seed}")
