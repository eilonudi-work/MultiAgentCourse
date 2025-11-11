"""Signal generation module for pure and mixed sinusoidal signals."""

import logging
from typing import Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate pure sinusoidal signals with configurable parameters."""

    def __init__(self, config: Dict):
        """
        Initialize signal generator with configuration.

        Args:
            config: Configuration dictionary with:
                - data.sampling_rate: int (samples per second)
                - data.time_range: List[float] ([start, end] in seconds)
                - data.frequencies: List[float] (available frequencies in Hz)

        Raises:
            ValueError: If configuration is invalid
        """
        self.sampling_rate = config['data']['sampling_rate']
        self.time_range = config['data']['time_range']
        self.frequencies = config['data']['frequencies']

        if self.sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.sampling_rate}")

        if len(self.time_range) != 2:
            raise ValueError(f"Time range must have 2 elements, got {len(self.time_range)}")

        if self.time_range[0] >= self.time_range[1]:
            raise ValueError(
                f"Time range start ({self.time_range[0]}) must be less than end ({self.time_range[1]})"
            )

        logger.debug(
            f"Initialized SignalGenerator with sampling_rate={self.sampling_rate}Hz, "
            f"time_range={self.time_range}s"
        )

    def generate_time_vector(self, duration: float) -> np.ndarray:
        """
        Create time vector with proper sampling.

        Args:
            duration: Duration in seconds

        Returns:
            Time array from 0 to duration with sampling_rate spacing

        Raises:
            ValueError: If duration is not positive
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")

        # Number of samples for given duration
        n_samples = int(duration * self.sampling_rate)

        # Create time vector: t = [0, dt, 2*dt, ..., (n-1)*dt]
        # where dt = 1/sampling_rate
        time_vector = np.arange(n_samples) / self.sampling_rate

        logger.debug(f"Generated time vector: {n_samples} samples over {duration}s")
        return time_vector

    def generate_sinusoid(
        self,
        frequency: float,
        amplitude: float,
        phase: float,
        duration: float
    ) -> np.ndarray:
        """
        Generate single pure sinusoid: A*sin(2πft + φ)

        Args:
            frequency: Frequency in Hz
            amplitude: Signal amplitude
            phase: Phase offset in radians
            duration: Signal duration in seconds

        Returns:
            Signal array of shape (n_samples,)

        Raises:
            ValueError: If parameters are invalid
        """
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")

        if amplitude < 0:
            raise ValueError(f"Amplitude must be non-negative, got {amplitude}")

        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")

        # Generate time vector
        t = self.generate_time_vector(duration)

        # Generate sinusoid: A * sin(2π * f * t + φ)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        logger.debug(
            f"Generated sinusoid: f={frequency}Hz, A={amplitude:.3f}, "
            f"φ={phase:.3f}rad, duration={duration}s"
        )

        return signal


class MixedSignalGenerator(SignalGenerator):
    """Generate mixed signals from multiple sinusoids."""

    def __init__(self, config: Dict):
        """
        Initialize mixed signal generator.

        Args:
            config: Configuration dictionary (same as SignalGenerator)
        """
        super().__init__(config)

        # Extract noise configuration
        self.noise_std = config['data']['noise']['std']

        if self.noise_std < 0:
            raise ValueError(f"Noise std must be non-negative, got {self.noise_std}")

        logger.debug(f"Initialized MixedSignalGenerator with noise_std={self.noise_std}")

    def add_gaussian_noise(self, signal: np.ndarray, std: float) -> np.ndarray:
        """
        Add Gaussian noise to signal.

        Args:
            signal: Input signal array
            std: Standard deviation of noise

        Returns:
            Signal with added Gaussian noise N(0, std²)

        Raises:
            ValueError: If std is negative
        """
        if std < 0:
            raise ValueError(f"Noise std must be non-negative, got {std}")

        if std == 0:
            return signal.copy()

        # Generate Gaussian noise: N(0, std²)
        noise = np.random.normal(loc=0.0, scale=std, size=signal.shape)

        noisy_signal = signal + noise

        logger.debug(
            f"Added Gaussian noise: std={std:.4f}, "
            f"signal_power={np.var(signal):.4f}, noise_power={np.var(noise):.4f}"
        )

        return noisy_signal

    def generate_mixed_signal(
        self,
        amplitudes: List[float],
        phases: List[float],
        add_noise: bool = True,
        noise_std: float = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate mixed signal and individual components.

        Mixed signal formula:
            S(t) = (1/4) * Σ[i=1 to 4] Sinus_i^noisy(t)

        where Sinus_i^noisy(t) = A_i * sin(2π * f_i * t + φ_i)

        Args:
            amplitudes: List of 4 amplitudes (one per frequency)
            phases: List of 4 phases in radians (one per frequency)
            add_noise: Whether to add Gaussian noise to mixed signal
            noise_std: Standard deviation of noise (uses config default if None)

        Returns:
            Tuple containing:
                - mixed_signal: Sum of all components + noise, shape (n_samples,)
                - components: List of 4 individual pure signals, each shape (n_samples,)

        Raises:
            ValueError: If amplitudes or phases don't have exactly 4 elements
        """
        if len(amplitudes) != 4:
            raise ValueError(f"Expected 4 amplitudes, got {len(amplitudes)}")

        if len(phases) != 4:
            raise ValueError(f"Expected 4 phases, got {len(phases)}")

        # Use configured noise std if not specified
        if noise_std is None:
            noise_std = self.noise_std

        # Calculate signal duration from time range
        duration = self.time_range[1] - self.time_range[0]

        # Generate individual sinusoid components
        components = []
        for i, (freq, amp, phase) in enumerate(zip(self.frequencies, amplitudes, phases)):
            component = self.generate_sinusoid(
                frequency=freq,
                amplitude=amp,
                phase=phase,
                duration=duration
            )
            components.append(component)

            logger.debug(
                f"Component {i}: f={freq}Hz, A={amp:.3f}, φ={phase:.3f}rad, "
                f"power={np.var(component):.4f}"
            )

        # Create mixed signal: (1/4) * sum of components
        # Stack components and sum along axis 0
        components_array = np.stack(components, axis=0)
        mixed_signal = np.mean(components_array, axis=0)  # Mean is same as (1/4)*sum

        logger.debug(f"Mixed signal (before noise): power={np.var(mixed_signal):.4f}")

        # Add Gaussian noise if requested
        if add_noise:
            mixed_signal = self.add_gaussian_noise(mixed_signal, noise_std)
            logger.debug(f"Mixed signal (after noise): power={np.var(mixed_signal):.4f}")

        return mixed_signal, components
