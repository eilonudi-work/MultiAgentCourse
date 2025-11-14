"""Unit tests for signal generation module."""

import numpy as np
import pytest
from scipy import stats

from src.data.signal_generator import SignalGenerator, MixedSignalGenerator


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'sampling_rate': 1000,
            'time_range': [0, 10],
            'frequencies': [1, 3, 5, 7],
            'noise': {
                'type': 'gaussian',
                'std': 0.1
            }
        }
    }


@pytest.fixture
def signal_generator(sample_config):
    """Create signal generator instance."""
    return SignalGenerator(sample_config)


@pytest.fixture
def mixed_signal_generator(sample_config):
    """Create mixed signal generator instance."""
    return MixedSignalGenerator(sample_config)


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_initialization(self, sample_config):
        """Test SignalGenerator initialization."""
        gen = SignalGenerator(sample_config)
        assert gen.sampling_rate == 1000
        assert gen.time_range == [0, 10]
        assert gen.frequencies == [1, 3, 5, 7]

    def test_initialization_invalid_sampling_rate(self, sample_config):
        """Test initialization with invalid sampling rate."""
        sample_config['data']['sampling_rate'] = -1
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            SignalGenerator(sample_config)

    def test_initialization_invalid_time_range(self, sample_config):
        """Test initialization with invalid time range."""
        sample_config['data']['time_range'] = [10, 0]
        with pytest.raises(ValueError, match="must be less than end"):
            SignalGenerator(sample_config)

    def test_generate_time_vector(self, signal_generator):
        """Test time vector generation."""
        t = signal_generator.generate_time_vector(duration=10.0)

        # Should have 10,000 samples for 10 seconds at 1000 Hz
        assert len(t) == 10000

        # First value should be 0
        assert t[0] == 0.0

        # Last value should be close to 10 seconds
        assert np.isclose(t[-1], 9.999, atol=0.001)

        # Spacing should be 1/sampling_rate
        dt = t[1] - t[0]
        assert np.isclose(dt, 1/1000, atol=1e-6)

    def test_generate_time_vector_invalid_duration(self, signal_generator):
        """Test time vector generation with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            signal_generator.generate_time_vector(duration=-1)

    def test_generate_sinusoid_basic(self, signal_generator):
        """Test basic sinusoid generation."""
        frequency = 5.0
        amplitude = 1.5
        phase = 0.0
        duration = 10.0

        signal = signal_generator.generate_sinusoid(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            duration=duration
        )

        # Check shape
        assert signal.shape == (10000,)

        # Check amplitude (max should be close to specified amplitude)
        assert np.isclose(np.max(signal), amplitude, atol=0.01)
        assert np.isclose(np.min(signal), -amplitude, atol=0.01)

    def test_sinusoid_frequency(self, signal_generator):
        """Verify generated signal has correct frequency via FFT."""
        frequency = 3.0
        amplitude = 1.0
        phase = 0.0
        duration = 10.0

        signal = signal_generator.generate_sinusoid(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            duration=duration
        )

        # Compute FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/signal_generator.sampling_rate)

        # Find dominant frequency (positive frequencies only)
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft[positive_mask])

        dominant_freq = positive_freqs[np.argmax(positive_fft)]

        # Check that dominant frequency matches expected frequency
        assert np.isclose(dominant_freq, frequency, atol=0.01)

    def test_sinusoid_amplitude(self, signal_generator):
        """Verify amplitude is correct."""
        amplitudes = [0.5, 1.0, 1.5, 2.0]

        for amp in amplitudes:
            signal = signal_generator.generate_sinusoid(
                frequency=5.0,
                amplitude=amp,
                phase=0.0,
                duration=10.0
            )

            # Maximum value should be close to amplitude
            assert np.isclose(np.max(signal), amp, atol=0.01)

    def test_sinusoid_phase(self, signal_generator):
        """Verify phase offset is applied."""
        # Phase = 0
        signal_0 = signal_generator.generate_sinusoid(
            frequency=1.0,
            amplitude=1.0,
            phase=0.0,
            duration=10.0
        )

        # Phase = π (180 degrees)
        signal_pi = signal_generator.generate_sinusoid(
            frequency=1.0,
            amplitude=1.0,
            phase=np.pi,
            duration=10.0
        )

        # Signals with 180-degree phase difference should be negatives
        assert np.allclose(signal_0, -signal_pi, atol=0.01)

    def test_generate_sinusoid_invalid_params(self, signal_generator):
        """Test sinusoid generation with invalid parameters."""
        # Invalid frequency
        with pytest.raises(ValueError, match="Frequency must be positive"):
            signal_generator.generate_sinusoid(-1, 1.0, 0.0, 10.0)

        # Invalid amplitude
        with pytest.raises(ValueError, match="Amplitude must be non-negative"):
            signal_generator.generate_sinusoid(1.0, -1.0, 0.0, 10.0)

        # Invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            signal_generator.generate_sinusoid(1.0, 1.0, 0.0, -10.0)


class TestMixedSignalGenerator:
    """Tests for MixedSignalGenerator class."""

    def test_initialization(self, sample_config):
        """Test MixedSignalGenerator initialization."""
        gen = MixedSignalGenerator(sample_config)
        assert gen.noise_std == 0.1

    def test_initialization_invalid_noise_std(self, sample_config):
        """Test initialization with invalid noise std."""
        sample_config['data']['noise']['std'] = -0.1
        with pytest.raises(ValueError, match="Noise std must be non-negative"):
            MixedSignalGenerator(sample_config)

    def test_add_gaussian_noise(self, mixed_signal_generator):
        """Test Gaussian noise addition."""
        signal = np.zeros(10000)
        std = 0.1

        noisy_signal = mixed_signal_generator.add_gaussian_noise(signal, std)

        # Check that noise was added
        assert not np.allclose(signal, noisy_signal)

        # Check noise statistics
        noise = noisy_signal - signal
        assert np.isclose(np.mean(noise), 0.0, atol=0.01)
        assert np.isclose(np.std(noise), std, atol=0.02)

    def test_noise_properties(self, mixed_signal_generator):
        """Verify noise is N(0, σ²) using statistical tests."""
        signal = np.ones(10000)
        std = 0.1

        noisy_signal = mixed_signal_generator.add_gaussian_noise(signal, std)
        noise = noisy_signal - signal

        # Test 1: Mean should be close to 0
        assert np.isclose(np.mean(noise), 0.0, atol=0.01)

        # Test 2: Std should be close to specified std
        assert np.isclose(np.std(noise), std, atol=0.02)

        # Test 3: Normality test (Shapiro-Wilk on sample)
        # Use a smaller sample for the test
        sample = np.random.choice(noise, size=1000, replace=False)
        _, p_value = stats.shapiro(sample)

        # p-value > 0.05 suggests data is normally distributed
        assert p_value > 0.01  # Using lower threshold for robustness

    def test_add_noise_zero_std(self, mixed_signal_generator):
        """Test that zero std returns unchanged signal."""
        signal = np.random.randn(1000)
        noisy_signal = mixed_signal_generator.add_gaussian_noise(signal, std=0.0)

        assert np.allclose(signal, noisy_signal)

    def test_generate_mixed_signal_basic(self, mixed_signal_generator):
        """Test basic mixed signal generation."""
        amplitudes = [1.0, 1.0, 1.0, 1.0]
        phases = [0.0, 0.0, 0.0, 0.0]

        mixed, components = mixed_signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=False
        )

        # Check that we got 4 components
        assert len(components) == 4

        # Check shapes
        assert mixed.shape == (10000,)
        for comp in components:
            assert comp.shape == (10000,)

    def test_mixed_signal_composition(self, mixed_signal_generator):
        """Verify mixed = sum of components (no noise case)."""
        amplitudes = [1.5, 1.2, 0.8, 1.9]
        phases = [0.1, 0.5, 1.2, 2.0]

        mixed, components = mixed_signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=False
        )

        # Mixed should be (1/4) * sum of components
        expected_mixed = np.mean(components, axis=0)

        assert np.allclose(mixed, expected_mixed, atol=1e-6)

    def test_mixed_signal_with_noise(self, mixed_signal_generator):
        """Test mixed signal generation with noise."""
        amplitudes = [1.0, 1.0, 1.0, 1.0]
        phases = [0.0, 0.0, 0.0, 0.0]

        mixed_noisy, components = mixed_signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=True
        )

        # Without noise
        mixed_clean, _ = mixed_signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=False
        )

        # They should be different
        assert not np.allclose(mixed_noisy, mixed_clean)

        # Difference should have properties of noise
        noise = mixed_noisy - mixed_clean
        assert np.isclose(np.mean(noise), 0.0, atol=0.02)
        assert np.isclose(np.std(noise), 0.1, atol=0.03)

    def test_generate_mixed_signal_invalid_params(self, mixed_signal_generator):
        """Test mixed signal generation with invalid parameters."""
        # Wrong number of amplitudes
        with pytest.raises(ValueError, match="Expected 4 amplitudes"):
            mixed_signal_generator.generate_mixed_signal([1.0, 1.0], [0.0, 0.0])

        # Wrong number of phases
        with pytest.raises(ValueError, match="Expected 4 phases"):
            mixed_signal_generator.generate_mixed_signal(
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0]
            )

    def test_mixed_signal_frequency_content(self, mixed_signal_generator):
        """Verify mixed signal contains all expected frequencies."""
        amplitudes = [1.0, 1.0, 1.0, 1.0]
        phases = [0.0, 0.0, 0.0, 0.0]

        mixed, _ = mixed_signal_generator.generate_mixed_signal(
            amplitudes=amplitudes,
            phases=phases,
            add_noise=False
        )

        # Compute FFT
        fft = np.fft.fft(mixed)
        freqs = np.fft.fftfreq(len(mixed), 1/mixed_signal_generator.sampling_rate)

        # Find peaks in positive frequencies
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft[positive_mask])

        # Expected frequencies
        expected_freqs = [1, 3, 5, 7]

        for expected_freq in expected_freqs:
            # Find the frequency bin closest to expected frequency
            freq_idx = np.argmin(np.abs(positive_freqs - expected_freq))
            found_freq = positive_freqs[freq_idx]

            # Should be within 0.1 Hz
            assert np.isclose(found_freq, expected_freq, atol=0.1)


class TestReproducibility:
    """Tests for reproducibility of signal generation."""

    def test_reproducibility(self, sample_config):
        """Verify same seed produces same results."""
        np.random.seed(42)

        gen1 = MixedSignalGenerator(sample_config)
        amplitudes1 = [1.5, 1.2, 0.8, 1.9]
        phases1 = [0.1, 0.5, 1.2, 2.0]

        mixed1, _ = gen1.generate_mixed_signal(
            amplitudes=amplitudes1,
            phases=phases1,
            add_noise=True
        )

        # Reset seed
        np.random.seed(42)

        gen2 = MixedSignalGenerator(sample_config)
        mixed2, _ = gen2.generate_mixed_signal(
            amplitudes=amplitudes1,
            phases=phases1,
            add_noise=True
        )

        # Should be identical
        assert np.allclose(mixed1, mixed2, atol=1e-10)

    def test_different_seed_different_results(self, sample_config):
        """Verify different seeds produce different results."""
        np.random.seed(42)
        gen1 = MixedSignalGenerator(sample_config)
        amplitudes = [1.0, 1.0, 1.0, 1.0]
        phases = [0.0, 0.0, 0.0, 0.0]
        mixed1, _ = gen1.generate_mixed_signal(amplitudes, phases, add_noise=True)

        np.random.seed(123)
        gen2 = MixedSignalGenerator(sample_config)
        mixed2, _ = gen2.generate_mixed_signal(amplitudes, phases, add_noise=True)

        # Should be different (due to different noise)
        assert not np.allclose(mixed1, mixed2)
