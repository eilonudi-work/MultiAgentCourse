"""Unit tests for parameter sampler module."""

import numpy as np
import pytest
from scipy import stats

from src.data.parameter_sampler import ParameterSampler


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'amplitude_range': [0.5, 2.0],
            'phase_range': [0, 2 * np.pi],
        }
    }


@pytest.fixture
def parameter_sampler(sample_config):
    """Create parameter sampler instance."""
    return ParameterSampler(sample_config, seed=42)


class TestParameterSampler:
    """Tests for ParameterSampler class."""

    def test_initialization(self, sample_config):
        """Test ParameterSampler initialization."""
        sampler = ParameterSampler(sample_config, seed=42)
        assert sampler.amplitude_range == [0.5, 2.0]
        assert sampler.phase_range == [0, 2 * np.pi]
        assert sampler.seed == 42

    def test_initialization_invalid_amplitude_range(self, sample_config):
        """Test initialization with invalid amplitude range."""
        sample_config['data']['amplitude_range'] = [2.0, 0.5]
        with pytest.raises(ValueError, match="Amplitude range invalid"):
            ParameterSampler(sample_config)

    def test_initialization_invalid_phase_range(self, sample_config):
        """Test initialization with invalid phase range."""
        sample_config['data']['phase_range'] = [2 * np.pi, 0]
        with pytest.raises(ValueError, match="Phase range invalid"):
            ParameterSampler(sample_config)

    def test_sample_amplitude(self, parameter_sampler):
        """Test amplitude sampling."""
        amplitude = parameter_sampler.sample_amplitude()

        # Should be within range
        assert 0.5 <= amplitude <= 2.0

        # Should be a float
        assert isinstance(amplitude, (float, np.floating))

    def test_sample_phase(self, parameter_sampler):
        """Test phase sampling."""
        phase = parameter_sampler.sample_phase()

        # Should be within range
        assert 0 <= phase <= 2 * np.pi

        # Should be a float
        assert isinstance(phase, (float, np.floating))

    def test_sample_parameters(self, parameter_sampler):
        """Test sampling all parameters."""
        amplitudes, phases = parameter_sampler.sample_parameters(n_frequencies=4)

        # Should have 4 of each
        assert len(amplitudes) == 4
        assert len(phases) == 4

        # All should be within range
        for amp in amplitudes:
            assert 0.5 <= amp <= 2.0

        for phase in phases:
            assert 0 <= phase <= 2 * np.pi

    def test_sample_parameters_invalid_n(self, parameter_sampler):
        """Test sampling with invalid n_frequencies."""
        with pytest.raises(ValueError, match="n_frequencies must be positive"):
            parameter_sampler.sample_parameters(n_frequencies=0)

        with pytest.raises(ValueError, match="n_frequencies must be positive"):
            parameter_sampler.sample_parameters(n_frequencies=-1)

    def test_amplitude_distribution(self, sample_config):
        """Test that amplitude distribution is uniform."""
        sampler = ParameterSampler(sample_config, seed=42)

        # Sample many amplitudes
        n_samples = 10000
        amplitudes = [sampler.sample_amplitude() for _ in range(n_samples)]

        # Check range
        assert all(0.5 <= a <= 2.0 for a in amplitudes)

        # Check mean (should be close to midpoint)
        mean_amp = np.mean(amplitudes)
        expected_mean = (0.5 + 2.0) / 2
        assert np.isclose(mean_amp, expected_mean, atol=0.05)

        # Kolmogorov-Smirnov test for uniformity
        ks_stat, p_value = stats.kstest(
            amplitudes,
            lambda x: stats.uniform.cdf(x, loc=0.5, scale=1.5)
        )

        # p-value should be high for uniform distribution
        assert p_value > 0.01

    def test_phase_distribution(self, sample_config):
        """Test that phase distribution is uniform."""
        sampler = ParameterSampler(sample_config, seed=42)

        # Sample many phases
        n_samples = 10000
        phases = [sampler.sample_phase() for _ in range(n_samples)]

        # Check range
        assert all(0 <= p <= 2 * np.pi for p in phases)

        # Check mean (should be close to midpoint)
        mean_phase = np.mean(phases)
        expected_mean = np.pi
        assert np.isclose(mean_phase, expected_mean, atol=0.1)

        # Kolmogorov-Smirnov test for uniformity
        ks_stat, p_value = stats.kstest(
            phases,
            lambda x: stats.uniform.cdf(x, loc=0, scale=2*np.pi)
        )

        # p-value should be high for uniform distribution
        assert p_value > 0.01

    def test_reproducibility_with_seed(self, sample_config):
        """Test reproducibility with same seed."""
        sampler1 = ParameterSampler(sample_config, seed=42)
        sampler2 = ParameterSampler(sample_config, seed=42)

        # Sample parameters
        amps1, phases1 = sampler1.sample_parameters(4)
        amps2, phases2 = sampler2.sample_parameters(4)

        # Should be identical
        assert np.allclose(amps1, amps2)
        assert np.allclose(phases1, phases2)

    def test_different_seeds_different_samples(self, sample_config):
        """Test that different seeds produce different samples."""
        sampler1 = ParameterSampler(sample_config, seed=42)
        sampler2 = ParameterSampler(sample_config, seed=123)

        # Sample parameters
        amps1, phases1 = sampler1.sample_parameters(4)
        amps2, phases2 = sampler2.sample_parameters(4)

        # Should be different
        assert not np.allclose(amps1, amps2)
        assert not np.allclose(phases1, phases2)

    def test_reset_seed(self, sample_config):
        """Test reset_seed functionality."""
        sampler = ParameterSampler(sample_config, seed=42)

        # Sample some values
        amps1, phases1 = sampler.sample_parameters(4)

        # Reset seed
        sampler.reset_seed(42)

        # Sample again - should be identical
        amps2, phases2 = sampler.sample_parameters(4)

        assert np.allclose(amps1, amps2)
        assert np.allclose(phases1, phases2)

    def test_no_seed(self, sample_config):
        """Test sampler without explicit seed."""
        sampler = ParameterSampler(sample_config, seed=None)

        # Should still work
        amps, phases = sampler.sample_parameters(4)

        assert len(amps) == 4
        assert len(phases) == 4

    def test_multiple_samples_coverage(self, sample_config):
        """Test that multiple samples cover the parameter space."""
        sampler = ParameterSampler(sample_config, seed=42)

        # Sample many parameter sets
        n_samples = 1000
        all_amps = []
        all_phases = []

        for _ in range(n_samples):
            amps, phases = sampler.sample_parameters(4)
            all_amps.extend(amps)
            all_phases.extend(phases)

        # Check that we have good coverage of the ranges
        # Min amplitude should be close to 0.5
        assert np.min(all_amps) < 0.6

        # Max amplitude should be close to 2.0
        assert np.max(all_amps) > 1.9

        # Min phase should be close to 0
        assert np.min(all_phases) < 0.2

        # Max phase should be close to 2π
        assert np.max(all_phases) > 2 * np.pi - 0.2

    def test_edge_case_single_frequency(self, sample_config):
        """Test sampling for single frequency."""
        sampler = ParameterSampler(sample_config, seed=42)

        amps, phases = sampler.sample_parameters(1)

        assert len(amps) == 1
        assert len(phases) == 1
        assert 0.5 <= amps[0] <= 2.0
        assert 0 <= phases[0] <= 2 * np.pi

    def test_edge_case_many_frequencies(self, sample_config):
        """Test sampling for many frequencies."""
        sampler = ParameterSampler(sample_config, seed=42)

        amps, phases = sampler.sample_parameters(100)

        assert len(amps) == 100
        assert len(phases) == 100

        # All should be valid
        assert all(0.5 <= a <= 2.0 for a in amps)
        assert all(0 <= p <= 2 * np.pi for p in phases)


class TestParameterStatistics:
    """Statistical tests for parameter sampling."""

    def test_amplitude_moments(self, sample_config):
        """Test statistical moments of amplitude distribution."""
        sampler = ParameterSampler(sample_config, seed=42)

        n_samples = 10000
        amplitudes = [sampler.sample_amplitude() for _ in range(n_samples)]

        # For Uniform(a, b):
        # Mean = (a + b) / 2
        # Variance = (b - a)² / 12

        a, b = 0.5, 2.0
        expected_mean = (a + b) / 2
        expected_var = (b - a) ** 2 / 12

        actual_mean = np.mean(amplitudes)
        actual_var = np.var(amplitudes)

        # Check mean
        assert np.isclose(actual_mean, expected_mean, atol=0.05)

        # Check variance
        assert np.isclose(actual_var, expected_var, atol=0.05)

    def test_phase_moments(self, sample_config):
        """Test statistical moments of phase distribution."""
        sampler = ParameterSampler(sample_config, seed=42)

        n_samples = 10000
        phases = [sampler.sample_phase() for _ in range(n_samples)]

        # For Uniform(0, 2π):
        # Mean = π
        # Variance = (2π)² / 12

        expected_mean = np.pi
        expected_var = (2 * np.pi) ** 2 / 12

        actual_mean = np.mean(phases)
        actual_var = np.var(phases)

        # Check mean
        assert np.isclose(actual_mean, expected_mean, atol=0.1)

        # Check variance
        assert np.isclose(actual_var, expected_var, atol=0.2)
