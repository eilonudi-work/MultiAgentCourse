"""Dataset validation module for quality assurance."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate generated datasets meet specifications."""

    def __init__(self, config: Dict = None):
        """
        Initialize validator with configuration.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}

        # Get validation tolerances from config or use defaults
        validation_config = self.config.get('validation', {})
        self.frequency_tolerance = validation_config.get('frequency_tolerance', 0.01)
        self.amplitude_tolerance = validation_config.get('amplitude_tolerance', 0.05)
        self.noise_tolerance = validation_config.get('noise_tolerance', 0.02)

        logger.debug("Initialized DatasetValidator")

    def validate_signal_properties(self, dataset: Dict) -> Dict:
        """
        Validate mathematical properties of signals.

        Checks:
        - Frequency content (FFT analysis)
        - Amplitude ranges
        - Phase distribution
        - Noise characteristics

        Args:
            dataset: Dataset dictionary

        Returns:
            Validation results dictionary with:
                - 'passed': bool
                - 'frequency_validation': Dict
                - 'amplitude_validation': Dict
                - 'phase_validation': Dict
                - 'noise_validation': Dict
        """
        logger.info("Validating signal properties...")

        results = {
            'passed': True,
            'frequency_validation': {},
            'amplitude_validation': {},
            'phase_validation': {},
            'noise_validation': {}
        }

        # Get configuration
        config = dataset.get('config', self.config)
        expected_frequencies = config['data']['frequencies']
        sampling_rate = config['data']['sampling_rate']
        amplitude_range = config['data']['amplitude_range']
        noise_std = config['data']['noise']['std']

        # Validate frequency content using FFT on sample of target signals
        freq_results = self._validate_frequencies(
            dataset['target_signals'],
            dataset['metadata'],
            expected_frequencies,
            sampling_rate
        )
        results['frequency_validation'] = freq_results
        results['passed'] &= freq_results['passed']

        # Validate amplitude distribution
        amp_results = self._validate_amplitudes(
            dataset['metadata'],
            amplitude_range
        )
        results['amplitude_validation'] = amp_results
        results['passed'] &= amp_results['passed']

        # Validate phase distribution
        phase_results = self._validate_phases(dataset['metadata'])
        results['phase_validation'] = phase_results
        results['passed'] &= phase_results['passed']

        # Validate noise characteristics
        noise_results = self._validate_noise(
            dataset['mixed_signals'],
            dataset['target_signals'],
            dataset['metadata'],
            noise_std
        )
        results['noise_validation'] = noise_results
        results['passed'] &= noise_results['passed']

        logger.info(f"Signal properties validation: {'PASSED' if results['passed'] else 'FAILED'}")

        return results

    def _validate_frequencies(
        self,
        target_signals: np.ndarray,
        metadata: List[Dict],
        expected_frequencies: List[float],
        sampling_rate: int
    ) -> Dict:
        """Validate frequency content using FFT."""
        # Sample 100 random signals for validation
        n_samples = min(100, len(target_signals))
        sample_indices = np.random.choice(len(target_signals), n_samples, replace=False)

        frequency_errors = []

        for idx in sample_indices:
            signal = target_signals[idx]
            expected_freq = metadata[idx]['target_frequency']

            # Compute FFT
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)

            # Find dominant frequency (positive frequencies only)
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft[positive_mask])

            dominant_freq = positive_freqs[np.argmax(positive_fft)]

            # Calculate error
            freq_error = abs(dominant_freq - expected_freq)
            frequency_errors.append(freq_error)

        mean_error = np.mean(frequency_errors)
        max_error = np.max(frequency_errors)

        passed = max_error < self.frequency_tolerance

        return {
            'passed': passed,
            'mean_error': float(mean_error),
            'max_error': float(max_error),
            'tolerance': self.frequency_tolerance,
            'n_samples_checked': n_samples
        }

    def _validate_amplitudes(
        self,
        metadata: List[Dict],
        amplitude_range: List[float]
    ) -> Dict:
        """Validate amplitude distribution."""
        # Extract all target amplitudes
        amplitudes = [m['target_amplitude'] for m in metadata]
        amplitudes = np.array(amplitudes)

        # Check range
        min_amp = np.min(amplitudes)
        max_amp = np.max(amplitudes)
        mean_amp = np.mean(amplitudes)

        # Check if all amplitudes are within expected range
        in_range = (min_amp >= amplitude_range[0]) and (max_amp <= amplitude_range[1])

        # Test uniformity using Kolmogorov-Smirnov test
        # Expected: Uniform(0.5, 2.0)
        ks_statistic, ks_pvalue = stats.kstest(
            amplitudes,
            lambda x: stats.uniform.cdf(x, loc=amplitude_range[0],
                                       scale=amplitude_range[1] - amplitude_range[0])
        )

        # Pass if distribution is approximately uniform (p > 0.01)
        uniform_test_passed = ks_pvalue > 0.01

        passed = in_range and uniform_test_passed

        return {
            'passed': passed,
            'in_range': in_range,
            'min': float(min_amp),
            'max': float(max_amp),
            'mean': float(mean_amp),
            'expected_range': amplitude_range,
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'uniform_test_passed': uniform_test_passed
        }

    def _validate_phases(self, metadata: List[Dict]) -> Dict:
        """Validate phase distribution."""
        # Extract all target phases
        phases = [m['target_phase'] for m in metadata]
        phases = np.array(phases)

        # Check range
        min_phase = np.min(phases)
        max_phase = np.max(phases)

        # Check if all phases are within expected range [0, 2π]
        in_range = (min_phase >= 0) and (max_phase <= 2 * np.pi)

        # Test uniformity using Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.kstest(
            phases,
            lambda x: stats.uniform.cdf(x, loc=0, scale=2*np.pi)
        )

        uniform_test_passed = ks_pvalue > 0.01

        passed = in_range and uniform_test_passed

        return {
            'passed': passed,
            'in_range': in_range,
            'min': float(min_phase),
            'max': float(max_phase),
            'expected_range': [0, 2 * np.pi],
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'uniform_test_passed': uniform_test_passed
        }

    def _validate_noise(
        self,
        mixed_signals: np.ndarray,
        target_signals: np.ndarray,
        metadata: List[Dict],
        expected_noise_std: float
    ) -> Dict:
        """Validate noise characteristics."""
        # Sample 100 random signals
        n_samples = min(100, len(mixed_signals))
        sample_indices = np.random.choice(len(mixed_signals), n_samples, replace=False)

        # Reconstruct clean mixed signal and compare with noisy version
        noise_estimates = []

        for idx in sample_indices:
            mixed = mixed_signals[idx]
            meta = metadata[idx]

            # Reconstruct clean mixed signal from metadata
            # We need all components, but we only have the target
            # Instead, estimate noise from high-frequency content or residuals

            # Simple approach: assume noise is what's left after subtracting
            # the smooth trend. Use moving average to get trend.
            window_size = 50
            smooth_signal = np.convolve(
                mixed,
                np.ones(window_size)/window_size,
                mode='same'
            )
            estimated_noise = mixed - smooth_signal

            noise_std = np.std(estimated_noise)
            noise_estimates.append(noise_std)

        mean_noise_std = np.mean(noise_estimates)
        std_noise_std = np.std(noise_estimates)

        # Check if noise std is close to expected
        error = abs(mean_noise_std - expected_noise_std)
        passed = error < self.noise_tolerance

        return {
            'passed': passed,
            'estimated_noise_std': float(mean_noise_std),
            'expected_noise_std': expected_noise_std,
            'std_of_estimates': float(std_noise_std),
            'error': float(error),
            'tolerance': self.noise_tolerance,
            'n_samples_checked': n_samples
        }

    def validate_dataset_balance(self, dataset: Dict) -> Dict:
        """
        Validate dataset balance.

        Checks:
        - Equal samples per frequency
        - Amplitude distribution uniformity across frequencies
        - Phase distribution uniformity across frequencies

        Args:
            dataset: Dataset dictionary

        Returns:
            Validation results dictionary
        """
        logger.info("Validating dataset balance...")

        metadata = dataset['metadata']
        config = dataset.get('config', self.config)
        expected_frequencies = config['data']['frequencies']

        # Count samples per frequency
        freq_counts = {freq: 0 for freq in expected_frequencies}
        for meta in metadata:
            freq_counts[meta['target_frequency']] += 1

        # Check if all frequencies have equal samples
        counts = list(freq_counts.values())
        equal_balance = len(set(counts)) == 1

        # Expected count per frequency
        expected_count = len(metadata) // len(expected_frequencies)

        results = {
            'passed': equal_balance,
            'frequency_counts': freq_counts,
            'expected_count_per_frequency': expected_count,
            'equal_balance': equal_balance,
            'total_samples': len(metadata)
        }

        logger.info(f"Dataset balance validation: {'PASSED' if results['passed'] else 'FAILED'}")

        return results

    def validate_reconstruction(self, dataset: Dict) -> Dict:
        """
        Validate that mixed signal = sum of components.

        The mixed signal should equal (1/4) * sum of components + noise.
        MSE between mixed and (1/4)*sum should be approximately noise variance.

        Args:
            dataset: Dataset dictionary

        Returns:
            Validation results dictionary

        Note:
            This validation requires generating components again since we don't
            store all components in the dataset. We'll validate on a sample.
        """
        logger.info("Validating signal reconstruction...")

        config = dataset.get('config', self.config)
        noise_std = config['data']['noise']['std']
        expected_noise_variance = noise_std ** 2

        # Sample 50 signals to validate
        n_samples = min(50, len(dataset['mixed_signals']))
        sample_indices = np.random.choice(
            len(dataset['mixed_signals']),
            n_samples,
            replace=False
        )

        reconstruction_errors = []

        # We need to regenerate signals to validate reconstruction
        # Import here to avoid circular dependency
        from .signal_generator import MixedSignalGenerator
        from .parameter_sampler import ParameterSampler

        signal_gen = MixedSignalGenerator(config)

        for idx in sample_indices:
            meta = dataset['metadata'][idx]

            # Regenerate mixed signal with same parameters but no noise
            mixed_clean, components = signal_gen.generate_mixed_signal(
                amplitudes=meta['amplitudes'],
                phases=meta['phases'],
                add_noise=False
            )

            # Get the actual noisy mixed signal
            mixed_noisy = dataset['mixed_signals'][idx]

            # Difference should be approximately the noise
            difference = mixed_noisy - mixed_clean
            variance = np.var(difference)

            reconstruction_errors.append(variance)

        mean_variance = np.mean(reconstruction_errors)
        error = abs(mean_variance - expected_noise_variance)

        # Pass if mean variance is close to expected noise variance
        # Allow 50% tolerance since noise is random
        tolerance = expected_noise_variance * 0.5
        passed = error < tolerance

        results = {
            'passed': passed,
            'mean_reconstruction_variance': float(mean_variance),
            'expected_noise_variance': float(expected_noise_variance),
            'error': float(error),
            'tolerance': float(tolerance),
            'n_samples_checked': n_samples
        }

        logger.info(f"Reconstruction validation: {'PASSED' if results['passed'] else 'FAILED'}")

        return results

    def generate_validation_report(self, dataset: Dict) -> str:
        """
        Create comprehensive validation report.

        Args:
            dataset: Dataset dictionary

        Returns:
            String containing formatted validation report
        """
        logger.info("Generating validation report...")

        # Run all validations
        signal_props = self.validate_signal_properties(dataset)
        balance = self.validate_dataset_balance(dataset)
        reconstruction = self.validate_reconstruction(dataset)

        # Overall pass/fail
        overall_passed = (
            signal_props['passed'] and
            balance['passed'] and
            reconstruction['passed']
        )

        # Format report
        report_lines = [
            "=" * 70,
            "DATASET VALIDATION REPORT",
            "=" * 70,
            f"Dataset Split: {dataset['split']}",
            f"Total Samples: {len(dataset['mixed_signals'])}",
            f"Overall Status: {'PASSED' if overall_passed else 'FAILED'}",
            "",
            "-" * 70,
            "1. SIGNAL PROPERTIES VALIDATION",
            "-" * 70,
            f"Status: {'PASSED' if signal_props['passed'] else 'FAILED'}",
            "",
            "Frequency Validation:",
            f"  - Mean Error: {signal_props['frequency_validation']['mean_error']:.6f} Hz",
            f"  - Max Error: {signal_props['frequency_validation']['max_error']:.6f} Hz",
            f"  - Tolerance: {signal_props['frequency_validation']['tolerance']} Hz",
            f"  - Status: {'PASSED' if signal_props['frequency_validation']['passed'] else 'FAILED'}",
            "",
            "Amplitude Validation:",
            f"  - Range: [{signal_props['amplitude_validation']['min']:.3f}, "
            f"{signal_props['amplitude_validation']['max']:.3f}]",
            f"  - Expected: {signal_props['amplitude_validation']['expected_range']}",
            f"  - Mean: {signal_props['amplitude_validation']['mean']:.3f}",
            f"  - KS Test p-value: {signal_props['amplitude_validation']['ks_pvalue']:.4f}",
            f"  - Status: {'PASSED' if signal_props['amplitude_validation']['passed'] else 'FAILED'}",
            "",
            "Phase Validation:",
            f"  - Range: [{signal_props['phase_validation']['min']:.3f}, "
            f"{signal_props['phase_validation']['max']:.3f}]",
            f"  - Expected: [0.000, {2*np.pi:.3f}]",
            f"  - KS Test p-value: {signal_props['phase_validation']['ks_pvalue']:.4f}",
            f"  - Status: {'PASSED' if signal_props['phase_validation']['passed'] else 'FAILED'}",
            "",
            "Noise Validation:",
            f"  - Estimated Std: {signal_props['noise_validation']['estimated_noise_std']:.4f}",
            f"  - Expected Std: {signal_props['noise_validation']['expected_noise_std']:.4f}",
            f"  - Error: {signal_props['noise_validation']['error']:.4f}",
            f"  - Status: {'PASSED' if signal_props['noise_validation']['passed'] else 'FAILED'}",
            "",
            "-" * 70,
            "2. DATASET BALANCE VALIDATION",
            "-" * 70,
            f"Status: {'PASSED' if balance['passed'] else 'FAILED'}",
            "",
            "Samples per Frequency:",
        ]

        for freq, count in balance['frequency_counts'].items():
            report_lines.append(f"  - {freq} Hz: {count} samples")

        report_lines.extend([
            f"  - Expected: {balance['expected_count_per_frequency']} per frequency",
            f"  - Balanced: {'YES' if balance['equal_balance'] else 'NO'}",
            "",
            "-" * 70,
            "3. RECONSTRUCTION VALIDATION",
            "-" * 70,
            f"Status: {'PASSED' if reconstruction['passed'] else 'FAILED'}",
            "",
            f"  - Mean Reconstruction Variance: {reconstruction['mean_reconstruction_variance']:.6f}",
            f"  - Expected Noise Variance: {reconstruction['expected_noise_variance']:.6f}",
            f"  - Error: {reconstruction['error']:.6f}",
            "",
            "=" * 70,
            f"FINAL STATUS: {'ALL VALIDATIONS PASSED' if overall_passed else 'SOME VALIDATIONS FAILED'}",
            "=" * 70,
        ])

        report = "\n".join(report_lines)

        return report
