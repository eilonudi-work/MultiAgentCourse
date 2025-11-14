"""Integration tests for complete dataset generation pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config.config_loader import ConfigLoader
from src.data.dataset_builder import SignalDatasetBuilder
from src.data.validators import DatasetValidator
from src.data.visualizers import DatasetVisualizer


@pytest.fixture
def test_config():
    """Test configuration with small dataset size."""
    return {
        'project': {
            'name': 'Integration Test',
            'random_seed': 42,
            'test_random_seed': 123
        },
        'data': {
            'sampling_rate': 1000,
            'time_range': [0, 10],
            'frequencies': [1, 3, 5, 7],
            'samples_per_frequency': {
                'train': 50,
                'test': 25
            },
            'amplitude_range': [0.5, 2.0],
            'phase_range': [0, 2 * np.pi],
            'noise': {
                'type': 'gaussian',
                'std': 0.1
            }
        },
        'paths': {
            'data_dir': 'data',
            'processed_data_dir': 'data/processed',
            'figures_dir': 'outputs/figures'
        },
        'validation': {
            'frequency_tolerance': 0.01,
            'amplitude_tolerance': 0.05,
            'noise_tolerance': 0.02
        }
    }


class TestDatasetGenerationPipeline:
    """Test complete dataset generation pipeline."""

    def test_dataset_generation_end_to_end(self, test_config):
        """Test complete dataset generation pipeline from start to finish."""
        # Initialize builder
        builder = SignalDatasetBuilder(test_config)

        # Generate train dataset
        train_dataset = builder.generate_dataset(split='train', show_progress=False)

        # Verify train dataset structure
        assert train_dataset['split'] == 'train'
        assert len(train_dataset['mixed_signals']) == 200  # 50 * 4

        # Generate test dataset
        test_dataset = builder.generate_dataset(split='test', show_progress=False)

        # Verify test dataset structure
        assert test_dataset['split'] == 'test'
        assert len(test_dataset['mixed_signals']) == 100  # 25 * 4

        # Verify datasets are different
        assert not np.allclose(
            train_dataset['mixed_signals'][0],
            test_dataset['mixed_signals'][0]
        )

    def test_dataset_saving_loading(self, test_config):
        """Test that datasets can be saved and loaded without data loss."""
        builder = SignalDatasetBuilder(test_config)
        original_dataset = builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_dataset.h5'

            # Save dataset
            builder.save_dataset(original_dataset, filepath)

            # Verify file exists
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Load dataset
            loaded_dataset = builder.load_dataset(filepath)

            # Verify exact match (should be bit-for-bit identical)
            assert np.array_equal(
                original_dataset['mixed_signals'],
                loaded_dataset['mixed_signals']
            )
            assert np.array_equal(
                original_dataset['target_signals'],
                loaded_dataset['target_signals']
            )
            assert np.array_equal(
                original_dataset['condition_vectors'],
                loaded_dataset['condition_vectors']
            )

            # Verify metadata
            assert len(original_dataset['metadata']) == len(loaded_dataset['metadata'])
            assert original_dataset['split'] == loaded_dataset['split']

            # Spot check a few metadata entries
            for i in [0, 10, 50]:
                assert (original_dataset['metadata'][i]['target_frequency'] ==
                       loaded_dataset['metadata'][i]['target_frequency'])
                assert np.isclose(
                    original_dataset['metadata'][i]['target_amplitude'],
                    loaded_dataset['metadata'][i]['target_amplitude']
                )

    def test_validation_pipeline(self, test_config):
        """Test dataset validation pipeline."""
        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        validator = DatasetValidator(test_config)

        # Test signal properties validation
        signal_results = validator.validate_signal_properties(dataset)
        assert 'passed' in signal_results
        assert 'frequency_validation' in signal_results
        assert 'amplitude_validation' in signal_results
        assert 'phase_validation' in signal_results
        assert 'noise_validation' in signal_results

        # Test balance validation
        balance_results = validator.validate_dataset_balance(dataset)
        assert balance_results['passed']
        assert balance_results['equal_balance']

        # Test reconstruction validation
        reconstruction_results = validator.validate_reconstruction(dataset)
        assert 'passed' in reconstruction_results

        # Test full validation report
        report = validator.generate_validation_report(dataset)
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'VALIDATION REPORT' in report

    def test_visualization_pipeline(self, test_config):
        """Test visualization pipeline."""
        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        visualizer = DatasetVisualizer(test_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test sample signals plot
            sample = {
                'mixed_signal': dataset['mixed_signals'][0],
                'target_signal': dataset['target_signals'][0],
                'metadata': dataset['metadata'][0]
            }
            sample_path = tmpdir / 'sample.png'
            visualizer.plot_sample_signals(sample, save_path=sample_path)
            assert sample_path.exists()

            # Test frequency spectrum plot
            spectrum_path = tmpdir / 'spectrum.png'
            visualizer.plot_frequency_spectrum(
                dataset['mixed_signals'][0],
                test_config['data']['sampling_rate'],
                save_path=spectrum_path
            )
            assert spectrum_path.exists()

            # Test parameter distributions plot
            params_path = tmpdir / 'params.png'
            visualizer.plot_parameter_distributions(
                dataset,
                save_path=params_path
            )
            assert params_path.exists()

            # Test dataset summary figure
            summary_path = tmpdir / 'summary.png'
            visualizer.create_dataset_summary_figure(
                dataset,
                save_path=summary_path
            )
            assert summary_path.exists()

            # Test multiple samples plot
            samples_path = tmpdir / 'samples.png'
            visualizer.plot_multiple_samples(
                dataset,
                n_samples=4,
                save_path=samples_path
            )
            assert samples_path.exists()

    def test_complete_workflow(self, test_config):
        """Test complete workflow: generate, save, load, validate, visualize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Generate dataset
            builder = SignalDatasetBuilder(test_config)
            dataset = builder.generate_dataset(split='train', show_progress=False)

            # Step 2: Save dataset
            dataset_path = tmpdir / 'dataset.h5'
            builder.save_dataset(dataset, dataset_path)
            assert dataset_path.exists()

            # Step 3: Load dataset
            loaded_dataset = builder.load_dataset(dataset_path)
            assert len(loaded_dataset['mixed_signals']) == len(dataset['mixed_signals'])

            # Step 4: Validate dataset
            validator = DatasetValidator(test_config)
            validation_report = validator.generate_validation_report(loaded_dataset)
            assert 'VALIDATION REPORT' in validation_report

            # Step 5: Visualize dataset
            visualizer = DatasetVisualizer(test_config)
            figure_path = tmpdir / 'summary.png'
            visualizer.create_dataset_summary_figure(
                loaded_dataset,
                save_path=figure_path
            )
            assert figure_path.exists()

    def test_multiple_dataset_generation(self, test_config):
        """Test generating multiple datasets with different splits."""
        builder = SignalDatasetBuilder(test_config)

        # Generate train dataset
        train_data = builder.generate_dataset(split='train', show_progress=False)

        # Generate test dataset
        test_data = builder.generate_dataset(split='test', show_progress=False)

        # Verify they are different
        assert not np.allclose(
            train_data['mixed_signals'][0],
            test_data['mixed_signals'][0]
        )

        # Verify sizes are correct
        assert len(train_data['mixed_signals']) == 200
        assert len(test_data['mixed_signals']) == 100

        # Verify both have correct structure
        for dataset in [train_data, test_data]:
            assert 'mixed_signals' in dataset
            assert 'target_signals' in dataset
            assert 'condition_vectors' in dataset
            assert 'metadata' in dataset
            assert 'split' in dataset
            assert 'config' in dataset


class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_config_validation(self, test_config):
        """Test configuration validation."""
        # Valid config should pass
        assert ConfigLoader.validate_config(test_config)

        # Invalid config should fail
        invalid_config = test_config.copy()
        invalid_config['data']['sampling_rate'] = -1

        with pytest.raises((ValueError, KeyError)):
            ConfigLoader.validate_config(invalid_config)

    def test_nested_config_access(self, test_config):
        """Test accessing nested configuration values."""
        value = ConfigLoader.get_nested(test_config, 'data.sampling_rate')
        assert value == 1000

        value = ConfigLoader.get_nested(test_config, 'data.noise.std')
        assert value == 0.1

        # Non-existent key
        value = ConfigLoader.get_nested(test_config, 'nonexistent.key', default=42)
        assert value == 42


class TestDatasetQuality:
    """Integration tests for dataset quality."""

    def test_frequency_separation(self, test_config):
        """Test that different frequency components are separable."""
        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        # For each frequency, check that target signals at that frequency
        # have strong frequency content at the target frequency
        for freq_idx, target_freq in enumerate(test_config['data']['frequencies']):
            # Find samples with this target frequency
            target_samples = []
            for i, meta in enumerate(dataset['metadata']):
                if meta['target_frequency'] == target_freq:
                    target_samples.append(dataset['target_signals'][i])

            # Check first few samples
            for signal in target_samples[:5]:
                # Compute FFT
                fft = np.fft.fft(signal)
                freqs = np.fft.fftfreq(
                    len(signal),
                    1/test_config['data']['sampling_rate']
                )

                # Find dominant frequency
                positive_mask = freqs > 0
                positive_freqs = freqs[positive_mask]
                positive_fft = np.abs(fft[positive_mask])
                dominant_freq = positive_freqs[np.argmax(positive_fft)]

                # Should match target frequency
                assert np.isclose(dominant_freq, target_freq, atol=0.1)

    def test_noise_consistency(self, test_config):
        """Test that noise level is consistent across dataset."""
        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        expected_noise_std = test_config['data']['noise']['std']

        # Sample a few signals and estimate noise
        noise_estimates = []

        for i in range(min(20, len(dataset['mixed_signals']))):
            mixed = dataset['mixed_signals'][i]

            # Simple noise estimation using high-frequency content
            window_size = 50
            smooth = np.convolve(mixed, np.ones(window_size)/window_size, mode='same')
            noise = mixed - smooth

            noise_estimates.append(np.std(noise))

        mean_noise = np.mean(noise_estimates)

        # Should be reasonably close to expected
        assert np.isclose(mean_noise, expected_noise_std, atol=0.05)

    def test_dataset_reproducibility(self, test_config):
        """Test that same configuration produces reproducible results."""
        builder1 = SignalDatasetBuilder(test_config)
        dataset1 = builder1.generate_dataset(split='train', show_progress=False)

        builder2 = SignalDatasetBuilder(test_config)
        dataset2 = builder2.generate_dataset(split='train', show_progress=False)

        # Should be identical
        assert np.allclose(dataset1['mixed_signals'], dataset2['mixed_signals'])
        assert np.allclose(dataset1['target_signals'], dataset2['target_signals'])

    def test_train_test_independence(self, test_config):
        """Test that train and test sets are independent."""
        builder = SignalDatasetBuilder(test_config)

        train_data = builder.generate_dataset(split='train', show_progress=False)
        test_data = builder.generate_dataset(split='test', show_progress=False)

        # Check that no signals are identical
        for train_signal in train_data['mixed_signals'][:10]:
            for test_signal in test_data['mixed_signals'][:10]:
                # Signals should not be identical
                assert not np.allclose(train_signal, test_signal)

    def test_balanced_frequency_distribution(self, test_config):
        """Test that frequencies are balanced in dataset."""
        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        # Count samples per frequency
        freq_counts = {}
        for meta in dataset['metadata']:
            freq = meta['target_frequency']
            freq_counts[freq] = freq_counts.get(freq, 0) + 1

        # All frequencies should have equal count
        counts = list(freq_counts.values())
        assert len(set(counts)) == 1  # All counts are the same

        # Each should have expected count
        expected_count = test_config['data']['samples_per_frequency']['train']
        for count in counts:
            assert count == expected_count


class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_invalid_split(self, test_config):
        """Test handling of invalid split parameter."""
        builder = SignalDatasetBuilder(test_config)

        with pytest.raises((ValueError, KeyError)):
            builder.generate_dataset(split='invalid')

    def test_load_nonexistent_file(self, test_config):
        """Test loading nonexistent dataset file."""
        builder = SignalDatasetBuilder(test_config)

        with pytest.raises(FileNotFoundError):
            builder.load_dataset(Path('nonexistent.h5'))

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            'data': {
                'sampling_rate': -1000  # Invalid
            }
        }

        with pytest.raises((ValueError, KeyError)):
            SignalDatasetBuilder(invalid_config)


class TestPerformance:
    """Performance tests for dataset generation."""

    def test_generation_speed(self, test_config):
        """Test that dataset generation completes in reasonable time."""
        import time

        builder = SignalDatasetBuilder(test_config)

        start_time = time.time()
        dataset = builder.generate_dataset(split='train', show_progress=False)
        elapsed_time = time.time() - start_time

        # 200 samples should generate in under 10 seconds
        assert elapsed_time < 10.0

        # Verify dataset was actually generated
        assert len(dataset['mixed_signals']) == 200

    def test_save_load_speed(self, test_config):
        """Test that save/load operations are fast."""
        import time

        builder = SignalDatasetBuilder(test_config)
        dataset = builder.generate_dataset(split='train', show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'

            # Test save speed
            start_time = time.time()
            builder.save_dataset(dataset, filepath)
            save_time = time.time() - start_time

            # Should save in under 5 seconds
            assert save_time < 5.0

            # Test load speed
            start_time = time.time()
            loaded = builder.load_dataset(filepath)
            load_time = time.time() - start_time

            # Should load in under 2 seconds
            assert load_time < 2.0

            # Verify data was loaded
            assert len(loaded['mixed_signals']) == 200
