"""
Integration tests for evaluation framework.

Tests comprehensive evaluation including:
- Model evaluation on test sets
- Statistical analysis
- Error analysis
- Visualization generation
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.pytorch_dataset import SignalDataset
from src.models.model_factory import ModelFactory
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.statistical_analyzer import StatisticalAnalyzer
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.visualizer import SignalVisualizer


@pytest.fixture
def temp_dataset():
    """Create temporary test dataset."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create small dataset
    num_samples = 8
    time_steps = 50
    num_frequencies = 4

    mixed_signals = np.random.randn(num_samples, time_steps).astype(np.float32) * 0.5
    target_signals = np.random.randn(num_samples, time_steps).astype(np.float32) * 0.3

    condition_vectors = np.zeros((num_samples, num_frequencies), dtype=np.float32)
    for i in range(num_samples):
        condition_vectors[i, i % num_frequencies] = 1

    metadata = {
        'frequencies': [1, 3, 5, 7],
        'num_samples': num_samples,
        'time_steps': time_steps
    }

    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('mixed_signals', data=mixed_signals)
        f.create_dataset('target_signals', data=target_signals)
        f.create_dataset('condition_vectors', data=condition_vectors)
        f.create_dataset('metadata', data=json.dumps(metadata))

    yield temp_path

    temp_path.unlink()


@pytest.fixture
def trained_model():
    """Create a simple trained model for testing."""
    config = {
        'model': {
            'lstm': {
                'input_size': 5,
                'hidden_size': 16,
                'num_layers': 1,
                'dropout': 0.0
            }
        }
    }

    model = ModelFactory.create_model(config, device='cpu')
    return model


class TestModelEvaluator:
    """Test model evaluation functionality."""

    def test_evaluator_initialization(self, trained_model, temp_dataset):
        """Test evaluator can be initialized."""
        dataset = SignalDataset(temp_dataset)
        evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

        assert evaluator is not None
        assert evaluator.model == trained_model
        assert evaluator.dataset == dataset
        assert len(evaluator.frequencies) == 4

    def test_full_dataset_evaluation(self, trained_model, temp_dataset):
        """Test full dataset evaluation."""
        dataset = SignalDataset(temp_dataset)
        evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

        results = evaluator.evaluate_full_dataset(
            batch_size=2,
            save_predictions=True
        )

        # Check structure
        assert 'overall_metrics' in results
        assert 'per_frequency_metrics' in results
        assert 'summary' in results
        assert 'per_sample_results' in results

        # Check overall metrics
        assert 'mse' in results['overall_metrics']
        assert 'mae' in results['overall_metrics']
        assert 'correlation' in results['overall_metrics']

        # Check all values are finite
        assert np.isfinite(results['overall_metrics']['mse'])
        assert np.isfinite(results['overall_metrics']['mae'])

    def test_per_frequency_metrics(self, trained_model, temp_dataset):
        """Test per-frequency metrics computation."""
        dataset = SignalDataset(temp_dataset)
        evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

        results = evaluator.evaluate_full_dataset(batch_size=2, save_predictions=False)

        # Should have metrics for each frequency
        assert len(results['per_frequency_metrics']) == 4

        # Each frequency should have metrics
        for freq_metrics in results['per_frequency_metrics']:
            assert 'frequency' in freq_metrics
            assert 'mse' in freq_metrics
            assert 'num_samples' in freq_metrics

    def test_prd_target_checking(self, trained_model, temp_dataset):
        """Test PRD target validation."""
        dataset = SignalDataset(temp_dataset)
        evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

        results = evaluator.evaluate_full_dataset(batch_size=2)

        # Check summary contains PRD validation
        assert 'summary' in results
        assert 'mse_target_met' in results['summary']
        assert 'mse_value' in results['summary']
        assert 'mse_target' in results['summary']

        # Target should be 0.01
        assert results['summary']['mse_target'] == 0.01

    def test_save_evaluation_results(self, trained_model, temp_dataset):
        """Test saving evaluation results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SignalDataset(temp_dataset)
            evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

            results = evaluator.evaluate_full_dataset(batch_size=2)

            output_path = Path(tmpdir) / 'results.json'
            evaluator.save_evaluation_results(results, output_path)

            # Check file was created
            assert output_path.exists()

            # Check can load JSON
            with open(output_path, 'r') as f:
                loaded = json.load(f)

            assert 'overall_metrics' in loaded

    def test_generate_evaluation_report(self, trained_model, temp_dataset):
        """Test generating evaluation report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SignalDataset(temp_dataset)
            evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

            results = evaluator.evaluate_full_dataset(batch_size=2)

            report_path = Path(tmpdir) / 'report.md'
            evaluator.generate_evaluation_report(results, report_path)

            # Check file was created
            assert report_path.exists()

            # Check it's markdown
            content = report_path.read_text()
            assert '# Model Evaluation Report' in content


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""

    def test_confidence_interval_computation(self):
        """Test confidence interval computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean, lower, upper = StatisticalAnalyzer.compute_confidence_interval(values, confidence=0.95)

        assert mean == 3.0
        assert lower < mean
        assert upper > mean

    def test_summary_statistics(self):
        """Test summary statistics computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = StatisticalAnalyzer.compute_summary_statistics(values)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0

    def test_effect_size_computation(self):
        """Test Cohen's d effect size."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])

        effect_size = StatisticalAnalyzer.compute_effect_size(group1, group2)

        # Should be finite
        assert np.isfinite(effect_size)

    def test_per_frequency_performance_analysis(self):
        """Test per-frequency performance analysis."""
        per_frequency_metrics = [
            {'frequency': 1, 'mse': 0.5, 'correlation': 0.9},
            {'frequency': 3, 'mse': 0.3, 'correlation': 0.95},
            {'frequency': 5, 'mse': 0.7, 'correlation': 0.85},
            {'frequency': 7, 'mse': 0.4, 'correlation': 0.92}
        ]

        analysis = StatisticalAnalyzer.analyze_per_frequency_performance(per_frequency_metrics)

        assert 'num_frequencies' in analysis
        assert analysis['num_frequencies'] == 4

        assert 'best_frequency' in analysis
        assert analysis['best_frequency']['frequency'] == 3  # Lowest MSE

        assert 'worst_frequency' in analysis
        assert analysis['worst_frequency']['frequency'] == 5  # Highest MSE


class TestErrorAnalyzer:
    """Test error analysis functionality."""

    def test_find_worst_predictions(self):
        """Test finding worst predictions."""
        per_sample_results = [
            {'frequency': 1, 'metrics': {'mse': 0.5}},
            {'frequency': 3, 'metrics': {'mse': 0.9}},  # Worst
            {'frequency': 5, 'metrics': {'mse': 0.3}},
            {'frequency': 7, 'metrics': {'mse': 0.7}}
        ]

        worst = ErrorAnalyzer.find_worst_predictions(per_sample_results, n=2, metric='mse')

        assert len(worst) == 2
        assert worst[0]['frequency'] == 3  # Highest MSE
        assert worst[1]['frequency'] == 7  # Second highest MSE

    def test_analyze_error_patterns(self):
        """Test error pattern analysis."""
        predictions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        targets = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        analysis = ErrorAnalyzer.analyze_error_patterns(predictions, targets)

        assert 'mean_error' in analysis
        assert 'std_error' in analysis
        assert 'mae' in analysis
        assert 'rmse' in analysis
        assert 'bias_present' in analysis

    def test_identify_systematic_biases(self):
        """Test systematic bias identification."""
        # Create data with systematic overprediction
        targets = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predictions = targets + 0.5  # Always overpredict

        biases = ErrorAnalyzer.identify_systematic_biases(predictions, targets)

        assert 'overall_bias' in biases
        assert 'overpredict_ratio' in biases
        assert biases['overall_bias'] > 0  # Positive bias
        assert biases['overpredict_ratio'] > 0.9  # Almost always overpredict


class TestSignalVisualizer:
    """Test visualization functionality."""

    def test_visualizer_initialization(self):
        """Test visualizer can be initialized."""
        visualizer = SignalVisualizer(dpi=100)

        assert visualizer is not None
        assert visualizer.dpi == 100

    def test_create_f2_detailed_plot(self):
        """Test creating Graph 1 (f₂ detailed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = SignalVisualizer(dpi=100)

            sample_data = {
                'target': np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100)),
                'mixed_signal': np.random.randn(100) * 0.1,
                'prediction': np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100)) + np.random.randn(100) * 0.05,
                'mse': 0.005
            }

            save_path = Path(tmpdir) / 'test_graph1.png'
            visualizer.create_f2_detailed_plot(sample_data, save_path)

            # Check file was created
            assert save_path.exists()

    def test_create_all_frequencies_plot(self):
        """Test creating Graph 2 (all frequencies)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = SignalVisualizer(dpi=100)

            # Create sample data for each frequency
            frequency_samples = {}
            for freq in [1, 3, 5, 7]:
                frequency_samples[freq] = {
                    'target': np.sin(2 * np.pi * freq * np.linspace(0, 1, 100)),
                    'prediction': np.sin(2 * np.pi * freq * np.linspace(0, 1, 100)) + np.random.randn(100) * 0.05,
                    'mse': 0.005,
                    'r2': 0.95
                }

            save_path = Path(tmpdir) / 'test_graph2.png'
            visualizer.create_all_frequencies_plot(frequency_samples, save_path=save_path)

            # Check file was created
            assert save_path.exists()

    def test_plot_per_frequency_metrics(self):
        """Test per-frequency metrics plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = SignalVisualizer(dpi=100)

            per_frequency_metrics = [
                {'frequency': 1, 'mse': 0.5, 'correlation': 0.9},
                {'frequency': 3, 'mse': 0.3, 'correlation': 0.95},
                {'frequency': 5, 'mse': 0.7, 'correlation': 0.85},
                {'frequency': 7, 'mse': 0.4, 'correlation': 0.92}
            ]

            save_path = Path(tmpdir) / 'test_freq_metrics.png'
            visualizer.plot_per_frequency_metrics(per_frequency_metrics, save_path)

            # Check file was created
            assert save_path.exists()


class TestEndToEndEvaluation:
    """Test complete end-to-end evaluation workflow."""

    def test_full_evaluation_workflow(self, trained_model, temp_dataset):
        """Test complete evaluation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load dataset
            dataset = SignalDataset(temp_dataset)

            # Create evaluator
            evaluator = ModelEvaluator(trained_model, dataset, device='cpu')

            # Run evaluation
            results = evaluator.evaluate_full_dataset(batch_size=2, save_predictions=True)

            # Save results
            results_file = Path(tmpdir) / 'results.json'
            evaluator.save_evaluation_results(results, results_file)
            assert results_file.exists()

            # Generate report
            report_file = Path(tmpdir) / 'report.md'
            evaluator.generate_evaluation_report(results, report_file)
            assert report_file.exists()

            # Statistical analysis
            analyzer = StatisticalAnalyzer()
            freq_analysis = analyzer.analyze_per_frequency_performance(
                results['per_frequency_metrics']
            )
            assert 'best_frequency' in freq_analysis

            # Error analysis
            error_analyzer = ErrorAnalyzer()
            worst = error_analyzer.find_worst_predictions(
                results['per_sample_results'],
                n=3
            )
            assert len(worst) <= 3

            # Create visualizations
            visualizer = SignalVisualizer(dpi=100)

            # Find 3 Hz sample for Graph 1
            f2_sample = None
            sample_idx = None
            for i, sample in enumerate(results['per_sample_results']):
                if sample['frequency'] == 3:
                    f2_sample = sample
                    sample_idx = i
                    break

            if f2_sample:
                graph1_data = {
                    'target': f2_sample['target'],
                    'prediction': f2_sample['prediction'],
                    'mixed_signal': dataset.mixed_signals[sample_idx],
                    'mse': f2_sample['metrics']['mse']
                }

                graph1_path = Path(tmpdir) / 'graph1.png'
                visualizer.create_f2_detailed_plot(graph1_data, graph1_path)
                assert graph1_path.exists()

            # Check overall success
            assert results['overall_metrics']['mse'] >= 0  # Should be non-negative
            assert 'summary' in results
            assert 'mse_target_met' in results['summary']
