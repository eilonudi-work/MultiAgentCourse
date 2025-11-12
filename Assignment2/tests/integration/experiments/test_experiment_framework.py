"""
Integration tests for experiment management framework.

Tests hyperparameter tuning infrastructure including:
- Experiment management and execution
- Experiment tracking and analysis
- Result comparison and visualization
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.pytorch_dataset import SignalDataset
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.experiment_tracker import ExperimentTracker
from src.experiments.experiment_comparator import ExperimentComparator


@pytest.fixture
def temp_dataset():
    """Create temporary test dataset."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create small dataset
    num_samples = 4
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
def test_config():
    """Create minimal test configuration."""
    return {
        'model': {
            'lstm': {
                'input_size': 5,
                'hidden_size': 16,
                'num_layers': 1,
                'dropout': 0.0
            }
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'criterion': 'mse',
            'grad_clip': 1.0,
            'seed': 42,
            'early_stop_patience': 5
        }
    }


class TestExperimentManager:
    """Test experiment management functionality."""

    def test_manager_initialization(self, test_config):
        """Test experiment manager can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            assert manager is not None
            assert manager.base_config == test_config
            assert manager.device == 'cpu'
            assert len(manager.experiments) == 0

    def test_search_space_definition(self, test_config):
        """Test search space can be defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir
            )

            search_space = manager.define_search_space()

            assert 'hidden_size' in search_space
            assert 'num_layers' in search_space
            assert 'learning_rate' in search_space
            assert isinstance(search_space['hidden_size'], list)
            assert len(search_space['hidden_size']) > 0

    def test_single_experiment_execution(self, test_config, temp_dataset):
        """Test single experiment can be run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            result = manager.run_experiment(
                config=test_config,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2,
                experiment_name='test_exp'
            )

            # Verify result structure
            assert result is not None
            assert 'experiment_name' in result
            assert 'config' in result
            assert 'metrics' in result
            assert 'success' in result

            # Verify experiment was tracked
            assert len(manager.experiments) == 1

            # Verify metrics
            if result['success']:
                assert 'best_val_loss' in result['metrics']
                assert 'best_train_loss' in result['metrics']
                assert np.isfinite(result['metrics']['best_val_loss'])

    def test_grid_search_execution(self, test_config, temp_dataset):
        """Test grid search can be executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            # Define small parameter grid
            param_grid = {
                'hidden_size': [16, 32],
                'learning_rate': [0.01, 0.001]
            }

            results = manager.run_grid_search(
                param_grid=param_grid,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2,
                max_experiments=4
            )

            # Should run all 4 combinations
            assert len(results) == 4
            assert len(manager.experiments) == 4

    def test_random_search_execution(self, test_config, temp_dataset):
        """Test random search can be executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            # Define search space
            search_space = {
                'hidden_size': [16, 32],
                'num_layers': [1, 2]
            }

            results = manager.run_random_search(
                search_space=search_space,
                train_dataset=dataset,
                val_dataset=dataset,
                n_experiments=3,
                num_epochs=2,
                seed=42
            )

            # Should run 3 experiments
            assert len(results) == 3
            assert len(manager.experiments) == 3

    def test_get_best_experiment(self, test_config, temp_dataset):
        """Test best experiment can be retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            # Run multiple experiments
            for i in range(3):
                config = test_config.copy()
                config['training']['learning_rate'] = 0.01 * (i + 1)

                manager.run_experiment(
                    config=config,
                    train_dataset=dataset,
                    val_dataset=dataset,
                    num_epochs=2
                )

            # Get best
            best = manager.get_best_experiment(metric='best_val_loss', mode='min')

            assert best is not None
            assert 'metrics' in best
            assert 'best_val_loss' in best['metrics']

    def test_experiment_persistence(self, test_config, temp_dataset):
        """Test experiments are saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manager and run experiment
            manager1 = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            manager1.run_experiment(
                config=test_config,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2
            )

            # Create new manager (should load existing experiments)
            manager2 = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            assert len(manager2.experiments) == 1

    def test_export_best_config(self, test_config, temp_dataset):
        """Test best configuration can be exported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            # Run experiment
            manager.run_experiment(
                config=test_config,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2
            )

            # Export best config
            output_path = Path(tmpdir) / 'best_config.yaml'
            result_path = manager.export_best_config(str(output_path))

            assert result_path is not None
            assert Path(result_path).exists()


class TestExperimentTracker:
    """Test experiment tracking functionality."""

    def test_tracker_initialization(self):
        """Test tracker can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            # Create empty experiments file
            with open(exp_file, 'w') as f:
                json.dump([], f)

            tracker = ExperimentTracker(str(exp_file))

            assert tracker is not None
            assert len(tracker.experiments) == 0

    def test_load_experiments(self, test_config, temp_dataset):
        """Test tracker can load experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and run experiments
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            manager.run_experiment(
                config=test_config,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2
            )

            # Load with tracker
            exp_file = Path(tmpdir) / 'experiments.json'
            tracker = ExperimentTracker(str(exp_file))

            assert len(tracker.experiments) == 1

    def test_filter_successful_experiments(self):
        """Test filtering successful experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            # Create mock experiments
            experiments = [
                {'success': True, 'metrics': {'best_val_loss': 0.5}},
                {'success': False, 'error': 'Test error'},
                {'success': True, 'metrics': {'best_val_loss': 0.3}}
            ]

            with open(exp_file, 'w') as f:
                json.dump(experiments, f)

            tracker = ExperimentTracker(str(exp_file))

            successful = tracker.get_successful_experiments()
            assert len(successful) == 2

            failed = tracker.get_failed_experiments()
            assert len(failed) == 1

    def test_get_best_n_experiments(self):
        """Test getting top N experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            # Create mock experiments with different losses
            experiments = [
                {'success': True, 'metrics': {'best_val_loss': 0.5}},
                {'success': True, 'metrics': {'best_val_loss': 0.3}},
                {'success': True, 'metrics': {'best_val_loss': 0.7}},
                {'success': True, 'metrics': {'best_val_loss': 0.2}}
            ]

            with open(exp_file, 'w') as f:
                json.dump(experiments, f)

            tracker = ExperimentTracker(str(exp_file))

            best_n = tracker.get_best_n_experiments(n=2, metric='best_val_loss', mode='min')

            assert len(best_n) == 2
            assert best_n[0]['metrics']['best_val_loss'] == 0.2
            assert best_n[1]['metrics']['best_val_loss'] == 0.3

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            experiments = [
                {
                    'experiment_name': 'exp1',
                    'success': True,
                    'metrics': {'best_val_loss': 0.5},
                    'config': {
                        'model': {'lstm': {'hidden_size': 64, 'num_layers': 2}},
                        'training': {'learning_rate': 0.001, 'batch_size': 32}
                    }
                }
            ]

            with open(exp_file, 'w') as f:
                json.dump(experiments, f)

            tracker = ExperimentTracker(str(exp_file))

            df = tracker.to_dataframe(include_config=True)

            assert not df.empty
            assert 'experiment_name' in df.columns
            assert 'best_val_loss' in df.columns
            assert 'hidden_size' in df.columns
            assert 'learning_rate' in df.columns

    def test_compute_statistics(self):
        """Test computing statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            experiments = [
                {'success': True, 'metrics': {'best_val_loss': 0.5}},
                {'success': True, 'metrics': {'best_val_loss': 0.3}},
                {'success': True, 'metrics': {'best_val_loss': 0.7}}
            ]

            with open(exp_file, 'w') as f:
                json.dump(experiments, f)

            tracker = ExperimentTracker(str(exp_file))

            stats = tracker.compute_statistics('best_val_loss')

            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert stats['count'] == 3
            assert stats['min'] == 0.3
            assert stats['max'] == 0.7


class TestExperimentComparator:
    """Test experiment comparison and visualization."""

    def test_comparator_initialization(self):
        """Test comparator can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            with open(exp_file, 'w') as f:
                json.dump([], f)

            tracker = ExperimentTracker(str(exp_file))
            comparator = ExperimentComparator(tracker, output_dir=tmpdir)

            assert comparator is not None
            assert comparator.tracker == tracker

    def test_create_comparison_table(self):
        """Test creating comparison table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_file = Path(tmpdir) / 'experiments.json'

            experiments = [
                {
                    'experiment_name': 'exp1',
                    'success': True,
                    'metrics': {'best_val_loss': 0.5, 'best_train_loss': 0.4},
                    'config': {
                        'model': {'lstm': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}},
                        'training': {'learning_rate': 0.001, 'batch_size': 32}
                    }
                },
                {
                    'experiment_name': 'exp2',
                    'success': True,
                    'metrics': {'best_val_loss': 0.3, 'best_train_loss': 0.2},
                    'config': {
                        'model': {'lstm': {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2}},
                        'training': {'learning_rate': 0.0005, 'batch_size': 64}
                    }
                }
            ]

            with open(exp_file, 'w') as f:
                json.dump(experiments, f)

            tracker = ExperimentTracker(str(exp_file))
            comparator = ExperimentComparator(tracker, output_dir=tmpdir)

            table = comparator.create_comparison_table(top_n=2)

            assert not table.empty
            assert len(table) == 2
            assert 'Experiment' in table.columns
            assert 'Hidden Size' in table.columns
            assert 'best_val_loss' in table.columns


class TestEndToEndTuning:
    """Test complete end-to-end tuning workflow."""

    def test_full_tuning_workflow(self, test_config, temp_dataset):
        """Test complete hyperparameter tuning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create experiment manager
            manager = ExperimentManager(
                base_config=test_config,
                output_dir=tmpdir,
                device='cpu'
            )

            dataset = SignalDataset(temp_dataset)

            # 2. Run baseline
            baseline_result = manager.run_experiment(
                config=test_config,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2,
                experiment_name='baseline'
            )

            assert baseline_result['success']

            # 3. Run small grid search
            param_grid = {'hidden_size': [16, 32], 'learning_rate': [0.01, 0.001]}

            grid_results = manager.run_grid_search(
                param_grid=param_grid,
                train_dataset=dataset,
                val_dataset=dataset,
                num_epochs=2,
                max_experiments=4
            )

            assert len(grid_results) == 4

            # 4. Get best experiment
            best = manager.get_best_experiment()
            assert best is not None

            # 5. Export best config
            best_config_path = manager.export_best_config(
                f'{tmpdir}/best_config.yaml'
            )
            assert Path(best_config_path).exists()

            # 6. Load with tracker and analyze
            tracker = ExperimentTracker(f'{tmpdir}/experiments.json')
            assert len(tracker.experiments) == 5  # 1 baseline + 4 grid

            # 7. Create comparator and generate report
            comparator = ExperimentComparator(tracker, output_dir=tmpdir)
            comparator.generate_summary_report(f'{tmpdir}/summary.md')

            assert Path(f'{tmpdir}/summary.md').exists()
