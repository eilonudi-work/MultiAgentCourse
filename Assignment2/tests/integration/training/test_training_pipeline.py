"""
Integration tests for complete training pipeline.

Tests training from end-to-end including:
    - Model training with real data
    - Metrics computation
    - Checkpoint saving/loading
    - Callbacks (early stopping, checkpointing)
"""

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback
)
from src.training.metrics import MetricsCalculator, MetricsTracker
from src.training.utils import create_optimizer, create_criterion, set_seed


@pytest.fixture
def temp_dataset():
    """Create temporary dataset for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create small test dataset
    num_samples = 8
    time_steps = 100  # Shorter for faster testing
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
    """Create test configuration."""
    return {
        'model': {
            'lstm': {
                'input_size': 5,
                'hidden_size': 16,  # Small for fast testing
                'num_layers': 1,
                'dropout': 0.0
            }
        }
    }


class TestMetricsCalculator:
    """Test metrics computation."""

    def test_compute_mse(self):
        """Test MSE computation."""
        calculator = MetricsCalculator()

        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[1.5, 2.5], [3.5, 4.5]])

        mse = calculator.compute_mse(pred, target)

        assert mse == pytest.approx(0.25, abs=1e-6)

    def test_compute_correlation(self):
        """Test correlation computation."""
        calculator = MetricsCalculator()

        # Perfect correlation
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([2.0, 4.0, 6.0, 8.0])

        corr = calculator.compute_correlation(pred, target)

        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        calculator = MetricsCalculator()

        pred = np.random.randn(10, 100)
        target = pred + np.random.randn(10, 100) * 0.1

        metrics = calculator.compute_metrics(pred, target)

        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'correlation' in metrics
        assert 'r2' in metrics
        assert 'snr' in metrics

        assert all(np.isfinite(v) for v in metrics.values())


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_update_and_get_latest(self):
        """Test updating and retrieving latest value."""
        tracker = MetricsTracker()

        tracker.update('loss', 0.5, epoch=0)
        tracker.update('loss', 0.3, epoch=1)
        tracker.update('loss', 0.2, epoch=2)

        assert tracker.get_latest('loss') == 0.2

    def test_get_best(self):
        """Test getting best value."""
        tracker = MetricsTracker()

        tracker.update('loss', 0.5, epoch=0)
        tracker.update('loss', 0.2, epoch=1)  # Best
        tracker.update('loss', 0.3, epoch=2)

        assert tracker.get_best('loss', mode='min') == 0.2
        assert tracker.get_best_epoch('loss', mode='min') == 1

    def test_has_improved(self):
        """Test improvement detection."""
        tracker = MetricsTracker()

        tracker.update('loss', 0.5, epoch=0)
        tracker.update('loss', 0.3, epoch=1)

        assert tracker.has_improved('loss', mode='min', patience=1)

        tracker.update('loss', 0.4, epoch=2)

        assert not tracker.has_improved('loss', mode='min', patience=1)


class TestTrainingPipeline:
    """Test complete training pipeline."""

    def test_trainer_initialization(self, test_config, temp_dataset):
        """Test trainer can be initialized."""
        # Setup
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)
        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
            config=test_config
        )

        assert trainer is not None
        assert trainer.model == model
        assert trainer.optimizer == optimizer

    def test_single_epoch_training(self, test_config, temp_dataset):
        """Test training for single epoch."""
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)
        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu'
        )

        # Train for 1 epoch
        summary = trainer.train(num_epochs=1)

        # Check metrics were tracked
        assert 'train_loss' in trainer.metrics_tracker.history
        assert len(trainer.metrics_tracker.get_history('train_loss')) == 1

        # Check loss is finite
        train_loss = trainer.metrics_tracker.get_latest('train_loss')
        assert np.isfinite(train_loss)

    def test_training_with_validation(self, test_config, temp_dataset):
        """Test training with validation set."""
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)

        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)
        val_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu'
        )

        # Train for 2 epochs
        summary = trainer.train(num_epochs=2)

        # Check both train and val metrics
        assert 'train_loss' in trainer.metrics_tracker.history
        assert 'val_loss' in trainer.metrics_tracker.history
        assert 'val_mse' in trainer.metrics_tracker.history
        assert 'val_correlation' in trainer.metrics_tracker.history

    def test_loss_decreases_with_training(self, test_config, temp_dataset):
        """Test that loss decreases during training."""
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)
        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu'
        )

        # Train for 5 epochs
        trainer.train(num_epochs=5)

        # Get loss history
        losses = trainer.metrics_tracker.get_history('train_loss')

        # Loss should generally decrease (allowing for some fluctuation)
        assert losses[-1] <= losses[0] * 1.5  # Allow some tolerance

    def test_checkpoint_callback(self, test_config, temp_dataset):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / 'checkpoints'

            set_seed(42)
            model = ModelFactory.create_model(test_config, device='cpu')
            dataset = SignalDataset(temp_dataset)
            train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)
            val_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=2)

            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            # Add checkpoint callback
            callbacks = [
                CheckpointCallback(
                    checkpoint_dir=checkpoint_dir,
                    save_best=True,
                    save_last=True,
                    monitor='val_loss'
                )
            ]

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                callbacks=callbacks
            )

            # Train
            trainer.train(num_epochs=3)

            # Check checkpoints were saved
            assert (checkpoint_dir / 'best_model.pt').exists()
            assert (checkpoint_dir / 'last_model.pt').exists()

            # Load best checkpoint
            checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint

    def test_early_stopping_callback(self, test_config, temp_dataset):
        """Test early stopping."""
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)
        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)
        val_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Very small LR
        criterion = nn.MSELoss()

        # Add early stopping with very low patience
        callbacks = [
            EarlyStoppingCallback(
                patience=2,
                min_delta=0.0,
                monitor='val_loss'
            )
        ]

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
            callbacks=callbacks
        )

        # Train (should stop early)
        trainer.train(num_epochs=20)

        # Should have stopped before 20 epochs
        actual_epochs = len(trainer.metrics_tracker.get_history('train_loss'))
        assert actual_epochs < 20

    def test_gradient_clipping(self, test_config, temp_dataset):
        """Test gradient clipping."""
        set_seed(42)
        model = ModelFactory.create_model(test_config, device='cpu')
        dataset = SignalDataset(temp_dataset)
        train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.1)  # High LR
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
            grad_clip_value=1.0
        )

        # Train should not explode with gradient clipping
        trainer.train(num_epochs=3)

        losses = trainer.metrics_tracker.get_history('train_loss')
        assert all(np.isfinite(losses))
        assert all(loss < 1e6 for loss in losses)  # No explosion


class TestUtilityFunctions:
    """Test training utility functions."""

    def test_create_optimizer(self, test_config):
        """Test optimizer creation."""
        from src.training.utils import create_optimizer

        model = ModelFactory.create_model(test_config, device='cpu')

        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        assert isinstance(optimizer, optim.Adam)

        optimizer = create_optimizer(model, 'sgd', learning_rate=0.01)
        assert isinstance(optimizer, optim.SGD)

    def test_create_criterion(self):
        """Test criterion creation."""
        from src.training.utils import create_criterion

        criterion = create_criterion('mse')
        assert isinstance(criterion, nn.MSELoss)

        criterion = create_criterion('mae')
        assert isinstance(criterion, nn.L1Loss)

    def test_set_seed(self):
        """Test seed setting."""
        from src.training.utils import set_seed

        set_seed(42)
        x1 = torch.randn(5)

        set_seed(42)
        x2 = torch.randn(5)

        assert torch.allclose(x1, x2)


class TestEndToEndTraining:
    """Test complete end-to-end training scenario."""

    def test_full_training_pipeline(self, test_config, temp_dataset):
        """Test complete training pipeline from start to finish."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / 'checkpoints'

            # Set seed
            set_seed(42)

            # Create model
            model = ModelFactory.create_model(test_config, device='cpu')

            # Create datasets
            dataset = SignalDataset(temp_dataset)
            train_loader = DataLoaderFactory.create_train_loader(dataset, batch_size=4)
            val_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=4)

            # Create optimizer and criterion
            optimizer = create_optimizer(model, 'adam', learning_rate=0.01)
            criterion = create_criterion('mse')

            # Setup callbacks
            callbacks = [
                CheckpointCallback(
                    checkpoint_dir=checkpoint_dir,
                    save_best=True,
                    save_last=True
                ),
                ProgressCallback(print_every_n_epochs=1)
            ]

            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                config=test_config,
                callbacks=callbacks,
                grad_clip_value=1.0
            )

            # Train
            summary = trainer.train(num_epochs=5)

            # Verify training completed
            assert len(trainer.metrics_tracker.get_history('train_loss')) == 5
            assert len(trainer.metrics_tracker.get_history('val_loss')) == 5

            # Verify checkpoints exist
            assert (checkpoint_dir / 'best_model.pt').exists()
            assert (checkpoint_dir / 'last_model.pt').exists()

            # Verify metrics are reasonable
            best_metrics = trainer.get_best_metrics()
            assert 'val_loss' in best_metrics
            assert np.isfinite(best_metrics['val_loss'])

            # Load best checkpoint and verify
            loaded_model = ModelFactory.create_from_checkpoint(
                checkpoint_dir / 'best_model.pt',
                device='cpu'
            )
            assert isinstance(loaded_model, type(model))
