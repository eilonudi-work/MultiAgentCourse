"""Integration tests for Phase 2 components working together."""

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.lstm_model import SignalExtractionLSTM
from src.models.state_manager import StatefulProcessor
from src.models.model_factory import ModelFactory
from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'model': {
            'lstm': {
                'input_size': 5,
                'hidden_size': 32,
                'num_layers': 2,
                'dropout': 0.1
            }
        }
    }


@pytest.fixture
def temp_dataset():
    """Create temporary dataset for integration testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = Path(f.name)

    # Create realistic test dataset
    num_samples = 10
    time_steps = 1000  # Shorter for faster testing
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


class TestModelCreationAndDataLoading:
    """Test model creation and data loading integration."""

    def test_model_factory_with_config(self, test_config):
        """Test creating model from config."""
        model = ModelFactory.create_model(test_config, device='cpu')

        assert isinstance(model, SignalExtractionLSTM)
        assert model.hidden_size == 32
        assert model.num_layers == 2

    def test_dataset_loading_and_model_forward(self, test_config, temp_dataset):
        """Test loading dataset and running model forward pass."""
        # Create model
        model = ModelFactory.create_model(test_config, device='cpu')

        # Load dataset
        dataset = SignalDataset(temp_dataset)
        loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=4)

        # Get batch and run forward pass
        batch = next(iter(loader))
        batch_size = batch['mixed_signal'].size(0)

        # Create input: [S(t), C1, C2, C3, C4]
        # For simplicity, test with first time step
        signal_t = batch['mixed_signal'][:, 0:1].unsqueeze(-1)  # (batch, 1, 1)
        condition = batch['condition_vector'].unsqueeze(1)  # (batch, 1, 4)
        input_t = torch.cat([signal_t, condition], dim=2)  # (batch, 1, 5)

        model.eval()
        with torch.no_grad():
            output, _ = model(input_t)

        assert output.shape == (batch_size, 1, 1)
        assert not torch.isnan(output).any()


class TestStatefulProcessingWithRealData:
    """Test stateful processing with real dataset."""

    def test_stateful_processor_with_dataset_sample(self, test_config, temp_dataset):
        """Test StatefulProcessor with real dataset sample."""
        # Create model and processor
        model = ModelFactory.create_model(test_config, device='cpu')
        processor = StatefulProcessor(model)

        # Load dataset
        dataset = SignalDataset(temp_dataset)
        sample = dataset[0]

        # Convert to numpy for processor
        sample_np = {
            'mixed_signal': sample['mixed_signal'].numpy(),
            'condition_vector': sample['condition_vector'].numpy()
        }

        # Process sample
        predictions = processor.process_sample(sample_np, reset_state=True)

        assert predictions.shape == (1000,)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()

    def test_batch_processing_with_dataloader(self, test_config, temp_dataset):
        """Test batch processing with DataLoader."""
        model = ModelFactory.create_model(test_config, device='cpu')
        processor = StatefulProcessor(model)

        dataset = SignalDataset(temp_dataset)
        loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=4)

        batch = next(iter(loader))

        # Process batch (manually construct proper format)
        batch_dict = {
            'mixed_signals': batch['mixed_signal'],
            'condition_vectors': batch['condition_vector']
        }

        predictions = processor.process_batch(batch_dict, reset_state=True)

        batch_size = batch['mixed_signal'].size(0)
        assert predictions.shape == (batch_size, 1000, 1)
        assert not torch.isnan(predictions).any()

    def test_multiple_samples_sequential_processing(self, test_config, temp_dataset):
        """Test processing multiple samples sequentially."""
        model = ModelFactory.create_model(test_config, device='cpu')
        processor = StatefulProcessor(model)

        dataset = SignalDataset(temp_dataset)

        predictions_list = []
        for i in range(3):
            sample = dataset[i]
            sample_np = {
                'mixed_signal': sample['mixed_signal'].numpy(),
                'condition_vector': sample['condition_vector'].numpy()
            }

            predictions = processor.process_sample(sample_np, reset_state=True)
            predictions_list.append(predictions)

        # All should have correct shape
        for predictions in predictions_list:
            assert predictions.shape == (1000,)
            assert not np.isnan(predictions).any()


class TestCheckpointSaveAndLoad:
    """Test model checkpoint saving and loading."""

    def test_save_and_load_checkpoint(self, test_config):
        """Test saving and loading model checkpoint."""
        # Create and modify model
        model = ModelFactory.create_model(test_config, device='cpu')

        # Get initial output
        x = torch.randn(1, 1, 5)
        model.eval()
        with torch.no_grad():
            output_before, _ = model(x)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_model.pt'

            ModelFactory.save_checkpoint(
                model,
                checkpoint_path,
                epoch=10,
                loss=0.123,
                config=test_config
            )

            assert checkpoint_path.exists()

            # Load checkpoint
            loaded_model = ModelFactory.create_from_checkpoint(
                checkpoint_path,
                device='cpu'
            )

            # Verify same output
            loaded_model.eval()
            with torch.no_grad():
                output_after, _ = loaded_model(x)

            assert torch.allclose(output_before, output_after, rtol=1e-5)

    def test_checkpoint_contains_metadata(self, test_config):
        """Test that checkpoint contains all metadata."""
        model = ModelFactory.create_model(test_config, device='cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_model.pt'

            ModelFactory.save_checkpoint(
                model,
                checkpoint_path,
                epoch=5,
                loss=0.05,
                config=test_config
            )

            # Load and inspect checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            assert 'model_state_dict' in checkpoint
            assert 'model_info' in checkpoint
            assert 'config' in checkpoint
            assert 'epoch' in checkpoint
            assert 'loss' in checkpoint

            assert checkpoint['epoch'] == 5
            assert checkpoint['loss'] == 0.05

    def test_load_checkpoint_without_config(self, test_config):
        """Test loading checkpoint that doesn't have config."""
        model = ModelFactory.create_model(test_config, device='cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_model.pt'

            # Save manually without config
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 1,
                'train_loss': 0.1
            }
            torch.save(checkpoint, checkpoint_path)

            # Should still load using state dict inference
            loaded_model = ModelFactory.create_from_checkpoint(checkpoint_path)

            assert isinstance(loaded_model, SignalExtractionLSTM)
            assert loaded_model.hidden_size == 32  # Should infer from state dict


class TestTrainingLoop:
    """Test basic training loop integration."""

    def test_single_batch_training_step(self, test_config, temp_dataset):
        """Test a single training step."""
        model = ModelFactory.create_model(test_config, device='cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = SignalDataset(temp_dataset)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=4)

        batch = next(iter(loader))
        batch_size = batch['mixed_signal'].size(0)

        model.train()

        # Process one time step for simplicity
        signal_t = batch['mixed_signal'][:, 0:1].unsqueeze(-1)
        condition = batch['condition_vector'].unsqueeze(1)
        input_t = torch.cat([signal_t, condition], dim=2)

        target_t = batch['target_signal'][:, 0:1].unsqueeze(-1)

        # Forward pass
        optimizer.zero_grad()
        output, _ = model(input_t)

        # Compute loss
        loss = criterion(output, target_t)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_multi_step_training(self, test_config, temp_dataset):
        """Test training for multiple steps."""
        model = ModelFactory.create_model(test_config, device='cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        dataset = SignalDataset(temp_dataset)
        loader = DataLoaderFactory.create_train_loader(dataset, batch_size=4, shuffle=False)

        model.train()
        losses = []

        # Train for a few steps
        for step, batch in enumerate(loader):
            if step >= 3:
                break

            batch_size = batch['mixed_signal'].size(0)

            # Simple training on first time step
            signal_t = batch['mixed_signal'][:, 0:1].unsqueeze(-1)
            condition = batch['condition_vector'].unsqueeze(1)
            input_t = torch.cat([signal_t, condition], dim=2)

            target_t = batch['target_signal'][:, 0:1].unsqueeze(-1)

            optimizer.zero_grad()
            output, _ = model(input_t)
            loss = criterion(output, target_t)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Check all losses are finite
        assert all(np.isfinite(losses))
        assert len(losses) == 3

    def test_overfitting_small_dataset(self, test_config):
        """Test that model can overfit to small dataset."""
        model = ModelFactory.create_model(test_config, device='cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Create tiny dataset
        x = torch.randn(8, 1, 5)
        y = torch.randn(8, 1, 1)

        model.train()
        initial_loss = None

        # Train for many steps
        for epoch in range(100):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()

        final_loss = loss.item()

        # Should overfit (loss should decrease significantly)
        assert final_loss < initial_loss * 0.3


class TestModelInfo:
    """Test model information and statistics."""

    def test_get_model_info(self, test_config):
        """Test getting comprehensive model info."""
        model = ModelFactory.create_model(test_config, device='cpu')
        info = ModelFactory.get_model_info(model)

        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'non_trainable_parameters' in info
        assert 'estimated_size_mb' in info

        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] == info['total_parameters']
        assert info['non_trainable_parameters'] == 0

    def test_count_parameters_by_layer(self, test_config):
        """Test counting parameters by layer."""
        model = ModelFactory.create_model(test_config, device='cpu')
        counts = ModelFactory.count_parameters(model)

        assert 'lstm' in counts
        assert 'fc' in counts
        assert 'total' in counts

        assert counts['lstm'] > 0
        assert counts['fc'] > 0
        assert counts['total'] == counts['lstm'] + counts['fc']

    def test_print_model_summary(self, test_config, capsys):
        """Test printing model summary."""
        model = ModelFactory.create_model(test_config, device='cpu')

        ModelFactory.print_model_summary(model)

        captured = capsys.readouterr()
        output = captured.out

        assert 'SignalExtractionLSTM' in output
        assert 'Parameters' in output
        assert str(model.hidden_size) in output


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_full_inference_pipeline(self, test_config, temp_dataset):
        """Test complete inference pipeline from dataset to predictions."""
        # 1. Create model
        model = ModelFactory.create_model(test_config, device='cpu')

        # 2. Load dataset
        dataset = SignalDataset(temp_dataset)

        # 3. Create processor
        processor = StatefulProcessor(model)

        # 4. Process sample
        sample = dataset[0]
        sample_np = {
            'mixed_signal': sample['mixed_signal'].numpy(),
            'condition_vector': sample['condition_vector'].numpy()
        }

        predictions = processor.process_sample(sample_np, reset_state=True)

        # 5. Verify output
        assert predictions.shape == (1000,)
        assert not np.isnan(predictions).any()

        # 6. Compare with target
        target = sample['target_signal'].numpy()
        assert predictions.shape == target.shape

    def test_full_training_pipeline(self, test_config, temp_dataset):
        """Test complete training pipeline."""
        # 1. Create model
        model = ModelFactory.create_model(test_config, device='cpu')
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 2. Load dataset
        dataset = SignalDataset(temp_dataset, normalize=True)
        train_loader = DataLoaderFactory.create_train_loader(
            dataset,
            batch_size=4,
            shuffle=True
        )

        # 3. Train for one epoch
        model.train()
        epoch_losses = []

        for batch in train_loader:
            # Simplified: train on first time step only
            signal_t = batch['mixed_signal'][:, 0:1].unsqueeze(-1)
            condition = batch['condition_vector'].unsqueeze(1)
            input_t = torch.cat([signal_t, condition], dim=2)

            target_t = batch['target_signal'][:, 0:1].unsqueeze(-1)

            optimizer.zero_grad()
            output, _ = model(input_t)
            loss = criterion(output, target_t)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # 4. Verify training ran
        assert len(epoch_losses) > 0
        assert all(np.isfinite(epoch_losses))

        # 5. Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'model.pt'
            ModelFactory.save_checkpoint(
                model,
                checkpoint_path,
                optimizer=optimizer,
                epoch=1,
                loss=np.mean(epoch_losses),
                config=test_config
            )

            assert checkpoint_path.exists()

            # 6. Load and verify
            loaded_model = ModelFactory.create_from_checkpoint(checkpoint_path)
            assert isinstance(loaded_model, SignalExtractionLSTM)

    def test_evaluation_pipeline(self, test_config, temp_dataset):
        """Test complete evaluation pipeline."""
        # 1. Create and setup model
        model = ModelFactory.create_model(test_config, device='cpu')
        model.eval()

        # 2. Load dataset
        dataset = SignalDataset(temp_dataset)
        eval_loader = DataLoaderFactory.create_eval_loader(dataset, batch_size=4)

        # 3. Create processor
        processor = StatefulProcessor(model)

        # 4. Evaluate all samples
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in eval_loader:
                for i in range(batch['mixed_signal'].size(0)):
                    sample_np = {
                        'mixed_signal': batch['mixed_signal'][i].numpy(),
                        'condition_vector': batch['condition_vector'][i].numpy()
                    }

                    predictions = processor.process_sample(sample_np, reset_state=True)
                    target = batch['target_signal'][i].numpy()

                    all_predictions.append(predictions)
                    all_targets.append(target)

        # 5. Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = np.mean((all_predictions - all_targets) ** 2)

        assert np.isfinite(mse)
        assert mse >= 0
        assert len(all_predictions) == len(dataset)
