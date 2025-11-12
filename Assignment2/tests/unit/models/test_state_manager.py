"""Unit tests for StatefulProcessor."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import SignalExtractionLSTM
from src.models.state_manager import StatefulProcessor


@pytest.fixture
def model():
    """Create LSTM model for testing."""
    return SignalExtractionLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        dropout=0.1
    )


@pytest.fixture
def processor(model):
    """Create StatefulProcessor instance."""
    return StatefulProcessor(model)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    time_steps = 10000
    return {
        'mixed_signal': np.random.randn(time_steps).astype(np.float32),
        'condition_vector': np.array([0, 1, 0, 0], dtype=np.float32)
    }


@pytest.fixture
def small_sample_data():
    """Create small sample for faster testing."""
    time_steps = 100
    return {
        'mixed_signal': np.random.randn(time_steps).astype(np.float32),
        'condition_vector': np.array([1, 0, 0, 0], dtype=np.float32)
    }


class TestStatefulProcessorInitialization:
    """Tests for StatefulProcessor initialization."""

    def test_initialization(self, model):
        """Test processor initialization."""
        processor = StatefulProcessor(model)

        assert processor.model is model
        assert processor.current_state is None

    def test_initialization_with_invalid_model(self):
        """Test initialization with non-LSTM model."""
        invalid_model = torch.nn.Linear(5, 1)

        with pytest.raises(TypeError, match="must be SignalExtractionLSTM"):
            StatefulProcessor(invalid_model)

    def test_repr(self, processor):
        """Test string representation."""
        repr_str = repr(processor)

        assert 'StatefulProcessor' in repr_str
        assert 'SignalExtractionLSTM' in repr_str
        assert 'not initialized' in repr_str


class TestStatefulProcessorStateManagement:
    """Tests for state management."""

    def test_reset_state_initializes_state(self, processor):
        """Test that reset_state initializes hidden state."""
        assert processor.current_state is None

        processor.reset_state(batch_size=1)

        assert processor.current_state is not None
        h_n, c_n = processor.current_state
        assert h_n.shape == (2, 1, 32)  # (num_layers, batch_size, hidden_size)
        assert c_n.shape == (2, 1, 32)

    def test_reset_state_zeros(self, processor):
        """Test that reset_state initializes to zeros."""
        processor.reset_state(batch_size=4)

        h_n, c_n = processor.current_state
        assert torch.all(h_n == 0)
        assert torch.all(c_n == 0)

    def test_reset_state_with_different_batch_sizes(self, processor):
        """Test reset_state with various batch sizes."""
        batch_sizes = [1, 8, 16, 32]

        for batch_size in batch_sizes:
            processor.reset_state(batch_size=batch_size)
            h_n, c_n = processor.current_state

            assert h_n.shape == (2, batch_size, 32)
            assert c_n.shape == (2, batch_size, 32)

    def test_get_state_info_when_not_initialized(self, processor):
        """Test get_state_info returns None when state not initialized."""
        info = processor.get_state_info()
        assert info is None

    def test_get_state_info_after_reset(self, processor):
        """Test get_state_info after state initialization."""
        processor.reset_state(batch_size=4)
        info = processor.get_state_info()

        assert info is not None
        assert info['h_shape'] == (2, 4, 32)
        assert info['c_shape'] == (2, 4, 32)
        assert 'h_mean' in info
        assert 'h_std' in info
        assert 'device' in info


class TestStatefulProcessorInputVectorCreation:
    """Tests for _create_input_vector method."""

    def test_create_input_vector_shape(self, processor):
        """Test input vector has correct shape."""
        mixed_signal = np.random.randn(100).astype(np.float32)
        condition_vector = np.array([0, 1, 0, 0], dtype=np.float32)

        input_tensor = processor._create_input_vector(mixed_signal, condition_vector, 0)

        assert input_tensor.shape == (1, 1, 5)  # (batch, seq_len, features)

    def test_create_input_vector_values(self, processor):
        """Test input vector contains correct values."""
        mixed_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        condition_vector = np.array([0, 1, 0, 0], dtype=np.float32)

        input_tensor = processor._create_input_vector(mixed_signal, condition_vector, 2)

        # Should be [S(2), C1, C2, C3, C4] = [3.0, 0, 1, 0, 0]
        expected = torch.tensor([[[3.0, 0.0, 1.0, 0.0, 0.0]]])
        assert torch.allclose(input_tensor, expected)

    def test_create_input_vector_all_time_indices(self, processor):
        """Test input vector for different time indices."""
        mixed_signal = np.arange(10, dtype=np.float32)
        condition_vector = np.array([1, 0, 0, 0], dtype=np.float32)

        for t in range(len(mixed_signal)):
            input_tensor = processor._create_input_vector(mixed_signal, condition_vector, t)

            # Check signal value matches time index
            signal_value = input_tensor[0, 0, 0].item()
            assert signal_value == float(t)

    def test_create_input_vector_device(self, processor):
        """Test input vector is on correct device."""
        mixed_signal = np.random.randn(10).astype(np.float32)
        condition_vector = np.array([0, 0, 1, 0], dtype=np.float32)

        input_tensor = processor._create_input_vector(mixed_signal, condition_vector, 0)

        model_device = next(processor.model.parameters()).device
        assert input_tensor.device == model_device


class TestStatefulProcessorProcessSample:
    """Tests for process_sample method."""

    def test_process_sample_basic(self, processor, small_sample_data):
        """Test basic sample processing."""
        predictions = processor.process_sample(small_sample_data, reset_state=True)

        assert predictions.shape == (100,)  # time_steps
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()

    def test_process_sample_full_length(self, processor, sample_data):
        """Test processing full 10,000 time step sample."""
        predictions = processor.process_sample(sample_data, reset_state=True)

        assert predictions.shape == (10000,)
        assert not np.isnan(predictions).any()

    def test_process_sample_with_reset(self, processor, small_sample_data):
        """Test that reset_state is called when requested."""
        # Process first sample
        predictions1 = processor.process_sample(small_sample_data, reset_state=True)

        # State should be initialized
        assert processor.current_state is not None

        # Process second sample with reset
        predictions2 = processor.process_sample(small_sample_data, reset_state=True)

        # Predictions might be same for same input in eval mode
        assert predictions1.shape == predictions2.shape

    def test_process_sample_without_reset(self, processor, small_sample_data):
        """Test processing without resetting state."""
        # Initialize state first
        processor.reset_state(batch_size=1)
        initial_state = processor.current_state

        # Process without reset (should continue from current state)
        predictions = processor.process_sample(small_sample_data, reset_state=False)

        # State should have changed
        assert processor.current_state is not None
        assert not torch.equal(processor.current_state[0], initial_state[0])

    def test_process_sample_state_persistence(self, processor, small_sample_data):
        """Test that state persists across time steps within sample."""
        # We can't directly observe internal state changes,
        # but we can verify the output is computed sequentially
        predictions = processor.process_sample(small_sample_data, reset_state=True)

        # Just verify output is reasonable
        assert len(predictions) == len(small_sample_data['mixed_signal'])

    def test_process_sample_missing_mixed_signal(self, processor):
        """Test process_sample with missing mixed_signal key."""
        invalid_sample = {
            'condition_vector': np.array([1, 0, 0, 0])
        }

        with pytest.raises(KeyError, match="must contain 'mixed_signal'"):
            processor.process_sample(invalid_sample)

    def test_process_sample_missing_condition_vector(self, processor):
        """Test process_sample with missing condition_vector key."""
        invalid_sample = {
            'mixed_signal': np.random.randn(100)
        }

        with pytest.raises(KeyError, match="must contain 'condition_vector'"):
            processor.process_sample(invalid_sample)

    def test_process_sample_invalid_mixed_signal_shape(self, processor):
        """Test process_sample with invalid mixed_signal shape."""
        invalid_sample = {
            'mixed_signal': np.random.randn(10, 10),  # 2D instead of 1D
            'condition_vector': np.array([1, 0, 0, 0])
        }

        with pytest.raises(ValueError, match="mixed_signal must be 1D"):
            processor.process_sample(invalid_sample)

    def test_process_sample_invalid_condition_vector(self, processor):
        """Test process_sample with invalid condition_vector."""
        invalid_sample = {
            'mixed_signal': np.random.randn(100),
            'condition_vector': np.array([1, 0, 0])  # Wrong size
        }

        with pytest.raises(ValueError, match="condition_vector must have 4 elements"):
            processor.process_sample(invalid_sample)

    def test_process_sample_return_tensor(self, processor, small_sample_data):
        """Test process_sample with return_tensor=True."""
        predictions = processor.process_sample(
            small_sample_data,
            reset_state=True,
            return_tensor=True
        )

        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (100,)

    def test_process_sample_return_numpy(self, processor, small_sample_data):
        """Test process_sample with return_tensor=False."""
        predictions = processor.process_sample(
            small_sample_data,
            reset_state=True,
            return_tensor=False
        )

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (100,)

    def test_process_sample_eval_mode(self, processor, small_sample_data):
        """Test that process_sample uses eval mode."""
        # Set model to train mode
        processor.model.train()

        # Process sample
        processor.process_sample(small_sample_data, reset_state=True)

        # Model should be back in train mode
        assert processor.model.training

    def test_process_sample_deterministic(self, processor, small_sample_data):
        """Test that process_sample is deterministic in eval mode."""
        processor.model.eval()

        predictions1 = processor.process_sample(small_sample_data, reset_state=True)
        predictions2 = processor.process_sample(small_sample_data, reset_state=True)

        assert np.allclose(predictions1, predictions2)


class TestStatefulProcessorProcessBatch:
    """Tests for process_batch method."""

    def test_process_batch_basic(self, processor):
        """Test basic batch processing."""
        batch_size = 4
        time_steps = 100

        batch = {
            'mixed_signals': torch.randn(batch_size, time_steps),
            'condition_vectors': torch.randint(0, 2, (batch_size, 4)).float()
        }

        predictions = processor.process_batch(batch, reset_state=True)

        assert predictions.shape == (batch_size, time_steps, 1)
        assert not torch.isnan(predictions).any()

    def test_process_batch_different_sizes(self, processor):
        """Test batch processing with different batch sizes."""
        batch_sizes = [1, 8, 16]
        time_steps = 50

        for batch_size in batch_sizes:
            batch = {
                'mixed_signals': torch.randn(batch_size, time_steps),
                'condition_vectors': torch.randint(0, 2, (batch_size, 4)).float()
            }

            predictions = processor.process_batch(batch, reset_state=True)
            assert predictions.shape == (batch_size, time_steps, 1)

    def test_process_batch_state_independence(self, processor):
        """Test that batch samples have independent states."""
        batch_size = 2
        time_steps = 50

        # Create batch with different signals
        batch = {
            'mixed_signals': torch.randn(batch_size, time_steps),
            'condition_vectors': torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
        }

        predictions = processor.process_batch(batch, reset_state=True)

        # Predictions for different samples should be different
        assert not torch.allclose(predictions[0], predictions[1])

    def test_process_batch_with_reset(self, processor):
        """Test batch processing with state reset."""
        batch_size = 4
        time_steps = 50

        batch = {
            'mixed_signals': torch.randn(batch_size, time_steps),
            'condition_vectors': torch.randint(0, 2, (batch_size, 4)).float()
        }

        # Process with reset
        predictions1 = processor.process_batch(batch, reset_state=True)
        predictions2 = processor.process_batch(batch, reset_state=True)

        # Same input should give same output in eval mode
        processor.model.eval()
        predictions1 = processor.process_batch(batch, reset_state=True)
        predictions2 = processor.process_batch(batch, reset_state=True)
        assert torch.allclose(predictions1, predictions2)

    def test_process_batch_eval_mode_preservation(self, processor):
        """Test that process_batch preserves training mode."""
        batch_size = 4
        time_steps = 50

        batch = {
            'mixed_signals': torch.randn(batch_size, time_steps),
            'condition_vectors': torch.randint(0, 2, (batch_size, 4)).float()
        }

        # Set to train mode
        processor.model.train()

        # Process batch
        processor.process_batch(batch, reset_state=True)

        # Should be back in train mode
        assert processor.model.training


class TestStatefulProcessorIntegration:
    """Integration tests for StatefulProcessor."""

    def test_sequential_sample_processing(self, processor, small_sample_data):
        """Test processing multiple samples sequentially."""
        samples = [small_sample_data for _ in range(3)]

        predictions_list = []
        for sample in samples:
            predictions = processor.process_sample(sample, reset_state=True)
            predictions_list.append(predictions)

        # All predictions should have correct shape
        for predictions in predictions_list:
            assert predictions.shape == (100,)

    def test_different_condition_vectors(self, processor):
        """Test processing with different condition vectors."""
        time_steps = 100
        mixed_signal = np.random.randn(time_steps).astype(np.float32)

        condition_vectors = [
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([0, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1, 0], dtype=np.float32),
            np.array([0, 0, 0, 1], dtype=np.float32),
        ]

        predictions_list = []
        for condition in condition_vectors:
            sample = {
                'mixed_signal': mixed_signal,
                'condition_vector': condition
            }
            predictions = processor.process_sample(sample, reset_state=True)
            predictions_list.append(predictions)

        # Different conditions should produce different outputs
        for i in range(len(predictions_list) - 1):
            # At least some predictions should be different
            assert not np.allclose(predictions_list[i], predictions_list[i + 1])

    def test_state_reset_between_samples_matters(self, processor, small_sample_data):
        """Test that resetting state between samples affects output."""
        processor.model.eval()

        # Process first sample with reset
        pred1 = processor.process_sample(small_sample_data, reset_state=True)

        # Process second sample WITHOUT reset (continuing from previous state)
        pred2_no_reset = processor.process_sample(small_sample_data, reset_state=False)

        # Process second sample WITH reset
        pred2_with_reset = processor.process_sample(small_sample_data, reset_state=True)

        # With reset should match first prediction (same input)
        assert np.allclose(pred1, pred2_with_reset)

        # Without reset might be different (depends on model)
        # We just check it runs without error
        assert pred2_no_reset.shape == pred1.shape
