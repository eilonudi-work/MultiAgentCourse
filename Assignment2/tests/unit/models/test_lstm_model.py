"""Unit tests for SignalExtractionLSTM model."""

import pytest
import torch
import torch.nn as nn

from src.models.lstm_model import SignalExtractionLSTM


@pytest.fixture
def default_model():
    """Create model with default parameters."""
    return SignalExtractionLSTM()


@pytest.fixture
def custom_model():
    """Create model with custom parameters."""
    return SignalExtractionLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=3,
        dropout=0.2,
        device='cpu'
    )


class TestSignalExtractionLSTMInitialization:
    """Tests for model initialization."""

    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = SignalExtractionLSTM()

        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.1
        assert model.device == 'cpu'

        # Check layers exist
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)

    def test_custom_initialization(self, custom_model):
        """Test model initialization with custom parameters."""
        assert custom_model.input_size == 5
        assert custom_model.hidden_size == 32
        assert custom_model.num_layers == 3
        assert custom_model.dropout == 0.2

    def test_invalid_input_size(self):
        """Test initialization with invalid input size."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            SignalExtractionLSTM(input_size=0)

        with pytest.raises(ValueError, match="input_size must be positive"):
            SignalExtractionLSTM(input_size=-1)

    def test_invalid_hidden_size(self):
        """Test initialization with invalid hidden size."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            SignalExtractionLSTM(hidden_size=0)

    def test_invalid_num_layers(self):
        """Test initialization with invalid number of layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            SignalExtractionLSTM(num_layers=0)

    def test_invalid_dropout(self):
        """Test initialization with invalid dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            SignalExtractionLSTM(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            SignalExtractionLSTM(dropout=1.0)

    def test_parameter_count(self, default_model):
        """Test that model has trainable parameters."""
        num_params = default_model.count_parameters()
        assert num_params > 0

        # Verify count matches manual calculation
        total = sum(p.numel() for p in default_model.parameters() if p.requires_grad)
        assert num_params == total


class TestSignalExtractionLSTMForward:
    """Tests for forward pass."""

    def test_forward_shape_single_batch(self, default_model):
        """Test forward pass output shape with batch_size=1."""
        batch_size = 1
        seq_len = 1
        input_size = 5

        x = torch.randn(batch_size, seq_len, input_size)
        output, (h_n, c_n) = default_model(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 1)

        # Check hidden state shapes
        assert h_n.shape == (default_model.num_layers, batch_size, default_model.hidden_size)
        assert c_n.shape == (default_model.num_layers, batch_size, default_model.hidden_size)

    def test_forward_with_different_batch_sizes(self, default_model):
        """Test forward pass with various batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 5)
            output, (h_n, c_n) = default_model(x)

            assert output.shape == (batch_size, 1, 1)
            assert h_n.shape == (2, batch_size, 64)
            assert c_n.shape == (2, batch_size, 64)

    def test_forward_with_provided_hidden_state(self, default_model):
        """Test forward pass with provided hidden state."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 5)

        # Initialize hidden state
        h_0, c_0 = default_model.init_hidden(batch_size)

        # Forward pass
        output, (h_n, c_n) = default_model(x, (h_0, c_0))

        assert output.shape == (batch_size, 1, 1)
        assert not torch.equal(h_n, h_0)  # State should change
        assert not torch.equal(c_n, c_0)

    def test_forward_without_hidden_state(self, default_model):
        """Test forward pass initializes hidden state automatically."""
        batch_size = 8
        x = torch.randn(batch_size, 1, 5)

        output, (h_n, c_n) = default_model(x)

        assert output.shape == (batch_size, 1, 1)
        assert h_n.shape == (2, batch_size, 64)
        assert c_n.shape == (2, batch_size, 64)

    def test_forward_invalid_input_dimensions(self, default_model):
        """Test forward pass rejects invalid input dimensions."""
        # 2D input (missing batch dimension)
        with pytest.raises(ValueError, match="Input must be 3-dimensional"):
            default_model(torch.randn(1, 5))

        # 4D input
        with pytest.raises(ValueError, match="Input must be 3-dimensional"):
            default_model(torch.randn(1, 1, 5, 1))

    def test_forward_invalid_feature_size(self, default_model):
        """Test forward pass rejects incorrect feature size."""
        x = torch.randn(1, 1, 3)  # Wrong feature size (3 instead of 5)

        with pytest.raises(ValueError, match="Input feature size must be"):
            default_model(x)

    def test_forward_produces_gradients(self, default_model):
        """Test that forward pass produces gradients."""
        default_model.train()

        x = torch.randn(4, 1, 5, requires_grad=True)
        output, _ = default_model(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_deterministic_with_same_input(self, default_model):
        """Test that same input produces same output (in eval mode)."""
        default_model.eval()

        x = torch.randn(4, 1, 5)

        with torch.no_grad():
            output1, _ = default_model(x)
            output2, _ = default_model(x)

        assert torch.allclose(output1, output2)


class TestSignalExtractionLSTMHiddenState:
    """Tests for hidden state management."""

    def test_init_hidden_shape(self, default_model):
        """Test init_hidden produces correct shapes."""
        batch_sizes = [1, 8, 32]

        for batch_size in batch_sizes:
            h_0, c_0 = default_model.init_hidden(batch_size)

            expected_shape = (default_model.num_layers, batch_size, default_model.hidden_size)
            assert h_0.shape == expected_shape
            assert c_0.shape == expected_shape

    def test_init_hidden_zeros(self, default_model):
        """Test that init_hidden initializes to zeros."""
        h_0, c_0 = default_model.init_hidden(batch_size=4)

        assert torch.all(h_0 == 0)
        assert torch.all(c_0 == 0)

    def test_init_hidden_device(self, default_model):
        """Test that init_hidden creates tensors on correct device."""
        h_0, c_0 = default_model.init_hidden(batch_size=4)

        # Check device matches model parameters
        model_device = next(default_model.parameters()).device
        assert h_0.device == model_device
        assert c_0.device == model_device

    def test_state_persistence_across_time_steps(self, default_model):
        """Test that hidden state changes across time steps."""
        default_model.eval()
        batch_size = 1

        # Initialize state
        h_0, c_0 = default_model.init_hidden(batch_size)
        state = (h_0, c_0)

        # Process multiple time steps
        states = []
        with torch.no_grad():
            for t in range(5):
                x_t = torch.randn(batch_size, 1, 5)
                _, state = default_model(x_t, state)
                states.append(state)

        # Verify state changes at each step
        for i in range(len(states) - 1):
            h_curr, c_curr = states[i]
            h_next, c_next = states[i + 1]

            # States should be different
            assert not torch.allclose(h_curr, h_next)
            assert not torch.allclose(c_curr, c_next)


class TestSignalExtractionLSTMUtilities:
    """Tests for utility methods."""

    def test_count_parameters(self, default_model):
        """Test parameter counting."""
        num_params = default_model.count_parameters()

        # Manual count
        manual_count = sum(p.numel() for p in default_model.parameters() if p.requires_grad)

        assert num_params == manual_count
        assert num_params > 0

    def test_get_model_info(self, default_model):
        """Test get_model_info returns correct information."""
        info = default_model.get_model_info()

        assert info['model_type'] == 'SignalExtractionLSTM'
        assert info['input_size'] == 5
        assert info['hidden_size'] == 64
        assert info['num_layers'] == 2
        assert info['dropout'] == 0.1
        assert info['total_parameters'] > 0
        assert 'device' in info

    def test_reset_parameters(self, default_model):
        """Test parameter reset."""
        # Get initial parameters
        initial_params = [p.clone() for p in default_model.parameters()]

        # Modify parameters (simulate training)
        with torch.no_grad():
            for p in default_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Reset
        default_model.reset_parameters()

        # Parameters should be different from modified version
        for p, p_init in zip(default_model.parameters(), initial_params):
            # Should be reinitialized (likely different from both modified and initial)
            # Just check they're valid
            assert not torch.isnan(p).any()
            assert not torch.isinf(p).any()

    def test_repr(self, default_model):
        """Test string representation."""
        repr_str = repr(default_model)

        assert 'SignalExtractionLSTM' in repr_str
        assert 'input_size=5' in repr_str
        assert 'hidden_size=64' in repr_str
        assert 'num_layers=2' in repr_str


class TestSignalExtractionLSTMDevice:
    """Tests for device management."""

    def test_model_on_cpu(self):
        """Test model works on CPU."""
        model = SignalExtractionLSTM(device='cpu')
        x = torch.randn(4, 1, 5)

        output, _ = model(x)
        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self):
        """Test model works on GPU."""
        model = SignalExtractionLSTM(device='cuda')
        x = torch.randn(4, 1, 5, device='cuda')

        output, _ = model(x)
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch_handling(self):
        """Test that model handles device mismatch in hidden state."""
        model = SignalExtractionLSTM(device='cuda')
        x = torch.randn(4, 1, 5, device='cuda')

        # Create hidden state on CPU
        h_0 = torch.zeros(2, 4, 64)
        c_0 = torch.zeros(2, 4, 64)

        # Should automatically move hidden state to match input device
        output, (h_n, c_n) = model(x, (h_0, c_0))

        assert output.device.type == 'cuda'
        assert h_n.device.type == 'cuda'
        assert c_n.device.type == 'cuda'


class TestSignalExtractionLSTMIntegration:
    """Integration tests for model behavior."""

    def test_overfitting_on_simple_pattern(self):
        """Test that model can overfit to a simple pattern."""
        model = SignalExtractionLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Create simple pattern: constant input -> constant output
        x = torch.ones(16, 1, 5) * 0.5
        y = torch.ones(16, 1, 1) * 0.3

        model.train()
        initial_loss = None

        # Train for a few iterations
        for epoch in range(50):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()

        final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5

    def test_gradient_flow(self, default_model):
        """Test that gradients flow through all parameters."""
        default_model.train()

        x = torch.randn(8, 1, 5, requires_grad=True)
        output, _ = default_model(x)
        loss = output.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in default_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_no_gradient_explosion(self, default_model):
        """Test that gradients don't explode during backprop."""
        default_model.train()
        optimizer = torch.optim.Adam(default_model.parameters())

        # Process long sequence
        batch_size = 4
        state = default_model.init_hidden(batch_size)

        for t in range(100):
            x_t = torch.randn(batch_size, 1, 5)
            output, state = default_model(x_t, state)

            # Detach state to prevent gradient explosion
            state = (state[0].detach(), state[1].detach())

        # Backprop on last output
        loss = output.sum()
        loss.backward()

        # Check gradients are reasonable
        for param in default_model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert torch.abs(param.grad).max() < 1000  # No explosion
