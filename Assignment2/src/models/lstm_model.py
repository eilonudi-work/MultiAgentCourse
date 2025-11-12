"""
LSTM model for extracting pure sinusoidal components from mixed signals.

This module implements a stateful LSTM network that processes signals with
sequence length L=1, maintaining hidden state across time steps within samples
but resetting between different samples.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SignalExtractionLSTM(nn.Module):
    """
    LSTM-based signal extraction model with stateful processing.

    Architecture:
        Input Layer (5) → LSTM Layer(s) → Dense Output (1)

    The model processes input vectors [S(t), C1, C2, C3, C4] where:
        - S(t): Mixed signal value at time t
        - C1-C4: One-hot encoded frequency selector

    Output is a scalar value representing the extracted pure sinusoid at time t.

    Key Features:
        - Sequence length L=1 (processes one time step at a time)
        - Stateful processing (maintains hidden state across time steps)
        - Hidden state must be reset between different samples
        - Supports both CPU and GPU execution

    Example:
        >>> model = SignalExtractionLSTM(input_size=5, hidden_size=64, num_layers=2)
        >>> input_t = torch.randn(1, 1, 5)  # [batch=1, seq_len=1, features=5]
        >>> output, hidden_state = model(input_t)
        >>> print(output.shape)  # torch.Size([1, 1, 1])
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize LSTM signal extraction model.

        Args:
            input_size: Number of input features (default: 5)
                       [S(t), C1, C2, C3, C4]
            hidden_size: Number of features in hidden state (default: 64)
            num_layers: Number of recurrent layers (default: 2)
            dropout: Dropout probability between LSTM layers (default: 0.1)
                    Note: dropout is only applied if num_layers > 1
            device: Device to run model on ('cpu' or 'cuda')

        Raises:
            ValueError: If input parameters are invalid
        """
        super().__init__()

        # Validate parameters
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        # LSTM layers
        # batch_first=True: input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer: hidden_size → 1 (scalar prediction)
        self.fc = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._initialize_weights()

        # Move to device
        self.to(device)

        logger.info(
            f"Initialized SignalExtractionLSTM: "
            f"input_size={input_size}, hidden_size={hidden_size}, "
            f"num_layers={num_layers}, dropout={dropout}, device={device}"
        )

    def _initialize_weights(self):
        """
        Initialize model weights using Xavier/Glorot initialization.

        LSTM weights are initialized with Xavier uniform.
        Linear layer weights are initialized with Xavier uniform.
        Biases are initialized to zero.
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # Biases
                nn.init.constant_(param.data, 0)
            elif 'fc.weight' in name:  # Output layer weights
                nn.init.xavier_uniform_(param.data)

        logger.debug("Initialized model weights")

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               For L=1: (batch_size, 1, 5)
            hidden_state: Optional tuple of (h_0, c_0) where:
                         - h_0: Initial hidden state (num_layers, batch_size, hidden_size)
                         - c_0: Initial cell state (num_layers, batch_size, hidden_size)
                         If None, hidden state is initialized to zeros.

        Returns:
            Tuple containing:
                - output: Model predictions of shape (batch_size, seq_len, 1)
                - (h_n, c_n): New hidden and cell states for next time step

        Raises:
            ValueError: If input tensor has incorrect shape

        Example:
            >>> model = SignalExtractionLSTM()
            >>> x = torch.randn(32, 1, 5)  # batch=32, seq=1, features=5
            >>> output, (h_n, c_n) = model(x)
            >>> print(output.shape)  # torch.Size([32, 1, 1])
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(
                f"Input must be 3-dimensional (batch, seq, features), got shape {x.shape}"
            )

        if x.size(2) != self.input_size:
            raise ValueError(
                f"Input feature size must be {self.input_size}, got {x.size(2)}"
            )

        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        # Ensure hidden state is on the same device as input
        h_0, c_0 = hidden_state
        if h_0.device != x.device:
            h_0 = h_0.to(x.device)
            c_0 = c_0.to(x.device)
            hidden_state = (h_0, c_0)

        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_size)
        # (h_n, c_n): new hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)

        # Extract last time step output
        # For seq_len=1, this is just lstm_out[:, 0, :]
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Pass through output layer
        prediction = self.fc(last_output)  # (batch_size, 1)

        # Reshape to (batch_size, seq_len, 1) for consistency
        prediction = prediction.unsqueeze(1)  # (batch_size, 1, 1)

        return prediction, (h_n, c_n)

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states to zeros.

        Args:
            batch_size: Batch size for hidden state initialization

        Returns:
            Tuple of (h_0, c_0):
                - h_0: Hidden state of shape (num_layers, batch_size, hidden_size)
                - c_0: Cell state of shape (num_layers, batch_size, hidden_size)

        Example:
            >>> model = SignalExtractionLSTM()
            >>> h_0, c_0 = model.init_hidden(batch_size=32)
            >>> print(h_0.shape)  # torch.Size([2, 32, 64])
        """
        device = next(self.parameters()).device

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        return h_0, c_0

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total number of parameters in the model

        Example:
            >>> model = SignalExtractionLSTM(hidden_size=64, num_layers=2)
            >>> num_params = model.count_parameters()
            >>> print(f"Parameters: {num_params:,}")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """
        Get model architecture information.

        Returns:
            Dictionary containing model configuration and statistics

        Example:
            >>> model = SignalExtractionLSTM()
            >>> info = model.get_model_info()
            >>> print(info['total_parameters'])
        """
        return {
            'model_type': 'SignalExtractionLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters(),
            'device': str(self.device),
        }

    def reset_parameters(self):
        """
        Reset all model parameters to initial values.

        Useful for retraining or reinitializing the model.
        """
        self._initialize_weights()
        logger.debug("Reset all model parameters")

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"SignalExtractionLSTM(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dropout={self.dropout},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
