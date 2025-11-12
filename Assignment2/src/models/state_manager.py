"""
State management for LSTM stateful processing.

This module handles LSTM hidden state management for sequence processing with
proper reset behavior between samples and state persistence within samples.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .lstm_model import SignalExtractionLSTM

logger = logging.getLogger(__name__)


class StatefulProcessor:
    """
    Manage LSTM state for sequence processing.

    Key Behaviors:
        - Initialize state at start of sample
        - Preserve state during time step iteration (t to t+1)
        - Reset state when moving to new sample
        - Handle batch processing correctly

    The processor ensures proper state management according to the requirement:
        - Sequence length L=1 (one time step per forward pass)
        - DO NOT reset state between consecutive time steps (t to t+1)
        - MUST reset state between different samples

    Example:
        >>> model = SignalExtractionLSTM()
        >>> processor = StatefulProcessor(model)
        >>>
        >>> # Process a sample (10,000 time steps)
        >>> predictions = processor.process_sample(sample, reset_state=True)
        >>> print(predictions.shape)  # torch.Size([10000, 1])
    """

    def __init__(self, model: SignalExtractionLSTM):
        """
        Initialize stateful processor.

        Args:
            model: SignalExtractionLSTM model instance

        Raises:
            TypeError: If model is not SignalExtractionLSTM instance
        """
        if not isinstance(model, SignalExtractionLSTM):
            raise TypeError(
                f"model must be SignalExtractionLSTM, got {type(model).__name__}"
            )

        self.model = model
        self.current_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        logger.debug(f"Initialized StatefulProcessor with {model}")

    def reset_state(self, batch_size: int = 1):
        """
        Reset hidden state to zeros.

        This should be called at the beginning of each new sample.

        Args:
            batch_size: Batch size for state initialization (default: 1)

        Example:
            >>> processor.reset_state(batch_size=32)
        """
        self.current_state = self.model.init_hidden(batch_size)
        logger.debug(f"Reset state for batch_size={batch_size}")

    def process_sample(
        self,
        sample: Dict[str, np.ndarray],
        reset_state: bool = True,
        return_tensor: bool = False
    ) -> torch.Tensor:
        """
        Process entire sample (10,000 time steps) maintaining state.

        This method processes a complete signal sample one time step at a time,
        maintaining LSTM hidden state across all time steps.

        Args:
            sample: Dictionary containing:
                   - 'mixed_signal': np.ndarray of shape (time_steps,)
                   - 'condition_vector': np.ndarray of shape (4,)
            reset_state: Whether to reset hidden state before processing
                        (default: True for new samples)
            return_tensor: If True, return torch.Tensor; else numpy array

        Returns:
            predictions: Tensor of shape (time_steps, 1) with model predictions

        Raises:
            KeyError: If required keys missing from sample
            ValueError: If sample has incorrect structure

        Example:
            >>> sample = {
            ...     'mixed_signal': np.random.randn(10000),
            ...     'condition_vector': np.array([0, 1, 0, 0])
            ... }
            >>> predictions = processor.process_sample(sample)
            >>> print(predictions.shape)  # torch.Size([10000, 1])
        """
        # Validate sample structure
        if 'mixed_signal' not in sample:
            raise KeyError("sample must contain 'mixed_signal'")
        if 'condition_vector' not in sample:
            raise KeyError("sample must contain 'condition_vector'")

        mixed_signal = sample['mixed_signal']
        condition_vector = sample['condition_vector']

        if len(mixed_signal.shape) != 1:
            raise ValueError(
                f"mixed_signal must be 1D, got shape {mixed_signal.shape}"
            )
        if len(condition_vector) != 4:
            raise ValueError(
                f"condition_vector must have 4 elements, got {len(condition_vector)}"
            )

        # Reset state if requested
        if reset_state:
            self.reset_state(batch_size=1)

        time_steps = len(mixed_signal)
        predictions = []

        # Set model to evaluation mode
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            # Process each time step
            for t in range(time_steps):
                # Create input vector: [S(t), C1, C2, C3, C4]
                input_t = self._create_input_vector(mixed_signal, condition_vector, t)

                # Forward pass with L=1 (single time step)
                # input_t shape: (1, 1, 5) - [batch=1, seq_len=1, features=5]
                output_t, self.current_state = self.model(input_t, self.current_state)

                # Store prediction
                predictions.append(output_t)

        # Restore training mode if needed
        if was_training:
            self.model.train()

        # Concatenate all predictions: (time_steps, 1, 1) -> (time_steps,)
        predictions = torch.cat(predictions, dim=0).squeeze()

        logger.debug(f"Processed sample: {time_steps} time steps")

        if return_tensor:
            return predictions
        else:
            return predictions.cpu().numpy()

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        reset_state: bool = True
    ) -> torch.Tensor:
        """
        Process batch of samples with proper state management.

        Each sample in the batch maintains independent hidden state.

        Args:
            batch: Dictionary containing:
                  - 'mixed_signals': Tensor of shape (batch_size, time_steps)
                  - 'condition_vectors': Tensor of shape (batch_size, 4)
            reset_state: Whether to reset states before processing

        Returns:
            predictions: Tensor of shape (batch_size, time_steps, 1)

        Example:
            >>> batch = {
            ...     'mixed_signals': torch.randn(32, 10000),
            ...     'condition_vectors': torch.randint(0, 2, (32, 4)).float()
            ... }
            >>> predictions = processor.process_batch(batch)
            >>> print(predictions.shape)  # torch.Size([32, 10000, 1])
        """
        mixed_signals = batch['mixed_signals']
        condition_vectors = batch['condition_vectors']

        batch_size = mixed_signals.size(0)
        time_steps = mixed_signals.size(1)

        # Reset state if requested
        if reset_state:
            self.reset_state(batch_size=batch_size)

        predictions = []

        # Set model to evaluation mode
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            # Process each time step
            for t in range(time_steps):
                # Create input for time step t
                # Shape: (batch_size, 1, 5)
                input_t = torch.cat([
                    mixed_signals[:, t:t+1].unsqueeze(-1),  # (batch_size, 1, 1)
                    condition_vectors.unsqueeze(1)  # (batch_size, 1, 4)
                ], dim=2)

                # Forward pass
                output_t, self.current_state = self.model(input_t, self.current_state)

                predictions.append(output_t)

        # Restore training mode
        if was_training:
            self.model.train()

        # Concatenate: (time_steps, batch_size, 1, 1) -> (batch_size, time_steps, 1)
        predictions = torch.cat(predictions, dim=1)

        logger.debug(f"Processed batch: batch_size={batch_size}, time_steps={time_steps}")

        return predictions

    def _create_input_vector(
        self,
        mixed_signal: np.ndarray,
        condition_vector: np.ndarray,
        time_index: int
    ) -> torch.Tensor:
        """
        Create 5-dimensional input vector [S(t), C1, C2, C3, C4].

        Args:
            mixed_signal: Mixed signal array of shape (time_steps,)
            condition_vector: One-hot condition vector of shape (4,)
            time_index: Index of current time step

        Returns:
            Input tensor of shape (1, 1, 5) ready for model input

        Example:
            >>> mixed_signal = np.random.randn(10000)
            >>> condition = np.array([0, 1, 0, 0])
            >>> input_t = processor._create_input_vector(mixed_signal, condition, 0)
            >>> print(input_t.shape)  # torch.Size([1, 1, 5])
        """
        # Get signal value at time t
        signal_value = mixed_signal[time_index]

        # Concatenate: [S(t), C1, C2, C3, C4]
        input_vector = np.concatenate([
            np.array([signal_value]),  # S(t)
            condition_vector  # [C1, C2, C3, C4]
        ])

        # Convert to tensor and reshape: (5,) -> (1, 1, 5)
        input_tensor = torch.from_numpy(input_vector).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # Move to model's device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        return input_tensor

    def get_state_info(self) -> Optional[Dict]:
        """
        Get information about current hidden state.

        Returns:
            Dictionary with state information, or None if state not initialized

        Example:
            >>> processor.reset_state(batch_size=1)
            >>> info = processor.get_state_info()
            >>> print(info['h_shape'])
        """
        if self.current_state is None:
            return None

        h_n, c_n = self.current_state

        return {
            'h_shape': tuple(h_n.shape),
            'c_shape': tuple(c_n.shape),
            'h_mean': h_n.mean().item(),
            'h_std': h_n.std().item(),
            'c_mean': c_n.mean().item(),
            'c_std': c_n.std().item(),
            'device': str(h_n.device),
        }

    def __repr__(self) -> str:
        """String representation."""
        state_status = "initialized" if self.current_state is not None else "not initialized"
        return f"StatefulProcessor(model={self.model.__class__.__name__}, state={state_status})"
