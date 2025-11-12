"""
Trainer class for LSTM signal extraction model.

This module provides the main training loop with support for:
    - Training and validation
    - Metrics tracking
    - Callbacks (checkpointing, early stopping, etc.)
    - Stateful LSTM processing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.lstm_model import SignalExtractionLSTM
from src.models.state_manager import StatefulProcessor
from .callbacks import Callback
from .metrics import MetricsCalculator, MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for LSTM signal extraction.

    Handles:
        - Training loop with stateful LSTM processing
        - Validation and metrics computation
        - Callback management
        - Progress tracking and logging

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     criterion=nn.MSELoss(),
        ...     device='cpu'
        ... )
        >>> trainer.train(num_epochs=50)
    """

    def __init__(
        self,
        model: SignalExtractionLSTM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        config: Optional[Dict] = None,
        callbacks: Optional[List[Callback]] = None,
        grad_clip_value: Optional[float] = None
    ):
        """
        Initialize trainer.

        Args:
            model: LSTM model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: Device to train on ('cpu' or 'cuda')
            config: Configuration dictionary
            callbacks: List of callback objects
            grad_clip_value: Gradient clipping value (None to disable)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        self.callbacks = callbacks or []
        self.grad_clip_value = grad_clip_value

        # Training state
        self.current_epoch = 0
        self.num_epochs = 0
        self.should_stop = False

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.metrics_calculator = MetricsCalculator()

        # Stateful processor
        self.processor = StatefulProcessor(model)

        logger.info(
            f"Initialized Trainer: device={device}, "
            f"train_samples={len(train_loader.dataset)}, "
            f"val_samples={len(val_loader.dataset) if val_loader else 0}"
        )

    def train(self, num_epochs: int):
        """
        Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Example:
            >>> trainer.train(num_epochs=50)
        """
        self.num_epochs = num_epochs
        self.should_stop = False

        # Call callbacks
        self._call_callbacks('on_train_begin')

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            if self.should_stop:
                logger.info(f"Training stopped early at epoch {epoch+1}")
                break

            self.current_epoch = epoch

            # Call callbacks
            self._call_callbacks('on_epoch_begin', epoch=epoch)

            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Update tracker
            for name, value in epoch_metrics.items():
                self.metrics_tracker.update(name, value, epoch)

            # Call callbacks
            self._call_callbacks('on_epoch_end', epoch=epoch, metrics=epoch_metrics)

        # Call callbacks
        self._call_callbacks('on_train_end')

        logger.info("Training complete")

        return self.metrics_tracker.get_summary()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Call callbacks
            self._call_callbacks('on_batch_begin', batch_idx=batch_idx)

            # Process batch
            loss = self._train_batch(batch)

            epoch_loss += loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.6f}'})

            # Call callbacks
            self._call_callbacks('on_batch_end', batch_idx=batch_idx, loss=loss)

        avg_loss = epoch_loss / num_batches

        metrics = {
            'train_loss': avg_loss
        }

        logger.debug(f"Epoch {epoch+1} train_loss: {avg_loss:.6f}")

        return metrics

    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Train on single batch.

        Args:
            batch: Batch dictionary with mixed_signal, target_signal, condition_vector

        Returns:
            Batch loss value
        """
        # Move to device
        mixed_signal = batch['mixed_signal'].to(self.device)  # (B, T)
        target_signal = batch['target_signal'].to(self.device)  # (B, T)
        condition_vector = batch['condition_vector'].to(self.device)  # (B, 4)

        batch_size = mixed_signal.size(0)
        time_steps = mixed_signal.size(1)

        # Zero gradients
        self.optimizer.zero_grad()

        # Initialize hidden state
        hidden_state = self.model.init_hidden(batch_size)

        # Accumulate loss over time steps
        total_loss = 0.0

        # Process each time step
        for t in range(time_steps):
            # Create input: [S(t), C1, C2, C3, C4]
            signal_t = mixed_signal[:, t:t+1].unsqueeze(-1)  # (B, 1, 1)
            condition = condition_vector.unsqueeze(1)  # (B, 1, 4)
            input_t = torch.cat([signal_t, condition], dim=2)  # (B, 1, 5)

            # Target at time t
            target_t = target_signal[:, t:t+1].unsqueeze(-1)  # (B, 1, 1)

            # Forward pass
            output, hidden_state = self.model(input_t, hidden_state)

            # Compute loss
            loss_t = self.criterion(output, target_t)
            total_loss += loss_t

            # Detach hidden state to prevent gradient explosion
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        # Average loss over time steps
        avg_loss = total_loss / time_steps

        # Backward pass
        avg_loss.backward()

        # Gradient clipping
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_value
            )

        # Optimizer step
        self.optimizer.step()

        return avg_loss.item()

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]",
            leave=False
        )

        with torch.no_grad():
            for batch in pbar:
                # Process batch
                predictions, targets, loss = self._validate_batch(batch)

                all_predictions.append(predictions)
                all_targets.append(targets)
                epoch_loss += loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.6f}'})

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.metrics_calculator.compute_metrics(all_predictions, all_targets)

        # Add validation prefix
        val_metrics = {
            'val_loss': avg_loss,
            **{f'val_{k}': v for k, v in metrics.items()}
        }

        logger.debug(
            f"Epoch {epoch+1} val_loss: {avg_loss:.6f}, "
            f"val_mse: {metrics['mse']:.6f}"
        )

        return val_metrics

    def _validate_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Validate on single batch.

        Args:
            batch: Batch dictionary

        Returns:
            Tuple of (predictions, targets, loss)
        """
        # Move to device
        mixed_signal = batch['mixed_signal'].to(self.device)
        target_signal = batch['target_signal'].to(self.device)
        condition_vector = batch['condition_vector'].to(self.device)

        batch_size = mixed_signal.size(0)
        time_steps = mixed_signal.size(1)

        # Initialize hidden state
        hidden_state = self.model.init_hidden(batch_size)

        # Collect predictions
        predictions = []
        total_loss = 0.0

        # Process each time step
        for t in range(time_steps):
            # Create input
            signal_t = mixed_signal[:, t:t+1].unsqueeze(-1)
            condition = condition_vector.unsqueeze(1)
            input_t = torch.cat([signal_t, condition], dim=2)

            # Target
            target_t = target_signal[:, t:t+1].unsqueeze(-1)

            # Forward pass
            output, hidden_state = self.model(input_t, hidden_state)

            # Compute loss
            loss_t = self.criterion(output, target_t)
            total_loss += loss_t

            # Store prediction
            predictions.append(output.cpu().squeeze(-1))

        # Stack predictions: (time_steps, batch_size, 1) -> (batch_size, time_steps)
        predictions = torch.stack(predictions, dim=1).squeeze(-1).numpy()
        targets = target_signal.cpu().numpy()

        avg_loss = (total_loss / time_steps).item()

        return predictions, targets, avg_loss

    def _call_callbacks(self, hook_name: str, **kwargs):
        """
        Call callback hooks.

        Args:
            hook_name: Name of callback method to call
            **kwargs: Arguments to pass to callbacks
        """
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)
            if method is not None:
                method(self, **kwargs)

    def save_checkpoint(self, path: Path):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        from src.models.model_factory import ModelFactory

        latest_metrics = {
            name: self.metrics_tracker.get_latest(name)
            for name in self.metrics_tracker.history.keys()
        }

        ModelFactory.save_checkpoint(
            self.model,
            path,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            config=self.config,
            metrics=latest_metrics
        )

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        from src.models.model_factory import ModelFactory

        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load epoch
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        logger.info(f"Loaded checkpoint from {path}")

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        Get full metrics history.

        Returns:
            Dictionary with all metrics history
        """
        return self.metrics_tracker.history

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best value for each metric.

        Returns:
            Dictionary with best values
        """
        best_metrics = {}

        for metric_name in self.metrics_tracker.history.keys():
            # Assume validation metrics should be minimized
            mode = 'min' if 'loss' in metric_name or 'mse' in metric_name else 'max'
            best_value = self.metrics_tracker.get_best(metric_name, mode=mode)

            if best_value is not None:
                best_metrics[metric_name] = best_value

        return best_metrics
