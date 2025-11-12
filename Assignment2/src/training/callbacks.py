"""
Callbacks for training loop management.

This module provides callback infrastructure for:
    - Checkpoint saving
    - Early stopping
    - Learning rate scheduling
    - TensorBoard logging
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks can hook into various points in the training loop:
        - on_train_begin
        - on_train_end
        - on_epoch_begin
        - on_epoch_end
        - on_batch_begin
        - on_batch_end
    """

    def on_train_begin(self, trainer: Any):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: Any, batch_idx: int):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float):
        """Called at the end of each batch."""
        pass


class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    Saves checkpoints:
        - At specified intervals (save_every_n_epochs)
        - When metric improves (save_best=True)
        - At end of training (save_last=True)

    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir='checkpoints',
        ...     save_best=True,
        ...     monitor='val_loss',
        ...     mode='min'
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: Optional[int] = None,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model based on monitored metric
            save_last: Whether to save last model at end of training
            save_every_n_epochs: Save every N epochs (None to disable)
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for optimization direction
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint based on configuration."""
        # Save at intervals
        if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
            filename = f'checkpoint_epoch_{epoch+1}.pt'
            self._save_checkpoint(trainer, filename, epoch, metrics)

        # Save best model
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]

            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )

            if is_better:
                self.best_value = current_value
                self.best_epoch = epoch

                filename = 'best_model.pt'
                self._save_checkpoint(trainer, filename, epoch, metrics)

                logger.info(
                    f"Saved best model: {self.monitor}={current_value:.6f} "
                    f"(epoch {epoch+1})"
                )

    def on_train_end(self, trainer: Any):
        """Save last checkpoint at end of training."""
        if self.save_last:
            filename = 'last_model.pt'
            metrics = trainer.metrics_tracker.get_summary()

            # Get latest metrics
            latest_metrics = {
                name: stats['latest']
                for name, stats in metrics.items()
            }

            self._save_checkpoint(
                trainer,
                filename,
                trainer.current_epoch,
                latest_metrics
            )

            logger.info("Saved final checkpoint")

    def _save_checkpoint(
        self,
        trainer: Any,
        filename: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save checkpoint to disk."""
        from src.models.model_factory import ModelFactory

        checkpoint_path = self.checkpoint_dir / filename

        ModelFactory.save_checkpoint(
            trainer.model,
            checkpoint_path,
            optimizer=trainer.optimizer,
            epoch=epoch,
            loss=metrics.get('train_loss', None),
            val_loss=metrics.get('val_loss', None),
            config=trainer.config,
            metrics=metrics
        )

        logger.debug(f"Saved checkpoint: {checkpoint_path}")


class EarlyStoppingCallback(Callback):
    """
    Stop training when monitored metric stops improving.

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     min_delta=0.001
        ... )
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore weights from best epoch
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop."""
        if self.monitor not in metrics:
            logger.warning(f"Metric '{self.monitor}' not found for early stopping")
            return

        current_value = metrics[self.monitor]

        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }

            logger.debug(
                f"Early stopping: improved to {current_value:.6f}, "
                f"resetting patience"
            )
        else:
            self.wait += 1
            logger.debug(
                f"Early stopping: no improvement for {self.wait}/{self.patience} epochs"
            )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True

                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs. "
                    f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch+1}"
                )

    def on_train_end(self, trainer: Any):
        """Restore best weights if requested."""
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            logger.info(f"Restored model weights from epoch {self.best_epoch+1}")


class LearningRateSchedulerCallback(Callback):
    """
    Adjust learning rate during training.

    Supports:
        - ReduceLROnPlateau: Reduce LR when metric plateaus
        - StepLR: Reduce LR at fixed intervals
        - CosineAnnealingLR: Cosine annealing schedule

    Example:
        >>> callback = LearningRateSchedulerCallback(
        ...     scheduler='plateau',
        ...     monitor='val_loss',
        ...     factor=0.5,
        ...     patience=5
        ... )
    """

    def __init__(
        self,
        scheduler: str = 'plateau',
        monitor: str = 'val_loss',
        **scheduler_kwargs
    ):
        """
        Initialize learning rate scheduler callback.

        Args:
            scheduler: Type of scheduler ('plateau', 'step', 'cosine')
            monitor: Metric to monitor (for plateau scheduler)
            **scheduler_kwargs: Additional arguments for scheduler
        """
        self.scheduler_type = scheduler
        self.monitor = monitor
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler = None

    def on_train_begin(self, trainer: Any):
        """Initialize scheduler."""
        if self.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                trainer.optimizer,
                mode='min',
                **self.scheduler_kwargs
            )
        elif self.scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(trainer.optimizer, **self.scheduler_kwargs)
        elif self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(trainer.optimizer, **self.scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        logger.info(f"Initialized {self.scheduler_type} scheduler")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Update learning rate."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
        else:
            self.scheduler.step()

        # Log current LR
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.debug(f"Learning rate: {current_lr:.6e}")


class TensorBoardCallback(Callback):
    """
    Log metrics to TensorBoard.

    Example:
        >>> callback = TensorBoardCallback(log_dir='runs/experiment1')
    """

    def __init__(self, log_dir: Path, log_every_n_batches: int = 10):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
            log_every_n_batches: Log batch metrics every N batches
        """
        self.log_dir = Path(log_dir)
        self.log_every_n_batches = log_every_n_batches
        self.writer = None

    def on_train_begin(self, trainer: Any):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging to: {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, logging disabled")
            self.writer = None

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float):
        """Log batch metrics."""
        if self.writer is None:
            return

        if batch_idx % self.log_every_n_batches == 0:
            global_step = trainer.current_epoch * len(trainer.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss, global_step)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        if self.writer is None:
            return

        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)

        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', lr, epoch)

    def on_train_end(self, trainer: Any):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            logger.info("Closed TensorBoard writer")


class ProgressCallback(Callback):
    """
    Print training progress to console.

    Example:
        >>> callback = ProgressCallback(print_every_n_epochs=1)
    """

    def __init__(self, print_every_n_epochs: int = 1):
        """
        Initialize progress callback.

        Args:
            print_every_n_epochs: Print progress every N epochs
        """
        self.print_every_n_epochs = print_every_n_epochs

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Print epoch results."""
        if (epoch + 1) % self.print_every_n_epochs == 0:
            metrics_str = ', '.join([
                f"{k}={v:.6f}" for k, v in metrics.items()
            ])

            logger.info(f"Epoch {epoch+1}/{trainer.num_epochs}: {metrics_str}")
