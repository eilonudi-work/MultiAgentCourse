"""
Metrics computation for signal extraction evaluation.

This module provides metrics for evaluating model performance including MSE,
correlation, and signal-to-noise ratio.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate evaluation metrics for signal extraction.

    Metrics include:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - Correlation coefficient
        - R² score

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.compute_metrics(predictions, targets)
        >>> print(f"MSE: {metrics['mse']:.6f}")
    """

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_mse(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Mean Squared Error.

        Args:
            predictions: Predicted values of shape (n_samples, time_steps)
            targets: Target values of same shape

        Returns:
            MSE value

        Example:
            >>> mse = calculator.compute_mse(pred, target)
        """
        return float(np.mean((predictions - targets) ** 2))

    def compute_rmse(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Root Mean Squared Error.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            RMSE value
        """
        mse = self.compute_mse(predictions, targets)
        return float(np.sqrt(mse))

    def compute_mae(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Mean Absolute Error.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            MAE value
        """
        return float(np.mean(np.abs(predictions - targets)))

    def compute_correlation(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Pearson correlation coefficient.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Correlation coefficient between -1 and 1
        """
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Compute correlation
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

        return float(correlation)

    def compute_r2_score(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute R² (coefficient of determination) score.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            R² score
        """
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)

        # Avoid division by zero
        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

    def compute_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray
    ) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.

        Args:
            signal: True signal
            noise: Noise (error = target - prediction)

        Returns:
            SNR in decibels
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        # Avoid division by zero
        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Dictionary with all computed metrics

        Example:
            >>> metrics = calculator.compute_metrics(pred, target)
            >>> print(metrics)
            {'mse': 0.005, 'rmse': 0.071, 'mae': 0.055,
             'correlation': 0.95, 'r2': 0.90, 'snr': 30.5}
        """
        metrics = {
            'mse': self.compute_mse(predictions, targets),
            'rmse': self.compute_rmse(predictions, targets),
            'mae': self.compute_mae(predictions, targets),
            'correlation': self.compute_correlation(predictions, targets),
            'r2': self.compute_r2_score(predictions, targets),
        }

        # Compute SNR (noise = error)
        noise = targets - predictions
        metrics['snr'] = self.compute_snr(targets, noise)

        return metrics


class MetricsTracker:
    """
    Track metrics across training epochs.

    Maintains history of metrics and provides utilities for:
        - Computing running averages
        - Finding best values
        - Checking for improvement

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update('train_loss', 0.5, epoch=0)
        >>> tracker.update('val_loss', 0.3, epoch=0)
        >>> best = tracker.get_best('val_loss', mode='min')
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.history: Dict[str, List[float]] = {}
        self.best_values: Dict[str, float] = {}
        self.best_epochs: Dict[str, int] = {}

    def update(self, metric_name: str, value: float, epoch: int):
        """
        Update metric with new value.

        Args:
            metric_name: Name of metric (e.g., 'train_loss', 'val_mse')
            value: Metric value
            epoch: Current epoch number
        """
        if metric_name not in self.history:
            self.history[metric_name] = []

        self.history[metric_name].append(value)

    def get_history(self, metric_name: str) -> List[float]:
        """
        Get full history for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            List of all recorded values
        """
        return self.history.get(metric_name, [])

    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get most recent value for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            Latest value or None if no history
        """
        history = self.get_history(metric_name)
        return history[-1] if history else None

    def get_best(
        self,
        metric_name: str,
        mode: str = 'min'
    ) -> Optional[float]:
        """
        Get best value for a metric.

        Args:
            metric_name: Name of metric
            mode: 'min' or 'max' for optimization direction

        Returns:
            Best value or None if no history
        """
        history = self.get_history(metric_name)
        if not history:
            return None

        if mode == 'min':
            return min(history)
        else:
            return max(history)

    def get_best_epoch(
        self,
        metric_name: str,
        mode: str = 'min'
    ) -> Optional[int]:
        """
        Get epoch with best value for a metric.

        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'

        Returns:
            Epoch number or None if no history
        """
        history = self.get_history(metric_name)
        if not history:
            return None

        if mode == 'min':
            return int(np.argmin(history))
        else:
            return int(np.argmax(history))

    def has_improved(
        self,
        metric_name: str,
        mode: str = 'min',
        patience: int = 1
    ) -> bool:
        """
        Check if metric has improved in last N epochs.

        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'
            patience: Number of epochs to look back

        Returns:
            True if metric improved, False otherwise
        """
        history = self.get_history(metric_name)
        if len(history) < patience + 1:
            return True  # Not enough history

        current = history[-1]
        best_in_window = min(history[-(patience+1):-1]) if mode == 'min' else max(history[-(patience+1):-1])

        if mode == 'min':
            return current < best_in_window
        else:
            return current > best_in_window

    def get_running_average(
        self,
        metric_name: str,
        window: int = 5
    ) -> Optional[float]:
        """
        Get running average over last N values.

        Args:
            metric_name: Name of metric
            window: Number of values to average

        Returns:
            Running average or None
        """
        history = self.get_history(metric_name)
        if not history:
            return None

        recent = history[-window:]
        return float(np.mean(recent))

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.

        Returns:
            Dictionary with summary stats for each metric
        """
        summary = {}

        for metric_name, values in self.history.items():
            if values:
                summary[metric_name] = {
                    'latest': values[-1],
                    'best': min(values),
                    'worst': max(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }

        return summary

    def reset(self):
        """Reset all tracked metrics."""
        self.history.clear()
        self.best_values.clear()
        self.best_epochs.clear()

        logger.debug("Reset metrics tracker")

    def __repr__(self) -> str:
        """String representation."""
        n_metrics = len(self.history)
        return f"MetricsTracker(n_metrics={n_metrics})"
