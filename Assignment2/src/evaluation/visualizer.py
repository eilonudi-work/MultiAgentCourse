"""
Signal Visualizer for publication-quality plots.

Creates PRD-required visualizations and additional analysis plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class SignalVisualizer:
    """
    Create publication-quality visualizations.

    Features:
    - PRD-required Graph 1: Detailed f₂ (3 Hz) analysis
    - PRD-required Graph 2: All frequencies comparison (2x2 grid)
    - Additional analysis plots
    - 300 DPI quality for publications

    """

    def __init__(self, dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            dpi: DPI for saved figures (default 300 for publication quality)
        """
        self.dpi = dpi

        # Set publication-quality style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

    def create_f2_detailed_plot(
        self,
        sample_data: Dict,
        save_path: str = 'outputs/figures/graph1_f2_detailed.png'
    ):
        """
        Create Graph 1: Detailed f₂ (3 Hz) analysis.

        PRD Requirement: Plot showing target signal, noisy mixed signal,
        and model prediction for f₂ = 3 Hz.

        Args:
            sample_data: Dictionary with 'target', 'mixed_signal', 'prediction', 'mse'
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract data
        target = np.array(sample_data['target'])
        mixed_signal = np.array(sample_data['mixed_signal'])
        prediction = np.array(sample_data['prediction'])
        mse = sample_data.get('mse', 0.0)

        # Create time vector
        time_steps = len(target)
        time = np.arange(time_steps) / 1000  # Convert to seconds (assuming 1000 Hz)

        # Limit to first few seconds for clarity
        display_samples = min(3000, time_steps)  # First 3 seconds
        time = time[:display_samples]
        target = target[:display_samples]
        mixed_signal = mixed_signal[:display_samples]
        prediction = prediction[:display_samples]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot signals
        ax.plot(time, target, 'g-', linewidth=1.5, label='Target Signal (Pure 3 Hz)', alpha=0.8)
        ax.plot(time, mixed_signal, 'gray', linewidth=0.8, label='Noisy Mixed Signal', alpha=0.5)
        ax.plot(time, prediction, 'b--', linewidth=1.2, label='Model Prediction', alpha=0.9)

        # Labels and title
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(
            f'LSTM Signal Extraction for f₂ = 3 Hz\nMSE = {mse:.6f}',
            fontsize=12,
            fontweight='bold'
        )

        # Legend
        ax.legend(loc='upper right', framealpha=0.95)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Graph 1 (f₂ detailed) saved to {save_path}")

        plt.close()

    def create_all_frequencies_plot(
        self,
        frequency_samples: Dict[int, Dict],
        frequencies: List[int] = [1, 3, 5, 7],
        save_path: str = 'outputs/figures/graph2_all_frequencies.png'
    ):
        """
        Create Graph 2: All frequencies comparison (2x2 grid).

        PRD Requirement: 4-panel plot showing target vs prediction for all frequencies.

        Args:
            frequency_samples: Dictionary mapping frequency to sample data
            frequencies: List of frequencies to plot
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Plot each frequency
        for idx, freq in enumerate(frequencies):
            ax = axes[idx]

            if freq not in frequency_samples:
                ax.text(0.5, 0.5, f'No data for {freq} Hz',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'f = {freq} Hz')
                continue

            # Extract data
            sample = frequency_samples[freq]
            target = np.array(sample['target'])
            prediction = np.array(sample['prediction'])
            mse = sample.get('mse', 0.0)
            r2 = sample.get('r2', 0.0)

            # Create time vector
            time_steps = len(target)
            time = np.arange(time_steps) / 1000  # Convert to seconds

            # Limit to first 3 seconds
            display_samples = min(3000, time_steps)
            time = time[:display_samples]
            target_plot = target[:display_samples]
            prediction_plot = prediction[:display_samples]

            # Plot
            ax.plot(time, target_plot, 'g-', linewidth=1.5, label='Target', alpha=0.8)
            ax.plot(time, prediction_plot, 'b--', linewidth=1.2, label='Prediction', alpha=0.9)

            # Labels and title
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(
                f'f = {freq} Hz\nMSE = {mse:.6f}, R² = {r2:.4f}',
                fontsize=11,
                fontweight='bold'
            )

            # Legend
            ax.legend(loc='upper right', fontsize=8, framealpha=0.95)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Overall title
        fig.suptitle(
            'LSTM Signal Extraction: All Frequencies Comparison',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Graph 2 (all frequencies) saved to {save_path}")

        plt.close()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: str = 'outputs/figures/training_history.png'
    ):
        """
        Plot training history (loss curves).

        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Loss curves
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)

        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')

        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Additional metrics
        ax2 = axes[1]
        if 'val_correlation' in history:
            ax2.plot(epochs, history['val_correlation'], 'g-', linewidth=2, label='Correlation')
        if 'val_r2' in history:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(epochs, history['val_r2'], 'purple', linewidth=2, label='R²')
            ax2_twin.set_ylabel('R² Score', fontsize=11, color='purple')
            ax2_twin.tick_params(axis='y', labelcolor='purple')
            ax2_twin.legend(loc='lower right')

        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Correlation', fontsize=11, color='g')
        ax2.set_title('Validation Metrics', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()

    def plot_per_frequency_metrics(
        self,
        per_frequency_metrics: List[Dict],
        save_path: str = 'outputs/figures/per_frequency_metrics.png'
    ):
        """
        Plot metrics comparison across frequencies.

        Args:
            per_frequency_metrics: List of metrics per frequency
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        frequencies = [m['frequency'] for m in per_frequency_metrics]
        mse_values = [m['mse'] for m in per_frequency_metrics]
        correlation_values = [m['correlation'] for m in per_frequency_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MSE comparison
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(frequencies)), mse_values, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(frequencies)))
        ax1.set_xticklabels([f'{f} Hz' for f in frequencies])
        ax1.set_ylabel('MSE', fontsize=11)
        ax1.set_title('MSE by Frequency', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars1, mse_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # Correlation comparison
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(frequencies)), correlation_values, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(frequencies)))
        ax2.set_xticklabels([f'{f} Hz' for f in frequencies])
        ax2.set_ylabel('Correlation', fontsize=11)
        ax2.set_title('Correlation by Frequency', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars2, correlation_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Per-frequency metrics saved to {save_path}")
        plt.close()

    def plot_error_distribution(
        self,
        errors: np.ndarray,
        save_path: str = 'outputs/figures/error_distribution.png'
    ):
        """
        Plot error distribution histogram.

        Args:
            errors: Array of prediction errors
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(errors.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')

        # Add vertical line at zero
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')

        # Add mean line
        mean_error = np.mean(errors)
        ax.axvline(mean_error, color='green', linestyle='--', linewidth=2,
                  label=f'Mean Error: {mean_error:.6f}')

        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Error distribution saved to {save_path}")
        plt.close()

    def plot_prediction_vs_target_scatter(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: str = 'outputs/figures/prediction_vs_target.png'
    ):
        """
        Scatter plot of predictions vs targets.

        Args:
            predictions: Predicted values
            targets: Target values
            save_path: Path to save figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Sample if too many points
        if len(pred_flat) > 10000:
            indices = np.random.choice(len(pred_flat), 10000, replace=False)
            pred_flat = pred_flat[indices]
            target_flat = target_flat[indices]

        # Scatter plot
        ax.scatter(target_flat, pred_flat, alpha=0.3, s=1, color='steelblue')

        # Perfect prediction line
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Target', fontsize=11)
        ax.set_ylabel('Prediction', fontsize=11)
        ax.set_title('Predictions vs Targets', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
        plt.close()
