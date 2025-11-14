"""Dataset visualization module for quality inspection."""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


class DatasetVisualizer:
    """Visualize dataset properties and quality."""

    def __init__(self, config: Dict = None):
        """
        Initialize visualizer with configuration.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        logger.debug("Initialized DatasetVisualizer")

    def plot_sample_signals(
        self,
        sample: Dict,
        save_path: Path = None,
        show: bool = False
    ):
        """
        Plot mixed signal and target component for one sample.

        Args:
            sample: Sample dictionary with 'mixed_signal', 'target_signal', 'metadata'
            save_path: Path to save figure (optional)
            show: Whether to display the figure
        """
        mixed = sample['mixed_signal']
        target = sample['target_signal']
        meta = sample['metadata']

        # Create time vector
        sampling_rate = meta['sampling_rate']
        duration = meta['duration']
        t = np.arange(len(mixed)) / sampling_rate

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot mixed signal
        axes[0].plot(t, mixed, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Mixed Signal S(t) = (1/4) × Σ[Sinus_i(t)] + Noise')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, duration])

        # Plot target signal
        target_freq = meta['target_frequency']
        target_amp = meta['target_amplitude']
        target_phase = meta['target_phase']

        axes[1].plot(t, target, 'r-', linewidth=0.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(
            f'Target Signal: f={target_freq}Hz, A={target_amp:.3f}, φ={target_phase:.3f}rad'
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, duration])

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved sample signals plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_frequency_spectrum(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        save_path: Path = None,
        show: bool = False,
        title: str = "Frequency Spectrum"
    ):
        """
        Plot FFT to verify frequency content.

        Args:
            signal: Signal array
            sampling_rate: Sampling rate in Hz
            save_path: Path to save figure (optional)
            show: Whether to display the figure
            title: Plot title
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)

        # Only plot positive frequencies
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft[positive_mask])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(positive_freqs, positive_fft, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 20])  # Focus on low frequencies where our signals are

        # Mark expected frequencies if config available
        if self.config and 'data' in self.config:
            expected_freqs = self.config['data'].get('frequencies', [])
            for freq in expected_freqs:
                ax.axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved frequency spectrum plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_parameter_distributions(
        self,
        dataset: Dict,
        save_path: Path = None,
        show: bool = False
    ):
        """
        Plot amplitude and phase distributions.

        Args:
            dataset: Dataset dictionary
            save_path: Path to save figure (optional)
            show: Whether to display the figure
        """
        metadata = dataset['metadata']

        # Extract amplitudes and phases
        amplitudes = [m['target_amplitude'] for m in metadata]
        phases = [m['target_phase'] for m in metadata]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot amplitude distribution
        axes[0].hist(amplitudes, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Amplitude')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Amplitude Distribution')
        axes[0].grid(True, alpha=0.3)

        # Add expected uniform distribution line
        config = dataset.get('config', self.config)
        if config and 'data' in config:
            amp_range = config['data']['amplitude_range']
            axes[0].axhline(
                y=1/(amp_range[1]-amp_range[0]),
                color='r',
                linestyle='--',
                label='Expected Uniform'
            )
            axes[0].legend()

        # Plot phase distribution
        axes[1].hist(phases, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Phase (rad)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Phase Distribution')
        axes[1].grid(True, alpha=0.3)

        # Add expected uniform distribution line
        axes[1].axhline(
            y=1/(2*np.pi),
            color='r',
            linestyle='--',
            label='Expected Uniform'
        )
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved parameter distributions plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_dataset_summary_figure(
        self,
        dataset: Dict,
        save_path: Path = None,
        show: bool = False
    ):
        """
        Create comprehensive summary figure for dataset.

        Args:
            dataset: Dataset dictionary
            save_path: Path to save figure (optional)
            show: Whether to display the figure
        """
        metadata = dataset['metadata']
        config = dataset.get('config', self.config)

        # Extract data
        amplitudes = [m['target_amplitude'] for m in metadata]
        phases = [m['target_phase'] for m in metadata]
        frequencies = [m['target_frequency'] for m in metadata]

        # Count samples per frequency
        expected_freqs = config['data']['frequencies']
        freq_counts = {freq: frequencies.count(freq) for freq in expected_freqs}

        # Select a random sample for visualization
        sample_idx = np.random.randint(0, len(dataset['mixed_signals']))
        mixed_sample = dataset['mixed_signals'][sample_idx]
        target_sample = dataset['target_signals'][sample_idx]
        sample_meta = metadata[sample_idx]

        sampling_rate = sample_meta['sampling_rate']
        duration = sample_meta['duration']
        t = np.arange(len(mixed_sample)) / sampling_rate

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Sample mixed signal
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(t[:2000], mixed_sample[:2000], 'b-', linewidth=0.8)  # First 2 seconds
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Sample Mixed Signal (first 2s)')
        ax1.grid(True, alpha=0.3)

        # 2. Sample target signal
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(t[:2000], target_sample[:2000], 'r-', linewidth=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(
            f'Sample Target Signal: {sample_meta["target_frequency"]}Hz (first 2s)'
        )
        ax2.grid(True, alpha=0.3)

        # 3. Frequency spectrum of mixed signal
        ax3 = fig.add_subplot(gs[2, :2])
        fft = np.fft.fft(mixed_sample)
        freqs_fft = np.fft.fftfreq(len(mixed_sample), 1/sampling_rate)
        positive_mask = freqs_fft > 0
        positive_freqs = freqs_fft[positive_mask]
        positive_fft = np.abs(fft[positive_mask])

        ax3.plot(positive_freqs, positive_fft, 'b-', linewidth=1)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Frequency Spectrum of Mixed Signal')
        ax3.set_xlim([0, 20])
        ax3.grid(True, alpha=0.3)

        # Mark expected frequencies
        for freq in expected_freqs:
            ax3.axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

        # 4. Amplitude distribution
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.hist(amplitudes, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Amplitude')
        ax4.set_ylabel('Density')
        ax4.set_title('Amplitude Distribution')
        ax4.grid(True, alpha=0.3)

        if config and 'data' in config:
            amp_range = config['data']['amplitude_range']
            ax4.axhline(
                y=1/(amp_range[1]-amp_range[0]),
                color='r',
                linestyle='--',
                linewidth=2
            )

        # 5. Phase distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(phases, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Phase (rad)')
        ax5.set_ylabel('Density')
        ax5.set_title('Phase Distribution')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=1/(2*np.pi), color='r', linestyle='--', linewidth=2)

        # 6. Frequency balance
        ax6 = fig.add_subplot(gs[2, 2])
        freqs_list = list(freq_counts.keys())
        counts_list = list(freq_counts.values())
        ax6.bar(range(len(freqs_list)), counts_list, color='purple', alpha=0.7)
        ax6.set_xticks(range(len(freqs_list)))
        ax6.set_xticklabels([f'{f}Hz' for f in freqs_list])
        ax6.set_ylabel('Sample Count')
        ax6.set_title('Samples per Frequency')
        ax6.grid(True, alpha=0.3, axis='y')

        # Add overall title
        fig.suptitle(
            f'Dataset Summary: {dataset["split"]} - {len(metadata)} samples',
            fontsize=16,
            fontweight='bold'
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved dataset summary figure to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_multiple_samples(
        self,
        dataset: Dict,
        n_samples: int = 4,
        save_path: Path = None,
        show: bool = False
    ):
        """
        Plot multiple random samples from dataset.

        Args:
            dataset: Dataset dictionary
            n_samples: Number of samples to plot
            save_path: Path to save figure (optional)
            show: Whether to display the figure
        """
        # Select random samples
        indices = np.random.choice(len(dataset['mixed_signals']), n_samples, replace=False)

        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            mixed = dataset['mixed_signals'][idx]
            target = dataset['target_signals'][idx]
            meta = dataset['metadata'][idx]

            sampling_rate = meta['sampling_rate']
            duration = meta['duration']
            t = np.arange(len(mixed)) / sampling_rate

            # Plot first 3 seconds
            plot_samples = min(3000, len(t))

            # Mixed signal
            axes[i, 0].plot(t[:plot_samples], mixed[:plot_samples], 'b-', linewidth=0.6)
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_title(f'Sample {idx}: Mixed Signal')
            axes[i, 0].grid(True, alpha=0.3)

            if i == n_samples - 1:
                axes[i, 0].set_xlabel('Time (s)')

            # Target signal
            axes[i, 1].plot(t[:plot_samples], target[:plot_samples], 'r-', linewidth=0.6)
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].set_title(
                f'Target: {meta["target_frequency"]}Hz, A={meta["target_amplitude"]:.2f}'
            )
            axes[i, 1].grid(True, alpha=0.3)

            if i == n_samples - 1:
                axes[i, 1].set_xlabel('Time (s)')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved multiple samples plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
