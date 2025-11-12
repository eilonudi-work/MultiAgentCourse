"""
Model Evaluator for comprehensive evaluation.

Evaluates trained models on test datasets and computes all required metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory
from src.models.lstm_model import SignalExtractionLSTM
from src.training.metrics import MetricsCalculator


class ModelEvaluator:
    """
    Comprehensive model evaluation.

    Features:
    - Evaluate on full test set
    - Compute all metrics (MSE, MAE, R², correlation, SNR)
    - Per-frequency analysis
    - Per-sample predictions
    - Statistical validation
    - PRD target checking

    Attributes:
        model: Trained LSTM model
        dataset: Test dataset
        device: Device for evaluation
        metrics_calculator: Metrics computation
    """

    def __init__(
        self,
        model: SignalExtractionLSTM,
        dataset: SignalDataset,
        device: str = 'cpu'
    ):
        """
        Initialize model evaluator.

        Args:
            model: Trained LSTM model
            dataset: Test dataset
            device: Device for evaluation ('cpu' or 'cuda')
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.metrics_calculator = MetricsCalculator()

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        # Get dataset info
        self.dataset_info = dataset.get_dataset_info()
        self.frequencies = self.dataset_info['frequencies']
        self.num_frequencies = len(self.frequencies)

    def evaluate_full_dataset(
        self,
        batch_size: int = 8,
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on full test dataset.

        Args:
            batch_size: Batch size for evaluation
            save_predictions: Whether to save all predictions

        Returns:
            Dictionary containing:
            - overall_metrics: Aggregated metrics across all samples
            - per_frequency_metrics: Metrics for each frequency
            - per_sample_results: Individual sample results (if save_predictions=True)
            - summary: Summary statistics and PRD validation
        """
        print(f"\n{'='*80}")
        print(f"Evaluating Model on Test Dataset")
        print(f"{'='*80}")
        print(f"Dataset: {len(self.dataset)} samples")
        print(f"Frequencies: {self.frequencies}")
        print(f"Device: {self.device}")

        # Create data loader
        data_loader = DataLoaderFactory.create_eval_loader(
            self.dataset,
            batch_size=batch_size
        )

        # Storage for results
        all_predictions = []
        all_targets = []
        all_conditions = []
        per_sample_results = []

        # Evaluate
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Get batch data
                mixed_signal = batch['mixed_signal'].to(self.device)
                target_signal = batch['target_signal'].to(self.device)
                condition = batch['condition_vector'].to(self.device)

                batch_size_actual = mixed_signal.size(0)
                time_steps = mixed_signal.size(1)

                # Initialize hidden state
                hidden_state = self.model.init_hidden(batch_size_actual)

                # Process each time step sequentially (L=1)
                predictions_t = []

                for t in range(time_steps):
                    # Get input at time t: [batch_size, 1]
                    signal_t = mixed_signal[:, t:t+1]

                    # Add feature dimension: [batch_size, 1, 1]
                    signal_t = signal_t.unsqueeze(2)

                    # Expand condition to match: [batch_size, 1, num_freq]
                    condition_t = condition.unsqueeze(1)

                    # Concatenate: [batch_size, 1, 1+num_freq]
                    input_t = torch.cat([signal_t, condition_t], dim=2)

                    # Forward pass
                    output, hidden_state = self.model(input_t, hidden_state)

                    # Detach hidden state
                    hidden_state = (
                        hidden_state[0].detach(),
                        hidden_state[1].detach()
                    )

                    # Store prediction: [batch_size, 1, 1]
                    predictions_t.append(output)

                # Concatenate predictions: [batch_size, time_steps, 1]
                predictions = torch.cat(predictions_t, dim=1)

                # Convert to numpy
                predictions_np = predictions.squeeze(-1).cpu().numpy()
                targets_np = target_signal.cpu().numpy()
                conditions_np = condition.cpu().numpy()

                # Store batch results
                all_predictions.append(predictions_np)
                all_targets.append(targets_np)
                all_conditions.append(conditions_np)

                # Compute per-sample metrics if requested
                if save_predictions:
                    for i in range(batch_size_actual):
                        sample_metrics = self._compute_sample_metrics(
                            predictions_np[i],
                            targets_np[i]
                        )

                        # Find frequency index
                        freq_idx = int(np.argmax(conditions_np[i]))

                        per_sample_results.append({
                            'frequency': self.frequencies[freq_idx],
                            'frequency_idx': freq_idx,
                            'prediction': predictions_np[i].tolist(),
                            'target': targets_np[i].tolist(),
                            'metrics': sample_metrics
                        })

        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_conditions = np.concatenate(all_conditions, axis=0)

        print(f"\nProcessed {len(all_predictions)} samples")

        # Compute overall metrics
        overall_metrics = self.metrics_calculator.compute_metrics(
            all_predictions,
            all_targets
        )

        print(f"\nOverall Metrics:")
        for metric_name, metric_value in overall_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")

        # Compute per-frequency metrics
        per_frequency_metrics = self._compute_per_frequency_metrics(
            all_predictions,
            all_targets,
            all_conditions
        )

        print(f"\nPer-Frequency Metrics:")
        for freq_idx, freq in enumerate(self.frequencies):
            metrics = per_frequency_metrics[freq_idx]
            print(f"  {freq} Hz - MSE: {metrics['mse']:.6f}, "
                  f"Correlation: {metrics['correlation']:.4f}")

        # Check PRD targets
        summary = self._check_prd_targets(overall_metrics)

        print(f"\n{'='*80}")
        print(f"PRD Target Validation:")
        print(f"  MSE < 0.01: {'✅ PASS' if summary['mse_target_met'] else '❌ FAIL'}")
        print(f"  MSE value: {overall_metrics['mse']:.6f}")
        print(f"{'='*80}\n")

        # Build result dictionary
        results = {
            'overall_metrics': overall_metrics,
            'per_frequency_metrics': per_frequency_metrics,
            'summary': summary,
            'dataset_info': {
                'num_samples': len(all_predictions),
                'num_frequencies': self.num_frequencies,
                'frequencies': self.frequencies
            }
        }

        if save_predictions:
            results['per_sample_results'] = per_sample_results

        return results

    def _compute_sample_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics for a single sample.

        Args:
            prediction: Predicted signal
            target: Target signal

        Returns:
            Dictionary of metrics
        """
        return {
            'mse': self.metrics_calculator.compute_mse(
                prediction.reshape(1, -1),
                target.reshape(1, -1)
            ),
            'mae': self.metrics_calculator.compute_mae(
                prediction.reshape(1, -1),
                target.reshape(1, -1)
            ),
            'correlation': self.metrics_calculator.compute_correlation(
                prediction.flatten(),
                target.flatten()
            ),
            'r2': self.metrics_calculator.compute_r2_score(
                prediction.reshape(1, -1),
                target.reshape(1, -1)
            ),
            'snr': self.metrics_calculator.compute_snr(
                target.reshape(1, -1),
                (target - prediction).reshape(1, -1)
            )
        }

    def _compute_per_frequency_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        conditions: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each frequency.

        Args:
            predictions: All predictions [num_samples, time_steps]
            targets: All targets [num_samples, time_steps]
            conditions: All conditions [num_samples, num_frequencies]

        Returns:
            List of metric dictionaries, one per frequency
        """
        per_frequency_metrics = []

        for freq_idx in range(self.num_frequencies):
            # Find samples for this frequency
            freq_mask = np.argmax(conditions, axis=1) == freq_idx

            if not np.any(freq_mask):
                # No samples for this frequency
                per_frequency_metrics.append({
                    'frequency': self.frequencies[freq_idx],
                    'num_samples': 0,
                    'mse': float('nan'),
                    'mae': float('nan'),
                    'rmse': float('nan'),
                    'correlation': float('nan'),
                    'r2': float('nan'),
                    'snr': float('nan')
                })
                continue

            # Get predictions and targets for this frequency
            freq_predictions = predictions[freq_mask]
            freq_targets = targets[freq_mask]

            # Compute metrics
            metrics = self.metrics_calculator.compute_metrics(
                freq_predictions,
                freq_targets
            )

            metrics['frequency'] = self.frequencies[freq_idx]
            metrics['num_samples'] = int(np.sum(freq_mask))

            per_frequency_metrics.append(metrics)

        return per_frequency_metrics

    def _check_prd_targets(
        self,
        overall_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if PRD targets are met.

        PRD Requirements:
        - MSE_test < 0.01
        - MSE_train < 0.01 (checked separately during training)
        - 0.9 < (MSE_test / MSE_train) < 1.1 (checked separately)

        Args:
            overall_metrics: Overall metrics dictionary

        Returns:
            Dictionary with validation results
        """
        mse = overall_metrics['mse']

        return {
            'mse_target_met': mse < 0.01,
            'mse_value': mse,
            'mse_target': 0.01,
            'all_metrics': overall_metrics
        }

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Evaluation results saved to {output_path}")

    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        output_path: str = 'outputs/evaluation/evaluation_report.md'
    ):
        """
        Generate comprehensive evaluation report in Markdown.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') if 'pd' in dir() else 'N/A'}\n\n")

            # Dataset info
            f.write("## Dataset Information\n\n")
            info = results['dataset_info']
            f.write(f"- Total Samples: {info['num_samples']}\n")
            f.write(f"- Number of Frequencies: {info['num_frequencies']}\n")
            f.write(f"- Frequencies: {info['frequencies']} Hz\n\n")

            # Overall metrics
            f.write("## Overall Metrics\n\n")
            metrics = results['overall_metrics']
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"| {metric_name.upper()} | {metric_value:.6f} |\n")
            f.write("\n")

            # Per-frequency metrics
            f.write("## Per-Frequency Metrics\n\n")
            f.write("| Frequency (Hz) | Samples | MSE | MAE | Correlation | R² | SNR (dB) |\n")
            f.write("|----------------|---------|-----|-----|-------------|----|---------|\n")

            for freq_metrics in results['per_frequency_metrics']:
                f.write(
                    f"| {freq_metrics['frequency']} | "
                    f"{freq_metrics['num_samples']} | "
                    f"{freq_metrics['mse']:.6f} | "
                    f"{freq_metrics['mae']:.6f} | "
                    f"{freq_metrics['correlation']:.4f} | "
                    f"{freq_metrics['r2']:.4f} | "
                    f"{freq_metrics['snr']:.2f} |\n"
                )
            f.write("\n")

            # PRD validation
            f.write("## PRD Target Validation\n\n")
            summary = results['summary']
            f.write(f"**MSE Target (< 0.01):** {'✅ PASS' if summary['mse_target_met'] else '❌ FAIL'}\n\n")
            f.write(f"- Target: {summary['mse_target']}\n")
            f.write(f"- Achieved: {summary['mse_value']:.6f}\n\n")

        print(f"Evaluation report saved to {output_path}")
