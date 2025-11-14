"""
Error Analyzer for identifying error patterns.

Analyzes prediction errors to identify patterns and problem areas.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np


class ErrorAnalyzer:
    """
    Analyze prediction errors.

    Features:
    - Find worst predictions
    - Analyze error patterns
    - Identify systematic biases
    - Compute error statistics

    """

    @staticmethod
    def find_worst_predictions(
        per_sample_results: List[Dict],
        n: int = 10,
        metric: str = 'mse'
    ) -> List[Dict]:
        """
        Find N worst predictions by metric.

        Args:
            per_sample_results: List of per-sample results
            n: Number of worst predictions to return
            metric: Metric to use for ranking

        Returns:
            List of worst prediction dictionaries
        """
        if not per_sample_results:
            return []

        # Sort by metric (descending for MSE/MAE, ascending for correlation/R2)
        if metric in ['mse', 'mae']:
            sorted_results = sorted(
                per_sample_results,
                key=lambda x: x['metrics'][metric],
                reverse=True
            )
        else:
            sorted_results = sorted(
                per_sample_results,
                key=lambda x: x['metrics'][metric],
                reverse=False
            )

        return sorted_results[:n]

    @staticmethod
    def analyze_error_patterns(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, any]:
        """
        Analyze error patterns across predictions.

        Args:
            predictions: Predicted values [num_samples, time_steps]
            targets: Target values [num_samples, time_steps]

        Returns:
            Dictionary with error pattern analysis
        """
        errors = predictions - targets

        # Overall error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        # Check for bias
        bias_present = abs(mean_error) > 0.01

        # Compute error distribution statistics
        error_flat = errors.flatten()

        analysis = {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'mae': float(mae),
            'rmse': float(rmse),
            'bias_present': bias_present,
            'bias_direction': 'positive' if mean_error > 0 else 'negative',
            'error_range': {
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'q25': float(np.percentile(error_flat, 25)),
                'q75': float(np.percentile(error_flat, 75))
            },
            'outliers': {
                'num_large_errors': int(np.sum(np.abs(errors) > 3 * std_error)),
                'threshold': float(3 * std_error)
            }
        }

        return analysis

    @staticmethod
    def compute_per_timestep_errors(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute error statistics per time step.

        Args:
            predictions: Predicted values [num_samples, time_steps]
            targets: Target values [num_samples, time_steps]

        Returns:
            Dictionary with per-timestep statistics
        """
        errors = predictions - targets

        return {
            'mean_error_per_timestep': np.mean(errors, axis=0),
            'std_error_per_timestep': np.std(errors, axis=0),
            'mae_per_timestep': np.mean(np.abs(errors), axis=0),
            'rmse_per_timestep': np.sqrt(np.mean(errors**2, axis=0))
        }

    @staticmethod
    def analyze_frequency_specific_errors(
        per_sample_results: List[Dict]
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze errors for each frequency separately.

        Args:
            per_sample_results: List of per-sample results

        Returns:
            Dictionary mapping frequency to error statistics
        """
        # Group by frequency
        frequency_groups = {}

        for sample in per_sample_results:
            freq = sample['frequency']

            if freq not in frequency_groups:
                frequency_groups[freq] = {
                    'predictions': [],
                    'targets': [],
                    'errors': []
                }

            pred = np.array(sample['prediction'])
            target = np.array(sample['target'])
            error = pred - target

            frequency_groups[freq]['predictions'].append(pred)
            frequency_groups[freq]['targets'].append(target)
            frequency_groups[freq]['errors'].append(error)

        # Compute statistics for each frequency
        frequency_analysis = {}

        for freq, data in frequency_groups.items():
            errors = np.array(data['errors'])

            frequency_analysis[freq] = {
                'num_samples': len(data['errors']),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'bias_present': abs(np.mean(errors)) > 0.01
            }

        return frequency_analysis

    @staticmethod
    def identify_systematic_biases(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, any]:
        """
        Identify systematic biases in predictions.

        Args:
            predictions: Predicted values [num_samples, time_steps]
            targets: Target values [num_samples, time_steps]

        Returns:
            Dictionary with bias analysis
        """
        errors = predictions - targets

        # Overall bias
        mean_error = np.mean(errors)

        # Check if model systematically over/under predicts
        overpredict_ratio = np.mean(predictions > targets)

        # Bias vs target magnitude
        target_flat = targets.flatten()
        error_flat = errors.flatten()

        # Divide into low/high magnitude targets
        median_target = np.median(np.abs(target_flat))
        low_mag_mask = np.abs(target_flat) < median_target
        high_mag_mask = np.abs(target_flat) >= median_target

        low_mag_error = np.mean(error_flat[low_mag_mask])
        high_mag_error = np.mean(error_flat[high_mag_mask])

        return {
            'overall_bias': float(mean_error),
            'overpredict_ratio': float(overpredict_ratio),
            'underpredict_ratio': float(1 - overpredict_ratio),
            'bias_by_magnitude': {
                'low_magnitude_bias': float(low_mag_error),
                'high_magnitude_bias': float(high_mag_error),
                'bias_correlation_with_magnitude': low_mag_error != high_mag_error
            },
            'interpretation': {
                'systematic_overprediction': overpredict_ratio > 0.6,
                'systematic_underprediction': overpredict_ratio < 0.4,
                'magnitude_dependent_bias': abs(low_mag_error - high_mag_error) > 0.01
            }
        }

    @staticmethod
    def generate_error_summary(
        per_sample_results: List[Dict]
    ) -> str:
        """
        Generate human-readable error summary.

        Args:
            per_sample_results: List of per-sample results

        Returns:
            String with error summary
        """
        if not per_sample_results:
            return "No data available for error analysis."

        # Collect all MSE values
        mse_values = [sample['metrics']['mse'] for sample in per_sample_results]

        # Find worst samples
        worst = ErrorAnalyzer.find_worst_predictions(per_sample_results, n=5, metric='mse')

        summary = []
        summary.append(f"Error Analysis Summary")
        summary.append(f"=" * 60)
        summary.append(f"Total samples analyzed: {len(per_sample_results)}")
        summary.append(f"Mean MSE: {np.mean(mse_values):.6f}")
        summary.append(f"Std MSE: {np.std(mse_values):.6f}")
        summary.append(f"Min MSE: {np.min(mse_values):.6f}")
        summary.append(f"Max MSE: {np.max(mse_values):.6f}")
        summary.append("")
        summary.append(f"Top 5 Worst Predictions:")
        summary.append(f"-" * 60)

        for i, sample in enumerate(worst, 1):
            summary.append(
                f"{i}. Frequency {sample['frequency']} Hz - "
                f"MSE: {sample['metrics']['mse']:.6f}, "
                f"Correlation: {sample['metrics']['correlation']:.4f}"
            )

        return "\n".join(summary)
