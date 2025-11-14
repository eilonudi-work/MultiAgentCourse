"""
Experiment Tracker for monitoring and analyzing experiments.

Provides utilities for tracking experiment progress and analyzing results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


class ExperimentTracker:
    """
    Track and analyze experiments.

    Features:
    - Load experiments from storage
    - Query experiments by criteria
    - Compute statistics across experiments
    - Export to various formats (DataFrame, CSV, JSON)

    Attributes:
        experiments_file: Path to experiments JSON file
        experiments: List of experiment dictionaries
    """

    def __init__(self, experiments_file: str = 'outputs/experiments/experiments.json'):
        """
        Initialize experiment tracker.

        Args:
            experiments_file: Path to experiments JSON file
        """
        self.experiments_file = Path(experiments_file)
        self.experiments = []

        if self.experiments_file.exists():
            self.load_experiments()

    def load_experiments(self):
        """Load experiments from JSON file."""
        with open(self.experiments_file, 'r') as f:
            self.experiments = json.load(f)

        print(f"Loaded {len(self.experiments)} experiments")

    def get_successful_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of successful experiments only.

        Returns:
            List of successful experiment dictionaries
        """
        return [exp for exp in self.experiments if exp.get('success', False)]

    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of failed experiments.

        Returns:
            List of failed experiment dictionaries
        """
        return [exp for exp in self.experiments if not exp.get('success', False)]

    def filter_experiments(
        self,
        **criteria
    ) -> List[Dict[str, Any]]:
        """
        Filter experiments by criteria.

        Args:
            **criteria: Key-value pairs to filter by
                       (e.g., hidden_size=64, num_layers=2)

        Returns:
            List of experiments matching all criteria
        """
        filtered = self.experiments

        for key, value in criteria.items():
            filtered = [
                exp for exp in filtered
                if self._get_nested_value(exp, key) == value
            ]

        return filtered

    def get_best_n_experiments(
        self,
        n: int = 10,
        metric: str = 'best_val_loss',
        mode: str = 'min'
    ) -> List[Dict[str, Any]]:
        """
        Get top N experiments by metric.

        Args:
            n: Number of experiments to return
            metric: Metric to sort by
            mode: 'min' or 'max'

        Returns:
            List of best N experiments
        """
        successful = self.get_successful_experiments()

        # Filter experiments with the metric
        with_metric = [
            exp for exp in successful
            if metric in exp.get('metrics', {})
        ]

        # Sort
        reverse = (mode == 'max')
        sorted_exps = sorted(
            with_metric,
            key=lambda x: x['metrics'][metric],
            reverse=reverse
        )

        return sorted_exps[:n]

    def to_dataframe(self, include_config: bool = True) -> pd.DataFrame:
        """
        Convert experiments to pandas DataFrame.

        Args:
            include_config: Whether to include configuration parameters as columns

        Returns:
            DataFrame with experiment data
        """
        if not self.experiments:
            return pd.DataFrame()

        rows = []

        for exp in self.experiments:
            row = {
                'experiment_name': exp.get('experiment_name', ''),
                'success': exp.get('success', False),
                'training_time': exp.get('training_time', 0.0),
                'total_epochs': exp.get('total_epochs', 0),
                'best_epoch': exp.get('best_epoch', -1),
                'timestamp': exp.get('timestamp', '')
            }

            # Add metrics
            for metric_name, metric_value in exp.get('metrics', {}).items():
                row[metric_name] = metric_value

            # Add config parameters if requested
            if include_config:
                config = exp.get('config', {})

                # Extract common parameters
                row['hidden_size'] = self._get_nested_value(
                    exp, 'hidden_size', default=None
                )
                row['num_layers'] = self._get_nested_value(
                    exp, 'num_layers', default=None
                )
                row['dropout'] = self._get_nested_value(
                    exp, 'dropout', default=None
                )
                row['learning_rate'] = self._get_nested_value(
                    exp, 'learning_rate', default=None
                )
                row['batch_size'] = self._get_nested_value(
                    exp, 'batch_size', default=None
                )
                row['optimizer'] = self._get_nested_value(
                    exp, 'optimizer', default=None
                )
                row['grad_clip'] = self._get_nested_value(
                    exp, 'grad_clip', default=None
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_statistics(
        self,
        metric: str = 'best_val_loss'
    ) -> Dict[str, float]:
        """
        Compute statistics for a metric across successful experiments.

        Args:
            metric: Metric name

        Returns:
            Dictionary with mean, std, min, max, median
        """
        successful = self.get_successful_experiments()

        values = [
            exp['metrics'][metric]
            for exp in successful
            if metric in exp.get('metrics', {})
        ]

        if not values:
            return {}

        df = pd.Series(values)

        return {
            'mean': float(df.mean()),
            'std': float(df.std()),
            'min': float(df.min()),
            'max': float(df.max()),
            'median': float(df.median()),
            'count': len(values)
        }

    def export_to_csv(self, output_path: str = 'outputs/experiments/experiments.csv'):
        """
        Export experiments to CSV file.

        Args:
            output_path: Path to save CSV file
        """
        df = self.to_dataframe()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Experiments exported to {output_path}")

    def print_summary(self):
        """Print summary of experiments."""
        total = len(self.experiments)
        successful = len(self.get_successful_experiments())
        failed = len(self.get_failed_experiments())

        print(f"\n{'='*60}")
        print(f"Experiment Summary")
        print(f"{'='*60}")
        print(f"Total experiments: {total}")
        print(f"Successful: {successful} ({100*successful/total if total > 0 else 0:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total if total > 0 else 0:.1f}%)")

        if successful > 0:
            stats = self.compute_statistics('best_val_loss')
            print(f"\nValidation Loss Statistics:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Median: {stats['median']:.6f}")

            # Get best experiment
            best_n = self.get_best_n_experiments(n=1, metric='best_val_loss', mode='min')
            if best_n:
                best = best_n[0]
                print(f"\nBest Experiment: {best['experiment_name']}")
                print(f"  Val Loss: {best['metrics']['best_val_loss']:.6f}")
                print(f"  Train Loss: {best['metrics']['best_train_loss']:.6f}")
                if 'best_val_correlation' in best['metrics']:
                    print(f"  Correlation: {best['metrics']['best_val_correlation']:.4f}")

        print(f"{'='*60}\n")

    def _get_nested_value(
        self,
        exp: Dict[str, Any],
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get nested value from experiment dictionary.

        Handles keys like 'hidden_size' which map to config['model']['lstm']['hidden_size']
        """
        # Map parameter names to config paths
        param_mapping = {
            'hidden_size': ('config', 'model', 'lstm', 'hidden_size'),
            'num_layers': ('config', 'model', 'lstm', 'num_layers'),
            'dropout': ('config', 'model', 'lstm', 'dropout'),
            'learning_rate': ('config', 'training', 'learning_rate'),
            'batch_size': ('config', 'training', 'batch_size'),
            'optimizer': ('config', 'training', 'optimizer'),
            'grad_clip': ('config', 'training', 'grad_clip')
        }

        if key in param_mapping:
            path = param_mapping[key]
        else:
            path = (key,)

        # Navigate through nested dict
        current = exp
        for part in path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current
