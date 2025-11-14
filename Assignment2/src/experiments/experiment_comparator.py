"""
Experiment Comparator for visualization and analysis.

Provides tools for comparing experiments and generating visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .experiment_tracker import ExperimentTracker


class ExperimentComparator:
    """
    Compare and visualize experiments.

    Features:
    - Create comparison tables
    - Plot training curves
    - Visualize hyperparameter effects
    - Generate summary reports

    Attributes:
        tracker: ExperimentTracker instance
        output_dir: Directory to save visualizations
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        output_dir: str = 'outputs/experiments/figures'
    ):
        """
        Initialize experiment comparator.

        Args:
            tracker: ExperimentTracker instance with loaded experiments
            output_dir: Directory to save figures
        """
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def create_comparison_table(
        self,
        metrics: Optional[List[str]] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Create comparison table for top experiments.

        Args:
            metrics: List of metrics to include (None for all)
            top_n: Number of top experiments to show

        Returns:
            DataFrame with experiment comparisons
        """
        if metrics is None:
            metrics = ['best_val_loss', 'best_train_loss', 'best_val_correlation']

        # Get best experiments
        best_exps = self.tracker.get_best_n_experiments(
            n=top_n,
            metric='best_val_loss',
            mode='min'
        )

        if not best_exps:
            return pd.DataFrame()

        rows = []

        for i, exp in enumerate(best_exps, 1):
            row = {
                'Rank': i,
                'Experiment': exp['experiment_name'],
                'Hidden Size': self.tracker._get_nested_value(exp, 'hidden_size'),
                'Num Layers': self.tracker._get_nested_value(exp, 'num_layers'),
                'Dropout': self.tracker._get_nested_value(exp, 'dropout'),
                'Learning Rate': self.tracker._get_nested_value(exp, 'learning_rate'),
                'Batch Size': self.tracker._get_nested_value(exp, 'batch_size')
            }

            # Add metrics
            for metric in metrics:
                if metric in exp.get('metrics', {}):
                    row[metric] = exp['metrics'][metric]

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_top_experiments(
        self,
        metric: str = 'best_val_loss',
        top_n: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart of top experiments by metric.

        Args:
            metric: Metric to plot
            top_n: Number of experiments to show
            save_path: Path to save figure (None for display only)
        """
        best_exps = self.tracker.get_best_n_experiments(n=top_n, metric=metric, mode='min')

        if not best_exps:
            print("No experiments to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = [exp['experiment_name'] for exp in best_exps]
        values = [exp['metrics'][metric] for exp in best_exps]

        bars = ax.barh(range(len(names)), values, color='steelblue', alpha=0.7)

        # Highlight best
        bars[0].set_color('darkgreen')
        bars[0].set_alpha(0.9)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Top {top_n} Experiments by {metric.replace("_", " ").title()}')
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_hyperparameter_effects(
        self,
        param_name: str,
        metric: str = 'best_val_loss',
        save_path: Optional[str] = None
    ):
        """
        Plot effect of hyperparameter on metric.

        Args:
            param_name: Hyperparameter name (e.g., 'hidden_size', 'learning_rate')
            metric: Metric to plot
            save_path: Path to save figure
        """
        df = self.tracker.to_dataframe(include_config=True)

        # Filter successful experiments with the metric
        df = df[df['success'] & df[metric].notna() & df[param_name].notna()]

        if df.empty:
            print(f"No data available for {param_name}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by parameter value
        grouped = df.groupby(param_name)[metric].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_index()

        x = grouped.index.tolist()
        y_mean = grouped['mean'].values
        y_std = grouped['std'].values
        counts = grouped['count'].values

        # Plot
        ax.errorbar(x, y_mean, yerr=y_std, marker='o', linestyle='-',
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    color='steelblue', label='Mean ± Std')

        # Annotate counts
        for i, (xi, yi, count) in enumerate(zip(x, y_mean, counts)):
            ax.annotate(f'n={count}', xy=(xi, yi), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=8)

        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Effect of {param_name.replace("_", " ").title()} on {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_parameter_heatmap(
        self,
        param1: str,
        param2: str,
        metric: str = 'best_val_loss',
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of metric for two hyperparameters.

        Args:
            param1: First hyperparameter
            param2: Second hyperparameter
            metric: Metric to visualize
            save_path: Path to save figure
        """
        df = self.tracker.to_dataframe(include_config=True)

        # Filter successful experiments
        df = df[df['success'] & df[metric].notna() &
                df[param1].notna() & df[param2].notna()]

        if df.empty:
            print(f"No data available for {param1} vs {param2}")
            return

        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index=param2,
            columns=param1,
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',  # Red for high (bad), green for low (good)
            ax=ax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )

        ax.set_xlabel(param1.replace('_', ' ').title())
        ax.set_ylabel(param2.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}: {param1} vs {param2}')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_loss_distribution(
        self,
        metric: str = 'best_val_loss',
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of loss values across experiments.

        Args:
            metric: Metric to plot
            save_path: Path to save figure
        """
        df = self.tracker.to_dataframe()

        # Filter successful experiments
        df = df[df['success'] & df[metric].notna()]

        if df.empty:
            print("No successful experiments")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(df[metric], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(df[metric].median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {df[metric].median():.4f}')
        ax1.axvline(df[metric].mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean: {df[metric].mean():.4f}')
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {metric.replace("_", " ").title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        box = ax2.boxplot([df[metric]], vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('steelblue')
        box['boxes'][0].set_alpha(0.7)
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title(f'Box Plot of {metric.replace("_", " ").title()}')
        ax2.set_xticklabels(['All Experiments'])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_training_time_comparison(
        self,
        save_path: Optional[str] = None
    ):
        """
        Compare training times across experiments.

        Args:
            save_path: Path to save figure
        """
        df = self.tracker.to_dataframe(include_config=True)
        df = df[df['success']]

        if df.empty:
            print("No successful experiments")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot by batch size
        if 'batch_size' in df.columns and df['batch_size'].notna().any():
            grouped = df.groupby('batch_size')['training_time'].agg(['mean', 'std'])

            x = grouped.index.tolist()
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values

            ax.errorbar(x, y_mean, yerr=y_std, marker='o', linestyle='-',
                       linewidth=2, markersize=8, capsize=5, capthick=2,
                       color='steelblue')

            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Training Time vs Batch Size')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_summary_report(
        self,
        output_path: str = 'outputs/experiments/summary_report.md'
    ):
        """
        Generate comprehensive summary report in Markdown.

        Args:
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Hyperparameter Tuning Summary Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall statistics
            f.write("## Overall Statistics\n\n")
            total = len(self.tracker.experiments)
            successful = len(self.tracker.get_successful_experiments())
            f.write(f"- Total experiments: {total}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Success rate: {100*successful/total:.1f}%\n\n")

            if successful > 0:
                # Validation loss stats
                stats = self.tracker.compute_statistics('best_val_loss')
                f.write("## Validation Loss Statistics\n\n")
                f.write(f"- Mean: {stats['mean']:.6f}\n")
                f.write(f"- Std: {stats['std']:.6f}\n")
                f.write(f"- Min: {stats['min']:.6f}\n")
                f.write(f"- Max: {stats['max']:.6f}\n")
                f.write(f"- Median: {stats['median']:.6f}\n\n")

                # Best experiments
                f.write("## Top 10 Experiments\n\n")
                comparison_table = self.create_comparison_table(top_n=10)
                f.write(comparison_table.to_markdown(index=False))
                f.write("\n\n")

                # Best experiment details
                best = self.tracker.get_best_n_experiments(n=1)[0]
                f.write("## Best Experiment Details\n\n")
                f.write(f"**Experiment:** {best['experiment_name']}\n\n")
                f.write("**Configuration:**\n")
                f.write(f"- Hidden Size: {self.tracker._get_nested_value(best, 'hidden_size')}\n")
                f.write(f"- Num Layers: {self.tracker._get_nested_value(best, 'num_layers')}\n")
                f.write(f"- Dropout: {self.tracker._get_nested_value(best, 'dropout')}\n")
                f.write(f"- Learning Rate: {self.tracker._get_nested_value(best, 'learning_rate')}\n")
                f.write(f"- Batch Size: {self.tracker._get_nested_value(best, 'batch_size')}\n")
                f.write(f"- Optimizer: {self.tracker._get_nested_value(best, 'optimizer')}\n\n")

                f.write("**Performance:**\n")
                for metric_name, metric_value in best['metrics'].items():
                    f.write(f"- {metric_name}: {metric_value:.6f}\n")
                f.write("\n")

        print(f"Summary report saved to {output_path}")

    def create_all_visualizations(self):
        """Create all standard visualizations."""
        print("Creating visualizations...")

        # 1. Top experiments
        self.plot_top_experiments(
            save_path=self.output_dir / 'top_experiments.png'
        )

        # 2. Hyperparameter effects
        for param in ['hidden_size', 'num_layers', 'learning_rate', 'batch_size']:
            self.plot_hyperparameter_effects(
                param,
                save_path=self.output_dir / f'effect_{param}.png'
            )

        # 3. Loss distribution
        self.plot_loss_distribution(
            save_path=self.output_dir / 'loss_distribution.png'
        )

        # 4. Training time
        self.plot_training_time_comparison(
            save_path=self.output_dir / 'training_time.png'
        )

        # 5. Heatmaps
        self.plot_parameter_heatmap(
            'hidden_size', 'num_layers',
            save_path=self.output_dir / 'heatmap_hidden_layers.png'
        )

        self.plot_parameter_heatmap(
            'learning_rate', 'batch_size',
            save_path=self.output_dir / 'heatmap_lr_batch.png'
        )

        print("All visualizations created!")
