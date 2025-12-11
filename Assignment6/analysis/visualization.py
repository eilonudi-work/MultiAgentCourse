"""
Visualization module for prompt engineering experiments.
Creates charts and graphs comparing different prompt variations.
"""

import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


class ExperimentVisualizer:
    """Create visualizations for experiment results."""

    def __init__(self, output_dir: str = "../visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def load_comparison_results(self, filepath: str) -> Dict[str, Any]:
        """Load comparison metrics from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def create_accuracy_comparison(self, data: Dict[str, Any], save: bool = True):
        """
        Create bar chart comparing accuracy across variations.

        Args:
            data: Comparison metrics data
            save: Whether to save the figure
        """
        variations = data['summary']

        # Extract data
        names = list(variations.keys())
        accuracies = [v['accuracy'] * 100 for v in variations.values()]
        f1_scores = [v['f1_score'] * 100 for v in variations.values()]

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        names = [names[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(names))
        width = 0.35

        # Plot bars
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)

        # Customize
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Accuracy Comparison Across Prompt Variations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 105])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'accuracy_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved accuracy comparison to {filepath}")

        return fig

    def create_metrics_comparison(self, data: Dict[str, Any], save: bool = True):
        """
        Create grouped bar chart with all metrics.

        Args:
            data: Comparison metrics data
            save: Whether to save the figure
        """
        variations = data['summary']

        # Extract data
        names = list(variations.keys())
        metrics = {
            'Accuracy': [v['accuracy'] * 100 for v in variations.values()],
            'Precision': [v['precision'] * 100 for v in variations.values()],
            'Recall': [v['recall'] * 100 for v in variations.values()],
            'F1 Score': [v['f1_score'] * 100 for v in variations.values()]
        }

        # Sort by accuracy
        sorted_indices = np.argsort(metrics['Accuracy'])[::-1]
        names = [names[i] for i in sorted_indices]
        for key in metrics:
            metrics[key] = [metrics[key][i] for i in sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(names))
        width = 0.2

        # Plot bars for each metric
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        for idx, (metric, values) in enumerate(metrics.items()):
            offset = width * (idx - 1.5)
            ax.bar(x + offset, values, width, label=metric, alpha=0.8, color=colors[idx])

        # Customize
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metrics comparison to {filepath}")

        return fig

    def create_distance_histogram(self, data: Dict[str, Any], save: bool = True):
        """
        Create histogram of distance distributions for each variation.

        Args:
            data: Full comparison data with metrics
            save: Whether to save the figure
        """
        variations = data['variations']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (name, metrics) in enumerate(variations.items()):
            ax = axes[idx]

            # Get distance distribution
            mean_dist = metrics['mean_distance']
            std_dist = metrics['std_distance']
            dist_info = metrics['distance_distribution']

            # Create histogram (simulate from statistics)
            # In real scenario, you'd have individual distances
            num_samples = metrics['successful_samples']
            simulated_distances = np.random.normal(mean_dist, std_dist, num_samples)
            simulated_distances = np.clip(simulated_distances, 0, 1)

            # Plot
            ax.hist(simulated_distances, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.3f}')

            # Customize
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('Distance Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Distance Distribution Across Variations', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'distance_distributions.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distance distributions to {filepath}")

        return fig

    def create_confusion_matrix_comparison(self, data: Dict[str, Any], save: bool = True):
        """
        Create confusion matrices for each variation.

        Args:
            data: Full comparison data
            save: Whether to save the figure
        """
        variations = data['variations']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (name, metrics) in enumerate(variations.items()):
            ax = axes[idx]

            cm_data = metrics['confusion_matrix']

            # Create confusion matrix
            matrix = np.array([
                [cm_data['true_positive'], cm_data['false_negative']],
                [cm_data['false_positive'], cm_data['true_negative']]
            ])

            # Plot heatmap
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Positive', 'Negative'],
                       yticklabels=['Positive', 'Negative'],
                       cbar=False)

            # Customize
            ax.set_title(f'{name}\nF1: {cm_data["f1_score"]:.1%}', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        plt.suptitle('Confusion Matrices by Variation', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrices to {filepath}")

        return fig

    def create_category_performance(self, data: Dict[str, Any], save: bool = True):
        """
        Create heatmap showing performance by category for each variation.

        Args:
            data: Full comparison data
            save: Whether to save the figure
        """
        variations = data['variations']

        # Extract category stats
        all_categories = set()
        for metrics in variations.values():
            if 'category_stats' in metrics:
                all_categories.update(metrics['category_stats'].keys())

        all_categories = sorted(all_categories)

        # Build matrix
        matrix_data = []
        variation_names = []

        for name, metrics in variations.items():
            variation_names.append(name)
            row = []
            for category in all_categories:
                if 'category_stats' in metrics and category in metrics['category_stats']:
                    accuracy = metrics['category_stats'][category]['accuracy']
                    row.append(accuracy * 100)
                else:
                    row.append(0)
            matrix_data.append(row)

        matrix = np.array(matrix_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                   xticklabels=all_categories,
                   yticklabels=variation_names,
                   cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)

        ax.set_title('Per-Category Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Variation', fontsize=12)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'category_performance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved category performance to {filepath}")

        return fig

    def create_all_visualizations(self, comparison_file: str):
        """
        Create all visualizations from a comparison file.

        Args:
            comparison_file: Path to comparison_metrics JSON file
        """
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60 + "\n")

        # Load data
        data = self.load_comparison_results(comparison_file)
        print(f"✓ Loaded data from {comparison_file}")
        print(f"  Model: {data.get('model', 'unknown')}")
        print(f"  Variations: {len(data['variations'])}")
        print()

        # Create visualizations
        self.create_accuracy_comparison(data)
        self.create_metrics_comparison(data)
        self.create_distance_histogram(data)
        self.create_confusion_matrix_comparison(data)
        self.create_category_performance(data)

        print("\n" + "="*60)
        print("All Visualizations Created! 🎨")
        print("="*60)
        print(f"\nSaved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  - accuracy_comparison.png")
        print("  - metrics_comparison.png")
        print("  - distance_distributions.png")
        print("  - confusion_matrices.png")
        print("  - category_performance.png")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations from experiment results")
    parser.add_argument(
        "--comparison-file",
        type=str,
        help="Path to comparison_metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest comparison file"
    )

    args = parser.parse_args()

    # Find comparison file
    if args.latest or not args.comparison_file:
        results_dir = "../results"
        comparison_files = [f for f in os.listdir(results_dir) if f.startswith("comparison_metrics_")]
        if not comparison_files:
            print("Error: No comparison files found in results/")
            print("Please run experiments first: ./run_all.sh")
            return

        # Get latest
        comparison_files.sort(reverse=True)
        comparison_file = os.path.join(results_dir, comparison_files[0])
        print(f"Using latest comparison file: {comparison_file}")
    else:
        comparison_file = args.comparison_file

    # Create visualizations
    visualizer = ExperimentVisualizer(args.output_dir)
    visualizer.create_all_visualizations(comparison_file)


if __name__ == "__main__":
    main()
