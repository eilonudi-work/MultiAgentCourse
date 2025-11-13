"""
Visualization Module for Translation Experiment Results

This script generates publication-quality graphs showing the relationship
between spelling error rate and semantic drift (vector distance).
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


def load_results(results_file: str = "results/experiment_results.json") -> dict:
    """
    Load experiment results from JSON file.

    Args:
        results_file: Path to results JSON

    Returns:
        Dictionary with experiment results
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_error_distance_plot(
    error_rates: List[float],
    cosine_distances: List[float],
    output_file: str = "results/error_vs_distance.png",
    title: str = "Semantic Drift vs Spelling Error Rate",
    figsize: Tuple[int, int] = (10, 6),
    show_regression: bool = True
):
    """
    Create a scatter plot with trend line showing error rate vs distance.

    Args:
        error_rates: List of error rates (as percentages, e.g., [0, 10, 25, 50])
        cosine_distances: List of corresponding cosine distances
        output_file: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        show_regression: Whether to show regression line
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(
        error_rates,
        cosine_distances,
        s=100,
        alpha=0.7,
        color='steelblue',
        edgecolors='navy',
        linewidth=1.5,
        zorder=3
    )

    # Add regression line if requested
    if show_regression and len(error_rates) >= 2:
        z = np.polyfit(error_rates, cosine_distances, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(error_rates), max(error_rates), 100)
        ax.plot(
            x_line,
            p(x_line),
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f'Linear fit: y = {z[0]:.5f}x + {z[1]:.5f}'
        )

        # Calculate R-squared
        y_pred = p(error_rates)
        ss_res = np.sum((np.array(cosine_distances) - y_pred) ** 2)
        ss_tot = np.sum((np.array(cosine_distances) - np.mean(cosine_distances)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ax.text(
            0.05, 0.95,
            f'R² = {r_squared:.4f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Labels and title
    ax.set_xlabel('Spelling Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    if show_regression:
        ax.legend(loc='upper left', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def create_comprehensive_plot(
    results: List[dict],
    output_file: str = "results/comprehensive_analysis.png"
):
    """
    Create a comprehensive multi-panel visualization.

    Args:
        results: List of result dictionaries
        output_file: Path to save the plot
    """
    # Sort results by error rate
    results = sorted(results, key=lambda x: x['error_rate'])

    error_rates = [r['error_rate'] * 100 for r in results]
    cosine_distances = [r['cosine_distance'] for r in results]
    cosine_similarities = [r['cosine_similarity'] for r in results]
    euclidean_distances = [r['euclidean_distance'] for r in results]

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Agent Translation Pipeline: Semantic Drift Analysis',
                 fontsize=16, fontweight='bold', y=0.998)

    # Plot 1: Cosine Distance
    axes[0, 0].scatter(error_rates, cosine_distances, s=100, alpha=0.7,
                       color='steelblue', edgecolors='navy', linewidth=1.5)
    if len(error_rates) >= 2:
        z = np.polyfit(error_rates, cosine_distances, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(error_rates), max(error_rates), 100)
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Spelling Error Rate (%)', fontweight='bold')
    axes[0, 0].set_ylabel('Cosine Distance', fontweight='bold')
    axes[0, 0].set_title('Cosine Distance vs Error Rate')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cosine Similarity
    axes[0, 1].scatter(error_rates, cosine_similarities, s=100, alpha=0.7,
                       color='forestgreen', edgecolors='darkgreen', linewidth=1.5)
    if len(error_rates) >= 2:
        z = np.polyfit(error_rates, cosine_similarities, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(error_rates), max(error_rates), 100)
        axes[0, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Spelling Error Rate (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Cosine Similarity', fontweight='bold')
    axes[0, 1].set_title('Cosine Similarity vs Error Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Euclidean Distance
    axes[1, 0].scatter(error_rates, euclidean_distances, s=100, alpha=0.7,
                       color='coral', edgecolors='red', linewidth=1.5)
    if len(error_rates) >= 2:
        z = np.polyfit(error_rates, euclidean_distances, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(error_rates), max(error_rates), 100)
        axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Spelling Error Rate (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Euclidean Distance', fontweight='bold')
    axes[1, 0].set_title('Euclidean Distance vs Error Rate')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Bar chart of distances at different error rates
    x_pos = np.arange(len(error_rates))
    width = 0.35
    axes[1, 1].bar(x_pos, cosine_distances, width, label='Cosine Dist',
                   alpha=0.7, color='steelblue')
    axes[1, 1].set_xlabel('Spelling Error Rate (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Distance', fontweight='bold')
    axes[1, 1].set_title('Distance Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'{int(er)}%' for er in error_rates])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {output_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    import sys

    # Check if results file exists
    results_file = "results/experiment_results.json"
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run the experiment first to generate results.")
        sys.exit(1)

    # Load results
    print("Loading results...")
    results = load_results(results_file)

    if not results:
        print("Error: No results found in file")
        sys.exit(1)

    print(f"Loaded {len(results)} data points")

    # Extract data
    results_sorted = sorted(results, key=lambda x: x['error_rate'])
    error_rates = [r['error_rate'] * 100 for r in results_sorted]
    cosine_distances = [r['cosine_distance'] for r in results_sorted]

    # Create simple plot
    print("\nGenerating main plot...")
    create_error_distance_plot(
        error_rates,
        cosine_distances,
        output_file="results/error_vs_distance.png",
        show_regression=True
    )

    # Create comprehensive plot
    print("\nGenerating comprehensive analysis...")
    create_comprehensive_plot(results, output_file="results/comprehensive_analysis.png")

    print("\n✓ Visualization complete!")
