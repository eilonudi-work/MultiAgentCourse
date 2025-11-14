#!/usr/bin/env python3
"""
Model Evaluation Script for LSTM Signal Extraction Model.

Performs comprehensive evaluation including:
- Full dataset evaluation with all metrics
- Per-frequency analysis
- Statistical validation
- PRD-required visualizations (Graph 1 & 2)
- Additional analysis plots
- Comprehensive reports

Usage:
    # Evaluate trained model
    python3 evaluate_model.py --checkpoint checkpoints/best_model.pt --dataset data/processed/test_dataset.h5

    # Quick demo
    python3 evaluate_model.py --checkpoint checkpoints/quick_demo/best_model.pt --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.pytorch_dataset import SignalDataset
from src.models.model_factory import ModelFactory
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.statistical_analyzer import StatisticalAnalyzer
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.visualizer import SignalVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LSTM Signal Extraction Model'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/processed/test_dataset.h5',
        help='Path to test dataset'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Directory for evaluation outputs'
    )

    parser.add_argument(
        '--figures-dir',
        type=str,
        default='outputs/figures',
        help='Directory for figures'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for evaluation'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick demo mode (limited samples)'
    )

    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip creating visualizations'
    )

    args = parser.parse_args()

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"LSTM Signal Extraction Model - Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Load model
    print(f"\nLoading model...")
    model = ModelFactory.create_from_checkpoint(checkpoint_path, device=args.device)
    print(f"Model loaded successfully")

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = SignalDataset(dataset_path)

    # Quick mode: use subset
    if args.quick:
        print(f"Quick mode: Using first 20 samples")
        dataset.mixed_signals = dataset.mixed_signals[:20]
        dataset.target_signals = dataset.target_signals[:20]
        dataset.condition_vectors = dataset.condition_vectors[:20]

    print(f"Dataset loaded: {len(dataset)} samples")

    # Create evaluator
    print(f"\nInitializing evaluator...")
    evaluator = ModelEvaluator(model, dataset, device=args.device)

    # Run evaluation
    print(f"\n{'='*80}")
    print(f"Running Evaluation")
    print(f"{'='*80}")

    results = evaluator.evaluate_full_dataset(
        batch_size=args.batch_size,
        save_predictions=True  # Save for visualization
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'test_metrics.json'
    evaluator.save_evaluation_results(results, results_file)

    # Generate report
    report_file = output_dir / 'evaluation_report.md'
    evaluator.generate_evaluation_report(results, report_file)

    # Statistical analysis
    print(f"\n{'='*80}")
    print(f"Statistical Analysis")
    print(f"{'='*80}")

    analyzer = StatisticalAnalyzer()

    # Per-frequency analysis
    freq_analysis = analyzer.analyze_per_frequency_performance(
        results['per_frequency_metrics']
    )

    print(f"\nPer-Frequency Performance Analysis:")
    if 'best_frequency' in freq_analysis and freq_analysis['best_frequency']:
        best = freq_analysis['best_frequency']
        print(f"  Best: {best['frequency']} Hz (MSE: {best['mse']:.6f})")

    if 'worst_frequency' in freq_analysis and freq_analysis['worst_frequency']:
        worst = freq_analysis['worst_frequency']
        print(f"  Worst: {worst['frequency']} Hz (MSE: {worst['mse']:.6f})")

    # Error analysis
    if 'per_sample_results' in results:
        print(f"\n{'='*80}")
        print(f"Error Analysis")
        print(f"{'='*80}")

        error_analyzer = ErrorAnalyzer()

        # Find worst predictions
        worst_predictions = error_analyzer.find_worst_predictions(
            results['per_sample_results'],
            n=5,
            metric='mse'
        )

        print(f"\nTop 5 Worst Predictions:")
        for i, sample in enumerate(worst_predictions, 1):
            print(f"  {i}. Frequency {sample['frequency']} Hz - "
                  f"MSE: {sample['metrics']['mse']:.6f}")

    # Create visualizations
    if not args.no_visualizations:
        print(f"\n{'='*80}")
        print(f"Creating Visualizations")
        print(f"{'='*80}")

        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        visualizer = SignalVisualizer(dpi=300)

        # PRD-required Graph 1: Detailed f₂ (3 Hz) analysis
        print(f"\nCreating Graph 1 (f₂ detailed)...")
        # Find a sample for 3 Hz
        f2_sample = None
        for sample in results['per_sample_results']:
            if sample['frequency'] == 3:
                f2_sample = sample
                break

        if f2_sample:
            # Need to get mixed signal too
            # Find the sample index
            sample_idx = results['per_sample_results'].index(f2_sample)

            # Get mixed signal from dataset
            mixed_signal = dataset.mixed_signals[sample_idx]

            graph1_data = {
                'target': f2_sample['target'],
                'prediction': f2_sample['prediction'],
                'mixed_signal': mixed_signal,
                'mse': f2_sample['metrics']['mse']
            }

            visualizer.create_f2_detailed_plot(
                graph1_data,
                save_path=figures_dir / 'graph1_f2_detailed.png'
            )
        else:
            print("Warning: No 3 Hz sample found for Graph 1")

        # PRD-required Graph 2: All frequencies comparison
        print(f"\nCreating Graph 2 (all frequencies)...")
        frequency_samples = {}

        # Get one sample per frequency
        frequencies = [1, 3, 5, 7]
        for freq in frequencies:
            for sample in results['per_sample_results']:
                if sample['frequency'] == freq:
                    frequency_samples[freq] = {
                        'target': sample['target'],
                        'prediction': sample['prediction'],
                        'mse': sample['metrics']['mse'],
                        'r2': sample['metrics']['r2']
                    }
                    break

        if frequency_samples:
            visualizer.create_all_frequencies_plot(
                frequency_samples,
                frequencies=frequencies,
                save_path=figures_dir / 'graph2_all_frequencies.png'
            )
        else:
            print("Warning: Insufficient data for Graph 2")

        # Additional visualizations
        print(f"\nCreating additional visualizations...")

        # Per-frequency metrics
        visualizer.plot_per_frequency_metrics(
            results['per_frequency_metrics'],
            save_path=figures_dir / 'per_frequency_metrics.png'
        )

        # Error distribution
        if 'per_sample_results' in results:
            all_predictions = np.array([s['prediction'] for s in results['per_sample_results']])
            all_targets = np.array([s['target'] for s in results['per_sample_results']])
            errors = all_predictions - all_targets

            visualizer.plot_error_distribution(
                errors,
                save_path=figures_dir / 'error_distribution.png'
            )

            # Scatter plot
            visualizer.plot_prediction_vs_target_scatter(
                all_predictions,
                all_targets,
                save_path=figures_dir / 'prediction_vs_target.png'
            )

    # Final summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - Metrics: {results_file}")
    print(f"  - Report: {report_file}")
    if not args.no_visualizations:
        print(f"  - Figures: {figures_dir}")

    print(f"\nKey Metrics:")
    print(f"  MSE: {results['overall_metrics']['mse']:.6f}")
    print(f"  MAE: {results['overall_metrics']['mae']:.6f}")
    print(f"  Correlation: {results['overall_metrics']['correlation']:.4f}")
    print(f"  R²: {results['overall_metrics']['r2']:.4f}")
    print(f"  SNR: {results['overall_metrics']['snr']:.2f} dB")

    print(f"\nPRD Target (MSE < 0.01): ", end='')
    if results['summary']['mse_target_met']:
        print("✅ PASS")
    else:
        print(f"❌ FAIL (MSE = {results['overall_metrics']['mse']:.6f})")

    print(f"\n")


if __name__ == '__main__':
    main()
