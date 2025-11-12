#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for LSTM Signal Extraction Model.

This script provides various modes for hyperparameter optimization:
- Baseline: Run baseline configuration
- Grid search: Systematic search over parameter combinations
- Random search: Sample random configurations
- Quick demo: Fast demonstration with small search space

Usage:
    # Run baseline
    python3 tune_hyperparameters.py --mode baseline

    # Grid search with limited combinations
    python3 tune_hyperparameters.py --mode grid --max-experiments 10

    # Random search with 20 experiments
    python3 tune_hyperparameters.py --mode random --n-experiments 20

    # Quick demo
    python3 tune_hyperparameters.py --mode quick
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.data.pytorch_dataset import SignalDataset
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.experiment_tracker import ExperimentTracker
from src.experiments.experiment_comparator import ExperimentComparator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_baseline(
    manager: ExperimentManager,
    train_dataset: SignalDataset,
    val_dataset: SignalDataset,
    num_epochs: int
):
    """Run baseline experiment with default configuration."""
    print("\n" + "="*80)
    print("Running Baseline Experiment")
    print("="*80)

    config = manager.base_config

    print("\nBaseline Configuration:")
    print(f"  Hidden Size: {config['model']['lstm']['hidden_size']}")
    print(f"  Num Layers: {config['model']['lstm']['num_layers']}")
    print(f"  Dropout: {config['model']['lstm']['dropout']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']}")

    result = manager.run_experiment(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        experiment_name='baseline',
        save_checkpoints=True
    )

    if result['success']:
        print("\n" + "="*80)
        print("Baseline Results:")
        print(f"  Best Val Loss: {result['metrics']['best_val_loss']:.6f}")
        print(f"  Best Train Loss: {result['metrics']['best_train_loss']:.6f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print("="*80 + "\n")


def run_grid_search(
    manager: ExperimentManager,
    train_dataset: SignalDataset,
    val_dataset: SignalDataset,
    num_epochs: int,
    max_experiments: int
):
    """Run grid search over hyperparameter space."""
    print("\n" + "="*80)
    print("Running Grid Search")
    print("="*80)

    # Define limited parameter grid for grid search
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2],
        'learning_rate': [5e-4, 1e-3, 5e-3]
    }

    print("\nParameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    results = manager.run_grid_search(
        param_grid=param_grid,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        max_experiments=max_experiments
    )

    # Show summary
    successful = [r for r in results if r['success']]
    print(f"\n{len(successful)}/{len(results)} experiments successful")

    if successful:
        best = min(successful, key=lambda x: x['metrics']['best_val_loss'])
        print(f"\nBest experiment: {best['experiment_name']}")
        print(f"  Val Loss: {best['metrics']['best_val_loss']:.6f}")


def run_random_search(
    manager: ExperimentManager,
    train_dataset: SignalDataset,
    val_dataset: SignalDataset,
    num_epochs: int,
    n_experiments: int
):
    """Run random search over hyperparameter space."""
    print("\n" + "="*80)
    print("Running Random Search")
    print("="*80)

    # Get full search space
    search_space = manager.define_search_space()

    print("\nSearch Space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")

    print(f"\nRunning {n_experiments} random experiments...")

    results = manager.run_random_search(
        search_space=search_space,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_experiments=n_experiments,
        num_epochs=num_epochs
    )

    # Show summary
    successful = [r for r in results if r['success']]
    print(f"\n{len(successful)}/{len(results)} experiments successful")

    if successful:
        best = min(successful, key=lambda x: x['metrics']['best_val_loss'])
        print(f"\nBest experiment: {best['experiment_name']}")
        print(f"  Val Loss: {best['metrics']['best_val_loss']:.6f}")


def run_quick_demo(
    manager: ExperimentManager,
    train_dataset: SignalDataset,
    val_dataset: SignalDataset
):
    """Run quick demo with small search space and few epochs."""
    print("\n" + "="*80)
    print("Running Quick Demo")
    print("="*80)

    # Small parameter grid
    param_grid = {
        'hidden_size': [32, 64],
        'learning_rate': [1e-3, 5e-3]
    }

    print("\nQuick Demo - Limited Search:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    results = manager.run_grid_search(
        param_grid=param_grid,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=5,  # Very short for demo
        max_experiments=4
    )

    print(f"\n{len([r for r in results if r['success']])}/{len(results)} experiments successful")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for LSTM Signal Extraction Model'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'grid', 'random', 'quick'],
        default='baseline',
        help='Tuning mode: baseline, grid, random, or quick'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--train-data',
        type=str,
        default='data/processed/train_dataset.h5',
        help='Path to training dataset'
    )

    parser.add_argument(
        '--val-data',
        type=str,
        default='data/processed/test_dataset.h5',
        help='Path to validation dataset (using test for simplicity)'
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=30,
        help='Number of training epochs per experiment'
    )

    parser.add_argument(
        '--max-experiments',
        type=int,
        default=20,
        help='Maximum experiments for grid search'
    )

    parser.add_argument(
        '--n-experiments',
        type=int,
        default=20,
        help='Number of experiments for random search'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/experiments',
        help='Directory for experiment outputs'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for training'
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.train_data).exists():
        print(f"Error: Training dataset not found: {args.train_data}")
        print("Please run dataset generation first:")
        print("  python3 generate_datasets.py")
        sys.exit(1)

    if not Path(args.val_data).exists():
        print(f"Error: Validation dataset not found: {args.val_data}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Load datasets
    print("Loading datasets...")
    print(f"  Train: {args.train_data}")
    print(f"  Val: {args.val_data}")

    train_dataset = SignalDataset(args.train_data)
    val_dataset = SignalDataset(args.val_data)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create experiment manager
    print(f"\nInitializing experiment manager...")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Device: {args.device}")

    manager = ExperimentManager(
        base_config=config,
        output_dir=args.output_dir,
        device=args.device
    )

    # Run experiments based on mode
    if args.mode == 'baseline':
        run_baseline(manager, train_dataset, val_dataset, args.num_epochs)

    elif args.mode == 'grid':
        run_grid_search(
            manager, train_dataset, val_dataset,
            args.num_epochs, args.max_experiments
        )

    elif args.mode == 'random':
        run_random_search(
            manager, train_dataset, val_dataset,
            args.num_epochs, args.n_experiments
        )

    elif args.mode == 'quick':
        run_quick_demo(manager, train_dataset, val_dataset)

    # Generate analysis
    print("\n" + "="*80)
    print("Generating Analysis")
    print("="*80)

    tracker = ExperimentTracker(f'{args.output_dir}/experiments.json')
    tracker.print_summary()

    # Export best config
    manager.export_best_config(f'{args.output_dir}/best_config.yaml')

    # Create comparator and visualizations
    print("\nCreating visualizations...")
    comparator = ExperimentComparator(
        tracker,
        output_dir=f'{args.output_dir}/figures'
    )

    # Create top experiments plot
    comparator.plot_top_experiments(
        save_path=f'{args.output_dir}/figures/top_experiments.png'
    )

    # Create loss distribution
    comparator.plot_loss_distribution(
        save_path=f'{args.output_dir}/figures/loss_distribution.png'
    )

    # Generate summary report
    comparator.generate_summary_report(
        output_path=f'{args.output_dir}/summary_report.md'
    )

    print("\n" + "="*80)
    print("Hyperparameter Tuning Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - experiments.json: All experiment data")
    print(f"  - best_config.yaml: Best configuration")
    print(f"  - summary_report.md: Analysis report")
    print(f"  - figures/: Visualization plots")
    print("\n")


if __name__ == '__main__':
    main()
