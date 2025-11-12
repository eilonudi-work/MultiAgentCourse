"""
Experiment Manager for hyperparameter tuning.

This module provides the ExperimentManager class for running and managing
hyperparameter optimization experiments.
"""

import copy
import itertools
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback
)
from src.training.utils import create_optimizer, create_criterion, set_seed


class ExperimentManager:
    """
    Manage hyperparameter tuning experiments.

    Features:
    - Define search spaces for hyperparameters
    - Run single experiments
    - Grid search over parameter combinations
    - Random search with sampling
    - Track and save all experiment results
    - Find best experiment based on metrics

    Attributes:
        base_config: Base configuration dictionary
        output_dir: Directory to save experiment results
        device: Device to run experiments on ('cpu' or 'cuda')
        experiments: List of completed experiments
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: str = 'outputs/experiments',
        device: str = 'cpu'
    ):
        """
        Initialize experiment manager.

        Args:
            base_config: Base configuration with default parameters
            output_dir: Directory to save experiment results
            device: Device for training ('cpu' or 'cuda')
        """
        self.base_config = copy.deepcopy(base_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.experiments = []

        # Load existing experiments if available
        self._load_experiments()

    def define_search_space(self) -> Dict[str, List[Any]]:
        """
        Define hyperparameter search space.

        Returns:
            Dictionary mapping parameter names to lists of values to try.

        Search space includes:
        - Model architecture: hidden_size, num_layers, dropout
        - Training: learning_rate, batch_size
        - Data: noise_std (if applicable)
        """
        search_space = {
            # Model parameters
            'hidden_size': [32, 64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.0, 0.1, 0.2, 0.3],

            # Training parameters
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [8, 16, 32, 64],

            # Optimizer
            'optimizer': ['adam', 'adamw', 'sgd'],

            # Gradient clipping
            'grad_clip': [0.5, 1.0, 2.0, None]
        }

        return search_space

    def run_experiment(
        self,
        config: Dict[str, Any],
        train_dataset: SignalDataset,
        val_dataset: SignalDataset,
        num_epochs: int = 50,
        experiment_name: Optional[str] = None,
        save_checkpoints: bool = False
    ) -> Dict[str, Any]:
        """
        Run single experiment with given configuration.

        Args:
            config: Experiment configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            experiment_name: Name for this experiment
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Dictionary with experiment results including:
            - config: Configuration used
            - metrics: Training and validation metrics
            - best_epoch: Epoch with best validation loss
            - training_time: Time taken for training
        """
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = f"exp_{len(self.experiments):04d}"

        print(f"\n{'='*80}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*80}")

        # Set random seed for reproducibility
        seed = config.get('training', {}).get('seed', 42)
        set_seed(seed)

        # Start timer
        start_time = time.time()

        try:
            # Create model
            model = ModelFactory.create_model(config, device=self.device)

            # Create data loaders
            batch_size = config.get('training', {}).get('batch_size', 32)
            train_loader = DataLoaderFactory.create_train_loader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoaderFactory.create_eval_loader(
                val_dataset, batch_size=batch_size
            )

            # Create optimizer and criterion
            optimizer_name = config.get('training', {}).get('optimizer', 'adam')
            learning_rate = config.get('training', {}).get('learning_rate', 0.001)
            optimizer = create_optimizer(model, optimizer_name, learning_rate)

            criterion_name = config.get('training', {}).get('criterion', 'mse')
            criterion = create_criterion(criterion_name)

            # Setup callbacks
            callbacks = []

            if save_checkpoints:
                checkpoint_dir = self.output_dir / experiment_name / 'checkpoints'
                callbacks.append(
                    CheckpointCallback(
                        checkpoint_dir=checkpoint_dir,
                        save_best=True,
                        save_last=False,
                        monitor='val_loss'
                    )
                )

            # Add early stopping
            early_stop_patience = config.get('training', {}).get('early_stop_patience', 15)
            callbacks.append(
                EarlyStoppingCallback(
                    patience=early_stop_patience,
                    min_delta=0.0001,
                    monitor='val_loss',
                    restore_best_weights=True
                )
            )

            # Add progress callback (quiet mode)
            callbacks.append(
                ProgressCallback(print_every_n_epochs=10)
            )

            # Create trainer
            grad_clip = config.get('training', {}).get('grad_clip', 1.0)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                config=config,
                callbacks=callbacks,
                grad_clip_value=grad_clip
            )

            # Train
            summary = trainer.train(num_epochs=num_epochs)

            # Get metrics
            training_time = time.time() - start_time
            best_metrics = trainer.get_best_metrics()

            # Create result dictionary
            result = {
                'experiment_name': experiment_name,
                'config': config,
                'metrics': {
                    'final_train_loss': trainer.metrics_tracker.get_latest('train_loss'),
                    'final_val_loss': trainer.metrics_tracker.get_latest('val_loss'),
                    'best_train_loss': trainer.metrics_tracker.get_best('train_loss', mode='min'),
                    'best_val_loss': trainer.metrics_tracker.get_best('val_loss', mode='min'),
                    'best_val_correlation': best_metrics.get('val_correlation', 0.0),
                    'best_val_r2': best_metrics.get('val_r2', 0.0),
                    'best_val_snr': best_metrics.get('val_snr', 0.0)
                },
                'best_epoch': trainer.metrics_tracker.get_best_epoch('val_loss', mode='min'),
                'total_epochs': len(trainer.metrics_tracker.get_history('train_loss')),
                'training_time': training_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': True
            }

            print(f"\nExperiment completed successfully!")
            print(f"Best validation loss: {result['metrics']['best_val_loss']:.6f}")
            print(f"Training time: {training_time:.2f}s")

        except Exception as e:
            # Handle failures
            training_time = time.time() - start_time
            result = {
                'experiment_name': experiment_name,
                'config': config,
                'metrics': {},
                'best_epoch': -1,
                'total_epochs': 0,
                'training_time': training_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'error': str(e)
            }

            print(f"\nExperiment failed: {e}")

        # Save result
        self.experiments.append(result)
        self._save_experiments()

        return result

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        train_dataset: SignalDataset,
        val_dataset: SignalDataset,
        num_epochs: int = 50,
        max_experiments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run grid search over parameter combinations.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs per experiment
            max_experiments: Maximum number of experiments to run (None for all)

        Returns:
            List of experiment results
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        all_combinations = list(itertools.product(*param_values))

        # Limit if requested
        if max_experiments is not None and len(all_combinations) > max_experiments:
            print(f"Grid search: {len(all_combinations)} combinations found, "
                  f"limiting to {max_experiments}")
            all_combinations = all_combinations[:max_experiments]
        else:
            print(f"Grid search: Running {len(all_combinations)} experiments")

        results = []

        for i, param_combination in enumerate(all_combinations):
            # Create config for this combination
            config = copy.deepcopy(self.base_config)

            # Update config with parameters
            for param_name, param_value in zip(param_names, param_combination):
                self._set_config_param(config, param_name, param_value)

            # Run experiment
            experiment_name = f"grid_{i+1:04d}"
            result = self.run_experiment(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=num_epochs,
                experiment_name=experiment_name,
                save_checkpoints=False
            )

            results.append(result)

            # Print progress
            print(f"\nGrid search progress: {i+1}/{len(all_combinations)} experiments completed")

        return results

    def run_random_search(
        self,
        search_space: Dict[str, List[Any]],
        train_dataset: SignalDataset,
        val_dataset: SignalDataset,
        n_experiments: int = 20,
        num_epochs: int = 50,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run random search by sampling parameter combinations.

        Args:
            search_space: Dictionary mapping parameter names to lists of values
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_experiments: Number of random experiments to run
            num_epochs: Number of epochs per experiment
            seed: Random seed for sampling

        Returns:
            List of experiment results
        """
        if seed is not None:
            random.seed(seed)

        print(f"Random search: Running {n_experiments} experiments")

        results = []

        for i in range(n_experiments):
            # Sample random configuration
            config = copy.deepcopy(self.base_config)

            for param_name, param_values in search_space.items():
                param_value = random.choice(param_values)
                self._set_config_param(config, param_name, param_value)

            # Run experiment
            experiment_name = f"random_{i+1:04d}"
            result = self.run_experiment(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=num_epochs,
                experiment_name=experiment_name,
                save_checkpoints=False
            )

            results.append(result)

            # Print progress
            print(f"\nRandom search progress: {i+1}/{n_experiments} experiments completed")

        return results

    def get_best_experiment(
        self,
        metric: str = 'best_val_loss',
        mode: str = 'min'
    ) -> Optional[Dict[str, Any]]:
        """
        Get best experiment based on metric.

        Args:
            metric: Metric name to use for comparison
            mode: 'min' or 'max'

        Returns:
            Best experiment result or None if no experiments
        """
        if not self.experiments:
            return None

        # Filter successful experiments
        successful = [exp for exp in self.experiments if exp.get('success', False)]

        if not successful:
            return None

        # Get experiments with the metric
        with_metric = [
            exp for exp in successful
            if metric in exp.get('metrics', {})
        ]

        if not with_metric:
            return None

        # Find best
        if mode == 'min':
            best = min(with_metric, key=lambda x: x['metrics'][metric])
        else:
            best = max(with_metric, key=lambda x: x['metrics'][metric])

        return best

    def _set_config_param(self, config: Dict[str, Any], param_name: str, param_value: Any):
        """
        Set parameter in config dictionary.

        Handles nested parameters like 'hidden_size' -> config['model']['lstm']['hidden_size']
        """
        # Map parameter names to config paths
        param_mapping = {
            'hidden_size': ('model', 'lstm', 'hidden_size'),
            'num_layers': ('model', 'lstm', 'num_layers'),
            'dropout': ('model', 'lstm', 'dropout'),
            'learning_rate': ('training', 'learning_rate'),
            'batch_size': ('training', 'batch_size'),
            'optimizer': ('training', 'optimizer'),
            'grad_clip': ('training', 'grad_clip')
        }

        if param_name in param_mapping:
            path = param_mapping[param_name]
            # Navigate to correct location
            current = config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            # Set value
            current[path[-1]] = param_value
        else:
            # Direct assignment
            config[param_name] = param_value

    def _save_experiments(self):
        """Save experiments to JSON file."""
        output_file = self.output_dir / 'experiments.json'
        with open(output_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def _load_experiments(self):
        """Load experiments from JSON file if it exists."""
        output_file = self.output_dir / 'experiments.json'
        if output_file.exists():
            with open(output_file, 'r') as f:
                self.experiments = json.load(f)
            print(f"Loaded {len(self.experiments)} previous experiments")

    def export_best_config(
        self,
        output_path: str = 'outputs/experiments/best_config.yaml'
    ) -> Optional[str]:
        """
        Export best configuration to YAML file.

        Args:
            output_path: Path to save best config

        Returns:
            Path to saved file or None if no experiments
        """
        best = self.get_best_experiment()

        if best is None:
            print("No successful experiments found")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(best['config'], f, default_flow_style=False, sort_keys=False)

        print(f"Best configuration saved to {output_path}")
        print(f"Best validation loss: {best['metrics']['best_val_loss']:.6f}")

        return str(output_path)
