#!/usr/bin/env python3
"""
Training script for LSTM Signal Extraction Model.

This script demonstrates the complete training pipeline:
1. Load configuration
2. Create datasets and dataloaders
3. Initialize model, optimizer, criterion
4. Setup callbacks (checkpointing, early stopping, logging)
5. Train model
6. Evaluate on test set

Usage:
    python3 train_model.py --config config/train_config.yaml
    python3 train_model.py --quick  # Quick demo with 5 epochs
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pytorch_dataset import SignalDataset, DataLoaderFactory
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    TensorBoardCallback,
    ProgressCallback
)
from src.training.utils import (
    create_optimizer,
    create_criterion,
    set_seed,
    get_device,
    print_training_config,
    validate_training_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_quick_config() -> dict:
    """Create quick demo configuration."""
    return {
        'model': {
            'lstm': {
                'input_size': 5,
                'hidden_size': 32,  # Smaller for quick demo
                'num_layers': 2,
                'dropout': 0.1
            }
        },
        'training': {
            'num_epochs': 5,  # Few epochs for quick demo
            'batch_size': 4,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'criterion': 'mse',
            'grad_clip': 1.0,
            'seed': 42
        },
        'data': {
            'train_path': 'data/processed/train_dataset.h5',
            'test_path': 'data/processed/test_dataset.h5',
            'normalize': False
        },
        'callbacks': {
            'checkpoint': {
                'enabled': True,
                'save_best': True,
                'save_last': True,
                'monitor': 'val_loss'
            },
            'early_stopping': {
                'enabled': False,  # Disabled for quick demo
                'patience': 10,
                'min_delta': 0.0001
            },
            'lr_scheduler': {
                'enabled': False  # Disabled for quick demo
            },
            'tensorboard': {
                'enabled': True,
                'log_dir': 'runs/quick_demo'
            }
        },
        'output': {
            'checkpoint_dir': 'checkpoints/quick_demo',
            'log_dir': 'logs/quick_demo'
        }
    }


def setup_datasets(config: dict):
    """Setup training and validation datasets."""
    data_config = config['data']
    train_config = config['training']

    # Paths
    train_path = Path(data_config['train_path'])
    test_path = Path(data_config.get('test_path', train_path))

    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    # Create datasets
    logger.info(f"Loading training dataset from: {train_path}")
    train_dataset = SignalDataset(
        train_path,
        normalize=data_config.get('normalize', False)
    )

    # Use portion of training data as validation if no test set specified
    if not test_path.exists() or test_path == train_path:
        logger.warning("Test dataset not found, using training data for validation")
        val_dataset = train_dataset
    else:
        logger.info(f"Loading validation dataset from: {test_path}")
        val_dataset = SignalDataset(
            test_path,
            normalize=data_config.get('normalize', False)
        )

    # Create dataloaders
    batch_size = train_config['batch_size']

    train_loader = DataLoaderFactory.create_train_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoaderFactory.create_eval_loader(
        val_dataset,
        batch_size=batch_size
    )

    logger.info(
        f"Datasets created: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, batch_size={batch_size}"
    )

    return train_loader, val_loader


def setup_model(config: dict, device: str):
    """Setup model, optimizer, and criterion."""
    # Create model
    logger.info("Creating model...")
    model = ModelFactory.create_model(config, device=device)

    # Print model info
    info = ModelFactory.get_model_info(model)
    logger.info(f"Model parameters: {info['total_parameters']:,}")

    # Create optimizer
    train_config = config['training']
    optimizer = create_optimizer(
        model,
        optimizer_type=train_config.get('optimizer', 'adam'),
        learning_rate=train_config['learning_rate']
    )

    # Create criterion
    criterion = create_criterion(
        criterion_type=train_config.get('criterion', 'mse')
    )

    return model, optimizer, criterion


def setup_callbacks(config: dict):
    """Setup training callbacks."""
    callbacks = []
    callbacks_config = config.get('callbacks', {})
    output_config = config.get('output', {})

    # Progress callback (always enabled)
    callbacks.append(ProgressCallback(print_every_n_epochs=1))

    # Checkpoint callback
    checkpoint_config = callbacks_config.get('checkpoint', {})
    if checkpoint_config.get('enabled', True):
        checkpoint_dir = Path(output_config.get('checkpoint_dir', 'checkpoints'))
        callbacks.append(
            CheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                save_best=checkpoint_config.get('save_best', True),
                save_last=checkpoint_config.get('save_last', True),
                monitor=checkpoint_config.get('monitor', 'val_loss')
            )
        )
        logger.info(f"Checkpoint callback enabled: {checkpoint_dir}")

    # Early stopping callback
    es_config = callbacks_config.get('early_stopping', {})
    if es_config.get('enabled', False):
        callbacks.append(
            EarlyStoppingCallback(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0.0),
                monitor=es_config.get('monitor', 'val_loss')
            )
        )
        logger.info("Early stopping callback enabled")

    # Learning rate scheduler callback
    lr_config = callbacks_config.get('lr_scheduler', {})
    if lr_config.get('enabled', False):
        callbacks.append(
            LearningRateSchedulerCallback(
                scheduler='plateau',
                monitor='val_loss',
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 5)
            )
        )
        logger.info("Learning rate scheduler callback enabled")

    # TensorBoard callback
    tb_config = callbacks_config.get('tensorboard', {})
    if tb_config.get('enabled', False):
        log_dir = Path(tb_config.get('log_dir', 'runs/experiment'))
        callbacks.append(
            TensorBoardCallback(log_dir=log_dir)
        )
        logger.info(f"TensorBoard callback enabled: {log_dir}")

    return callbacks


def train(config: dict):
    """Main training function."""
    # Print configuration
    print_training_config(config)

    # Validate configuration
    validate_training_config(config)

    # Set seed for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device(prefer_cuda=True)

    # Setup datasets
    train_loader, val_loader = setup_datasets(config)

    # Setup model
    model, optimizer, criterion = setup_model(config, device)

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        callbacks=callbacks,
        grad_clip_value=config['training'].get('grad_clip', None)
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    summary = trainer.train(num_epochs=config['training']['num_epochs'])

    # Print summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    logger.info("\nBest metrics:")
    best_metrics = trainer.get_best_metrics()
    for name, value in best_metrics.items():
        logger.info(f"  {name}: {value:.6f}")

    return trainer, summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train LSTM Signal Extraction Model')
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick demo with 5 epochs'
    )
    args = parser.parse_args()

    try:
        # Load or create configuration
        if args.quick:
            logger.info("Running quick demo...")
            config = create_quick_config()
        elif args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("No config specified, using quick demo")
            config = create_quick_config()

        # Train model
        trainer, summary = train(config)

        logger.info("\n✅ Training completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
