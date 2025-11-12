"""
Utility functions for training.

This module provides helper functions for:
    - Creating optimizers
    - Setting up training configuration
    - Computing class weights
    - Reproducibility (seed setting)
"""

import logging
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    **kwargs
) -> optim.Optimizer:
    """
    Create PyTorch optimizer.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw', 'rmsprop')
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        PyTorch optimizer

    Example:
        >>> optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")

    return optimizer


def create_criterion(
    criterion_type: str = 'mse',
    **kwargs
) -> nn.Module:
    """
    Create loss function.

    Args:
        criterion_type: Type of criterion ('mse', 'mae', 'huber', 'smooth_l1')
        **kwargs: Additional criterion arguments

    Returns:
        PyTorch loss function

    Example:
        >>> criterion = create_criterion('mse')
    """
    criterion_type = criterion_type.lower()

    if criterion_type == 'mse':
        criterion = nn.MSELoss(**kwargs)
    elif criterion_type == 'mae':
        criterion = nn.L1Loss(**kwargs)
    elif criterion_type == 'huber':
        criterion = nn.HuberLoss(**kwargs)
    elif criterion_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss(**kwargs)
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")

    logger.info(f"Created {criterion_type} criterion")

    return criterion


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts

    Example:
        >>> counts = count_parameters(model)
        >>> print(f"Total: {counts['total']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cpu' or 'cuda')

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if prefer_cuda and torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU")

    return device


def print_training_config(config: Dict):
    """
    Print training configuration in readable format.

    Args:
        config: Configuration dictionary

    Example:
        >>> print_training_config(config)
    """
    print("\n" + "=" * 60)
    print("  Training Configuration")
    print("=" * 60)

    def print_dict(d: Dict, indent: int = 0):
        """Recursively print dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(config)
    print("=" * 60 + "\n")


def validate_training_config(config: Dict) -> bool:
    """
    Validate training configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise

    Example:
        >>> validate_training_config(config)
    """
    required_keys = ['model', 'training', 'data']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate training config
    training_config = config['training']
    required_training_keys = ['num_epochs', 'batch_size', 'learning_rate']

    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training key: {key}")

    # Validate positive values
    if training_config['num_epochs'] <= 0:
        raise ValueError("num_epochs must be positive")
    if training_config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    if training_config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")

    logger.info("Training configuration validated successfully")

    return True
