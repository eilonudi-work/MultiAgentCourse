"""
Model factory for creating and loading LSTM models.

Provides utilities for model instantiation, checkpoint loading, and
model inspection.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from .lstm_model import SignalExtractionLSTM

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating and managing LSTM models.

    Provides static methods for:
        - Creating models from configuration
        - Loading models from checkpoints
        - Getting model information
        - Counting parameters

    Example:
        >>> config = {'model': {'lstm': {'hidden_size': 64, 'num_layers': 2}}}
        >>> model = ModelFactory.create_model(config)
        >>> print(model)
    """

    @staticmethod
    def create_model(config: Dict, device: str = 'cpu') -> SignalExtractionLSTM:
        """
        Create model from configuration.

        Args:
            config: Configuration dictionary with structure:
                   {
                       'model': {
                           'lstm': {
                               'input_size': int,
                               'hidden_size': int,
                               'num_layers': int,
                               'dropout': float
                           }
                       }
                   }
            device: Device to create model on ('cpu' or 'cuda')

        Returns:
            SignalExtractionLSTM model instance

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If configuration values are invalid

        Example:
            >>> config = {
            ...     'model': {
            ...         'lstm': {
            ...             'input_size': 5,
            ...             'hidden_size': 64,
            ...             'num_layers': 2,
            ...             'dropout': 0.1
            ...         }
            ...     }
            ... }
            >>> model = ModelFactory.create_model(config)
        """
        try:
            lstm_config = config['model']['lstm']
        except KeyError as e:
            raise KeyError(
                f"Configuration missing required key: {e}. "
                f"Expected structure: config['model']['lstm']"
            )

        # Extract parameters with defaults
        input_size = lstm_config.get('input_size', 5)
        hidden_size = lstm_config.get('hidden_size', 64)
        num_layers = lstm_config.get('num_layers', 2)
        dropout = lstm_config.get('dropout', 0.1)

        # Create model
        model = SignalExtractionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

        logger.info(
            f"Created model from config: "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"parameters={model.count_parameters():,}"
        )

        return model

    @staticmethod
    def create_from_checkpoint(
        checkpoint_path: Path,
        device: str = 'cpu',
        strict: bool = True
    ) -> SignalExtractionLSTM:
        """
        Load model from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pt or .pth)
            device: Device to load model on ('cpu' or 'cuda')
            strict: Whether to strictly enforce that keys in checkpoint match
                   model architecture (default: True)

        Returns:
            SignalExtractionLSTM model loaded from checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint cannot be loaded

        Example:
            >>> model = ModelFactory.create_from_checkpoint(
            ...     Path('checkpoints/best_model.pt'),
            ...     device='cuda'
            ... )
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract model configuration from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                model = ModelFactory.create_model(config, device=device)
            else:
                # Try to infer configuration from state dict
                logger.warning(
                    "No config found in checkpoint, attempting to infer from state dict"
                )
                model = ModelFactory._infer_model_from_state_dict(
                    checkpoint['model_state_dict'],
                    device=device
                )

            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

            # Log checkpoint information
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('val_loss', checkpoint.get('train_loss', 'unknown'))

            logger.info(
                f"Loaded model from checkpoint: {checkpoint_path.name}, "
                f"epoch={epoch}, loss={loss}"
            )

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    @staticmethod
    def _infer_model_from_state_dict(
        state_dict: Dict,
        device: str = 'cpu'
    ) -> SignalExtractionLSTM:
        """
        Infer model architecture from state dictionary.

        Args:
            state_dict: Model state dictionary
            device: Device to create model on

        Returns:
            SignalExtractionLSTM model instance

        Note:
            This is a fallback method when config is not available.
            It may not work for all model configurations.
        """
        # Infer input_size from LSTM weights
        lstm_weight = state_dict['lstm.weight_ih_l0']
        input_size = lstm_weight.size(1)

        # Infer hidden_size
        hidden_size = lstm_weight.size(0) // 4  # LSTM has 4 gates

        # Infer num_layers
        num_layers = sum(1 for key in state_dict if 'lstm.weight_ih_l' in key)

        # Dropout cannot be inferred, use default
        dropout = 0.1

        logger.warning(
            f"Inferred model config: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}"
        )

        return SignalExtractionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

    @staticmethod
    def get_model_info(model: SignalExtractionLSTM) -> Dict:
        """
        Get comprehensive model information.

        Args:
            model: SignalExtractionLSTM instance

        Returns:
            Dictionary with model architecture and statistics

        Example:
            >>> model = ModelFactory.create_model(config)
            >>> info = ModelFactory.get_model_info(model)
            >>> print(f"Parameters: {info['total_parameters']:,}")
        """
        info = model.get_model_info()

        # Add additional information
        info['trainable_parameters'] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        info['non_trainable_parameters'] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        # Memory estimation (rough)
        param_size_mb = info['total_parameters'] * 4 / (1024 ** 2)  # 4 bytes per float32
        info['estimated_size_mb'] = round(param_size_mb, 2)

        return info

    @staticmethod
    def count_parameters(model: SignalExtractionLSTM) -> Dict[str, int]:
        """
        Count parameters by layer.

        Args:
            model: SignalExtractionLSTM instance

        Returns:
            Dictionary with parameter counts by component

        Example:
            >>> params = ModelFactory.count_parameters(model)
            >>> print(params)
            {'lstm': 123456, 'fc': 65, 'total': 123521}
        """
        counts = {
            'lstm': 0,
            'fc': 0,
            'total': 0
        }

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                counts['total'] += param_count

                if 'lstm' in name:
                    counts['lstm'] += param_count
                elif 'fc' in name:
                    counts['fc'] += param_count

        return counts

    @staticmethod
    def print_model_summary(model: SignalExtractionLSTM):
        """
        Print detailed model summary.

        Args:
            model: SignalExtractionLSTM instance

        Example:
            >>> ModelFactory.print_model_summary(model)
            Model: SignalExtractionLSTM
            ================================
            ...
        """
        info = ModelFactory.get_model_info(model)
        param_counts = ModelFactory.count_parameters(model)

        print("=" * 60)
        print(f"Model: {info['model_type']}")
        print("=" * 60)
        print(f"Input Size:      {info['input_size']}")
        print(f"Hidden Size:     {info['hidden_size']}")
        print(f"Num Layers:      {info['num_layers']}")
        print(f"Dropout:         {info['dropout']}")
        print(f"Device:          {info['device']}")
        print("-" * 60)
        print(f"LSTM Parameters: {param_counts['lstm']:,}")
        print(f"FC Parameters:   {param_counts['fc']:,}")
        print(f"Total Parameters: {param_counts['total']:,}")
        print(f"Estimated Size:  {info['estimated_size_mb']:.2f} MB")
        print("=" * 60)
        print()
        print("Architecture:")
        print(model)
        print("=" * 60)

    @staticmethod
    def save_checkpoint(
        model: SignalExtractionLSTM,
        save_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Save model checkpoint.

        Args:
            model: SignalExtractionLSTM instance
            save_path: Path to save checkpoint
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            loss: Optional loss value
            config: Optional configuration dictionary
            **kwargs: Additional data to save in checkpoint

        Example:
            >>> ModelFactory.save_checkpoint(
            ...     model, Path('checkpoints/model.pt'),
            ...     optimizer=optimizer, epoch=10, loss=0.005
            ... )
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info(),
            'config': config,
            'epoch': epoch,
            'loss': loss
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add any additional kwargs
        checkpoint.update(kwargs)

        torch.save(checkpoint, save_path)

        logger.info(f"Saved checkpoint to {save_path}")
