"""
Generate training and test datasets for LSTM Signal Extraction.

Usage:
    python scripts/generate_datasets.py --config config/default.yaml
    python scripts/generate_datasets.py --train-only
    python scripts/generate_datasets.py --test-only
    python scripts/generate_datasets.py --validate-only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.data.dataset_builder import SignalDatasetBuilder
from src.data.validators import DatasetValidator
from src.data.visualizers import DatasetVisualizer


def setup_logging(log_dir: Path, verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        verbose: Whether to enable verbose logging
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_dir / 'dataset_generation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def generate_and_validate_dataset(
    builder: SignalDatasetBuilder,
    validator: DatasetValidator,
    visualizer: DatasetVisualizer,
    split: str,
    output_dir: Path,
    figures_dir: Path,
    skip_validation: bool = False
):
    """
    Generate, validate, and save a dataset.

    Args:
        builder: Dataset builder instance
        validator: Dataset validator instance
        visualizer: Dataset visualizer instance
        split: 'train' or 'test'
        output_dir: Directory to save dataset
        figures_dir: Directory to save figures
        skip_validation: Skip validation step

    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Generate dataset
        logger.info(f"=" * 70)
        logger.info(f"Generating {split} dataset...")
        logger.info(f"=" * 70)

        dataset = builder.generate_dataset(split=split, show_progress=True)

        # Save dataset
        output_file = output_dir / f"{split}_dataset.h5"
        logger.info(f"Saving {split} dataset to {output_file}...")
        builder.save_dataset(dataset, output_file)

        if not skip_validation:
            # Validate dataset
            logger.info(f"Validating {split} dataset...")
            report = validator.generate_validation_report(dataset)

            # Print report
            print("\n" + report + "\n")

            # Save report
            report_file = output_dir / f"{split}_validation_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Saved validation report to {report_file}")

            # Generate visualizations
            logger.info(f"Generating visualizations for {split} dataset...")

            # Summary figure
            summary_fig = figures_dir / f"{split}_dataset_summary.png"
            visualizer.create_dataset_summary_figure(
                dataset,
                save_path=summary_fig,
                show=False
            )

            # Parameter distributions
            params_fig = figures_dir / f"{split}_parameter_distributions.png"
            visualizer.plot_parameter_distributions(
                dataset,
                save_path=params_fig,
                show=False
            )

            # Multiple samples
            samples_fig = figures_dir / f"{split}_sample_signals.png"
            visualizer.plot_multiple_samples(
                dataset,
                n_samples=4,
                save_path=samples_fig,
                show=False
            )

            logger.info(f"Visualizations saved to {figures_dir}")

        logger.info(f"Successfully completed {split} dataset generation")
        return True

    except Exception as e:
        logger.error(f"Error generating {split} dataset: {e}", exc_info=True)
        return False


def validate_existing_dataset(
    config: dict,
    split: str,
    dataset_path: Path,
    figures_dir: Path
):
    """
    Validate an existing dataset file.

    Args:
        config: Configuration dictionary
        split: 'train' or 'test'
        dataset_path: Path to dataset file
        figures_dir: Directory to save figures

    Returns:
        True if validation passed, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading {split} dataset from {dataset_path}...")

        builder = SignalDatasetBuilder(config)
        dataset = builder.load_dataset(dataset_path)

        validator = DatasetValidator(config)
        visualizer = DatasetVisualizer(config)

        logger.info(f"Validating {split} dataset...")
        report = validator.generate_validation_report(dataset)

        # Print report
        print("\n" + report + "\n")

        # Save report
        report_file = dataset_path.parent / f"{split}_validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved validation report to {report_file}")

        # Generate visualizations
        logger.info(f"Generating visualizations...")

        summary_fig = figures_dir / f"{split}_dataset_summary.png"
        visualizer.create_dataset_summary_figure(
            dataset,
            save_path=summary_fig,
            show=False
        )

        params_fig = figures_dir / f"{split}_parameter_distributions.png"
        visualizer.plot_parameter_distributions(
            dataset,
            save_path=params_fig,
            show=False
        )

        logger.info(f"Validation complete for {split} dataset")
        return True

    except Exception as e:
        logger.error(f"Error validating {split} dataset: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate training and test datasets for LSTM Signal Extraction'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )

    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Generate only training dataset'
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Generate only test dataset'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing datasets (no generation)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step during generation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = ConfigLoader.load(args.config)
        ConfigLoader.validate_config(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Setup logging
    log_dir = Path(config['paths']['logs_dir'])
    setup_logging(log_dir, verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting dataset generation pipeline")
    logger.info(f"Configuration: {args.config}")

    # Create output directories
    output_dir = Path(config['paths']['processed_data_dir'])
    figures_dir = Path(config['paths']['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Validate-only mode
    if args.validate_only:
        logger.info("Running in validate-only mode")

        success = True

        if not args.test_only:
            train_path = output_dir / "train_dataset.h5"
            if train_path.exists():
                success &= validate_existing_dataset(
                    config, 'train', train_path, figures_dir
                )
            else:
                logger.warning(f"Train dataset not found at {train_path}")
                success = False

        if not args.train_only:
            test_path = output_dir / "test_dataset.h5"
            if test_path.exists():
                success &= validate_existing_dataset(
                    config, 'test', test_path, figures_dir
                )
            else:
                logger.warning(f"Test dataset not found at {test_path}")
                success = False

        return 0 if success else 1

    # Initialize components
    builder = SignalDatasetBuilder(config)
    validator = DatasetValidator(config)
    visualizer = DatasetVisualizer(config)

    success = True

    # Generate training dataset
    if not args.test_only:
        success &= generate_and_validate_dataset(
            builder, validator, visualizer,
            'train', output_dir, figures_dir,
            skip_validation=args.skip_validation
        )

    # Generate test dataset
    if not args.train_only:
        success &= generate_and_validate_dataset(
            builder, validator, visualizer,
            'test', output_dir, figures_dir,
            skip_validation=args.skip_validation
        )

    if success:
        logger.info("=" * 70)
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Datasets saved to: {output_dir}")
        logger.info(f"Figures saved to: {figures_dir}")
        logger.info(f"Logs saved to: {log_dir}")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("Dataset generation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
