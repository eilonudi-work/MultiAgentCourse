#!/usr/bin/env python3
"""
Quick demo of the LSTM Signal Extraction System
Run this to see your Phase 1 implementation in action!
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_io import DatasetIO
from src.data.visualizers import DatasetVisualizer
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("LSTM Signal Extraction - Quick Demo")
    print("=" * 60)
    print()

    # 1. Check datasets
    print("📊 1. Checking Datasets...")
    print("-" * 60)

    train_path = Path('data/processed/train_dataset.h5')
    test_path = Path('data/processed/test_dataset.h5')

    if train_path.exists():
        print(f"✓ Training dataset found: {train_path}")
        train_info = DatasetIO.get_dataset_info(train_path)
        print(f"  - Samples: {train_info.get('num_samples', 'N/A')}")
        print(f"  - Frequencies: {train_info.get('frequencies', 'N/A')}")
    else:
        print(f"✗ Training dataset not found at {train_path}")

    if test_path.exists():
        print(f"✓ Test dataset found: {test_path}")
        test_info = DatasetIO.get_dataset_info(test_path)
        print(f"  - Samples: {test_info.get('num_samples', 'N/A')}")
    else:
        print(f"✗ Test dataset not found at {test_path}")

    print()

    # 2. Load and explore a sample
    print("🔍 2. Loading Sample Data...")
    print("-" * 60)

    if train_path.exists():
        # Load dataset
        data = DatasetIO.load_hdf5(train_path)

        # Get first sample
        sample = data['samples'][0]

        print(f"Sample structure:")
        print(f"  - Mixed signal shape: {sample['mixed_signal'].shape}")
        print(f"  - Target signal shape: {sample['target_signal'].shape}")
        print(f"  - Condition vector: {sample['condition_vector']}")
        print(f"  - Target frequency: {sample['metadata']['frequency']} Hz")
        print(f"  - Amplitude: {sample['metadata']['amplitude']:.3f}")
        print(f"  - Phase: {sample['metadata']['phase']:.3f} rad")
        print()

        # 3. Basic statistics
        print("📈 3. Dataset Statistics...")
        print("-" * 60)

        import numpy as np

        mixed_signal = sample['mixed_signal']
        target_signal = sample['target_signal']

        print(f"Mixed signal statistics:")
        print(f"  - Mean: {np.mean(mixed_signal):.6f}")
        print(f"  - Std: {np.std(mixed_signal):.6f}")
        print(f"  - Min: {np.min(mixed_signal):.6f}")
        print(f"  - Max: {np.max(mixed_signal):.6f}")
        print()

        print(f"Target signal statistics:")
        print(f"  - Mean: {np.mean(target_signal):.6f}")
        print(f"  - Std: {np.std(target_signal):.6f}")
        print(f"  - Min: {np.min(target_signal):.6f}")
        print(f"  - Max: {np.max(target_signal):.6f}")
        print()

        # 4. Visualize a sample
        print("📊 4. Creating Visualization...")
        print("-" * 60)

        try:
            visualizer = DatasetVisualizer()
            output_path = Path('outputs/figures/demo_sample.png')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            visualizer.plot_sample_signals(sample, output_path)
            print(f"✓ Visualization saved to: {output_path}")
            print()
        except Exception as e:
            print(f"✗ Could not create visualization: {e}")
            print()

    # 5. Summary
    print("=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. View the visualization: open outputs/figures/demo_sample.png")
    print("  2. Run full tests: pytest tests/ -v")
    print("  3. Read GETTING_STARTED.md for more options")
    print("  4. Start Phase 2: LSTM model implementation")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
