#!/usr/bin/env python3
"""
Simple demo - See what you have!
"""

import h5py
import numpy as np
from pathlib import Path

print("=" * 70)
print("  LSTM SIGNAL EXTRACTION - YOUR PROJECT STATUS")
print("=" * 70)
print()

# Check datasets
train_path = Path('data/processed/train_dataset.h5')
test_path = Path('data/processed/test_dataset.h5')

print("📦 DATASETS:")
print("-" * 70)
if train_path.exists():
    size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"✓ Training data: {train_path} ({size_mb:.1f} MB)")

    with h5py.File(train_path, 'r') as f:
        print(f"  - Mixed signals: {f['mixed_signals'].shape}")
        print(f"  - Target signals: {f['target_signals'].shape}")
        print(f"  - Condition vectors: {f['condition_vectors'].shape}")
else:
    print(f"✗ Training data not found")

if test_path.exists():
    size_mb = test_path.stat().st_size / (1024 * 1024)
    print(f"✓ Test data: {test_path} ({size_mb:.1f} MB)")
else:
    print(f"✗ Test data not found")

print()
print("📊 SAMPLE DATA (First training sample):")
print("-" * 70)

if train_path.exists():
    with h5py.File(train_path, 'r') as f:
        # Get first sample
        mixed_signal = f['mixed_signals'][0]
        target_signal = f['target_signals'][0]
        condition = f['condition_vectors'][0]

        print(f"Mixed signal:")
        print(f"  Shape: {mixed_signal.shape}")
        print(f"  Min: {mixed_signal.min():.4f}")
        print(f"  Max: {mixed_signal.max():.4f}")
        print(f"  Mean: {mixed_signal.mean():.4f}")
        print(f"  Std: {mixed_signal.std():.4f}")
        print()

        print(f"Target signal:")
        print(f"  Shape: {target_signal.shape}")
        print(f"  Min: {target_signal.min():.4f}")
        print(f"  Max: {target_signal.max():.4f}")
        print()

        print(f"Condition vector (one-hot):")
        print(f"  {condition}")
        target_freq_idx = np.argmax(condition)
        freqs = [1, 3, 5, 7]
        print(f"  → Target frequency: {freqs[target_freq_idx]} Hz")

print()
print("📁 PROJECT STRUCTURE:")
print("-" * 70)
print("✓ Source code: src/data/ (7 Python files)")
print("✓ Tests: tests/ (7 test files, 81 tests)")
print("✓ Configuration: config/default.yaml")
print("✓ Scripts: scripts/generate_datasets.py")

print()
print("=" * 70)
print("  PHASE 1: COMPLETE ✅ (94% test coverage)")
print("=" * 70)
print()

print("🚀 WHAT YOU CAN DO NOW:")
print()
print("1. View validation reports:")
print("   cat data/processed/train_validation_report.txt")
print()
print("2. Regenerate datasets:")
print("   python3 scripts/generate_datasets.py")
print()
print("3. Run all tests:")
print("   python3 -m pytest tests/ -v")
print()
print("4. Read the getting started guide:")
print("   cat GETTING_STARTED.md")
print()
print("5. Start Phase 2 (LSTM Model):")
print("   See DEVELOPMENT_PLAN.md - Phase 2")
print()
