#!/bin/bash
# LSTM Signal Extraction - Project Setup Script

set -e  # Exit on error

echo "🚀 Setting up LSTM Signal Extraction Project..."
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "2️⃣  Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "3️⃣  Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "4️⃣  Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "5️⃣  Installing dependencies..."
pip install -r requirements.txt
echo "   ✓ Dependencies installed"
echo ""

# Install project in development mode
echo "6️⃣  Installing project in development mode..."
pip install -e .
echo "   ✓ Project installed"
echo ""

# Create necessary directories
echo "7️⃣  Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs/figures
mkdir -p outputs/logs
mkdir -p outputs/validation
mkdir -p checkpoints
echo "   ✓ Directories created"
echo ""

# Verify installation
echo "8️⃣  Verifying installation..."
python -c "
import sys
import numpy
import scipy
import h5py
import yaml
import matplotlib
import tqdm
import pytest
print('✓ All core packages imported successfully')
print(f'✓ Python: {sys.version}')
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ SciPy: {scipy.__version__}')
print(f'✓ H5PY: {h5py.__version__}')
"
echo ""

# Run tests
echo "9️⃣  Running tests to verify setup..."
pytest tests/ -v --tb=short -x
echo ""

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Generate datasets: python scripts/generate_datasets.py"
echo "   3. Run tests: pytest tests/ -v"
echo ""
echo "📚 Documentation:"
echo "   - README.md - Project overview and usage"
echo "   - DEVELOPMENT_PLAN.md - Development roadmap"
echo "   - LSTM_Signal_Extraction_PRD.md - Full requirements"
echo ""
