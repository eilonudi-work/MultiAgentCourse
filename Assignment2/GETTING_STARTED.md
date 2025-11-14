# Getting Started with LSTM Signal Extraction

## Quick Start (5 minutes)

### Method 1: Automated Setup (Easiest)

```bash
# Run the setup script
./setup_project.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Set up project structure
- Run verification tests

### Method 2: Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install project
pip install -e .

# 4. Verify installation
python -c "import src; print('Success!')"
```

---

## Running the Project

### 1. Generate Datasets (Phase 1 - Already Complete!)

**Check existing datasets:**
```bash
ls -lh data/processed/
```

You should see:
- `train_dataset.h5` (~2.2 MB, 40,000 samples)
- `test_dataset.h5` (~1.1 MB, 40,000 samples)

**Regenerate datasets (if needed):**
```bash
# Generate both train and test
python scripts/generate_datasets.py

# Generate only training set
python scripts/generate_datasets.py --train-only

# Generate only test set
python scripts/generate_datasets.py --test-only

# Use custom configuration
python scripts/generate_datasets.py --config config/custom.yaml
```

### 2. Validate Datasets

```bash
# Validate existing datasets
python scripts/generate_datasets.py --validate-only

# Check validation reports
cat data/processed/train_validation_report.txt
cat data/processed/test_validation_report.txt
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_signal_generation.py -v

# Run integration tests only
pytest tests/integration/ -v

# View coverage report
open htmlcov/index.html
```

### 4. Explore the Data

```python
# Interactive Python session
python

# Load and explore dataset
from src.data.dataset_io import DatasetIO
from pathlib import Path

# Load training data
train_data = DatasetIO.load_hdf5(Path('data/processed/train_dataset.h5'))

# Check structure
print(f"Number of samples: {len(train_data['samples'])}")
print(f"Frequencies: {train_data['metadata']['frequencies']}")

# Get a sample
sample = train_data['samples'][0]
print(f"Mixed signal shape: {sample['mixed_signal'].shape}")
print(f"Target signal shape: {sample['target_signal'].shape}")
print(f"Condition vector: {sample['condition_vector']}")
print(f"Metadata: {sample['metadata']}")
```

---

## Project Structure

```
Assignment2/
├── config/                   # Configuration files
│   ├── default.yaml         # Main configuration
│   └── test_small.yaml      # Small dataset for testing
├── src/                     # Source code
│   ├── config/              # Config loader
│   └── data/                # Data generation modules
│       ├── signal_generator.py
│       ├── parameter_sampler.py
│       ├── dataset_builder.py
│       ├── dataset_io.py
│       ├── validators.py
│       └── visualizers.py
├── scripts/                 # Executable scripts
│   └── generate_datasets.py
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── data/                    # Data directory
│   ├── raw/                 # Raw data (if any)
│   └── processed/           # Generated datasets
├── outputs/                 # Output files
│   ├── figures/             # Visualizations
│   ├── logs/                # Log files
│   └── validation/          # Validation reports
└── checkpoints/             # Model checkpoints (Phase 2+)
```

---

## Common Commands

### Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Install new package
pip install package-name
pip freeze > requirements.txt  # Update requirements

# Format code
black src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/

# Type check
mypy src/
```

### Testing Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_signal_generation.py::test_sinusoid_frequency

# Run tests in parallel (faster)
pytest -n auto

# Run only failed tests from last run
pytest --lf

# Stop on first failure
pytest -x
```

### Dataset Commands

```bash
# Generate datasets
python scripts/generate_datasets.py

# Check dataset info
python -c "
from src.data.dataset_io import DatasetIO
from pathlib import Path
info = DatasetIO.get_dataset_info(Path('data/processed/train_dataset.h5'))
print(info)
"

# Validate datasets
python scripts/generate_datasets.py --validate-only
```

---

## Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Project installed in development mode
- [ ] All tests passing (81 tests)
- [ ] Datasets exist in `data/processed/`
- [ ] Can import project modules: `import src`
- [ ] No errors when running scripts

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:** Install project in development mode:
```bash
pip install -e .
```

### Issue: `FileNotFoundError` when running scripts
**Solution:** Run scripts from project root:
```bash
cd "/Users/eilonudi/Desktop/HW/LLMs in multiagent env/MultiAgentCourse/Assignment2"
python scripts/generate_datasets.py
```

### Issue: Import errors for dependencies
**Solution:** Install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: Tests failing
**Solution:** Check your Python version and reinstall:
```bash
python3 --version  # Should be 3.8+
pip install --upgrade -r requirements.txt
```

### Issue: Permission denied for setup script
**Solution:** Make it executable:
```bash
chmod +x setup_project.sh
```

---

## What's Next?

Now that Phase 1 (Dataset Generation) is complete, you can:

### Immediate Actions:
1. **Explore the code** - Review implementation in `src/data/`
2. **Check test coverage** - Open `htmlcov/index.html`
3. **Read validation reports** - See `data/processed/*_validation_report.txt`
4. **Review documentation** - Read `DEVELOPMENT_PLAN.md`

### Next Development Phase (Phase 2):
Start implementing the LSTM model architecture:
- PyTorch LSTM model with stateful processing
- Dataset and DataLoader classes
- Model testing and validation

See `DEVELOPMENT_PLAN.md` Phase 2 for details.

---

## Additional Resources

- **PRD:** `LSTM_Signal_Extraction_PRD.md` - Full product requirements
- **Dev Plan:** `DEVELOPMENT_PLAN.md` - 6-week development roadmap
- **README:** `README.md` - Project overview
- **Phase 1 Summary:** `PHASE1_SUMMARY.md` - Implementation details

---

## Quick Reference

### Python Environment
```bash
# Create: python3 -m venv venv
# Activate: source venv/bin/activate
# Deactivate: deactivate
```

### Essential Commands
```bash
# Generate data: python scripts/generate_datasets.py
# Run tests: pytest tests/ -v
# Check coverage: pytest --cov=src --cov-report=html
# Format code: black .
# Lint code: flake8 .
```

### Getting Help
```bash
# Script help
python scripts/generate_datasets.py --help

# Pytest help
pytest --help

# Python help
python -c "from src.data import SignalGenerator; help(SignalGenerator)"
```
