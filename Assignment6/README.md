# Prompt Engineering Assignment - POC

A proof-of-concept implementation comparing different prompt engineering techniques using sentiment analysis with **local Ollama models**.

## 🚀 Quick Start (2 Commands)

```bash
./setup.sh    # Automatic installation (Ollama + dependencies)
./run.sh      # Run experiments
```

**That's it!** See [QUICKSTART.md](QUICKSTART.md) for more details.

---

## Overview

This project systematically evaluates how different prompting strategies affect LLM performance on sentiment classification tasks using local models via Ollama. We test:
- Baseline prompts
- Improved prompts with role definition
- Few-shot learning
- Chain of thought reasoning

**Benefits of using Ollama:**
- ✨ No API costs - completely free
- 🔒 Privacy - all processing is local
- ⚡ Fast iteration - no rate limits
- 🔄 Unlimited experimentation

## Project Structure

```
.
├── data/                   # Dataset storage
│   └── sentiment_dataset.json
├── src/                    # Source code
├── results/                # Experiment results
├── analysis/               # Analysis scripts
├── visualizations/         # Generated graphs
├── requirements.txt        # Python dependencies
└── .env.example           # API configuration template
```

## Quick Start (Automated Setup)

### One-Command Setup

```bash
./setup.sh
```

This script will automatically:
- ✓ Install Ollama (if not present)
- ✓ Start Ollama service
- ✓ Download the llama2 model
- ✓ Create Python virtual environment
- ✓ Install all dependencies
- ✓ Configure environment variables
- ✓ Run setup verification tests

**Use a different model:**
```bash
./setup.sh mistral  # or llama3, phi, etc.
```

### Run Experiments

```bash
./run.sh
```

**With options:**
```bash
./run.sh --model mistral      # Use different model
./run.sh --show-errors        # Show misclassified examples
./run.sh --help              # Show all options
```

---

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

### 2. Pull a Model

```bash
# Pull llama2 (recommended for sentiment analysis)
ollama pull llama2

# Alternative models:
# ollama pull mistral
# ollama pull llama3
# ollama pull phi
```

### 3. Verify Ollama is Running

```bash
ollama list
# Should show your downloaded models

# Test the model
ollama run llama2 "Hello"
```

### 4. Create Virtual Environment

```bash
cd Assignment6
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env if you want to use a different model or host
```

### 7. Test Setup

```bash
python test_setup.py
# Should pass all verification tests
```

## Dataset

- **Size**: 30 examples (POC version)
- **Balance**: 15 positive, 15 negative
- **Categories**: entertainment, product, food, service, technology, hospitality, general
- **Format**: JSON with text, ground_truth, and category fields

## Running Experiments

### Automated Way (Recommended)

```bash
./run.sh
```

### Manual Way

```bash
cd src
python baseline_experiment.py
```

**Command-line options:**
```bash
# Use a different model
python baseline_experiment.py --model mistral

# Don't save results
python baseline_experiment.py --no-save

# Show error cases
python baseline_experiment.py --show-errors

# Custom dataset
python baseline_experiment.py --dataset path/to/dataset.json
```

### Test Setup Only

```bash
python test_setup.py
```

This verifies:
- Ollama connection
- Model availability
- Sentiment classification functionality
- Dataset integrity

### Expected Output

The baseline experiment will:
1. Load the sentiment dataset (30 examples)
2. Classify each example using the baseline prompt
3. Calculate accuracy and distance metrics
4. Generate confusion matrix and per-category stats
5. Save results to `results/` directory

Sample output:
```
=== Running Baseline Experiment ===
Model: llama2
Prompt: Classify the sentiment of this text as 'positive' or 'negative': {text}

✓ Loaded 30 examples from data/sentiment_dataset.json
Processing examples: 100%|████████| 30/30

=== Experiment Summary ===
Total samples: 30
Successful: 30
Failed: 0
Success rate: 100.0%

Accuracy: 86.7%
Mean distance: 0.1234
Std distance: 0.0567

Confusion Matrix:
  Precision: 85.7%
  Recall:    88.2%
  F1 Score:  86.9%
```

## Run All Prompt Variations (Phase 3) ✓

Compare different prompting strategies:

```bash
./run_all.sh
```

This runs **6 prompt variations**:
- **Baseline**: Simple direct instruction
- **Role-Based**: Expert role definition
- **Few-Shot**: Learning from 3 examples
- **Chain of Thought**: Step-by-step reasoning
- **Structured Output**: Formatted response requests
- **Contrastive**: Comparing positive vs negative aspects

**Options:**
```bash
# Run specific variations only
./run_all.sh --variations baseline few_shot chain_of_thought

# Use different model
./run_all.sh --model mistral

# Don't save results
./run_all.sh --no-save
```

**Manual execution:**
```bash
cd src
python run_all_experiments.py
python improved_prompts.py  # View all prompt variations
```

### Comparison Output

After running all variations, you'll get a comparison table:

```
Variation            Accuracy   F1 Score  Precision     Recall
--------------------------------------------------------------------------------
few_shot               93.3%      92.5%      91.2%      93.8%
chain_of_thought       90.0%      89.8%      88.5%      91.1%
role_based             86.7%      86.1%      85.0%      87.3%
baseline               83.3%      82.9%      81.5%      84.3%
```

## Analysis & Visualization (Phase 4) ✓

Generate comprehensive analysis and visualizations:

```bash
./analyze.sh
```

This creates:
- **5 Visualization Charts** (PNG format)
  - Accuracy comparison bar chart
  - Comprehensive metrics comparison
  - Distance distribution histograms
  - Confusion matrix heatmaps
  - Per-category performance heatmap

- **Statistical Analysis** (JSON format)
  - Rankings by all metrics
  - Improvements over baseline
  - Consistency analysis
  - Category-wise performance
  - Key findings

- **Final Report** (Markdown format)
  - Executive summary
  - Detailed methodology
  - Complete results
  - Insights and recommendations
  - Conclusion

**Manual execution:**
```bash
cd analysis
python statistical_analysis.py --latest      # Statistical analysis
python visualization.py --latest             # Generate charts
python generate_report.py --latest           # Create report
```

**View results:**
```bash
open visualizations/                         # View charts
cat EXPERIMENT_REPORT.md                     # Read report
cat analysis/statistical_analysis_*.json | jq  # View analysis
```

---

## Complete Workflow

```bash
# 1. Initial setup (one-time)
./setup.sh

# 2. Run baseline experiment
./run.sh

# 3. Run all prompt variations
./run_all.sh

# 4. Generate analysis and visualizations
./analyze.sh

# 5. View final report
cat EXPERIMENT_REPORT.md
```

## Requirements

- Python 3.9+
- Ollama installed and running
- ~8GB RAM (16GB recommended for larger models)
- ~4-7GB disk space for model weights
- No API costs - completely free!
