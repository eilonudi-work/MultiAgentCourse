# Prompt Engineering Assignment - POC

A proof-of-concept implementation comparing different prompt engineering techniques using sentiment analysis with **local Ollama models**.

## Overview

This project systematically evaluates how different prompting strategies affect LLM performance on sentiment classification tasks using local models via Ollama. We test:
- Baseline prompts
- Improved prompts with role definition
- Few-shot learning
- Chain of thought reasoning

**Benefits of using Ollama:**
- No API costs
- Unlimited experimentation
- Privacy (all processing is local)
- Fast iteration

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

## Setup Instructions

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

### 7. Verify Dataset

```bash
python -c "import json; print(len(json.load(open('data/sentiment_dataset.json'))))"
# Should output: 30
```

### 8. Test Configuration

```bash
python src/config.py
# Should display your Ollama configuration
```

## Dataset

- **Size**: 30 examples (POC version)
- **Balance**: 15 positive, 15 negative
- **Categories**: entertainment, product, food, service, technology, hospitality, general
- **Format**: JSON with text, ground_truth, and category fields

## Running Experiments

### Quick Setup Test

```bash
python test_setup.py
```

This will verify:
- Ollama connection
- Model availability
- Sentiment classification functionality
- Dataset integrity

### Run Baseline Experiment

```bash
cd src
python baseline_experiment.py
```

Options:
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

## Next Steps

Phase 3 will implement:
1. Improved prompt variations (role definition, few-shot, CoT)
2. Batch experiment runner for all variations
3. Comparison visualizations and analysis

## Requirements

- Python 3.9+
- Ollama installed and running
- ~8GB RAM (16GB recommended for larger models)
- ~4-7GB disk space for model weights
- No API costs - completely free!
