# Quick Start Guide

Get up and running in 2 minutes!

## For the Impatient

```bash
# 1. Automated setup (installs everything)
./setup.sh

# 2. Run experiments
./run.sh
```

Done! 🎉

---

## What Just Happened?

### `./setup.sh` installed:
- ✓ Ollama (local LLM runtime)
- ✓ llama2 model (~4GB)
- ✓ Python virtual environment
- ✓ All Python dependencies
- ✓ Configured environment variables
- ✓ Verified everything works

### `./run.sh` did:
- ✓ Started Ollama service
- ✓ Ran sentiment analysis on 30 examples
- ✓ Calculated accuracy metrics
- ✓ Saved results to `results/` directory

---

## View Your Results

```bash
# List result files
ls -lh results/

# View metrics (use latest timestamp)
cat results/baseline_metrics_*.json | jq '.metrics'
```

---

## Try Different Models

```bash
# Setup with mistral (faster, smaller)
./setup.sh mistral

# Run with mistral
./run.sh --model mistral

# Setup with llama3 (more capable)
./setup.sh llama3
./run.sh --model llama3
```

---

## Common Options

```bash
# Show which examples the model got wrong
./run.sh --show-errors

# Use different model without re-setup
./run.sh --model phi

# Don't save results (just test)
./run.sh --no-save

# Get help
./run.sh --help
```

---

## File Structure

```
Assignment6/
├── setup.sh                    # One-time setup
├── run.sh                      # Run experiments
├── test_setup.py              # Verify setup
│
├── data/
│   └── sentiment_dataset.json # 30 test examples
│
├── src/
│   ├── config.py              # Configuration
│   ├── ollama_client.py       # LLM client
│   ├── metrics.py             # Metrics calculator
│   └── baseline_experiment.py # Main experiment
│
└── results/                    # Generated results
    ├── baseline_results_*.json
    └── baseline_metrics_*.json
```

---

## Troubleshooting

### "ModuleNotFoundError" or dependency issues
```bash
# Upgrade build tools and reinstall
source venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt

# Verify with quick test
python quick_test.py
```

### "Ollama connection failed"
```bash
# Start Ollama manually
ollama serve

# In another terminal, run:
./run.sh
```

### "Model not found"
```bash
# Pull the model
ollama pull llama2

# Run again
./run.sh
```

### "Virtual environment not found"
```bash
# Re-run setup
./setup.sh
```

### Quick verification
```bash
# Run quick integration test
python quick_test.py

# Or full diagnostic test
python test_setup.py
```

---

## Understanding the Results

### Accuracy Metrics

- **Accuracy**: % of correct classifications (higher is better)
- **Mean Distance**: Semantic distance from truth (lower is better)
- **Precision**: % of positive predictions that are correct
- **Recall**: % of actual positives found
- **F1 Score**: Harmonic mean of precision and recall

### Example Output

```json
{
  "accuracy": 0.867,          // 86.7% correct
  "mean_distance": 0.123,     // Low distance = good
  "precision": 0.857,         // 85.7% precision
  "recall": 0.882,            // 88.2% recall
  "f1_score": 0.869           // 86.9% F1
}
```

---

## Next Steps

1. **Examine results**:
   ```bash
   cat results/baseline_metrics_*.json | jq
   ```

2. **Try different models**:
   ```bash
   ./run.sh --model mistral
   ./run.sh --model llama3
   ```

3. **Implement Phase 3** - Prompt variations:
   - Few-shot learning
   - Chain of thought
   - Role-based prompts

4. **Create visualizations** (Phase 4):
   - Comparison charts
   - Distribution histograms
   - Category breakdowns

---

## Resources

- **Ollama Documentation**: https://ollama.ai/docs
- **Available Models**: https://ollama.ai/library
- **Project README**: [README.md](README.md)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

---

## Tips

💡 **Use smaller models for testing**:
```bash
./setup.sh phi  # ~2GB, faster
```

💡 **Use larger models for accuracy**:
```bash
./setup.sh llama3  # ~7GB, more accurate
```

💡 **Check system resources**:
```bash
# CPU usage
top | grep ollama

# Disk space
du -sh ~/.ollama/models
```

💡 **Clean up models**:
```bash
# List models
ollama list

# Remove unused models
ollama rm model_name
```

---

Happy experimenting! 🚀
