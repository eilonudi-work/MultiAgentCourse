# Assignment 5: Context Windows in Practice

## Experiment 1: Needle in Haystack

### Overview
This experiment demonstrates context window phenomena in LLMs by testing their ability to retrieve facts embedded at different positions within a document.

### Setup
- **Model:** tinyllama (very small LLM for clear demonstration)
- **Document Size:** ~1400 words
- **Test Positions:** Start, Middle, End
- **Trials:** 10 per position (30 total)
- **Test Fact:** "The CEO of the company is David Cohen"

### Results

The experiment revealed a **recency bias** in the tinyllama model:

- **START position:** 70.0% accuracy (7/10 trials)
- **MIDDLE position:** 90.0% accuracy (9/10 trials)
- **END position:** 100.0% accuracy (10/10 trials)

### Key Findings

The results demonstrate a clear recency effect where information at the end of the context window is most reliably retrieved (100% accuracy), followed by the middle (90%), with the beginning showing the lowest accuracy (70%). This pattern shows that the small model has a strong bias toward recently processed information in its context window.

While this differs from the classic "Lost in the Middle" phenomenon (where both start and end perform well, but middle performs poorly), it demonstrates an important related concept: smaller models exhibit strong recency bias when context length approaches their effective attention span. The model successfully retrieves facts from the end almost perfectly, but struggles increasingly as information appears earlier in the document.

### Files Generated
- `/results/exp1_results.csv` - Raw experimental data (30 trials)
- `/results/exp1_chart.png` - Visualization showing accuracy by position
- `experiment_1.py` - Complete experiment implementation

### How to Run
```bash
# Ensure Ollama is installed and tinyllama model is available
ollama pull tinyllama

# Install Python dependencies
pip install ollama matplotlib pandas

# Run the experiment
python3 experiment_1.py
```

### Technical Notes
Using tinyllama at 1400 words provides a sweet spot where the model shows variable performance across positions. At smaller document sizes (e.g., 1200 words), the model achieves 100% accuracy across all positions. At larger sizes (e.g., 1600+ words), only the END position maintains high accuracy, showing pure recency bias.
