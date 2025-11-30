# Context Windows Lab - Experiment Plan (POC)

**Assignment:** Assignment 5 - Context Windows in Practice
**Version:** 2.0 (Concise POC)
**Date:** November 2025

---

## Overview

This is a **proof-of-concept experiment** to demonstrate context window phenomena. Focus is on quick implementation and clear results, not production-grade code.

**Goal:** Run 4 experiments showing context window challenges and solutions
**Approach:** Simple, direct implementation with visualization of results

---

## Tech Stack (Minimal)

- **LLM:** Ollama with **mistral:7b** - a 7B parameter model with 8K context window, validated in research to demonstrate the U-shaped "Lost in the Middle" phenomenon
- **Python Libraries:** ollama, matplotlib, pandas
- **Optional:** LangChain (Exp 2+), ChromaDB (Exp 3+)

---

## Experiment 1: Needle in Haystack

**Goal:** Demonstrate "Lost in the Middle" phenomenon
**Duration:** ~15 min to run
**Difficulty:** Basic

### What to Build

1. **Document Generator** (`experiment_1.py`)
   - Generate documents with ~2000 words each (optimal for challenging mistral:7b)
   - Embed a fact (e.g., "CEO is David Cohen") at start/middle/end
   - Use simple filler text (business-related sentences)
   - **Use mistral:7b** - validated in research to show the U-shaped effect

2. **Experiment Runner**
   - For each position (start/middle/end):
     - Create document with fact at that position
     - Query LLM: "Who is the CEO?"
     - Check if answer is correct
   - Run 10 times per position for statistical validity

3. **Results & Visualization**
   - Calculate accuracy for each position
   - Create bar chart: Position vs Accuracy
   - Save results to CSV

### Implementation Steps

```
1. Setup (5-10 min)
   - Create project folder structure
   - Install: pip install ollama matplotlib pandas
   - Pull Ollama model: ollama pull mistral:7b
   - Note: mistral:7b was validated in "Lost in the Middle" research

2. Code (30-45 min)
   - Write document generator function
   - Write fact embedding function
   - Write LLM query function
   - Write experiment loop

3. Run & Analyze (15 min)
   - Execute experiment
   - Generate visualization
   - Document findings

4. Output
   - results/exp1_results.csv
   - results/exp1_accuracy_by_position.png
```

### Expected Result
High accuracy at **start** and **end**, low accuracy in **middle** - creating a **U-shaped curve** (classic "Lost in the Middle" phenomenon)

**Why this setup works:**
- **mistral:7b** is a 7B parameter model with 8K token context window
- Validated in [research](https://arxiv.org/abs/2307.03172) to demonstrate primacy and recency bias
- **2000-word documents** provide enough context to show performance degradation in the middle
- Creates the classic U-shaped performance curve: high at edges, low in middle

### Sample Code Structure

```python
# experiment_1.py
def generate_filler_text(words):
    """Create simple filler text"""
    pass

def embed_fact(text, fact, position):
    """Insert fact at start/middle/end"""
    pass

def query_llm(document, question):
    """Query Ollama"""
    pass

def check_accuracy(response, expected):
    """Simple string matching"""
    pass

def run_experiment():
    """Main experiment loop"""
    positions = ['start', 'middle', 'end']
    results = []

    for position in positions:
        for run in range(10):
            doc = generate_document_with_fact(position)
            response = query_llm(doc, "Who is the CEO?")
            accuracy = check_accuracy(response, "David Cohen")
            results.append({'position': position, 'run': run, 'accuracy': accuracy})

    # Save and visualize
    df = pd.DataFrame(results)
    df.to_csv('results/exp1_results.csv')
    plot_results(df)

if __name__ == "__main__":
    run_experiment()
```

### Deliverables (Minimal)

- [ ] `experiment_1.py` - working script
- [ ] `results/exp1_results.csv` - raw data
- [ ] `results/exp1_chart.png` - bar chart
- [ ] Brief findings (3-4 sentences in README)

---

## Notes

- **Keep it simple:** This is a POC, not production code
- **No tests required:** Focus on getting results
- **Minimal documentation:** Code comments + brief README section
- **Statistical validity:** 10 runs per condition is enough
- **Visualization:** Simple matplotlib bar chart is sufficient
