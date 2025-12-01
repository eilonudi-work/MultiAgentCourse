# Experiment 4: Context Engineering Strategies - Plan (POC)

**Assignment:** Assignment 5 - Context Windows in Practice
**Version:** 1.0 (Concise POC)
**Date:** November 2025

---

## Overview

**Goal:** Compare context management strategies in a multi-step agent scenario
**Duration:** ~30 min to run
**Difficulty:** Advanced

---

## Experiment Objective

Simulate a multi-step agent performing 10 sequential actions, where each action adds output to the context. Test three strategies for managing growing context:

1. **SELECT** - Use RAG to retrieve only relevant history
2. **COMPRESS** - Automatically summarize history when it grows too large
3. **WRITE** - Store key facts in external memory (scratchpad)
4. **BASELINE** - No context management (accumulate everything)

**Key Metrics:**
- Context size (tokens) over time
- Response latency
- Answer accuracy/quality
- Memory usage

---

## Tech Stack

- **LLM:** mistral:7b
- **Embeddings:** nomic-embed-text (for SELECT strategy)
- **Vector Store:** ChromaDB (for SELECT strategy)
- **Python Libraries:** ollama, chromadb, matplotlib, pandas

---

## Scenario Design

### Multi-Step Agent Task
**Scenario:** Research assistant gathering information about a complex topic across 10 steps

**Sequential Actions (10 steps):**
1. Find definition of "machine learning"
2. Explain supervised learning
3. Explain unsupervised learning
4. Describe neural networks
5. Explain deep learning
6. Find applications of ML in healthcare
7. Find applications of ML in finance
8. Discuss ethical concerns in ML
9. Explain bias in ML models
10. Summarize key ML trends for 2024

Each step:
- Receives query
- Has access to growing history (outputs from previous steps)
- Generates response
- Adds response to history

---

## Context Management Strategies

### 1. BASELINE (No Management)
```python
def baseline_strategy(history, query):
    # Simply concatenate all history
    full_context = "\n\n".join(history)
    return query_llm(full_context, query)
```
**Expected:** Performance degrades as context grows

### 2. SELECT (RAG-based)
```python
def select_strategy(history, query):
    # Embed all history items
    # Retrieve top-k=3 most relevant
    relevant = vector_store.similarity_search(query, k=3)
    return query_llm(relevant, query)
```
**Expected:** Maintains performance, uses only relevant history

### 3. COMPRESS (Summarization)
```python
def compress_strategy(history, query):
    if len(history) > MAX_ITEMS:
        # Summarize oldest items, keep recent items
        summary = summarize(history[:-3])
        context = summary + history[-3:]
    else:
        context = history
    return query_llm(context, query)
```
**Expected:** Bounded context size, some information loss

### 4. WRITE (External Memory/Scratchpad)
```python
def write_strategy(history, query, scratchpad):
    # Extract key facts from each history item
    for item in history:
        facts = extract_key_facts(item)
        scratchpad[fact_id] = facts

    # Retrieve relevant facts
    relevant_facts = scratchpad.retrieve(query)
    return query_llm(relevant_facts, query)
```
**Expected:** Compact representation, maintains key info

---

## Implementation Steps

### 1. Setup (10 min)
```bash
# Already have models from Experiment 3
ollama list  # Verify mistral:7b and nomic-embed-text

# Dependencies already installed
pip3 list | grep -E "(chromadb|ollama|matplotlib|pandas)"
```

### 2. Create Agent Simulator (30 min)

**Code Structure:**
```python
# experiment_4.py

class MultiStepAgent:
    def __init__(self, strategy):
        self.strategy = strategy
        self.history = []

    def execute_step(self, query):
        # Apply strategy to manage context
        response = self.strategy(self.history, query)
        self.history.append(response)
        return response

# Define 10 sequential queries
queries = [
    "What is machine learning?",
    "Explain supervised learning",
    # ... 8 more
]

# Run each strategy
strategies = {
    'baseline': baseline_strategy,
    'select': select_strategy,
    'compress': compress_strategy,
    'write': write_strategy
}

results = []
for strategy_name, strategy_func in strategies.items():
    agent = MultiStepAgent(strategy_func)
    for step, query in enumerate(queries):
        start = time.time()
        response = agent.execute_step(query)
        latency = time.time() - start

        results.append({
            'strategy': strategy_name,
            'step': step,
            'query': query,
            'latency': latency,
            'tokens': count_tokens(agent.history),
            'history_size': len(agent.history)
        })
```

### 3. Run Experiment (30 min)

**For each strategy:**
- Execute all 10 steps sequentially
- Track metrics at each step
- Save results

### 4. Analyze & Visualize (15 min)

**Visualizations:**
1. **Context Growth:** Tokens over time (4 lines, one per strategy)
2. **Latency Growth:** Response time over steps
3. **Efficiency Comparison:** Final tokens/latency by strategy
4. **Strategy Comparison Table**

---

## Expected Results

### Context Size Growth

| Step | BASELINE | SELECT | COMPRESS | WRITE |
|------|----------|--------|----------|-------|
| 1    | ~500     | ~500   | ~500     | ~100  |
| 5    | ~2500    | ~1500  | ~1000    | ~300  |
| 10   | ~5000    | ~1500  | ~1200    | ~500  |

### Performance at Step 10

| Strategy | Tokens | Latency | Quality |
|----------|--------|---------|---------|
| BASELINE | ~5000  | High    | Lower (distracted) |
| SELECT   | ~1500  | Medium  | High (focused) |
| COMPRESS | ~1200  | Medium  | Medium (lossy) |
| WRITE    | ~500   | Low     | High (structured) |

### Key Insights

1. **BASELINE:** Context accumulation → performance degradation
2. **SELECT:** Effective but requires embedding overhead
3. **COMPRESS:** Balances size and information, some loss
4. **WRITE:** Most efficient, requires fact extraction

---

## Sample Code Structure

```python
# experiment_4.py

import ollama
import chromadb
import time
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
MODEL = "mistral:7b"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_HISTORY_ITEMS = 5  # For COMPRESS strategy

# 10 Sequential Queries
QUERIES = [
    "What is machine learning?",
    "Explain supervised learning",
    "Explain unsupervised learning",
    "Describe neural networks",
    "Explain deep learning",
    "Find applications of ML in healthcare",
    "Find applications of ML in finance",
    "Discuss ethical concerns in ML",
    "Explain bias in ML models",
    "Summarize key ML trends for 2024"
]

def baseline_strategy(history, query):
    """No context management - accumulate everything"""
    context = "\n\n".join(history) if history else ""
    prompt = f"Previous context:\n{context}\n\nNew question: {query}\nAnswer:"
    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response']

def select_strategy(history, query, vector_store):
    """RAG-based selection of relevant history"""
    if not history:
        return baseline_strategy([], query)

    # Retrieve top-3 relevant history items
    query_embedding = embed_text(query)
    results = vector_store.query(query_embeddings=[query_embedding], n_results=min(3, len(history)))
    relevant_context = "\n\n".join(results['documents'][0])

    prompt = f"Relevant context:\n{relevant_context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response']

def compress_strategy(history, query):
    """Summarize old context, keep recent"""
    if len(history) <= MAX_HISTORY_ITEMS:
        return baseline_strategy(history, query)

    # Keep last 3, summarize the rest
    old_context = "\n\n".join(history[:-3])
    summary_prompt = f"Summarize the following in 2-3 sentences:\n{old_context}"
    summary = ollama.generate(model=MODEL, prompt=summary_prompt)['response']

    recent_context = "\n\n".join(history[-3:])
    context = f"Summary: {summary}\n\nRecent:\n{recent_context}"

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response']

def write_strategy(history, query, scratchpad):
    """Extract and store key facts"""
    # Extract facts from history
    for i, item in enumerate(history):
        if f"fact_{i}" not in scratchpad:
            fact_prompt = f"Extract 2-3 key facts from:\n{item}\nFacts (bullet points):"
            facts = ollama.generate(model=MODEL, prompt=fact_prompt)['response']
            scratchpad[f"fact_{i}"] = facts

    # Use scratchpad facts as context
    context = "\n".join(scratchpad.values()) if scratchpad else ""
    prompt = f"Key facts:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.generate(model=MODEL, prompt=prompt)
    return response['response']

def run_experiment():
    """Main experiment loop"""
    results = []

    # Initialize storage for each strategy
    vector_store = setup_vector_store()
    scratchpad = {}

    strategies = {
        'baseline': lambda h, q: baseline_strategy(h, q),
        'select': lambda h, q: select_strategy(h, q, vector_store),
        'compress': lambda h, q: compress_strategy(h, q),
        'write': lambda h, q: write_strategy(h, q, scratchpad)
    }

    for strategy_name, strategy_func in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running {strategy_name.upper()} strategy")
        print(f"{'='*60}")

        history = []

        for step, query in enumerate(QUERIES):
            print(f"\nStep {step + 1}/10: {query}")

            start_time = time.time()
            response = strategy_func(history, query)
            latency = time.time() - start_time

            history.append(response)

            # Calculate metrics
            total_tokens = sum(count_tokens(item) for item in history)

            results.append({
                'strategy': strategy_name,
                'step': step + 1,
                'query': query,
                'latency': latency,
                'total_tokens': total_tokens,
                'history_items': len(history),
                'response_preview': response[:100]
            })

            print(f"  Latency: {latency:.2f}s | Tokens: {total_tokens} | History: {len(history)} items")

    # Save and visualize
    df = pd.DataFrame(results)
    df.to_csv('results/exp4_results.csv', index=False)
    visualize_results(df)
    print_summary(df)

if __name__ == "__main__":
    run_experiment()
```

---

## Deliverables (Minimal)

- [ ] `experiment_4.py` - Complete implementation
- [ ] `results/exp4_results.csv` - Raw data (40 rows: 4 strategies × 10 steps)
- [ ] `results/exp4_context_growth.png` - Token growth over steps
- [ ] `results/exp4_latency_comparison.png` - Latency comparison
- [ ] `results/exp4_strategy_summary.png` - Final comparison table
- [ ] Brief findings (4-5 sentences)

---

## Notes

- **Keep it simple:** POC focused on demonstrating concepts
- **Sequential execution:** Run one strategy at a time to avoid resource contention
- **Metrics focus:** Prioritize token count and latency over subjective quality
- **Visualization:** Clear charts showing context growth and performance impact
- **Real-world relevance:** Demonstrates practical context management challenges in multi-turn agents
