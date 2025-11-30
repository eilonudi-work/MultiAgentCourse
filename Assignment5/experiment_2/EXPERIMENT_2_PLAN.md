# Experiment 2: Context Window Size Impact (POC)

**Assignment:** Assignment 5 - Context Windows in Practice
**Version:** 1.0 (Concise POC)
**Date:** November 2025

---

## Overview

**Goal:** Demonstrate how increasing context window size affects accuracy and performance
**Approach:** Measure accuracy degradation and latency growth as we add more documents

---

## Tech Stack (Minimal)

- **LLM:** Ollama with **mistral:7b** (same as Experiment 1)
- **Python Libraries:** ollama, matplotlib, pandas, tiktoken (for token counting)
- **Optional:** LangChain (can be added but not required for POC)

---

## Experiment 2: Context Window Size Impact

**Goal:** Show that accuracy decreases and latency increases as context grows
**Duration:** ~20 min to run
**Difficulty:** Medium

### What to Build

1. **Document Generator** (`experiment_2.py`)
   - Reuse document generation from Experiment 1
   - Create multiple documents with embedded fact
   - Concatenate varying numbers of documents: **2, 5, 10, 20, 50**
   - Embed the fact in the MIDDLE of the concatenated context

2. **Metrics Collection**
   - For each context size, measure:
     - **Accuracy** - Can the LLM find the fact?
     - **Latency** - How long does the query take?
     - **Token count** - How many tokens in the context?
   - Run 5 trials per size for statistical validity

3. **Results & Visualization**
   - Line plot: Accuracy vs Number of Documents
   - Line plot: Latency vs Number of Documents
   - Line plot: Token Count vs Number of Documents
   - Save results to CSV

### Implementation Steps

```
1. Setup (5 min)
   - Use same environment as Experiment 1
   - Install tiktoken: pip install tiktoken
   - Reuse mistral:7b model

2. Code (45-60 min)
   - Reuse document generation from Experiment 1
   - Write function to concatenate N documents
   - Write function to count tokens (using tiktoken)
   - Write function to measure latency
   - Write experiment loop over doc counts: [2, 5, 10, 20, 50]
   - Write visualization functions (3 line plots)

3. Run & Analyze (20 min)
   - Execute experiment (5 trials per size = 25 total trials)
   - Generate visualizations
   - Document findings

4. Output
   - results/exp2_results.csv
   - results/exp2_accuracy_chart.png
   - results/exp2_latency_chart.png
   - results/exp2_tokens_chart.png
```

### Expected Results

**Accuracy Degradation:**
- **2-5 docs:** High accuracy (70-100%)
- **10-20 docs:** Moderate accuracy (30-60%)
- **50 docs:** Low accuracy (0-20%)

**Latency Growth:**
- Increases linearly or exponentially with document count
- 2 docs: ~2-5 seconds
- 50 docs: ~30-60 seconds

**Token Growth:**
- Linear growth with number of documents
- ~2000 words/doc × 1.3 tokens/word = ~2600 tokens/doc

**Why this happens:**
- Larger contexts dilute attention across more content
- The critical fact gets buried in noise
- Model struggles to maintain focus across long contexts

### Sample Code Structure

```python
# experiment_2.py
import time
import tiktoken

MODEL = "mistral:7b"
DOC_COUNTS = [2, 5, 10, 20, 50]
TRIALS_PER_SIZE = 5
FACT = "The CEO of the company is David Cohen"

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def create_multi_doc_context(num_docs, fact_position='middle'):
    """Generate N documents and embed fact in the middle"""
    docs = []
    for i in range(num_docs):
        doc = generate_filler_text(2000)
        docs.append(doc)

    # Embed fact in middle document
    middle_idx = len(docs) // 2
    docs[middle_idx] = embed_fact(docs[middle_idx], FACT, 'middle')

    return "\n\n---\n\n".join(docs)

def run_experiment():
    """Main experiment loop"""
    results = []

    for num_docs in DOC_COUNTS:
        print(f"\nTesting with {num_docs} documents")

        for trial in range(TRIALS_PER_SIZE):
            # Create context
            context = create_multi_doc_context(num_docs)
            tokens = count_tokens(context)

            # Measure latency
            start_time = time.time()
            response = query_llm(context, "Who is the CEO?")
            latency = time.time() - start_time

            # Check accuracy
            accuracy = check_accuracy(response, "David Cohen")

            results.append({
                'num_docs': num_docs,
                'trial': trial + 1,
                'tokens': tokens,
                'latency': latency,
                'accuracy': accuracy
            })

    return pd.DataFrame(results)

def plot_results(df):
    """Create 3 line plots"""
    # 1. Accuracy vs docs
    # 2. Latency vs docs
    # 3. Tokens vs docs
    pass
```

### Deliverables (Minimal)

- [ ] `experiment_2.py` - working script
- [ ] `results/exp2_results.csv` - raw data
- [ ] `results/exp2_accuracy_chart.png` - accuracy degradation
- [ ] `results/exp2_latency_chart.png` - latency growth
- [ ] `results/exp2_tokens_chart.png` - token count growth
- [ ] Brief findings (3-4 sentences in README)

---

## Key Insights to Demonstrate

1. **Accuracy Degradation:** Performance drops significantly as context grows
2. **Latency Growth:** Processing time increases with context size
3. **Token Scaling:** Context size grows linearly with document count
4. **Practical Limit:** There's a practical limit to how much context an LLM can effectively use

This experiment shows why techniques like RAG (Experiment 3) are necessary for handling large document collections.

---

## Notes

- **Keep it simple:** POC focused on demonstrating the phenomenon
- **Reuse code:** Leverage Experiment 1's document generation
- **Focus on trends:** Clear visualization of degradation patterns
- **Statistical validity:** 5 trials per size is sufficient for POC
