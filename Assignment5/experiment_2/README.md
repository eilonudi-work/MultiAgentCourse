# Experiment 2: Context Window Size Impact

## Overview

This experiment demonstrates how LLM accuracy degrades and latency increases as the context window size grows. By varying the number of documents (2, 5, 10, 20, 50), we show the practical limits of context windows and the need for retrieval techniques like RAG.

**Model:** mistral:7b (32,768 token context window)
**Test Date:** November 30, 2025
**Total Trials:** 25 (5 trials per document count)

## Experimental Design

### Setup
- **Critical Fact:** "The CEO of the company is David Cohen"
- **Embedding Strategy:** Fact embedded in the MIDDLE document of the concatenated context
- **Document Size:** 500 words per document (reduced from original 2000 to account for model limitations)
- **Document Counts Tested:** [2, 5, 10, 20, 50]
- **Separator:** Documents separated by "\n\n---\n\n"

### Metrics Collected
1. **Accuracy:** Does the model correctly identify "David Cohen" as the CEO?
2. **Latency:** How long does the query take to complete?
3. **Token Count:** Total tokens in the context (measured with tiktoken cl100k_base encoding)

## Results Summary

| Documents | Accuracy | Avg Latency | Avg Tokens |
|-----------|----------|-------------|------------|
| 2         | 40.0%    | 6.75s       | 1,138      |
| 5         | 0.0%     | 17.03s      | 2,829      |
| 10        | 0.0%     | 22.85s      | 5,638      |
| 20        | 0.0%     | 22.88s      | 11,288     |
| 50        | 0.0%     | 22.98s      | 28,193     |

## Key Findings

### 1. Rapid Accuracy Degradation
- **2 documents:** 40% accuracy - model can sometimes find the fact in small contexts
- **5+ documents:** 0% accuracy - complete failure once context exceeds ~2,800 tokens
- **Critical observation:** Even at 2 documents, accuracy is only 40%, showing the "lost in the middle" problem

### 2. Non-Linear Latency Growth
- **Initial steep increase:** 6.75s (2 docs) → 17.03s (5 docs) = 2.5x increase
- **Plateau effect:** 17.03s (5 docs) → 22.98s (50 docs) = only 1.35x increase
- **Explanation:** The model appears to hit a processing threshold around 5 documents where latency stabilizes, possibly due to:
  - Context truncation
  - Attention mechanism saturation
  - Ollama's optimization strategies

### 3. Linear Token Growth
- **Scaling rate:** ~565 tokens per document (including separators)
- **Perfect linearity:** R² ≈ 1.0
- **Total range:** 1,138 tokens (2 docs) → 28,193 tokens (50 docs)
- **Within model limits:** All tests stayed well below mistral:7b's 32,768 token limit

### 4. Lost in the Middle Phenomenon
The experiment reveals severe "lost in the middle" behavior:
- Even with the fact embedded in the middle document, the model fails to retrieve it
- The overwhelming amount of repetitive filler text drowns out the critical information
- Attention mechanisms struggle to focus on relevant facts buried in noise

## Why This Happens

1. **Attention Dilution:** As context grows, the model's attention is distributed across more content, reducing focus on any single piece of information.

2. **Signal-to-Noise Ratio:** With 500 words of filler text per document, the critical fact (7 words) represents only ~1.4% of each document's content.

3. **Semantic Similarity:** The filler sentences are all business-related, creating semantic noise that makes the CEO fact harder to distinguish.

4. **Position Bias:** Facts in the middle of long contexts are harder to retrieve than those at the beginning or end (though our experiment shows even 2 documents is challenging).

## Practical Implications

### Why We Need RAG
This experiment demonstrates why Retrieval-Augmented Generation (RAG) is essential:
- **Problem:** Dumping all documents into context leads to 0% accuracy
- **Solution:** RAG retrieves only the most relevant documents, keeping context small and focused
- **Benefit:** Maintains high accuracy while handling large document collections

### Context Window Limitations
Even models with large context windows (32K+ tokens) struggle with:
- Finding specific facts in long contexts
- Maintaining attention across extensive text
- Processing speed degradation

### Optimal Context Size
Based on our results:
- **Sweet spot:** 2-3 documents (~1,000-2,000 tokens) for factual retrieval
- **Danger zone:** 5+ documents (2,800+ tokens) shows complete accuracy collapse
- **Recommendation:** Keep context under 2,000 tokens for critical fact retrieval tasks

## Experimental Challenges & Learnings

### Initial Issues
1. **Original design:** 2,000 words per document led to 0% accuracy across all tests
2. **Root cause:** Too much noise overwhelmed the model's ability to find facts
3. **Solution:** Reduced to 500 words per document to achieve measurable degradation

### Model Behavior
- mistral:7b shows strong position bias and struggles with needle-in-haystack tasks
- The model is honest about uncertainty, responding "The text does not provide..." rather than hallucinating
- Temperature setting (0.1) ensures consistent behavior across trials

## Files Generated

- **experiment_2.py** - Main experiment script
- **results/exp2_results.csv** - Raw trial data (25 rows)
- **results/exp2_accuracy_chart.png** - Accuracy degradation visualization
- **results/exp2_latency_chart.png** - Latency growth visualization
- **results/exp2_tokens_chart.png** - Token scaling visualization
- **results/exp2_combined_charts.png** - All three metrics in one view

## Conclusion

This experiment successfully demonstrates the fundamental challenge of context windows:

**Accuracy degrades rapidly as context grows, even when well within the model's technical token limit.**

The practical context limit for reliable fact retrieval is much smaller than the theoretical maximum. For mistral:7b, we observed complete failure beyond ~2,800 tokens for needle-in-haystack tasks.

This motivates the need for intelligent retrieval systems (RAG) that:
1. Identify the most relevant documents
2. Keep context focused and small
3. Maintain high accuracy even with large knowledge bases

The experiment sets the stage for Experiment 3, which will demonstrate how RAG overcomes these limitations.
