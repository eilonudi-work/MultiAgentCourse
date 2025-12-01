# Experiment 4: Context Engineering Strategies

## Overview
This experiment compares four different context management strategies in a multi-step agent scenario, demonstrating how different approaches handle growing conversation history and their impact on performance.

## Setup
- **Model:** mistral:7b
- **Embedding Model:** nomic-embed-text
- **Scenario:** Research assistant answering 10 sequential questions about machine learning
- **Strategies Tested:** 4 (BASELINE, SELECT, COMPRESS, WRITE)

## The Challenge

In multi-step agent interactions, conversation history accumulates over time, leading to:
- Growing context windows
- Increased latency
- Higher token costs
- Potential information overload
- Performance degradation

This experiment tests four strategies to manage this challenge.

## Four Context Management Strategies

### 1. BASELINE (No Management)
**Approach:** Accumulate all conversation history without any management

```python
def baseline_strategy(history, query):
    context = "\n\n".join(history)  # All history
    return query_llm(context, query)
```

**Expected:**
- Context grows linearly with each step
- Performance degrades over time
- Highest token usage
- Model becomes "distracted" by irrelevant history

### 2. SELECT (RAG-based Retrieval)
**Approach:** Use semantic search to retrieve only relevant history

```python
def select_strategy(history, query):
    relevant = vector_store.similarity_search(query, k=3)
    return query_llm(relevant, query)
```

**Expected:**
- Bounded context size (top-3 items)
- Maintains performance
- Focuses on relevant information
- Additional embedding overhead

### 3. COMPRESS (Summarization)
**Approach:** Summarize old history, keep recent items

```python
def compress_strategy(history, query):
    if len(history) > MAX_ITEMS:
        summary = summarize(history[:-3])
        context = summary + history[-3:]  # Summary + recent
    return query_llm(context, query)
```

**Expected:**
- Controlled context growth
- Preserves recent details
- Some information loss in summary
- Balance between efficiency and completeness

### 4. WRITE (External Memory/Scratchpad)
**Approach:** Extract and store key facts externally

```python
def write_strategy(history, query, scratchpad):
    for item in history:
        facts = extract_key_facts(item)
        scratchpad.store(facts)
    relevant_facts = scratchpad.retrieve(query)
    return query_llm(relevant_facts, query)
```

**Expected:**
- Most compact representation
- Structured fact storage
- Maintains key information
- Requires fact extraction overhead

## Sequential Queries (10 Steps)

The experiment simulates a research assistant answering progressively related questions:

1. "What is machine learning?"
2. "Explain supervised learning in detail"
3. "Explain unsupervised learning in detail"
4. "Describe neural networks and how they work"
5. "Explain deep learning and its relationship to neural networks"
6. "What are the main applications of machine learning in healthcare?"
7. "What are the main applications of machine learning in finance?"
8. "Discuss the main ethical concerns in machine learning"
9. "Explain the problem of bias in machine learning models"
10. "Summarize the key machine learning trends for 2024"

Each step builds on previous context, creating a realistic multi-turn conversation scenario.

## Metrics Tracked

For each strategy and step:
- **Total Tokens:** Cumulative tokens in context
- **Latency:** Response time (seconds)
- **History Items:** Number of items in history
- **Response Length:** Size of generated response

## Expected Results

### Context Growth Comparison

| Step | BASELINE | SELECT | COMPRESS | WRITE |
|------|----------|--------|----------|-------|
| 1    | ~500     | ~500   | ~500     | ~100  |
| 5    | ~2500    | ~1500  | ~1000    | ~300  |
| 10   | ~5000    | ~1500  | ~1200    | ~500  |

### Final Performance (Step 10)

| Strategy | Tokens | Latency | Efficiency |
|----------|--------|---------|------------|
| BASELINE | ~5000  | High    | 0% (reference) |
| SELECT   | ~1500  | Medium  | ~70% reduction |
| COMPRESS | ~1200  | Medium  | ~75% reduction |
| WRITE    | ~500   | Low     | ~90% reduction |

## Visualizations Generated

1. **Context Growth Over Time:** Line chart showing token accumulation across strategies
2. **Latency Over Time:** Line chart showing response time trends
3. **Final Comparison:** Bar chart comparing final metrics at step 10

## Key Findings

### BASELINE Strategy
- ❌ **Unlimited growth:** Context grows without bound
- ❌ **Performance degradation:** Slower with each step
- ❌ **Information overload:** Model overwhelmed by irrelevant history
- ✅ **Complete history:** No information loss

### SELECT Strategy
- ✅ **Bounded growth:** Consistent ~1500 tokens
- ✅ **Maintained performance:** Focused on relevant context
- ✅ **Semantic relevance:** Retrieves contextually appropriate history
- ⚠️ **Embedding overhead:** Requires vector operations

### COMPRESS Strategy
- ✅ **Controlled growth:** Slower growth than baseline
- ✅ **Recent detail:** Keeps full recent context
- ⚠️ **Information loss:** Older details compressed
- ⚠️ **Summarization cost:** Additional LLM calls

### WRITE Strategy
- ✅ **Most efficient:** Minimal token usage
- ✅ **Structured knowledge:** Organized fact storage
- ✅ **Scalable:** Works well with many steps
- ⚠️ **Extraction overhead:** Requires fact extraction
- ⚠️ **Lossy:** Only preserves extracted facts

## Implementation Highlights

### Multi-Strategy Runner
```python
strategies = {
    'baseline': baseline_strategy,
    'select': select_strategy,
    'compress': compress_strategy,
    'write': write_strategy
}

for strategy_name, strategy_func in strategies.items():
    history = []
    for step, query in enumerate(QUERIES):
        response = strategy_func(history, query)
        history.append(response)
        track_metrics(strategy_name, step, response, history)
```

### ChromaDB Integration (SELECT)
```python
vector_store = chromadb.Client().create_collection("agent_history")

# Add to vector store
embedding = embed_text(history_item)
vector_store.add(embeddings=[embedding], documents=[history_item])

# Retrieve relevant
results = vector_store.query(query_embeddings=[query_embedding], n_results=3)
```

### Fact Extraction (WRITE)
```python
scratchpad = {}

for i, item in enumerate(history):
    fact_prompt = f"Extract 2-3 key facts from: {item}"
    facts = query_llm(fact_prompt)
    scratchpad[f"fact_{i}"] = facts
```

## Files Generated
- `/results/exp4_results.csv` - Raw experimental data (40 rows: 4 strategies × 10 steps)
- `/results/exp4_comparison.png` - Combined comparison visualization
- `/results/exp4_context_growth.png` - Context growth over time chart
- `experiment_4.py` - Complete implementation

## How to Run

```bash
# Ensure Ollama is running with required models
ollama list  # Should show mistral:7b and nomic-embed-text

# Navigate to experiment directory
cd experiment_4

# Run experiment
python3 experiment_4.py
```

## Expected Runtime

- **Total time:** ~30-40 minutes
- **Per strategy:** ~7-10 minutes
- **Per step:** ~30-60 seconds (varies by strategy)

**Note:** BASELINE and COMPRESS strategies take longer due to larger context sizes and summarization overhead.

## Real-World Applications

This experiment demonstrates practical patterns for:

1. **Chatbots:** Managing long conversations efficiently
2. **Research Assistants:** Maintaining context across complex queries
3. **Customer Support:** Tracking conversation history without bloating context
4. **Code Assistants:** Remembering relevant code context across sessions

## Conclusion

**No single strategy is universally best** - the choice depends on use case:

- **Need complete history?** → BASELINE (but costly)
- **Need semantic relevance?** → SELECT (balanced)
- **Need recent detail?** → COMPRESS (good middle ground)
- **Need maximum efficiency?** → WRITE (most compact)

For most production applications, a **hybrid approach** combining SELECT and COMPRESS often provides the best balance of performance, cost, and quality.

## Technical Notes

- **Model:** mistral:7b handles the sequential reasoning well
- **Embeddings:** nomic-embed-text provides quality semantic search
- **ChromaDB:** In-memory mode sufficient for this POC
- **Token Counting:** Approximate (words × 1.3), actual may vary
- **Error Handling:** Graceful fallbacks for LLM/embedding errors
