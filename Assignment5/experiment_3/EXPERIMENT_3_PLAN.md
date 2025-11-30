# Experiment 3: RAG Impact - Plan (POC)

**Assignment:** Assignment 5 - Context Windows in Practice
**Version:** 1.0 (Concise POC)
**Date:** November 2025

---

## Overview

**Goal:** Compare RAG-based retrieval vs. full context loading to demonstrate efficiency gains
**Duration:** ~25 min to run
**Difficulty:** Medium+

---

## Experiment Objective

Demonstrate the difference between:
- **Without RAG:** Loading all 20 documents into context window
- **With RAG:** Using semantic search to retrieve only relevant documents (top-k=3)

**Key Metrics:**
- Accuracy of answers
- Response latency
- Token usage

---

## Tech Stack

- **LLM:** Ollama with **mistral:7b**
- **Embeddings:** nomic-embed-text (via Ollama)
- **Vector Store:** ChromaDB
- **Python Libraries:** ollama, chromadb, matplotlib, pandas, tiktoken

---

## Dataset

**20 Hebrew documents** covering 3 topics:
- Technology (7 docs)
- Law (7 docs)
- Medicine (6 docs)

**Test Query:** "מהן תופעות הלוואי של תרופה X?" (What are the side effects of drug X?)

**Expected:** Only 2-3 medicine documents are relevant, but full context loads all 20.

---

## Implementation Steps

### 1. Setup (10 min)
```bash
# Install dependencies
pip install chromadb ollama tiktoken matplotlib pandas

# Pull Ollama models
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### 2. Prepare Dataset (15 min)
- Create 20 synthetic Hebrew documents in `data/documents/`
- Topics: tech, law, medicine
- Each ~300-500 words
- Save as `doc_001_tech.txt`, `doc_002_law.txt`, etc.

### 3. Build RAG Pipeline (30 min)

**Code Structure:**
```python
# experiment_3.py

# Step 1: Load documents
def load_documents(doc_dir):
    """Load all 20 Hebrew documents"""
    pass

# Step 2: Chunk documents
def chunk_documents(documents, chunk_size=500):
    """Split into chunks (500 chars each)"""
    pass

# Step 3: Generate embeddings
def embed_chunks(chunks):
    """Use nomic-embed-text via Ollama"""
    pass

# Step 4: Store in ChromaDB
def create_vector_store(chunks, embeddings):
    """Initialize and populate ChromaDB"""
    pass

# Step 5: Compare modes
def compare_retrieval_modes(query):
    # Mode A: Full Context
    full_context = concatenate_all_documents()
    full_response = query_llm(full_context, query)

    # Mode B: RAG
    relevant_docs = vector_store.similarity_search(query, k=3)
    rag_response = query_llm(relevant_docs, query)

    return compare_metrics(full_response, rag_response)
```

### 4. Run Experiment (25 min)

**Test Cases:**
1. Medicine query → Should retrieve medicine docs
2. Technology query → Should retrieve tech docs
3. Law query → Should retrieve law docs

**Run 5 times per query** for statistical validity.

### 5. Measure & Visualize (15 min)

**Metrics to track:**
- Response time (seconds)
- Token count
- Answer accuracy (manual evaluation: correct/partially correct/wrong)
- Relevance of retrieved docs

**Visualizations:**
- Bar chart: RAG vs Full Context latency
- Bar chart: RAG vs Full Context accuracy
- Table: Token usage comparison

---

## Sample Code Structure

```python
# experiment_3.py
import ollama
import chromadb
import time
import pandas as pd
import matplotlib.pyplot as plt

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("hebrew_docs")

def embed_text(text):
    """Generate embeddings using nomic-embed-text"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def query_with_full_context(query):
    """Load all 20 docs into context"""
    all_docs = load_all_documents()
    full_context = "\n\n".join(all_docs)

    start = time.time()
    response = ollama.generate(
        model="mistral:7b",
        prompt=f"Context:\n{full_context}\n\nQuestion: {query}\nAnswer:"
    )
    latency = time.time() - start

    return {
        'answer': response['response'],
        'latency': latency,
        'tokens': count_tokens(full_context)
    }

def query_with_rag(query, k=3):
    """Retrieve top-k relevant docs"""
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    relevant_docs = results['documents'][0]

    context = "\n\n".join(relevant_docs)

    start = time.time()
    response = ollama.generate(
        model="mistral:7b",
        prompt=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    latency = time.time() - start

    return {
        'answer': response['response'],
        'latency': latency,
        'tokens': count_tokens(context),
        'retrieved_docs': len(relevant_docs)
    }

def run_experiment():
    queries = [
        "מהן תופעות הלוואי של תרופה X?",  # Medicine
        "מהן החובות החוקיות של מעסיק?",    # Law
        "מהם היתרונות של בינה מלאכותית?"  # Technology
    ]

    results = []

    for query in queries:
        for run in range(5):
            full = query_with_full_context(query)
            rag = query_with_rag(query, k=3)

            results.append({
                'query_type': detect_query_type(query),
                'run': run,
                'full_latency': full['latency'],
                'rag_latency': rag['latency'],
                'full_tokens': full['tokens'],
                'rag_tokens': rag['tokens'],
                'full_answer': full['answer'],
                'rag_answer': rag['answer']
            })

    df = pd.DataFrame(results)
    df.to_csv('results/exp3_results.csv')
    plot_comparison(df)

if __name__ == "__main__":
    run_experiment()
```

---

## Expected Results

### Without RAG (Full Context):
- **High latency** (~10-15 seconds)
- **High token usage** (~10,000+ tokens)
- **Lower accuracy** (noise from irrelevant docs)
- Model struggles with irrelevant information

### With RAG:
- **Low latency** (~2-4 seconds)
- **Low token usage** (~1,500 tokens)
- **High accuracy** (focused on relevant docs)
- Clean, precise answers

**Key Insight:** RAG reduces context noise and improves both speed and accuracy.

---

## Deliverables (Minimal)

- [ ] `experiment_3.py` - working RAG pipeline
- [ ] `data/documents/` - 20 Hebrew documents
- [ ] `results/exp3_results.csv` - raw comparison data
- [ ] `results/exp3_latency_comparison.png` - bar chart
- [ ] `results/exp3_accuracy_comparison.png` - bar chart
- [ ] Brief findings (4-5 sentences)

---

## Notes

- **Hebrew Support:** Ensure UTF-8 encoding for all files
- **ChromaDB:** Use in-memory mode for POC (no persistence needed)
- **Evaluation:** Manual accuracy check (correct/partial/wrong)
- **Embeddings:** nomic-embed-text supports multilingual text including Hebrew
- **Keep it simple:** Focus on demonstrating RAG benefits, not production optimization
