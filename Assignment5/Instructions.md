## **📝 Option 1 for Homework Assignment No. 5**

**Lab: Context Windows in Practice**

- **Planning an Experiment: Context Windows in Practice**
- **Student:**
- **Version:** 1.0
- **Date:** November 2025
- **All rights reserved \- Dr. Segal Yoram**

## **📚 Table of Contents**

| Topic / Section                                  | Exp. | Page |
| :----------------------------------------------- | :--- | :--- |
| Introduction                                     | 1    | 3    |
| Lab Objectives                                   | 2    | 3    |
| **Experiment 1: Needle in Haystack**             | 3    | 3    |
| Experiment Details                               | 3.1  | 3    |
| Data                                             | 3.2  | 3    |
| Pseudocode                                       | 3.3  | 4    |
| **Experiment 2: Context Window Size Impact**     | 4    | 4    |
| Experiment Details                               | 4.1  | 4    |
| Data                                             | 4.2  | 4    |
| Pseudocode                                       | 4.3  | 5    |
| **Experiment 3: RAG Impact**                     | 5    | 5    |
| Experiment Details                               | 5.1  | 5    |
| Data                                             | 5.2  | 5    |
| Pseudocode                                       | 5.3  | 6    |
| Expected Results                                 | 5.4  | 6    |
| **Experiment 4: Context Engineering Strategies** | 6    | 6    |
| Experiment Details                               | 6.1  | 6    |
| Data                                             | 6.2  | 7    |
| Pseudocode                                       | 6.3  | 7    |
| Summary Table                                    | 7    | 7    |
| Summary                                          | 8    | 8    |
| Submission Instructions                          | 9    | 8    |

## **1\. Introduction**

This assignment deals with the analysis and study of **Context Windows**.

The following proposed series of experiments serves as a general conceptual framework, and you are invited to interpret, develop, and research the topics in any way you see fit.

For each experiment, you must define research questions, perform the experiments, and analyze the findings, preferably by presenting a statistical and visual analysis (using graphs or tables). It is recommended to repeat each experiment a number of times to ensure the **statistical validity** of the result.

**Note:** Your conclusions do not have to align with the material presented in the lecture. You are allowed to reach **independent insights**, provided you justify them well; in such cases, it is recommended to use external references and offer an explanation for the discrepancies you discovered. Take the experiments in a direction that interests you and towards your personal line of inquiry—these instructions are intended to serve as a 'brainstorming' and are not closed definitions.

## **2\. Lab Objectives**

The student will understand and experience in code two central phenomena:

1. **Lost in the Middle**: Decrease in accuracy when relevant information is located in the middle of the context window.
2. **Context Accumulation Problem**: How data accumulates in agents and causes failure.

The lab is divided into **four modular sub-experiments**; each experiment is independent but builds upon the previous one.

## **3\. Experiment 1: Needle in Haystack**

### **3.1 Experiment Details**

- **Duration:** Approx. 15 minutes
- **Difficulty Level:** Basic
- **Goal:** Demonstration of **Lost in the Middle**

### **3.2 Data**

- **Synthetic text:** 5 documents, each with 200 words.
- Each document contains one critical fact (e.g., "The company CEO is David Cohen").
- The fact will appear in different locations: **Start / Middle / End**.

### **3.3 Pseudocode**

\# Experiment 1: Lost in the Middle Simulation

```
# Experiment 1: Lost in the Middle Simulation

# Generate synthetic documents with embedded facts
def create_documents(num_docs=5, words_per_doc=200):
    documents = []
    for i in range(num_docs):
        doc = generate_filler_text(words_per_doc)
        fact_position = random.choice(['start', 'middle', 'end'])
        doc = embed_critical_fact(doc, fact, fact_position)
        documents.append(doc)
    return documents

# Query LLM and measure accuracy by fact position
def measure_accuracy_by_position(documents, query):
    results = {'start': [], 'middle': [], 'end': []}
    for doc in documents:
        response = ollama_query(doc, query)
        accuracy = evaluate_response(response, expected_answer)
        results[doc.fact_position].append(accuracy)
    return calculate_averages(results)

# Expected: High accuracy at start/end, low at middle

```

\# Expected: High accuracy at start/end, low at middle

**Expected Result:** High accuracy at the start/end, low in the middle.

## **4\. Experiment 2: Context Window Size Impact**

### **4.1 Experiment Details**

- **Duration:** Approx. 20 minutes
- **Difficulty Level:** Medium
- **Goal:** Demonstration of **how context window size affects accuracy**

### **4.2 Data**

- Gradual increase in the number of documents: 2, 5, 10, 20, 50\.
- For each size: Measure **response time \+ accuracy \+ actual context length**.

### **4.3 Pseudocode**

```
# Experiment 2: Context Window Size Analysis

# Measure performance across different context sizes
def analyze_context_sizes(doc_counts=[2, 5, 10, 20, 50]):
    results = []
    for num_docs in doc_counts:
        documents = load_documents(num_docs)
        context = concatenate_documents(documents)
        start_time = time.time()
        response = langchain_query(context, query)
        latency = time.time() - start_time
        results.append({
            'num_docs': num_docs,
            'tokens_used': count_tokens(context),
            'latency': latency,
            'accuracy': evaluate_accuracy(response)
        })
    return results

# Plot: Accuracy degradation vs context size
# Expected: Accuracy decreases as window grows


```

**Results:** A graph showing the decrease in accuracy with the growing window.

## **5\. Experiment 3: RAG Impact**

### **5.1 Experiment Details**

- **Duration:** Approx. 25 minutes
- **Difficulty Level:** Medium+
- **Goal:** Comparison between two strategies:
  - **Without RAG:** All documents in the window.
  - **With RAG:** Only the relevant documents (using similarity search).

### **5.2 Data**

- A repository of **20 documents in Hebrew** (topics: Technology, Law, Medicine).
- **Query:** "What are the side effects of drug X?"

### **5.3 Pseudocode**

```
# Experiment 3: RAG vs Full Context Comparison

# Step 1: Chunking - split documents into chunks
chunks = split_documents(documents, chunk_size=500)

# Step 2: Embedding - convert to vectors
embeddings = nomic_embed_text(chunks)

# Step 3: Store in ChromaDB
vector_store = ChromaDB()
vector_store.add(chunks, embeddings)

# Step 4: Compare two retrieval modes
def compare_modes(query):
    # Mode A: Full context (all documents)
    full_response = query_with_full_context(all_documents, query)

    # Mode B: RAG (only similar documents)
    relevant_docs = vector_store.similarity_search(query, k=3)
    rag_response = query_with_context(relevant_docs, query)

    return {
        'full_accuracy': evaluate(full_response),
        'rag_accuracy': evaluate(rag_response),
        'full_latency': full_response.latency,
        'rag_latency': rag_response.latency
    }

# Expected: RAG accurate & fast, Full noisy & slow


```

### **5.4 Expected Results**

- **RAG:** Accurate and fast answers.
- **Full Context:** Noise and suffering (low quality), less accurate answers.

## **6\. Experiment 4: Context Engineering Strategies**

### **6.1 Experiment Details**

- **Duration:** Approx. 30 minutes
- **Difficulty Level:** Advanced
- **Goal:** Testing context management strategies: **Write, Select, Compress, Isolate**.

### **6.2 Data**

- Simulation of a multi-step agent performing **10 sequential actions**.
- Each action creates an output that is added to the context.

### **6.3 Pseudocode**

```
# Experiment 4: Context Engineering Strategies

# Strategy 1: SELECT - Use RAG for relevant retrieval only
def select_strategy(history, query):
    relevant = rag_search(history, query, k=5)
    return query_llm(relevant, query)

# Strategy 2: COMPRESS - Automatic history summarization
def compress_strategy(history, query):
    if len(history) > MAX_TOKENS:
        history = summarize(history)
    return query_llm(history, query)

# Strategy 3: WRITE - External memory (scratchpad)
def write_strategy(history, query, scratchpad):
    key_facts = extract_key_facts(history)
    scratchpad.store(key_facts)
    return query_llm(scratchpad.retrieve(query), query)

# Compare all strategies across 10 sequential actions
def benchmark_strategies(num_actions=10):
    results = {'select': [], 'compress': [], 'write': []}
    for action in range(num_actions):
        output = agent.execute(action)
        history.append(output)
        for strategy in ['select', 'compress', 'write']:
            result = evaluate_strategy(strategy, history)
            results[strategy].append(result)
    return results

```

## **7\. Summary Table**

**Table 1: Experiments Summary**

| Exp. | Topic / Output | Tools               | Duration (Min.) | Output                     |
| :--- | :------------- | :------------------ | :-------------- | :------------------------- |
| 1    | Lost in Middle | Ollama \+ Python    | 15              | Accuracy by position graph |
| 2    | Context Size   | Ollama \+ LangChain | 20              | Latency vs size graph      |
| 3    | RAG Impact     | Ollama \+ Chroma    | 25              | Performance comparison     |
| 4    | Engineering    | LangChain \+ Memory | 30              | Strategy performance table |

## **8\. Summary**

These experiments demonstrate the key challenges in working with large context windows:

1. **The Lost in the Middle problem**: Information in the middle of the window tends to be lost.
2. **Context window size**: As the window grows, accuracy decreases.
3. **RAG efficiency**: Focused retrieval improves accuracy and speed.
4. **Management strategies**: Select, Compress, Write provide various solutions.

## **9\. Submission Instructions**

The student must plan and consider a convincing way to present the results of the experimental investigation, and the conclusions from the experiment. It is recommended to validate the results with **graphs** at the student's discretion.
