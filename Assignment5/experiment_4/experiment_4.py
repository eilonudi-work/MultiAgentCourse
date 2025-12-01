"""
Experiment 4: Context Engineering Strategies
Comparing different context management approaches in a multi-step agent scenario
"""

import ollama
import chromadb
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "mistral:7b"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_HISTORY_ITEMS = 5  # For COMPRESS strategy

# 10 Sequential Queries - ML Research Assistant Scenario
QUERIES = [
    "What is machine learning?",
    "Explain supervised learning in detail",
    "Explain unsupervised learning in detail",
    "Describe neural networks and how they work",
    "Explain deep learning and its relationship to neural networks",
    "What are the main applications of machine learning in healthcare?",
    "What are the main applications of machine learning in finance?",
    "Discuss the main ethical concerns in machine learning",
    "Explain the problem of bias in machine learning models",
    "Summarize the key machine learning trends for 2024"
]

def count_tokens(text):
    """Estimate token count (rough approximation)"""
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)  # Average ~1.3 tokens per word

def embed_text(text):
    """Generate embeddings using nomic-embed-text"""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"  Error generating embedding: {e}")
        return None

def setup_vector_store():
    """Initialize ChromaDB for SELECT strategy"""
    client = chromadb.Client()

    # Delete collection if it exists
    try:
        client.delete_collection(name="agent_history")
    except:
        pass  # Collection doesn't exist yet

    # Create fresh collection
    collection = client.create_collection(
        name="agent_history",
        metadata={"description": "Multi-step agent conversation history"}
    )
    return collection

def baseline_strategy(history, query):
    """
    BASELINE: No context management - accumulate everything
    Expected: Performance degrades as context grows
    """
    if not history:
        context = ""
    else:
        context = "\n\n---\n\n".join(history)

    prompt = f"""You are a helpful AI assistant. Answer the question based on the previous conversation.

Previous conversation:
{context}

New question: {query}

Answer:"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        # Return response AND context tokens
        return response['response'], count_tokens(context)
    except Exception as e:
        return f"Error: {e}", 0

def select_strategy(history, query, vector_store, step):
    """
    SELECT: RAG-based selection of relevant history
    Expected: Maintains performance, uses only relevant context
    """
    if not history:
        return baseline_strategy([], query)

    # Add current history to vector store (if not already added)
    try:
        # Only embed the latest history item
        if len(history) > 0:
            latest_idx = len(history) - 1
            embedding = embed_text(history[-1])
            if embedding:
                vector_store.add(
                    embeddings=[embedding],
                    documents=[history[-1]],
                    ids=[f"step_{step}_{latest_idx}"]
                )
    except Exception as e:
        print(f"  Error adding to vector store: {e}")

    # Retrieve top-3 relevant history items
    query_embedding = embed_text(query)
    if not query_embedding:
        return baseline_strategy(history, query)

    try:
        k = min(3, vector_store.count())
        if k == 0:
            return baseline_strategy([], query)

        results = vector_store.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        relevant_context = "\n\n---\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"  Error querying vector store: {e}")
        relevant_context = "\n\n---\n\n".join(history[-3:])  # Fallback to recent

    prompt = f"""You are a helpful AI assistant. Answer the question based on relevant previous context.

Relevant context:
{relevant_context}

Question: {query}

Answer:"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        # Return response AND context tokens
        return response['response'], count_tokens(relevant_context)
    except Exception as e:
        return f"Error: {e}", 0

def compress_strategy(history, query):
    """
    COMPRESS: Summarize old context, keep recent
    Expected: Bounded context size, some information loss
    """
    if len(history) <= MAX_HISTORY_ITEMS:
        return baseline_strategy(history, query)

    # Keep last 3, summarize the rest
    old_context = "\n\n---\n\n".join(history[:-3])

    summary_prompt = f"""Summarize the following conversation in 3-4 concise sentences, preserving key facts:

{old_context}

Summary:"""

    try:
        summary_response = ollama.generate(model=MODEL, prompt=summary_prompt)
        summary = summary_response['response']
    except Exception as e:
        print(f"  Error summarizing: {e}")
        summary = "Previous conversation covered machine learning topics."

    recent_context = "\n\n---\n\n".join(history[-3:])
    context = f"Summary of earlier conversation:\n{summary}\n\n---\n\nRecent conversation:\n{recent_context}"

    prompt = f"""You are a helpful AI assistant. Answer the question based on the context.

{context}

Question: {query}

Answer:"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        # Return response AND context tokens
        return response['response'], count_tokens(context)
    except Exception as e:
        return f"Error: {e}", 0

def write_strategy(history, query, scratchpad):
    """
    WRITE: Extract and store key facts in external memory
    Expected: Most efficient, compact representation
    """
    # Extract facts from new history items
    for i, item in enumerate(history):
        fact_key = f"fact_{i}"
        if fact_key not in scratchpad:
            # Extract key facts
            fact_prompt = f"""Extract 2-3 key facts from the following text as bullet points:

{item}

Key facts:"""

            try:
                fact_response = ollama.generate(model=MODEL, prompt=fact_prompt)
                facts = fact_response['response']
                scratchpad[fact_key] = facts
            except Exception as e:
                print(f"  Error extracting facts: {e}")
                scratchpad[fact_key] = f"- {item[:100]}..."

    # Use scratchpad facts as context
    if scratchpad:
        context = "\n\n".join(scratchpad.values())
    else:
        context = ""

    prompt = f"""You are a helpful AI assistant. Answer the question based on these key facts.

Key facts from previous conversation:
{context}

Question: {query}

Answer:"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        # Return response AND context tokens
        return response['response'], count_tokens(context)
    except Exception as e:
        return f"Error: {e}", 0

def run_experiment():
    """Main experiment runner"""
    print("=" * 60)
    print("EXPERIMENT 4: CONTEXT ENGINEERING STRATEGIES")
    print("=" * 60)

    results = []

    # Initialize storage for each strategy
    vector_store = setup_vector_store()
    scratchpad = {}

    strategies = {
        'baseline': {
            'func': lambda h, q, s: baseline_strategy(h, q),
            'desc': 'No context management (accumulate all)'
        },
        'select': {
            'func': lambda h, q, s: select_strategy(h, q, vector_store, s),
            'desc': 'RAG-based selection (top-3 relevant)'
        },
        'compress': {
            'func': lambda h, q, s: compress_strategy(h, q),
            'desc': 'Summarization (compress old, keep recent)'
        },
        'write': {
            'func': lambda h, q, s: write_strategy(h, q, scratchpad),
            'desc': 'External memory (extract key facts)'
        }
    }

    for strategy_name, strategy_info in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Running {strategy_name.upper()} Strategy")
        print(f"Description: {strategy_info['desc']}")
        print(f"{'=' * 60}")

        history = []
        strategy_func = strategy_info['func']

        # Reset strategy-specific storage
        if strategy_name == 'select':
            vector_store = setup_vector_store()
        elif strategy_name == 'write':
            scratchpad = {}

        for step, query in enumerate(QUERIES):
            print(f"\nStep {step + 1}/10: {query}")

            start_time = time.time()
            response, context_tokens = strategy_func(history, query, step)
            latency = time.time() - start_time

            # Add response to history
            history.append(response)

            # Calculate metrics - context_tokens is what we send to LLM, not accumulated
            results.append({
                'strategy': strategy_name,
                'step': step + 1,
                'query': query,
                'latency': latency,
                'context_tokens': context_tokens,  # Changed: tokens in context window
                'history_items': len(history),
                'response_length': len(response),
                'response_preview': response[:150] + "..." if len(response) > 150 else response
            })

            print(f"  Latency: {latency:.2f}s | Context Tokens: {context_tokens} | History: {len(history)} items")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'exp4_results.csv', index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'exp4_results.csv'}")

    # Generate visualizations
    visualize_results(df)

    # Print summary
    print_summary(df)

def visualize_results(df):
    """Create visualizations comparing strategies"""
    print("\nGenerating visualizations...")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Context Growth Over Time
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        axes[0].plot(strategy_data['step'], strategy_data['context_tokens'],
                    marker='o', label=strategy.upper(), linewidth=2)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Context Window Tokens')
    axes[0].set_title('Context Window Size Over Time')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Latency Over Time
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        axes[1].plot(strategy_data['step'], strategy_data['latency'],
                    marker='s', label=strategy.upper(), linewidth=2)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Latency (seconds)')
    axes[1].set_title('Latency Over Time')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Final Comparison (Step 10)
    final_data = df[df['step'] == 10]
    strategies = final_data['strategy'].tolist()
    tokens = final_data['context_tokens'].tolist()
    latencies = final_data['latency'].tolist()

    x = range(len(strategies))
    width = 0.35

    bars1 = axes[2].bar([i - width/2 for i in x], tokens, width, label='Context Tokens', alpha=0.8)
    ax2 = axes[2].twinx()
    bars2 = ax2.bar([i + width/2 for i in x], latencies, width, label='Latency (s)', alpha=0.8, color='orange')

    axes[2].set_xlabel('Strategy')
    axes[2].set_ylabel('Context Window Tokens', color='tab:blue')
    axes[2].set_title('Final Comparison (Step 10)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([s.upper() for s in strategies])
    axes[2].tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_ylabel('Latency (s)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)

    axes[2].legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp4_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved to {RESULTS_DIR / 'exp4_comparison.png'}")

    # Individual chart: Context growth
    plt.figure(figsize=(10, 6))
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        plt.plot(strategy_data['step'], strategy_data['context_tokens'],
                marker='o', label=strategy.upper(), linewidth=2.5, markersize=8)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Context Window Tokens', fontsize=12)
    plt.title('Context Window Growth: Strategy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp4_context_growth.png', dpi=300, bbox_inches='tight')
    print(f"✓ Context growth chart saved to {RESULTS_DIR / 'exp4_context_growth.png'}")

def print_summary(df):
    """Print experiment summary"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Final metrics at step 10
    final_data = df[df['step'] == 10]

    for strategy in ['baseline', 'select', 'compress', 'write']:
        strategy_data = final_data[final_data['strategy'] == strategy]
        if not strategy_data.empty:
            row = strategy_data.iloc[0]
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Context Window Tokens: {row['context_tokens']}")
            print(f"  Final Latency: {row['latency']:.2f}s")
            print(f"  History Items: {row['history_items']}")

    # Calculate efficiency metrics
    baseline_tokens = final_data[final_data['strategy'] == 'baseline']['context_tokens'].values[0]

    print(f"\n{'=' * 60}")
    print("EFFICIENCY COMPARISON (vs BASELINE)")
    print(f"{'=' * 60}")

    for strategy in ['select', 'compress', 'write']:
        strategy_tokens = final_data[final_data['strategy'] == strategy]['context_tokens'].values[0]
        reduction = (1 - strategy_tokens / baseline_tokens) * 100
        print(f"{strategy.upper()}: {reduction:.1f}% token reduction")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Check if models are available
    print("Checking models...")
    try:
        ollama.list()
        print("✓ Ollama is running")
    except:
        print("✗ Error: Ollama is not running. Please start Ollama first.")
        exit(1)

    # Run experiment
    run_experiment()

    print("\n✓ Experiment 4 completed successfully!")
