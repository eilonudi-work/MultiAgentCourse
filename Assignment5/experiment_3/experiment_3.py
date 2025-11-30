"""
Experiment 3: RAG Impact
Comparing full context vs RAG retrieval for Hebrew document QA
"""

import ollama
import chromadb
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data/documents")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "mistral:7b"  # Using same model as other experiments
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 3  # Number of documents to retrieve in RAG mode

# Test queries in Hebrew
QUERIES = [
    {
        "query": "מהן תופעות הלוואי של תרופה X?",
        "type": "medicine",
        "expected_topic": "medicine"
    },
    {
        "query": "מהם היתרונות של בינה מלאכותית?",
        "type": "technology",
        "expected_topic": "technology"
    },
    {
        "query": "מהן החובות החוקיות של מעסיק?",
        "type": "law",
        "expected_topic": "law"
    }
]

def load_documents():
    """Load all documents from the data directory"""
    documents = []
    doc_files = sorted(DATA_DIR.glob("*.txt"))

    for doc_file in doc_files:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': doc_file.name,
                'content': content,
                'topic': doc_file.name.split('_')[-1].replace('.txt', '')
            })

    print(f"Loaded {len(documents)} documents")
    return documents

def count_tokens(text):
    """Estimate token count (rough approximation for Hebrew)"""
    # Hebrew words are typically 1-2 tokens
    words = len(text.split())
    return int(words * 1.5)

def embed_text(text):
    """Generate embeddings using nomic-embed-text"""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def setup_vector_store(documents):
    """Initialize ChromaDB and populate with documents"""
    print("Setting up vector store...")

    # Create a new client (in-memory)
    client = chromadb.Client()

    # Create collection
    collection = client.create_collection(
        name="hebrew_docs",
        metadata={"description": "Hebrew documents for RAG experiment"}
    )

    # Add documents with embeddings
    for i, doc in enumerate(tqdm(documents, desc="Embedding documents")):
        embedding = embed_text(doc['content'])
        if embedding:
            collection.add(
                embeddings=[embedding],
                documents=[doc['content']],
                metadatas=[{"filename": doc['filename'], "topic": doc['topic']}],
                ids=[f"doc_{i}"]
            )

    print(f"Vector store ready with {collection.count()} documents")
    return collection

def query_with_full_context(documents, query):
    """Query LLM with all documents in context"""
    # Concatenate all documents
    full_context = "\n\n---\n\n".join([doc['content'] for doc in documents])

    prompt = f"""אתה עוזר מועיל. ענה על השאלה הבאה בהתבסס על המסמכים המצורפים.

מסמכים:
{full_context}

שאלה: {query}

תשובה (בעברית):"""

    start_time = time.time()

    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt
        )
        latency = time.time() - start_time

        return {
            'answer': response['response'],
            'latency': latency,
            'tokens': count_tokens(full_context),
            'num_docs': len(documents)
        }
    except Exception as e:
        print(f"Error in full context query: {e}")
        return {
            'answer': f"Error: {e}",
            'latency': 0,
            'tokens': 0,
            'num_docs': 0
        }

def query_with_rag(collection, query, k=TOP_K):
    """Query LLM with RAG (retrieve top-k relevant docs)"""
    # Get query embedding
    query_embedding = embed_text(query)

    if not query_embedding:
        return {
            'answer': "Error: Could not generate query embedding",
            'latency': 0,
            'tokens': 0,
            'num_docs': 0,
            'retrieved_topics': []
        }

    # Retrieve similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    relevant_docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    retrieved_topics = [m['topic'] for m in metadatas]

    # Build context from retrieved docs
    context = "\n\n---\n\n".join(relevant_docs)

    prompt = f"""אתה עוזר מועיל. ענה על השאלה הבאה בהתבסס על המסמכים המצורפים.

מסמכים רלוונטיים:
{context}

שאלה: {query}

תשובה (בעברית):"""

    start_time = time.time()

    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt
        )
        latency = time.time() - start_time

        return {
            'answer': response['response'],
            'latency': latency,
            'tokens': count_tokens(context),
            'num_docs': k,
            'retrieved_topics': retrieved_topics
        }
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return {
            'answer': f"Error: {e}",
            'latency': 0,
            'tokens': 0,
            'num_docs': 0,
            'retrieved_topics': []
        }

def evaluate_retrieval_quality(retrieved_topics, expected_topic):
    """Check if retrieved documents match expected topic"""
    relevant_count = sum(1 for topic in retrieved_topics if topic == expected_topic)
    return relevant_count / len(retrieved_topics) if retrieved_topics else 0

def run_experiment():
    """Main experiment runner"""
    print("=" * 60)
    print("EXPERIMENT 3: RAG IMPACT")
    print("=" * 60)

    # Load documents
    documents = load_documents()

    # Setup vector store for RAG
    collection = setup_vector_store(documents)

    print("\n" + "=" * 60)
    print("RUNNING QUERIES")
    print("=" * 60)

    results = []

    # Run each query multiple times
    for query_info in QUERIES:
        query = query_info['query']
        query_type = query_info['type']
        expected_topic = query_info['expected_topic']

        print(f"\nQuery Type: {query_type}")
        print(f"Query: {query}")
        print("-" * 60)

        # Run 5 trials per query
        for trial in range(5):
            print(f"\nTrial {trial + 1}/5")

            # Full Context Mode
            print("  Running Full Context mode...")
            full_result = query_with_full_context(documents, query)

            # RAG Mode
            print("  Running RAG mode...")
            rag_result = query_with_rag(collection, query, k=TOP_K)

            # Evaluate retrieval quality
            retrieval_quality = evaluate_retrieval_quality(
                rag_result.get('retrieved_topics', []),
                expected_topic
            )

            results.append({
                'query_type': query_type,
                'trial': trial,
                'full_latency': full_result['latency'],
                'rag_latency': rag_result['latency'],
                'full_tokens': full_result['tokens'],
                'rag_tokens': rag_result['tokens'],
                'full_num_docs': full_result['num_docs'],
                'rag_num_docs': rag_result['num_docs'],
                'retrieval_quality': retrieval_quality,
                'retrieved_topics': ','.join(rag_result.get('retrieved_topics', [])),
                'full_answer': full_result['answer'][:200],  # First 200 chars
                'rag_answer': rag_result['answer'][:200]
            })

            print(f"  Full Context: {full_result['latency']:.2f}s, {full_result['tokens']} tokens")
            print(f"  RAG: {rag_result['latency']:.2f}s, {rag_result['tokens']} tokens")
            print(f"  Retrieval Quality: {retrieval_quality:.1%}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'exp3_results.csv', index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'exp3_results.csv'}")

    # Generate visualizations
    visualize_results(df)

    # Print summary
    print_summary(df)

def visualize_results(df):
    """Create visualizations comparing Full Context vs RAG"""
    print("\nGenerating visualizations...")

    # Calculate averages
    avg_full_latency = df['full_latency'].mean()
    avg_rag_latency = df['rag_latency'].mean()
    avg_full_tokens = df['full_tokens'].mean()
    avg_rag_tokens = df['rag_tokens'].mean()
    avg_retrieval_quality = df['retrieval_quality'].mean()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Latency Comparison
    axes[0].bar(['Full Context', 'RAG'],
                [avg_full_latency, avg_rag_latency],
                color=['#ff6b6b', '#4ecdc4'])
    axes[0].set_ylabel('Average Latency (seconds)')
    axes[0].set_title('Response Latency Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, v in enumerate([avg_full_latency, avg_rag_latency]):
        axes[0].text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')

    # 2. Token Usage Comparison
    axes[1].bar(['Full Context', 'RAG'],
                [avg_full_tokens, avg_rag_tokens],
                color=['#ff6b6b', '#4ecdc4'])
    axes[1].set_ylabel('Average Tokens')
    axes[1].set_title('Token Usage Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, v in enumerate([avg_full_tokens, avg_rag_tokens]):
        axes[1].text(i, v + 100, f'{int(v)}', ha='center', va='bottom')

    # 3. Retrieval Quality
    axes[2].bar(['Retrieval\nQuality'],
                [avg_retrieval_quality * 100],
                color=['#95e1d3'])
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title('RAG Retrieval Quality\n(Relevant Docs Retrieved)')
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].text(0, avg_retrieval_quality * 100 + 2,
                 f'{avg_retrieval_quality:.1%}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp3_combined_charts.png', dpi=300, bbox_inches='tight')
    print(f"✓ Charts saved to {RESULTS_DIR / 'exp3_combined_charts.png'}")

    # Individual charts
    # Latency by query type
    fig, ax = plt.subplots(figsize=(10, 6))
    query_types = df['query_type'].unique()
    x = range(len(query_types))
    width = 0.35

    full_latencies = [df[df['query_type'] == qt]['full_latency'].mean() for qt in query_types]
    rag_latencies = [df[df['query_type'] == qt]['rag_latency'].mean() for qt in query_types]

    ax.bar([i - width/2 for i in x], full_latencies, width, label='Full Context', color='#ff6b6b')
    ax.bar([i + width/2 for i in x], rag_latencies, width, label='RAG', color='#4ecdc4')

    ax.set_xlabel('Query Type')
    ax.set_ylabel('Average Latency (seconds)')
    ax.set_title('Latency Comparison by Query Type')
    ax.set_xticks(x)
    ax.set_xticklabels(query_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp3_latency_by_type.png', dpi=300, bbox_inches='tight')
    print(f"✓ Latency chart saved to {RESULTS_DIR / 'exp3_latency_by_type.png'}")

def print_summary(df):
    """Print experiment summary"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nFull Context Mode:")
    print(f"  Average Latency: {df['full_latency'].mean():.2f}s")
    print(f"  Average Tokens: {df['full_tokens'].mean():.0f}")
    print(f"  Documents Used: {df['full_num_docs'].iloc[0]}")

    print("\nRAG Mode:")
    print(f"  Average Latency: {df['rag_latency'].mean():.2f}s")
    print(f"  Average Tokens: {df['rag_tokens'].mean():.0f}")
    print(f"  Documents Used: {df['rag_num_docs'].iloc[0]}")
    print(f"  Retrieval Quality: {df['retrieval_quality'].mean():.1%}")

    print("\nSpeedup and Efficiency:")
    speedup = df['full_latency'].mean() / df['rag_latency'].mean()
    token_reduction = (1 - df['rag_tokens'].mean() / df['full_tokens'].mean()) * 100

    print(f"  RAG Speedup: {speedup:.2f}x faster")
    print(f"  Token Reduction: {token_reduction:.1f}%")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Check if model is available
    print("Checking models...")
    try:
        ollama.list()
        print("✓ Ollama is running")
    except:
        print("✗ Error: Ollama is not running. Please start Ollama first.")
        exit(1)

    # Run experiment
    run_experiment()

    print("\n✓ Experiment 3 completed successfully!")
