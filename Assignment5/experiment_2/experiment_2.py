"""
Experiment 2: Context Window Size Impact
Demonstrates how accuracy degrades and latency increases as context window grows

This experiment varies the number of documents in the context and measures:
- Accuracy: Can the LLM find the fact embedded in the middle?
- Latency: How long does processing take?
- Token Count: How many tokens are in the context?
"""

import ollama
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import tiktoken
import os
from typing import Literal

# Configuration
MODEL = "mistral:7b"
DOC_COUNTS = [2, 5, 10, 20, 50]  # Varying numbers of documents to test
TRIALS_PER_SIZE = 5  # Number of trials per document count
FACT = "The CEO of the company is David Cohen"
QUESTION = "Who is the CEO of the company?"
EXPECTED_ANSWER = "David Cohen"
DOCUMENT_WORDS = 500  # Words per document (reduced from 2000 to allow model to find facts)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def generate_filler_text(num_words: int) -> str:
    """
    Generate simple filler text to pad the document.
    Uses varied business-related sentences to create realistic context.

    Reused from Experiment 1.
    """
    sentences = [
        "The company has been operating in the technology sector for many years.",
        "Our quarterly earnings report shows significant growth in revenue.",
        "The board of directors met last week to discuss strategic initiatives.",
        "Employee satisfaction scores have improved significantly this year.",
        "We are committed to sustainable business practices and environmental responsibility.",
        "The marketing team launched a successful campaign last quarter.",
        "Product development continues to innovate with new features and capabilities.",
        "Customer feedback has been overwhelmingly positive across all demographics.",
        "Supply chain optimization remains a key priority for operational efficiency.",
        "The company culture emphasizes collaboration, creativity, and continuous improvement.",
        "International expansion plans are currently under review by senior management.",
        "Technology infrastructure investments have modernized our operations significantly.",
        "Research and development spending increased by fifteen percent this fiscal year.",
        "Partnership agreements with major vendors strengthen our market position substantially.",
        "Training programs for employees have been enhanced with digital learning platforms.",
        "Financial projections indicate sustained growth for the foreseeable future ahead.",
        "Competitive analysis shows we maintain strong advantages in key market segments.",
        "Brand recognition has grown substantially in our target demographic groups.",
        "Quality assurance processes ensure excellence in all products and services delivered.",
        "Corporate social responsibility initiatives support communities where we operate actively.",
    ]

    text = []
    words_generated = 0

    while words_generated < num_words:
        sentence = random.choice(sentences)
        text.append(sentence)
        words_generated += len(sentence.split())

    return " ".join(text)


def embed_fact(filler_text: str, fact: str, position: Literal["start", "middle", "end"]) -> str:
    """
    Embed the critical fact at the specified position in the document.

    Reused from Experiment 1.

    Args:
        filler_text: The filler text to use
        fact: The fact to embed
        position: Where to place the fact (start, middle, or end)

    Returns:
        Complete document with fact embedded
    """
    words = filler_text.split()
    total_words = len(words)

    if position == "start":
        return f"{fact} {filler_text}"
    elif position == "middle":
        midpoint = total_words // 2
        first_half = " ".join(words[:midpoint])
        second_half = " ".join(words[midpoint:])
        return f"{first_half} {fact} {second_half}"
    else:  # end
        return f"{filler_text} {fact}"


def query_llm(document: str, question: str) -> str:
    """
    Query the LLM with the document and question.

    Reused from Experiment 1.

    Args:
        document: The document containing the fact
        question: The question to ask

    Returns:
        The LLM's response
    """
    prompt = f"""Based on the following document, please answer the question.

Document:
{document}

Question: {question}

Answer:"""

    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low temperature for more consistent answers
                "num_predict": 50,   # Limit response length
            }
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return ""


def check_accuracy(response: str, expected: str) -> bool:
    """
    Check if the response contains the expected answer.
    Uses simple substring matching (case-insensitive).

    Reused from Experiment 1.

    Args:
        response: The LLM's response
        expected: The expected answer

    Returns:
        True if the answer is correct, False otherwise
    """
    response_lower = response.lower()
    expected_lower = expected.lower()
    return expected_lower in response_lower


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.
    Uses the cl100k_base encoding (GPT-4 tokenizer).

    Args:
        text: The text to count tokens for

    Returns:
        Number of tokens in the text
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def create_multi_doc_context(num_docs: int) -> str:
    """
    Generate N documents and embed the fact in the MIDDLE document.
    Documents are separated by a clear separator.

    Args:
        num_docs: Number of documents to generate

    Returns:
        Concatenated context with fact embedded in middle document
    """
    docs = []

    # Generate all documents
    for i in range(num_docs):
        doc = generate_filler_text(DOCUMENT_WORDS)
        docs.append(doc)

    # Embed fact in the middle document
    middle_idx = len(docs) // 2
    docs[middle_idx] = embed_fact(docs[middle_idx], FACT, 'middle')

    # Join documents with separator
    return "\n\n---\n\n".join(docs)


def run_experiment():
    """
    Main experiment loop.
    Tests varying document counts and measures accuracy, latency, and token count.
    """
    print(f"Starting Context Window Size Impact Experiment")
    print(f"Model: {MODEL}")
    print(f"Document counts to test: {DOC_COUNTS}")
    print(f"Trials per size: {TRIALS_PER_SIZE}")
    print(f"Words per document: {DOCUMENT_WORDS}")
    print("=" * 60)

    results = []

    for num_docs in DOC_COUNTS:
        print(f"\nTesting with {num_docs} documents")

        for trial in range(TRIALS_PER_SIZE):
            # Create multi-document context
            context = create_multi_doc_context(num_docs)

            # Count tokens
            token_count = count_tokens(context)

            # Measure latency
            start_time = time.time()
            response = query_llm(context, QUESTION)
            latency = time.time() - start_time

            # Check accuracy
            is_correct = check_accuracy(response, EXPECTED_ANSWER)

            # Store results
            results.append({
                'num_docs': num_docs,
                'trial': trial + 1,
                'tokens': token_count,
                'latency': latency,
                'correct': is_correct,
                'response': response
            })

            # Progress indicator
            status = "✓" if is_correct else "✗"
            print(f"  Trial {trial + 1}/{TRIALS_PER_SIZE}: {status} | "
                  f"Tokens: {token_count:,} | Latency: {latency:.2f}s")

        # Summary for this document count
        trials_for_size = [r for r in results if r['num_docs'] == num_docs]
        correct_count = sum(1 for r in trials_for_size if r['correct'])
        avg_latency = sum(r['latency'] for r in trials_for_size) / len(trials_for_size)
        avg_tokens = sum(r['tokens'] for r in trials_for_size) / len(trials_for_size)
        accuracy = (correct_count / TRIALS_PER_SIZE) * 100

        print(f"  Summary: Accuracy={accuracy:.1f}% | "
              f"Avg Latency={avg_latency:.2f}s | "
              f"Avg Tokens={avg_tokens:,.0f}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    csv_path = "results/exp2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # Generate visualizations
    plot_results(df)

    return df


def plot_results(df: pd.DataFrame):
    """
    Create and save three line plots:
    1. Accuracy vs Number of Documents
    2. Latency vs Number of Documents
    3. Token Count vs Number of Documents
    """

    # Calculate aggregated metrics by document count
    metrics = df.groupby('num_docs').agg({
        'correct': 'mean',  # Accuracy (as fraction)
        'latency': 'mean',   # Average latency
        'tokens': 'mean'     # Average token count
    }).reset_index()

    # Also calculate standard deviations for error bars
    std_metrics = df.groupby('num_docs').agg({
        'correct': 'std',
        'latency': 'std',
        'tokens': 'std'
    }).reset_index()

    # Convert accuracy to percentage
    metrics['accuracy'] = metrics['correct'] * 100
    std_metrics['accuracy_std'] = std_metrics['correct'] * 100

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Accuracy vs Number of Documents ---
    ax1 = axes[0]
    ax1.plot(metrics['num_docs'], metrics['accuracy'],
             marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.errorbar(metrics['num_docs'], metrics['accuracy'],
                 yerr=std_metrics['accuracy_std'],
                 fmt='none', ecolor='#e74c3c', alpha=0.3, capsize=5)
    ax1.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Degradation vs Context Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, 105)
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # --- Plot 2: Latency vs Number of Documents ---
    ax2 = axes[1]
    ax2.plot(metrics['num_docs'], metrics['latency'],
             marker='s', linewidth=2, markersize=8, color='#3498db')
    ax2.errorbar(metrics['num_docs'], metrics['latency'],
                 yerr=std_metrics['latency'],
                 fmt='none', ecolor='#3498db', alpha=0.3, capsize=5)
    ax2.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency Growth vs Context Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # --- Plot 3: Token Count vs Number of Documents ---
    ax3 = axes[2]
    ax3.plot(metrics['num_docs'], metrics['tokens'],
             marker='^', linewidth=2, markersize=8, color='#2ecc71')
    ax3.errorbar(metrics['num_docs'], metrics['tokens'],
                 yerr=std_metrics['tokens'],
                 fmt='none', ecolor='#2ecc71', alpha=0.3, capsize=5)
    ax3.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Token Count', fontsize=12, fontweight='bold')
    ax3.set_title('Token Growth vs Number of Documents', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Format y-axis for token count with commas
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Save combined chart
    combined_path = "results/exp2_combined_charts.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined charts saved to {combined_path}")

    # Save individual charts
    # Chart 1: Accuracy
    fig1, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics['num_docs'], metrics['accuracy'],
            marker='o', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.errorbar(metrics['num_docs'], metrics['accuracy'],
                yerr=std_metrics['accuracy_std'],
                fmt='none', ecolor='#e74c3c', alpha=0.3, capsize=5)
    ax.set_xlabel('Number of Documents', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Accuracy Degradation with Growing Context\nModel: {MODEL}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    accuracy_path = "results/exp2_accuracy_chart.png"
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy chart saved to {accuracy_path}")
    plt.close()

    # Chart 2: Latency
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics['num_docs'], metrics['latency'],
            marker='s', linewidth=2.5, markersize=10, color='#3498db')
    ax.errorbar(metrics['num_docs'], metrics['latency'],
                yerr=std_metrics['latency'],
                fmt='none', ecolor='#3498db', alpha=0.3, capsize=5)
    ax.set_xlabel('Number of Documents', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'Query Latency Growth with Context Size\nModel: {MODEL}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    latency_path = "results/exp2_latency_chart.png"
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    print(f"✓ Latency chart saved to {latency_path}")
    plt.close()

    # Chart 3: Token Count
    fig3, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics['num_docs'], metrics['tokens'],
            marker='^', linewidth=2.5, markersize=10, color='#2ecc71')
    ax.errorbar(metrics['num_docs'], metrics['tokens'],
                yerr=std_metrics['tokens'],
                fmt='none', ecolor='#2ecc71', alpha=0.3, capsize=5)
    ax.set_xlabel('Number of Documents', fontsize=13, fontweight='bold')
    ax.set_ylabel('Token Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Token Count Scaling with Document Count\nModel: {MODEL}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout()
    tokens_path = "results/exp2_tokens_chart.png"
    plt.savefig(tokens_path, dpi=300, bbox_inches='tight')
    print(f"✓ Token count chart saved to {tokens_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Docs':>5} | {'Accuracy':>9} | {'Latency':>12} | {'Tokens':>10}")
    print("-" * 60)
    for _, row in metrics.iterrows():
        print(f"{int(row['num_docs']):5d} | {row['accuracy']:8.1f}% | "
              f"{row['latency']:10.2f}s | {int(row['tokens']):10,d}")
    print("=" * 60)


if __name__ == "__main__":
    # Run the experiment
    df = run_experiment()

    print("\nExperiment complete!")
    print("Check the results/ directory for output files:")
    print("  - exp2_results.csv (raw data)")
    print("  - exp2_accuracy_chart.png (accuracy degradation)")
    print("  - exp2_latency_chart.png (latency growth)")
    print("  - exp2_tokens_chart.png (token count growth)")
    print("  - exp2_combined_charts.png (all three metrics)")
