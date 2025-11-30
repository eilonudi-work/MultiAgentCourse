"""
Experiment 1: Needle in Haystack
Demonstrates the "Lost in the Middle" phenomenon in LLMs

This experiment embeds a critical fact at different positions in a document
and measures the LLM's ability to retrieve it, showing that facts in the
middle of long contexts are often missed.
"""

import ollama
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from typing import Literal

# Configuration
MODEL = "mistral:7b"  # 7B parameter model known to demonstrate U-shaped "Lost in the Middle" effect
FACT = "The CEO of the company is David Cohen"
QUESTION = "Who is the CEO of the company?"
EXPECTED_ANSWER = "David Cohen"
TRIALS_PER_POSITION = 10
DOCUMENT_WORDS = 2000  # Target word count for documents (optimal for demonstrating U-shaped phenomenon with mistral)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def generate_filler_text(num_words: int) -> str:
    """
    Generate simple filler text to pad the document.
    Uses varied business-related sentences to create realistic context.
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
        # Insert fact at the very beginning
        return f"{fact} {filler_text}"
    elif position == "middle":
        # Insert fact at the midpoint
        midpoint = total_words // 2
        first_half = " ".join(words[:midpoint])
        second_half = " ".join(words[midpoint:])
        return f"{first_half} {fact} {second_half}"
    else:  # end
        # Insert fact at the very end
        return f"{filler_text} {fact}"


def query_llm(document: str, question: str) -> str:
    """
    Query the LLM with the document and question.

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

    Args:
        response: The LLM's response
        expected: The expected answer

    Returns:
        True if the answer is correct, False otherwise
    """
    response_lower = response.lower()
    expected_lower = expected.lower()

    # Check if the expected name appears in the response
    return expected_lower in response_lower


def run_experiment():
    """
    Main experiment loop.
    Runs trials for each position and collects results.
    """
    print(f"Starting Needle in Haystack Experiment")
    print(f"Model: {MODEL}")
    print(f"Trials per position: {TRIALS_PER_POSITION}")
    print(f"Document size: ~{DOCUMENT_WORDS} words")
    print("=" * 60)

    positions = ['start', 'middle', 'end']
    results = []

    for position in positions:
        print(f"\nTesting position: {position.upper()}")
        correct_count = 0

        for run in range(TRIALS_PER_POSITION):
            # Generate fresh filler text for each trial
            filler = generate_filler_text(DOCUMENT_WORDS)
            document = embed_fact(filler, FACT, position)

            # Query the LLM
            response = query_llm(document, QUESTION)

            # Check accuracy
            is_correct = check_accuracy(response, EXPECTED_ANSWER)
            if is_correct:
                correct_count += 1

            # Store results
            results.append({
                'position': position,
                'run': run + 1,
                'correct': is_correct,
                'response': response
            })

            # Progress indicator
            status = "✓" if is_correct else "✗"
            print(f"  Trial {run + 1}/10: {status}")

        accuracy = (correct_count / TRIALS_PER_POSITION) * 100
        print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{TRIALS_PER_POSITION})")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    csv_path = "results/exp1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # Generate visualization
    plot_results(df)

    return df


def plot_results(df: pd.DataFrame):
    """
    Create and save visualization of results.
    Shows accuracy by position as a bar chart.
    """
    # Calculate accuracy for each position
    accuracy_by_position = df.groupby('position')['correct'].mean() * 100

    # Ensure correct order
    positions = ['start', 'middle', 'end']
    accuracy_values = [accuracy_by_position[pos] for pos in positions]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(positions, accuracy_values, color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.8)

    # Customize chart
    plt.xlabel('Fact Position in Document', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Needle in Haystack: Lost in the Middle Phenomenon\n' +
              f'Model: {MODEL}, Document Size: ~{DOCUMENT_WORDS} words',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, accuracy_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add horizontal line at 100%
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    # Save chart
    chart_path = "results/exp1_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {chart_path}")

    # Display summary statistics
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for pos in positions:
        acc = accuracy_by_position[pos]
        print(f"{pos.upper():8} position: {acc:5.1f}% accuracy")
    print("=" * 60)


if __name__ == "__main__":
    # Run the experiment
    df = run_experiment()

    print("\nExperiment complete!")
    print("Check the results/ directory for output files.")
