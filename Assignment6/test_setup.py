#!/usr/bin/env python3
"""
Quick test script to verify the setup is working.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from ollama_client import OllamaClient


def main():
    """Test the setup."""
    print("=" * 50)
    print("Testing Prompt Engineering Assignment Setup")
    print("=" * 50)
    print()

    # 1. Test configuration
    print("1. Configuration:")
    Config.display_config()
    print()

    # 2. Test Ollama connection
    print("2. Testing Ollama connection...")
    client = OllamaClient()

    if not client.check_connection():
        print("✗ FAILED: Cannot connect to Ollama")
        print("\nPlease make sure:")
        print("  1. Ollama is installed")
        print("  2. Ollama is running (try: ollama serve)")
        print("  3. You have pulled a model (try: ollama pull llama2)")
        return False

    print("✓ Ollama connection successful!")
    print()

    # 3. Test sentiment classification
    print("3. Testing sentiment classification...")
    test_cases = [
        ("This is amazing!", "positive"),
        ("This is terrible!", "negative"),
    ]

    success_count = 0
    for text, expected in test_cases:
        result = client.classify_sentiment(
            text,
            "Classify the sentiment as 'positive' or 'negative': {text}"
        )

        if result["success"]:
            predicted = result["sentiment"]
            status = "✓" if predicted == expected else "✗"
            print(f"{status} '{text}' -> {predicted} (expected: {expected})")
            if predicted == expected:
                success_count += 1
        else:
            print(f"✗ '{text}' -> ERROR: {result['error']}")

    print()

    # 4. Test dataset
    print("4. Checking dataset...")
    dataset_path = "data/sentiment_dataset.json"
    if os.path.exists(dataset_path):
        import json
        with open(dataset_path) as f:
            data = json.load(f)
        print(f"✓ Dataset found with {len(data)} examples")
    else:
        print(f"✗ Dataset not found at {dataset_path}")
        return False

    print()

    # Final summary
    print("=" * 50)
    if success_count >= 1:
        print("✓ Setup test PASSED!")
        print("\nYou're ready to run experiments:")
        print("  python src/baseline_experiment.py")
    else:
        print("✗ Setup test FAILED")
        print("\nPlease check the errors above and try again.")
    print("=" * 50)

    return success_count >= 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
