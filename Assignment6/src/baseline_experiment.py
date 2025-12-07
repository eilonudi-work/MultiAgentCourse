"""
Baseline experiment for sentiment analysis using simple prompts.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from ollama_client import OllamaClient
from metrics import SentimentMetrics


class BaselineExperiment:
    """Run baseline sentiment analysis experiment."""

    def __init__(self, dataset_path: str, model_name: str = None):
        """
        Initialize baseline experiment.

        Args:
            dataset_path: Path to sentiment dataset JSON
            model_name: Optional model name override
        """
        self.dataset_path = dataset_path
        self.client = OllamaClient(model_name)
        self.metrics_calculator = SentimentMetrics()
        self.results = []

        # Baseline prompt template
        self.prompt_template = "Classify the sentiment of this text as 'positive' or 'negative': {text}"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file.

        Returns:
            List of dataset items
        """
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"✓ Loaded {len(dataset)} examples from {self.dataset_path}")
        return dataset

    def run_experiment(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the baseline experiment on the entire dataset.

        Args:
            save_results: Whether to save results to file

        Returns:
            Metrics dictionary
        """
        print("\n=== Running Baseline Experiment ===")
        print(f"Model: {self.client.model_name}")
        print(f"Prompt: {self.prompt_template}")
        print()

        # Check Ollama connection
        if not self.client.check_connection():
            raise ConnectionError("Cannot connect to Ollama. Please make sure it's running.")

        # Load dataset
        dataset = self.load_dataset()

        # Process each example
        self.results = []
        for item in tqdm(dataset, desc="Processing examples"):
            text = item["text"]
            ground_truth = item["ground_truth"]
            category = item.get("category", "unknown")

            # Get prediction
            result = self.client.classify_sentiment(text, self.prompt_template)

            # Store result
            result_entry = {
                "text": text,
                "ground_truth": ground_truth,
                "predicted": result.get("sentiment", "error"),
                "success": result.get("success", False),
                "category": category,
                "response": result.get("response", ""),
                "error": result.get("error"),
                "total_duration": result.get("total_duration"),
                "eval_count": result.get("eval_count")
            }
            self.results.append(result_entry)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(self.results)

        # Print summary
        self._print_summary(metrics)

        # Save results if requested
        if save_results:
            self._save_results(metrics)

        return metrics

    def _print_summary(self, metrics: Dict[str, Any]):
        """Print experiment summary."""
        print("\n=== Experiment Summary ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Successful: {metrics['successful_samples']}")
        print(f"Failed: {metrics['failed_samples']}")
        print(f"Success rate: {metrics['success_rate']:.1%}")
        print()
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"Mean distance: {metrics['mean_distance']:.4f}")
        print(f"Std distance: {metrics['std_distance']:.4f}")
        print()

        # Confusion matrix
        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        print(f"  True Positive:  {cm['true_positive']}")
        print(f"  True Negative:  {cm['true_negative']}")
        print(f"  False Positive: {cm['false_positive']}")
        print(f"  False Negative: {cm['false_negative']}")
        print()
        print(f"  Precision: {cm['precision']:.1%}")
        print(f"  Recall:    {cm['recall']:.1%}")
        print(f"  F1 Score:  {cm['f1_score']:.1%}")
        print()

        # Per-category stats
        if metrics.get('category_stats'):
            print("Per-Category Accuracy:")
            for category, stats in metrics['category_stats'].items():
                print(f"  {category:15s}: {stats['accuracy']:.1%} ({stats['count']} samples)")

        print("=" * 30)

    def _save_results(self, metrics: Dict[str, Any]):
        """Save results to JSON files."""
        # Create results directory if needed
        os.makedirs("results", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = f"results/baseline_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment": "baseline",
                "timestamp": timestamp,
                "model": self.client.model_name,
                "prompt_template": self.prompt_template,
                "results": self.results
            }, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")

        # Save metrics summary
        metrics_file = f"results/baseline_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "experiment": "baseline",
                "timestamp": timestamp,
                "model": self.client.model_name,
                "metrics": metrics
            }, f, indent=2)
        print(f"✓ Metrics saved to {metrics_file}")

    def show_examples(self, n: int = 5, show_errors: bool = False):
        """
        Show example results.

        Args:
            n: Number of examples to show
            show_errors: Whether to show error cases
        """
        print(f"\n=== Example Results (showing {n}) ===\n")

        examples = self.results[:n] if not show_errors else [
            r for r in self.results if r['predicted'] != r['ground_truth']
        ][:n]

        for i, result in enumerate(examples, 1):
            correct = "✓" if result['predicted'] == result['ground_truth'] else "✗"
            print(f"{i}. {correct} Text: {result['text'][:60]}...")
            print(f"   Ground Truth: {result['ground_truth']}")
            print(f"   Predicted:    {result['predicted']}")
            if result['predicted'] != result['ground_truth']:
                print(f"   Response:     {result['response'][:80]}...")
            print()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline sentiment analysis experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sentiment_dataset.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (overrides config)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show error cases after experiment"
    )

    args = parser.parse_args()

    # Run experiment
    experiment = BaselineExperiment(args.dataset, args.model)
    metrics = experiment.run_experiment(save_results=not args.no_save)

    # Show examples
    if args.show_errors:
        experiment.show_examples(n=10, show_errors=True)
    else:
        experiment.show_examples(n=5, show_errors=False)


if __name__ == "__main__":
    main()
