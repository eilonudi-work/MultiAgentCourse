"""
Batch experiment runner for all prompt variations.
Runs baseline and all improved prompts, compares results.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from ollama_client import OllamaClient
from metrics import SentimentMetrics
from improved_prompts import PromptVariations


class BatchExperimentRunner:
    """Run experiments across all prompt variations and compare results."""

    def __init__(self, dataset_path: str, model_name: str = None):
        """
        Initialize batch experiment runner.

        Args:
            dataset_path: Path to sentiment dataset JSON
            model_name: Optional model name override
        """
        # Resolve dataset path relative to project root
        if not os.path.isabs(dataset_path):
            # Get project root (parent of src directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            dataset_path = os.path.join(project_root, dataset_path)

        self.dataset_path = dataset_path
        self.client = OllamaClient(model_name)
        self.metrics_calculator = SentimentMetrics()
        self.all_results = {}
        self.all_metrics = {}

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"✓ Loaded {len(dataset)} examples from {self.dataset_path}")
        return dataset

    def run_single_variation(
        self,
        variation_name: str,
        prompt_template: str,
        system_prompt: str,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run experiment for a single prompt variation.

        Args:
            variation_name: Name of the variation
            prompt_template: Prompt template with {text} placeholder
            system_prompt: Optional system prompt
            dataset: List of dataset items

        Returns:
            Dict with results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Running: {variation_name}")
        print(f"Description: {PromptVariations.describe_variation(variation_name)}")
        print(f"{'='*60}\n")

        results = []

        for item in tqdm(dataset, desc=f"Processing {variation_name}"):
            text = item["text"]
            ground_truth = item["ground_truth"]
            category = item.get("category", "unknown")

            # Format prompt
            prompt = prompt_template.format(text=text)

            # Get prediction
            if system_prompt:
                # Combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{prompt}"
                result = self.client.generate(full_prompt)
            else:
                result = self.client.generate(prompt)

            # Extract sentiment from response
            if result["success"]:
                response_text = result["response"].lower()

                # Simple extraction logic
                if "positive" in response_text and "negative" not in response_text:
                    sentiment = "positive"
                elif "negative" in response_text and "positive" not in response_text:
                    sentiment = "negative"
                else:
                    # Try to find first occurrence
                    pos_idx = response_text.find("positive")
                    neg_idx = response_text.find("negative")

                    if pos_idx != -1 and (neg_idx == -1 or pos_idx < neg_idx):
                        sentiment = "positive"
                    elif neg_idx != -1:
                        sentiment = "negative"
                    else:
                        sentiment = "unknown"

                result["sentiment"] = sentiment
            else:
                result["sentiment"] = "error"

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
            results.append(result_entry)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(results)

        return {
            "variation_name": variation_name,
            "results": results,
            "metrics": metrics,
            "prompt_template": prompt_template,
            "system_prompt": system_prompt,
            "description": PromptVariations.describe_variation(variation_name)
        }

    def run_all_variations(self, variations: List[str] = None) -> Dict[str, Any]:
        """
        Run experiments for all or selected variations.

        Args:
            variations: List of variation names to run (None = all)

        Returns:
            Dict mapping variation names to results
        """
        print("\n" + "="*60)
        print("Batch Experiment Runner - All Prompt Variations")
        print("="*60)
        print(f"Model: {self.client.model_name}")
        print()

        # Check Ollama connection
        if not self.client.check_connection():
            raise ConnectionError("Cannot connect to Ollama. Please make sure it's running.")

        # Load dataset
        dataset = self.load_dataset()

        # Get variations to run
        all_variations = PromptVariations.get_all_variations()
        if variations:
            selected = {k: v for k, v in all_variations.items() if k in variations}
        else:
            selected = all_variations

        print(f"\nRunning {len(selected)} variations: {', '.join(selected.keys())}\n")

        # Run each variation
        for variation_name, (prompt_template, system_prompt) in selected.items():
            result = self.run_single_variation(
                variation_name,
                prompt_template,
                system_prompt,
                dataset
            )

            self.all_results[variation_name] = result
            self.all_metrics[variation_name] = result["metrics"]

            # Print summary for this variation
            self._print_variation_summary(variation_name, result["metrics"])

        # Print comparison
        self._print_comparison()

        return self.all_results

    def _print_variation_summary(self, name: str, metrics: Dict[str, Any]):
        """Print summary for a single variation."""
        print(f"\n--- {name} Summary ---")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"F1 Score: {metrics['confusion_matrix']['f1_score']:.1%}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")

    def _print_comparison(self):
        """Print comparison table of all variations."""
        print("\n" + "="*80)
        print("COMPARISON OF ALL VARIATIONS")
        print("="*80)
        print()

        # Header
        print(f"{'Variation':<20} {'Accuracy':>10} {'F1 Score':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 80)

        # Sort by accuracy
        sorted_variations = sorted(
            self.all_metrics.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        # Print rows
        for name, metrics in sorted_variations:
            accuracy = metrics['accuracy']
            f1 = metrics['confusion_matrix']['f1_score']
            precision = metrics['confusion_matrix']['precision']
            recall = metrics['confusion_matrix']['recall']

            print(f"{name:<20} {accuracy:>9.1%} {f1:>9.1%} {precision:>9.1%} {recall:>9.1%}")

        print("="*80)

        # Find best variation
        best_name = sorted_variations[0][0]
        best_accuracy = sorted_variations[0][1]['accuracy']
        print(f"\n🏆 Best performing: {best_name} with {best_accuracy:.1%} accuracy")

    def save_all_results(self):
        """Save all results to JSON files."""
        # Get project root and create results directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results for each variation
        for variation_name, result in self.all_results.items():
            results_file = os.path.join(results_dir, f"{variation_name}_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump({
                    "experiment": variation_name,
                    "timestamp": timestamp,
                    "model": self.client.model_name,
                    "description": result["description"],
                    "prompt_template": result["prompt_template"],
                    "system_prompt": result["system_prompt"],
                    "results": result["results"]
                }, f, indent=2)
            print(f"✓ Saved {variation_name} results to {results_file}")

        # Save comparison metrics
        comparison_file = os.path.join(results_dir, f"comparison_metrics_{timestamp}.json")
        with open(comparison_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "model": self.client.model_name,
                "variations": self.all_metrics,
                "summary": {
                    name: {
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["confusion_matrix"]["f1_score"],
                        "precision": metrics["confusion_matrix"]["precision"],
                        "recall": metrics["confusion_matrix"]["recall"]
                    }
                    for name, metrics in self.all_metrics.items()
                }
            }, f, indent=2)
        print(f"✓ Saved comparison to {comparison_file}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all prompt variation experiments")
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
        "--variations",
        type=str,
        nargs="+",
        default=None,
        help="Specific variations to run (default: all)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    # Run experiments
    runner = BatchExperimentRunner(args.dataset, args.model)
    results = runner.run_all_variations(args.variations)

    # Save results
    if not args.no_save:
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)
        runner.save_all_results()

    print("\n" + "="*60)
    print("All Experiments Complete! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
