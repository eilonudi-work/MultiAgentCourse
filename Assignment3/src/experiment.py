"""
Main Experiment Script for Multi-Agent Translation Pipeline

This script orchestrates the complete experiment:
1. Inject spelling errors at various rates
2. Run translation pipeline (manual or automated)
3. Calculate vector distances
4. Generate visualization
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from src.error_injector import ErrorInjector, ErrorStats
from src.embeddings import EmbeddingGenerator, DistanceCalculator


class TranslationExperiment:
    """Manage the complete translation experiment workflow."""

    def __init__(
        self,
        original_sentence: str,
        error_levels: List[float],
        embedding_model: str = "all-MiniLM-L6-v2",
        seed: int = 42,
        output_dir: str = "results/"
    ):
        """
        Initialize experiment.

        Args:
            original_sentence: Clean English sentence to start with
            error_levels: List of error rates to test (e.g., [0.0, 0.1, 0.25, 0.5])
            embedding_model: Model for generating embeddings
            seed: Random seed for reproducibility
            output_dir: Directory for saving results
        """
        self.original_sentence = original_sentence
        self.error_levels = error_levels
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.error_injector = ErrorInjector(seed=seed)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.distance_calculator = DistanceCalculator()

        # Storage for results
        self.results = []

    def inject_errors(self) -> Dict[float, Tuple[str, ErrorStats]]:
        """
        Inject errors at all specified levels.

        Returns:
            Dictionary mapping error_rate -> (corrupted_text, stats)
        """
        results = {}
        for error_rate in self.error_levels:
            injector = ErrorInjector(seed=self.seed)
            corrupted, stats = injector.inject(self.original_sentence, error_rate)
            results[error_rate] = (corrupted, stats)

        return results

    def record_translation_result(
        self,
        error_rate: float,
        corrupted_input: str,
        final_output: str,
        intermediate_translations: Dict[str, str] = None
    ):
        """
        Record the result of running translation pipeline.

        Args:
            error_rate: Error rate used for this run
            corrupted_input: Input sentence with errors
            final_output: Final English output after 3 translations
            intermediate_translations: Optional dict with intermediate steps
                {'french': '...', 'hebrew': '...'}
        """
        # Generate embeddings
        original_emb = self.embedding_generator.embed(self.original_sentence)
        final_emb = self.embedding_generator.embed(final_output)

        # Calculate distances
        metrics = self.distance_calculator.calculate_all_metrics(
            original_emb[0], final_emb[0]
        )

        # Store result
        result = {
            'error_rate': error_rate,
            'original': self.original_sentence,
            'corrupted_input': corrupted_input,
            'final_output': final_output,
            'intermediate_translations': intermediate_translations or {},
            'cosine_distance': metrics['cosine_distance'],
            'cosine_similarity': metrics['cosine_similarity'],
            'euclidean_distance': metrics['euclidean_distance'],
            'manhattan_distance': metrics['manhattan_distance'],
        }

        self.results.append(result)

    def save_results(self, filename: str = "experiment_results.json"):
        """
        Save all results to JSON file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")

    def load_results(self, filename: str = "experiment_results.json"):
        """
        Load results from JSON file.

        Args:
            filename: Input filename
        """
        input_path = self.output_dir / filename
        with open(input_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        print(f"Loaded {len(self.results)} results from: {input_path}")

    def get_plot_data(self) -> Tuple[List[float], List[float]]:
        """
        Extract data for plotting.

        Returns:
            Tuple of (error_rates, cosine_distances)
        """
        # Sort by error rate
        sorted_results = sorted(self.results, key=lambda x: x['error_rate'])

        error_rates = [r['error_rate'] * 100 for r in sorted_results]  # Convert to percentage
        cosine_distances = [r['cosine_distance'] for r in sorted_results]

        return error_rates, cosine_distances

    def print_summary(self):
        """Print a summary of experimental results."""
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        print(f"\nOriginal sentence:")
        print(f"  {self.original_sentence}")
        print(f"  (Length: {len(self.original_sentence.split())} words)")

        print(f"\nError levels tested: {len(self.results)}")

        for result in sorted(self.results, key=lambda x: x['error_rate']):
            error_pct = result['error_rate'] * 100
            print(f"\n--- {error_pct:.0f}% Error Rate ---")
            print(f"  Input:  {result['corrupted_input']}")
            print(f"  Output: {result['final_output']}")
            print(f"  Cosine Distance: {result['cosine_distance']:.6f}")
            print(f"  Cosine Similarity: {result['cosine_similarity']:.6f}")

        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # Define test sentence
    original = "Artificial intelligence is rapidly transforming the modern world by enabling machines to learn from data and make intelligent decisions"

    # Define error levels to test
    error_levels = [0.0, 0.10, 0.25, 0.50]

    # Initialize experiment
    experiment = TranslationExperiment(
        original_sentence=original,
        error_levels=error_levels,
        embedding_model="all-MiniLM-L6-v2",
        seed=42
    )

    # Step 1: Generate corrupted inputs
    print("Generating corrupted inputs...")
    corrupted_inputs = experiment.inject_errors()

    for error_rate, (corrupted, stats) in corrupted_inputs.items():
        print(f"\n{error_rate*100:.0f}% errors: {corrupted}")
        print(f"  Changes: {stats.changes}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. For each corrupted input above:")
    print("   a. Run through translation pipeline:")
    print("      - Agent 1: English → French")
    print("      - Agent 2: French → Hebrew")
    print("      - Agent 3: Hebrew → English")
    print("   b. Record results using:")
    print("      experiment.record_translation_result(error_rate, corrupted, final_output)")
    print("\n2. Save results:")
    print("   experiment.save_results()")
    print("\n3. Generate visualization:")
    print("   python src/visualize.py")
