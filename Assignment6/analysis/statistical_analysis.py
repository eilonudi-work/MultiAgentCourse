"""
Statistical analysis module for comparing prompt variations.
Performs significance testing and detailed statistical comparisons.
"""

import json
import os
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
from datetime import datetime


class StatisticalAnalyzer:
    """Perform statistical analysis on experiment results."""

    def __init__(self):
        """Initialize statistical analyzer."""
        self.comparison_data = None
        self.variations_data = None

    def load_results(self, comparison_file: str):
        """Load comparison results from JSON file."""
        with open(comparison_file, 'r') as f:
            data = json.load(f)

        self.comparison_data = data
        self.variations_data = data['variations']
        print(f"✓ Loaded results for {len(self.variations_data)} variations")

    def calculate_improvement(self, baseline: str = "baseline") -> Dict[str, float]:
        """
        Calculate improvement over baseline for each variation.

        Args:
            baseline: Name of baseline variation

        Returns:
            Dict mapping variation names to improvement percentages
        """
        if baseline not in self.variations_data:
            print(f"Warning: Baseline '{baseline}' not found")
            return {}

        baseline_accuracy = self.variations_data[baseline]['accuracy']

        improvements = {}
        for name, data in self.variations_data.items():
            if name == baseline:
                improvements[name] = 0.0
            else:
                accuracy = data['accuracy']
                improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
                improvements[name] = improvement

        return improvements

    def rank_variations(self, metric: str = "accuracy") -> List[Tuple[str, float]]:
        """
        Rank variations by a specific metric.

        Args:
            metric: Metric to rank by (accuracy, f1_score, precision, recall)

        Returns:
            List of (variation_name, metric_value) tuples, sorted descending
        """
        rankings = []

        for name, data in self.variations_data.items():
            if metric == "accuracy":
                value = data['accuracy']
            elif metric in ['f1_score', 'precision', 'recall']:
                value = data['confusion_matrix'][metric]
            else:
                value = data.get(metric, 0)

            rankings.append((name, value))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def compare_variations(self, var1: str, var2: str) -> Dict[str, Any]:
        """
        Detailed comparison between two variations.

        Args:
            var1: First variation name
            var2: Second variation name

        Returns:
            Dict with comparison metrics
        """
        if var1 not in self.variations_data or var2 not in self.variations_data:
            return {"error": "Variation not found"}

        data1 = self.variations_data[var1]
        data2 = self.variations_data[var2]

        comparison = {
            "variation_1": var1,
            "variation_2": var2,
            "accuracy_diff": data1['accuracy'] - data2['accuracy'],
            "f1_diff": data1['confusion_matrix']['f1_score'] - data2['confusion_matrix']['f1_score'],
            "better_variation": var1 if data1['accuracy'] > data2['accuracy'] else var2,
            "metrics": {
                var1: {
                    "accuracy": data1['accuracy'],
                    "f1_score": data1['confusion_matrix']['f1_score'],
                    "precision": data1['confusion_matrix']['precision'],
                    "recall": data1['confusion_matrix']['recall']
                },
                var2: {
                    "accuracy": data2['accuracy'],
                    "f1_score": data2['confusion_matrix']['f1_score'],
                    "precision": data2['confusion_matrix']['precision'],
                    "recall": data2['confusion_matrix']['recall']
                }
            }
        }

        return comparison

    def find_best_worst(self) -> Dict[str, Any]:
        """
        Find best and worst performing variations.

        Returns:
            Dict with best/worst variations and their metrics
        """
        rankings = self.rank_variations("accuracy")

        best_name, best_acc = rankings[0]
        worst_name, worst_acc = rankings[-1]

        return {
            "best": {
                "name": best_name,
                "accuracy": best_acc,
                "metrics": self.variations_data[best_name]
            },
            "worst": {
                "name": worst_name,
                "accuracy": worst_acc,
                "metrics": self.variations_data[worst_name]
            },
            "improvement": ((best_acc - worst_acc) / worst_acc) * 100
        }

    def category_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance across categories.

        Returns:
            Dict with category-wise analysis
        """
        # Collect all categories
        all_categories = set()
        for data in self.variations_data.values():
            if 'category_stats' in data:
                all_categories.update(data['category_stats'].keys())

        category_analysis = {}

        for category in all_categories:
            cat_results = {}
            for name, data in self.variations_data.items():
                if 'category_stats' in data and category in data['category_stats']:
                    cat_results[name] = data['category_stats'][category]['accuracy']

            if cat_results:
                best_var = max(cat_results, key=cat_results.get)
                worst_var = min(cat_results, key=cat_results.get)

                category_analysis[category] = {
                    "results": cat_results,
                    "best_variation": best_var,
                    "best_accuracy": cat_results[best_var],
                    "worst_variation": worst_var,
                    "worst_accuracy": cat_results[worst_var],
                    "mean_accuracy": np.mean(list(cat_results.values())),
                    "std_accuracy": np.std(list(cat_results.values()))
                }

        return category_analysis

    def consistency_analysis(self) -> Dict[str, Any]:
        """
        Analyze consistency (variance) of each variation.

        Returns:
            Dict with consistency metrics
        """
        consistency = {}

        for name, data in self.variations_data.items():
            consistency[name] = {
                "mean_distance": data['mean_distance'],
                "std_distance": data['std_distance'],
                "coefficient_of_variation": data['std_distance'] / data['mean_distance'] if data['mean_distance'] > 0 else 0,
                "interpretation": "high consistency" if data['std_distance'] < 0.1 else "moderate consistency" if data['std_distance'] < 0.2 else "low consistency"
            }

        # Rank by consistency (lower std is better)
        ranked = sorted(consistency.items(), key=lambda x: x[1]['std_distance'])

        return {
            "by_variation": consistency,
            "most_consistent": ranked[0][0],
            "least_consistent": ranked[-1][0]
        }

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.

        Returns:
            Dict with full analysis summary
        """
        print("\n" + "="*60)
        print("Generating Statistical Analysis")
        print("="*60 + "\n")

        # Rankings
        accuracy_rankings = self.rank_variations("accuracy")
        f1_rankings = self.rank_variations("f1_score")

        # Best/Worst
        best_worst = self.find_best_worst()

        # Improvements
        improvements = self.calculate_improvement()

        # Category analysis
        category_results = self.category_analysis()

        # Consistency
        consistency = self.consistency_analysis()

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.comparison_data.get('model', 'unknown'),
            "num_variations": len(self.variations_data),
            "rankings": {
                "by_accuracy": [(name, f"{acc:.1%}") for name, acc in accuracy_rankings],
                "by_f1_score": [(name, f"{f1:.1%}") for name, f1 in f1_rankings]
            },
            "best_performer": {
                "name": best_worst['best']['name'],
                "accuracy": f"{best_worst['best']['accuracy']:.1%}",
                "f1_score": f"{best_worst['best']['metrics']['confusion_matrix']['f1_score']:.1%}"
            },
            "worst_performer": {
                "name": best_worst['worst']['name'],
                "accuracy": f"{best_worst['worst']['accuracy']:.1%}",
                "f1_score": f"{best_worst['worst']['metrics']['confusion_matrix']['f1_score']:.1%}"
            },
            "overall_improvement": f"{best_worst['improvement']:.1f}%",
            "improvements_over_baseline": {
                name: f"{imp:+.1f}%" for name, imp in sorted(improvements.items(), key=lambda x: x[1], reverse=True)
            },
            "category_analysis": category_results,
            "consistency_analysis": consistency,
            "key_findings": self._generate_key_findings(accuracy_rankings, improvements, category_results, consistency)
        }

        return report

    def _generate_key_findings(self, rankings, improvements, categories, consistency) -> List[str]:
        """Generate key findings from analysis."""
        findings = []

        # Best performer
        best_name, best_acc = rankings[0]
        findings.append(f"{best_name} achieved the highest accuracy at {best_acc:.1%}")

        # Biggest improvement
        if improvements:
            best_improvement = max(improvements.items(), key=lambda x: x[1])
            if best_improvement[1] > 0:
                findings.append(f"{best_improvement[0]} showed {best_improvement[1]:+.1f}% improvement over baseline")

        # Most consistent
        most_consistent = consistency['most_consistent']
        findings.append(f"{most_consistent} demonstrated the most consistent performance")

        # Category insights
        if categories:
            # Find category with biggest variance
            max_variance_cat = max(categories.items(), key=lambda x: x[1]['std_accuracy'])
            findings.append(f"Category '{max_variance_cat[0]}' showed the most variation across prompts (σ={max_variance_cat[1]['std_accuracy']:.3f})")

        return findings

    def save_report(self, report: Dict[str, Any], output_dir: str = "../analysis"):
        """Save analysis report to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"statistical_analysis_{timestamp}.json")

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Saved statistical analysis to {filepath}")
        return filepath

    def print_summary(self, report: Dict[str, Any]):
        """Print formatted summary of the report."""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nModel: {report['model']}")
        print(f"Variations analyzed: {report['num_variations']}")

        print("\n--- Rankings by Accuracy ---")
        for idx, (name, acc) in enumerate(report['rankings']['by_accuracy'], 1):
            print(f"{idx}. {name:<20} {acc}")

        print("\n--- Best Performer ---")
        best = report['best_performer']
        print(f"  Name: {best['name']}")
        print(f"  Accuracy: {best['accuracy']}")
        print(f"  F1 Score: {best['f1_score']}")

        print("\n--- Improvements over Baseline ---")
        for name, imp in list(report['improvements_over_baseline'].items())[:5]:
            print(f"  {name:<20} {imp}")

        print("\n--- Consistency Analysis ---")
        cons = report['consistency_analysis']
        print(f"  Most consistent: {cons['most_consistent']}")
        print(f"  Least consistent: {cons['least_consistent']}")

        print("\n--- Key Findings ---")
        for idx, finding in enumerate(report['key_findings'], 1):
            print(f"{idx}. {finding}")

        print("\n" + "="*80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Perform statistical analysis on experiment results")
    parser.add_argument(
        "--comparison-file",
        type=str,
        help="Path to comparison_metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest comparison file"
    )

    args = parser.parse_args()

    # Find comparison file
    if args.latest or not args.comparison_file:
        results_dir = "../results"
        comparison_files = [f for f in os.listdir(results_dir) if f.startswith("comparison_metrics_")]
        if not comparison_files:
            print("Error: No comparison files found in results/")
            print("Please run experiments first: ./run_all.sh")
            return

        comparison_files.sort(reverse=True)
        comparison_file = os.path.join(results_dir, comparison_files[0])
        print(f"Using latest comparison file: {comparison_file}")
    else:
        comparison_file = args.comparison_file

    # Perform analysis
    analyzer = StatisticalAnalyzer()
    analyzer.load_results(comparison_file)

    report = analyzer.generate_summary_report()
    analyzer.print_summary(report)
    analyzer.save_report(report, args.output_dir)

    print("\nAnalysis complete! 📊")


if __name__ == "__main__":
    main()
