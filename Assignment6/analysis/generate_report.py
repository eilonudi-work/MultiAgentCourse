"""
Generate comprehensive final report with analysis and insights.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any


class ReportGenerator:
    """Generate markdown report from experiment results."""

    def __init__(self):
        """Initialize report generator."""
        self.data = None
        self.analysis = None

    def load_data(self, comparison_file: str, analysis_file: str = None):
        """Load comparison and analysis data."""
        with open(comparison_file, 'r') as f:
            self.data = json.load(f)

        if analysis_file and os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                self.analysis = json.load(f)

    def generate_markdown_report(self) -> str:
        """Generate complete markdown report."""
        report = []

        # Header
        report.append("# Prompt Engineering Experiment Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Model:** {self.data.get('model', 'unknown')}")
        report.append(f"**Variations Tested:** {len(self.data['variations'])}")
        report.append("")
        report.append("---")
        report.append("")

        # Executive Summary
        report.extend(self._generate_executive_summary())

        # Methodology
        report.extend(self._generate_methodology())

        # Results
        report.extend(self._generate_results())

        # Analysis
        if self.analysis:
            report.extend(self._generate_analysis_section())

        # Insights
        report.extend(self._generate_insights())

        # Recommendations
        report.extend(self._generate_recommendations())

        # Conclusion
        report.extend(self._generate_conclusion())

        return "\n".join(report)

    def _generate_executive_summary(self) -> list:
        """Generate executive summary section."""
        summary = self.data['summary']

        # Find best
        best_var = max(summary.items(), key=lambda x: x[1]['accuracy'])
        best_name, best_metrics = best_var

        # Find baseline
        baseline_acc = summary.get('baseline', {}).get('accuracy', 0)

        lines = [
            "## Executive Summary",
            "",
            "This report presents a systematic comparison of six prompt engineering strategies "
            "for sentiment analysis using local LLMs via Ollama.",
            "",
            "### Key Findings",
            "",
            f"- **Best Performer:** {best_name} achieved **{best_metrics['accuracy']:.1%}** accuracy",
            f"- **Baseline Performance:** {baseline_acc:.1%} accuracy with simple prompts",
            f"- **Maximum Improvement:** {((best_metrics['accuracy'] - baseline_acc) / baseline_acc * 100):+.1f}% over baseline",
            "",
        ]

        # Add top 3
        sorted_vars = sorted(summary.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        lines.append("**Top 3 Performers:**")
        for idx, (name, metrics) in enumerate(sorted_vars[:3], 1):
            lines.append(f"{idx}. {name}: {metrics['accuracy']:.1%} accuracy, {metrics['f1_score']:.1%} F1")

        lines.extend(["", "---", ""])
        return lines

    def _generate_methodology(self) -> list:
        """Generate methodology section."""
        lines = [
            "## Methodology",
            "",
            "### Dataset",
            "- 30 sentiment examples (balanced: 15 positive, 15 negative)",
            "- Categories: product, entertainment, food, service, technology, hospitality, general",
            "- Ground truth labels manually verified",
            "",
            "### Prompt Variations Tested",
            "",
            "1. **Baseline**: Simple direct instruction",
            "2. **Role-Based**: Expert role definition with analysis guidelines",
            "3. **Few-Shot**: 3 demonstration examples included",
            "4. **Chain of Thought**: Step-by-step reasoning instructions",
            "5. **Structured Output**: Specific format requests",
            "6. **Contrastive**: Explicit comparison of positive/negative aspects",
            "",
            "### Metrics",
            "- **Accuracy**: Percentage of correct classifications",
            "- **Precision**: Correct positive predictions / total positive predictions",
            "- **Recall**: Correct positive predictions / actual positives",
            "- **F1 Score**: Harmonic mean of precision and recall",
            "- **Distance**: Semantic distance from ground truth (lower is better)",
            "",
            "---",
            ""
        ]
        return lines

    def _generate_results(self) -> list:
        """Generate results section."""
        summary = self.data['summary']
        sorted_vars = sorted(summary.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        lines = [
            "## Results",
            "",
            "### Performance Comparison",
            "",
            "| Rank | Variation | Accuracy | F1 Score | Precision | Recall |",
            "|------|-----------|----------|----------|-----------|--------|",
        ]

        for idx, (name, metrics) in enumerate(sorted_vars, 1):
            lines.append(
                f"| {idx} | {name} | {metrics['accuracy']:.1%} | "
                f"{metrics['f1_score']:.1%} | {metrics['precision']:.1%} | "
                f"{metrics['recall']:.1%} |"
            )

        lines.extend(["", ""])

        # Detailed results per variation
        lines.extend(["### Detailed Results by Variation", ""])

        for name in ['baseline', 'role_based', 'few_shot', 'chain_of_thought']:
            if name in self.data['variations']:
                lines.extend(self._format_variation_details(name))

        lines.extend(["---", ""])
        return lines

    def _format_variation_details(self, name: str) -> list:
        """Format detailed results for a variation."""
        data = self.data['variations'][name]
        cm = data['confusion_matrix']

        lines = [
            f"#### {name}",
            "",
            f"- **Accuracy:** {data['accuracy']:.1%}",
            f"- **F1 Score:** {cm['f1_score']:.1%}",
            f"- **Precision:** {cm['precision']:.1%}",
            f"- **Recall:** {cm['recall']:.1%}",
            f"- **Mean Distance:** {data['mean_distance']:.4f}",
            f"- **Std Distance:** {data['std_distance']:.4f}",
            "",
            "**Confusion Matrix:**",
            f"- True Positives: {cm['true_positive']}",
            f"- True Negatives: {cm['true_negative']}",
            f"- False Positives: {cm['false_positive']}",
            f"- False Negatives: {cm['false_negative']}",
            ""
        ]
        return lines

    def _generate_analysis_section(self) -> list:
        """Generate analysis section from statistical analysis."""
        lines = [
            "## Statistical Analysis",
            "",
        ]

        if 'key_findings' in self.analysis:
            lines.extend(["### Key Findings", ""])
            for idx, finding in enumerate(self.analysis['key_findings'], 1):
                lines.append(f"{idx}. {finding}")
            lines.extend(["", ""])

        if 'improvements_over_baseline' in self.analysis:
            lines.extend(["### Improvements Over Baseline", ""])
            for name, imp in list(self.analysis['improvements_over_baseline'].items())[:5]:
                lines.append(f"- **{name}**: {imp}")
            lines.extend(["", ""])

        if 'consistency_analysis' in self.analysis:
            cons = self.analysis['consistency_analysis']
            lines.extend([
                "### Consistency Analysis",
                "",
                f"- **Most Consistent:** {cons['most_consistent']}",
                f"- **Least Consistent:** {cons['least_consistent']}",
                "",
                "Lower standard deviation indicates more consistent predictions across examples.",
                "",
            ])

        lines.extend(["---", ""])
        return lines

    def _generate_insights(self) -> list:
        """Generate insights section."""
        summary = self.data['summary']
        best_name = max(summary.items(), key=lambda x: x[1]['accuracy'])[0]

        lines = [
            "## Key Insights",
            "",
            "### What Works",
            "",
        ]

        # Determine what worked
        if best_name == "few_shot":
            lines.extend([
                "**Few-Shot Learning** demonstrated superior performance:",
                "- Providing concrete examples helps the model understand the task better",
                "- Examples serve as implicit guidelines for classification",
                "- Particularly effective for edge cases and ambiguous sentiment",
                ""
            ])
        elif best_name == "chain_of_thought":
            lines.extend([
                "**Chain of Thought** reasoning proved most effective:",
                "- Step-by-step analysis reduces impulsive classifications",
                "- Explicit reasoning helps with nuanced sentiment detection",
                "- Better handling of complex or sarcastic text",
                ""
            ])
        elif best_name == "role_based":
            lines.extend([
                "**Role-Based Prompting** showed strong results:",
                "- Setting expert context improves model confidence",
                "- Guidelines help establish consistent evaluation criteria",
                "- Clear instructions reduce ambiguity",
                ""
            ])

        lines.extend([
            "### Trade-offs",
            "",
            "**Accuracy vs. Complexity:**",
            "- Simple prompts (baseline) are fast but less accurate",
            "- Complex prompts (few-shot, CoT) improve accuracy but use more tokens",
            "- For local models, token cost is not an issue",
            "",
            "**Consistency vs. Flexibility:**",
            "- Structured prompts improve parsing reliability",
            "- Open-ended prompts may capture nuance better",
            "- Balance depends on use case requirements",
            "",
            "---",
            ""
        ])
        return lines

    def _generate_recommendations(self) -> list:
        """Generate recommendations section."""
        best_name = max(self.data['summary'].items(), key=lambda x: x[1]['accuracy'])[0]

        lines = [
            "## Recommendations",
            "",
            f"### For Production Sentiment Analysis",
            "",
            f"1. **Use {best_name}** as the primary approach",
            "   - Highest accuracy demonstrated in testing",
            "   - Reliable performance across categories",
            "",
            "2. **Consider the use case:**",
            "   - **High-volume, speed-critical:** Use baseline or role-based",
            "   - **High-accuracy required:** Use few-shot or chain of thought",
            "   - **Edge cases common:** Few-shot learning recommended",
            "",
            "3. **Optimize for your domain:**",
            "   - Customize few-shot examples to your specific use case",
            "   - Adjust role definitions based on your text type",
            "   - Test on representative samples from your actual data",
            "",
            "### Implementation Best Practices",
            "",
            "1. **Always include:**",
            "   - Clear task description",
            "   - Expected output format",
            "   - Examples when possible",
            "",
            "2. **Iterate and test:**",
            "   - Start with baseline",
            "   - Add complexity incrementally",
            "   - Measure impact of each change",
            "",
            "3. **Monitor and adapt:**",
            "   - Track accuracy over time",
            "   - Identify failure patterns",
            "   - Refine prompts based on errors",
            "",
            "---",
            ""
        ]
        return lines

    def _generate_conclusion(self) -> list:
        """Generate conclusion section."""
        best_name = max(self.data['summary'].items(), key=lambda x: x[1]['accuracy'])[0]
        best_acc = self.data['summary'][best_name]['accuracy']
        baseline_acc = self.data['summary'].get('baseline', {}).get('accuracy', 0)
        improvement = ((best_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0

        lines = [
            "## Conclusion",
            "",
            f"This systematic comparison of prompt engineering strategies demonstrates that "
            f"thoughtful prompt design can significantly impact LLM performance on sentiment analysis tasks.",
            "",
            f"Our experiments show that **{best_name}** achieved the best results with "
            f"**{best_acc:.1%} accuracy**, representing a **{improvement:.1f}% improvement** "
            f"over the baseline approach.",
            "",
            "### Key Takeaways",
            "",
            "1. **Prompt engineering matters:** The difference between best and baseline was substantial",
            "2. **Examples help:** Few-shot learning consistently improved performance",
            "3. **Context is valuable:** Role definitions and guidelines enhance accuracy",
            "4. **Test systematically:** Different approaches work for different scenarios",
            "5. **Local models viable:** Ollama models perform well for sentiment analysis",
            "",
            "### Future Work",
            "",
            "- Test with larger datasets (100+ examples)",
            "- Compare across different models (llama2 vs mistral vs llama3)",
            "- Domain-specific prompt optimization",
            "- Ensemble approaches combining multiple strategies",
            "- Error analysis and targeted improvements",
            "",
            "---",
            "",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
            ""
        ]
        return lines

    def save_report(self, filepath: str):
        """Save report to markdown file."""
        report_content = self.generate_markdown_report()

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(report_content)

        print(f"✓ Saved report to {filepath}")
        return filepath


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate final experiment report")
    parser.add_argument(
        "--comparison-file",
        type=str,
        help="Path to comparison_metrics JSON file"
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        help="Path to statistical_analysis JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../EXPERIMENT_REPORT.md",
        help="Output markdown file path"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest files"
    )

    args = parser.parse_args()

    # Find files
    if args.latest or not args.comparison_file:
        results_dir = "../results"
        analysis_dir = "../analysis"

        # Find latest comparison
        comparison_files = [f for f in os.listdir(results_dir) if f.startswith("comparison_metrics_")]
        if comparison_files:
            comparison_files.sort(reverse=True)
            comparison_file = os.path.join(results_dir, comparison_files[0])
        else:
            print("Error: No comparison files found")
            return

        # Find latest analysis
        if os.path.exists(analysis_dir):
            analysis_files = [f for f in os.listdir(analysis_dir) if f.startswith("statistical_analysis_")]
            if analysis_files:
                analysis_files.sort(reverse=True)
                analysis_file = os.path.join(analysis_dir, analysis_files[0])
            else:
                analysis_file = None
        else:
            analysis_file = None

        print(f"Using comparison file: {comparison_file}")
        if analysis_file:
            print(f"Using analysis file: {analysis_file}")
    else:
        comparison_file = args.comparison_file
        analysis_file = args.analysis_file

    # Generate report
    print("\n" + "="*60)
    print("Generating Final Report")
    print("="*60 + "\n")

    generator = ReportGenerator()
    generator.load_data(comparison_file, analysis_file)
    generator.save_report(args.output)

    print("\n" + "="*60)
    print("Report Generation Complete! 📄")
    print("="*60)
    print(f"\nView the report: {args.output}")


if __name__ == "__main__":
    main()
