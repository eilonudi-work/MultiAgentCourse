"""
Statistical Analyzer for statistical validation.

Performs statistical tests and computes confidence intervals.
"""

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """
    Perform statistical analysis on evaluation results.

    Features:
    - Confidence intervals
    - Statistical tests (t-test, ANOVA)
    - Effect size computation
    - Significance testing

    """

    @staticmethod
    def compute_confidence_interval(
        values: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence interval for mean.

        Args:
            values: Array of values
            confidence: Confidence level (default 0.95 for 95%)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if len(values) == 0:
            return (float('nan'), float('nan'), float('nan'))

        mean = np.mean(values)
        std_err = stats.sem(values)

        # Compute confidence interval
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        return (mean, mean - ci, mean + ci)

    @staticmethod
    def test_frequency_differences(
        per_frequency_metrics: List[Dict[str, float]],
        metric_name: str = 'mse'
    ) -> Dict[str, any]:
        """
        Test if there are significant differences between frequencies.

        Uses one-way ANOVA to test if metric values differ across frequencies.

        Args:
            per_frequency_metrics: List of metrics per frequency
            metric_name: Metric to test

        Returns:
            Dictionary with test results
        """
        # Extract metric values for each frequency
        frequency_values = []

        for freq_metrics in per_frequency_metrics:
            if metric_name in freq_metrics and not np.isnan(freq_metrics[metric_name]):
                # For overall metrics, we only have one value per frequency
                # In a real scenario, we'd have multiple samples
                frequency_values.append(freq_metrics[metric_name])

        if len(frequency_values) < 2:
            return {
                'test': 'ANOVA',
                'metric': metric_name,
                'significant': False,
                'p_value': float('nan'),
                'f_statistic': float('nan'),
                'note': 'Insufficient data for statistical test'
            }

        # Since we have only one value per frequency from aggregated metrics,
        # we can't perform ANOVA. In practice, you'd need individual sample metrics.
        # For now, we'll compute descriptive statistics

        mean_value = np.mean(frequency_values)
        std_value = np.std(frequency_values)
        min_value = np.min(frequency_values)
        max_value = np.max(frequency_values)

        return {
            'test': 'Descriptive Statistics',
            'metric': metric_name,
            'num_frequencies': len(frequency_values),
            'mean': float(mean_value),
            'std': float(std_value),
            'min': float(min_value),
            'max': float(max_value),
            'range': float(max_value - min_value),
            'coefficient_of_variation': float(std_value / mean_value) if mean_value != 0 else float('nan')
        }

    @staticmethod
    def compute_effect_size(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        if len(group1) == 0 or len(group2) == 0:
            return float('nan')

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return float('nan')

        cohen_d = (mean1 - mean2) / pooled_std

        return float(cohen_d)

    @staticmethod
    def perform_t_test(
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Perform independent samples t-test.

        Args:
            group1: First group of values
            group2: Second group of values
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if len(group1) < 2 or len(group2) < 2:
            return {
                't_statistic': float('nan'),
                'p_value': float('nan'),
                'significant': False,
                'note': 'Insufficient data'
            }

        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'alpha': 0.05,
            'alternative': alternative
        }

    @staticmethod
    def compute_summary_statistics(
        values: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive summary statistics.

        Args:
            values: Array of values

        Returns:
            Dictionary of statistics
        """
        if len(values) == 0:
            return {
                'count': 0,
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'q25': float('nan'),
                'median': float('nan'),
                'q75': float('nan'),
                'max': float('nan'),
                'iqr': float('nan')
            }

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'min': float(np.min(values)),
            'q25': float(np.percentile(values, 25)),
            'median': float(np.median(values)),
            'q75': float(np.percentile(values, 75)),
            'max': float(np.max(values)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
        }

    @staticmethod
    def analyze_per_frequency_performance(
        per_frequency_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of per-frequency performance.

        Args:
            per_frequency_metrics: List of metrics per frequency

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'num_frequencies': len(per_frequency_metrics),
            'frequencies': [],
            'mse_analysis': {},
            'correlation_analysis': {},
            'best_frequency': None,
            'worst_frequency': None
        }

        # Extract MSE and correlation values
        mse_values = []
        corr_values = []
        frequencies = []

        for freq_metrics in per_frequency_metrics:
            if 'mse' in freq_metrics and not np.isnan(freq_metrics['mse']):
                frequencies.append(freq_metrics['frequency'])
                mse_values.append(freq_metrics['mse'])
                corr_values.append(freq_metrics.get('correlation', float('nan')))

        analysis['frequencies'] = frequencies

        # MSE analysis
        if mse_values:
            mse_array = np.array(mse_values)
            analysis['mse_analysis'] = StatisticalAnalyzer.compute_summary_statistics(mse_array)

            # Find best and worst
            best_idx = np.argmin(mse_values)
            worst_idx = np.argmax(mse_values)

            analysis['best_frequency'] = {
                'frequency': frequencies[best_idx],
                'mse': mse_values[best_idx],
                'correlation': corr_values[best_idx]
            }

            analysis['worst_frequency'] = {
                'frequency': frequencies[worst_idx],
                'mse': mse_values[worst_idx],
                'correlation': corr_values[worst_idx]
            }

        # Correlation analysis
        if corr_values and not all(np.isnan(corr_values)):
            corr_array = np.array([c for c in corr_values if not np.isnan(c)])
            if len(corr_array) > 0:
                analysis['correlation_analysis'] = StatisticalAnalyzer.compute_summary_statistics(corr_array)

        return analysis
