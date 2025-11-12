"""
Evaluation framework for comprehensive model assessment.

This module provides tools for:
- Computing evaluation metrics on test sets
- Statistical analysis and validation
- Error pattern analysis
- Publication-quality visualizations
- Performance reporting
"""

from .model_evaluator import ModelEvaluator
from .statistical_analyzer import StatisticalAnalyzer
from .error_analyzer import ErrorAnalyzer
from .visualizer import SignalVisualizer

__all__ = [
    'ModelEvaluator',
    'StatisticalAnalyzer',
    'ErrorAnalyzer',
    'SignalVisualizer'
]
