"""
Metrics calculation for sentiment analysis results.
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class SentimentMetrics:
    """Calculate distance and accuracy metrics for sentiment classification."""

    def __init__(self):
        """Initialize metrics calculator with embedding model."""
        # Use a lightweight model for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def exact_match(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate exact match score (1.0 or 0.0).

        Args:
            predicted: Predicted sentiment
            ground_truth: True sentiment

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_clean = predicted.lower().strip()
        truth_clean = ground_truth.lower().strip()
        return 1.0 if pred_clean == truth_clean else 0.0

    def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate semantic similarity using embeddings (0.0 to 1.0).

        Args:
            predicted: Predicted sentiment
            ground_truth: True sentiment

        Returns:
            Cosine similarity score
        """
        # Get embeddings
        pred_embedding = self.embedding_model.encode([predicted])[0]
        truth_embedding = self.embedding_model.encode([ground_truth])[0]

        # Calculate cosine similarity
        similarity = np.dot(pred_embedding, truth_embedding) / (
            np.linalg.norm(pred_embedding) * np.linalg.norm(truth_embedding)
        )

        # Convert to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1) / 2

    def distance_score(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate distance score (0.0 = exact match, 1.0 = complete mismatch).

        Args:
            predicted: Predicted sentiment
            ground_truth: True sentiment

        Returns:
            Distance score
        """
        similarity = self.semantic_similarity(predicted, ground_truth)
        return 1.0 - similarity

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a list of results.

        Args:
            results: List of result dicts with 'predicted', 'ground_truth', 'success'

        Returns:
            Dictionary with accuracy, distance metrics, and statistics
        """
        if not results:
            return {
                "accuracy": 0.0,
                "mean_distance": 0.0,
                "std_distance": 0.0,
                "total_samples": 0,
                "successful_samples": 0,
                "failed_samples": 0
            }

        exact_matches = []
        distances = []
        successful_results = []

        for result in results:
            if result.get("success", False):
                predicted = result.get("predicted", "")
                ground_truth = result.get("ground_truth", "")

                # Calculate metrics
                exact_match = self.exact_match(predicted, ground_truth)
                distance = self.distance_score(predicted, ground_truth)

                exact_matches.append(exact_match)
                distances.append(distance)
                successful_results.append(result)

        total_samples = len(results)
        successful_samples = len(successful_results)
        failed_samples = total_samples - successful_samples

        # Calculate statistics
        accuracy = np.mean(exact_matches) if exact_matches else 0.0
        mean_distance = np.mean(distances) if distances else 0.0
        std_distance = np.std(distances) if distances else 0.0

        # Per-category breakdown
        category_stats = self._calculate_category_stats(successful_results)

        # Confusion matrix
        confusion = self._calculate_confusion_matrix(successful_results)

        return {
            "accuracy": float(accuracy),
            "mean_distance": float(mean_distance),
            "std_distance": float(std_distance),
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "success_rate": successful_samples / total_samples if total_samples > 0 else 0.0,
            "category_stats": category_stats,
            "confusion_matrix": confusion,
            "distance_distribution": {
                "min": float(np.min(distances)) if distances else 0.0,
                "max": float(np.max(distances)) if distances else 0.0,
                "median": float(np.median(distances)) if distances else 0.0,
                "q25": float(np.percentile(distances, 25)) if distances else 0.0,
                "q75": float(np.percentile(distances, 75)) if distances else 0.0
            }
        }

    def _calculate_category_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy per category."""
        category_results = {}

        for result in results:
            category = result.get("category", "unknown")
            if category not in category_results:
                category_results[category] = []

            predicted = result.get("predicted", "")
            ground_truth = result.get("ground_truth", "")
            exact_match = self.exact_match(predicted, ground_truth)
            category_results[category].append(exact_match)

        # Calculate stats per category
        stats = {}
        for category, matches in category_results.items():
            stats[category] = {
                "accuracy": float(np.mean(matches)),
                "count": len(matches)
            }

        return stats

    def _calculate_confusion_matrix(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate confusion matrix for binary classification."""
        tp = 0  # True Positive
        tn = 0  # True Negative
        fp = 0  # False Positive
        fn = 0  # False Negative

        for result in results:
            predicted = result.get("predicted", "").lower().strip()
            ground_truth = result.get("ground_truth", "").lower().strip()

            if ground_truth == "positive":
                if predicted == "positive":
                    tp += 1
                else:
                    fn += 1
            elif ground_truth == "negative":
                if predicted == "negative":
                    tn += 1
                else:
                    fp += 1

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }


if __name__ == "__main__":
    # Test metrics
    metrics = SentimentMetrics()

    # Test exact match
    print("Testing exact match:")
    print(f"Match: {metrics.exact_match('positive', 'positive')}")  # Should be 1.0
    print(f"No match: {metrics.exact_match('positive', 'negative')}")  # Should be 0.0

    # Test semantic similarity
    print("\nTesting semantic similarity:")
    print(f"Same: {metrics.semantic_similarity('positive', 'positive')}")
    print(f"Different: {metrics.semantic_similarity('positive', 'negative')}")

    # Test full metrics calculation
    print("\nTesting full metrics:")
    test_results = [
        {"predicted": "positive", "ground_truth": "positive", "success": True, "category": "test"},
        {"predicted": "negative", "ground_truth": "negative", "success": True, "category": "test"},
        {"predicted": "positive", "ground_truth": "negative", "success": True, "category": "test"},
    ]
    metrics_result = metrics.calculate_metrics(test_results)
    print(f"Accuracy: {metrics_result['accuracy']:.2f}")
    print(f"Mean distance: {metrics_result['mean_distance']:.2f}")
