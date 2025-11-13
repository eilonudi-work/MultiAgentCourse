"""
Embeddings and Distance Calculation Module

This module provides sentence embedding generation and semantic distance
calculation for the multi-agent translation pipeline experiment.
"""

import numpy as np
from typing import List, Union, Optional, Dict
from pathlib import Path
import json
import hashlib


class EmbeddingGenerator:
    """Generate sentence embeddings using Sentence-BERT or OpenAI models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "cache/",
        use_cache: bool = True
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Model to use. Options:
                - "all-MiniLM-L6-v2" (Sentence-BERT, 384 dims, fast)
                - "all-mpnet-base-v2" (Sentence-BERT, 768 dims, higher quality)
                - "openai" (requires OPENAI_API_KEY env var)
            cache_dir: Directory for caching embeddings
            use_cache: Whether to cache embeddings to disk
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Lazy load model when first needed
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            if self.model_name == "openai":
                import openai
                self._model = "openai"
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
        return self._model

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache first
        if self.use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cached = self._load_from_cache(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # If all are cached, return immediately
            if not uncached_texts:
                result = np.zeros((len(texts), cached_embeddings[0][1].shape[0]))
                for idx, emb in cached_embeddings:
                    result[idx] = emb
                return result

            # Compute uncached embeddings
            if uncached_texts:
                new_embeddings = self._compute_embeddings(uncached_texts)

                # Save to cache
                for text, emb in zip(uncached_texts, new_embeddings):
                    self._save_to_cache(text, emb)

                # Combine cached and new embeddings
                if cached_embeddings:
                    result = np.zeros((len(texts), new_embeddings.shape[1]))
                    for idx, emb in cached_embeddings:
                        result[idx] = emb
                    for idx, emb in zip(uncached_indices, new_embeddings):
                        result[idx] = emb
                    return result
                else:
                    return new_embeddings
        else:
            return self._compute_embeddings(texts)

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings without caching."""
        if self.model_name == "openai":
            import openai
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        else:
            # Sentence-BERT
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model_name}_{text_hash}.npy"

    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache if exists."""
        cache_file = self.cache_dir / self._get_cache_key(text)
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_file = self.cache_dir / self._get_cache_key(text)
        np.save(cache_file, embedding)


class DistanceCalculator:
    """Calculate various distance metrics between embeddings."""

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.

        Cosine distance = 1 - cosine similarity
        Range: [0, 2], where 0 = identical, 2 = opposite

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine distance
        """
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1.0 - similarity

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Range: [-1, 1], where 1 = identical, -1 = opposite

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Euclidean (L2) distance between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(emb1 - emb2)

    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Manhattan distance
        """
        return np.sum(np.abs(emb1 - emb2))

    @staticmethod
    def calculate_all_metrics(emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Calculate all distance metrics.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Dictionary with all distance metrics
        """
        return {
            'cosine_distance': DistanceCalculator.cosine_distance(emb1, emb2),
            'cosine_similarity': DistanceCalculator.cosine_similarity(emb1, emb2),
            'euclidean_distance': DistanceCalculator.euclidean_distance(emb1, emb2),
            'manhattan_distance': DistanceCalculator.manhattan_distance(emb1, emb2),
        }


# Example usage
if __name__ == "__main__":
    # Example sentences
    original = "Artificial intelligence is rapidly transforming the modern world by enabling machines to learn from data and make intelligent decisions"
    final = "Artificial intelligence is rapidly changing the modern world by enabling machines to learn from data and make smart decisions"

    # Generate embeddings
    print("Generating embeddings...")
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    emb1 = generator.embed(original)
    emb2 = generator.embed(final)

    print(f"Embedding shape: {emb1.shape}")

    # Calculate distances
    calculator = DistanceCalculator()
    metrics = calculator.calculate_all_metrics(emb1[0], emb2[0])

    print(f"\nDistance Metrics:")
    print(f"  Cosine Distance: {metrics['cosine_distance']:.6f}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"  Euclidean Distance: {metrics['euclidean_distance']:.6f}")
    print(f"  Manhattan Distance: {metrics['manhattan_distance']:.6f}")
