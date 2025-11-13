"""
Error Injection Module for Multi-Agent Translation Pipeline

This module provides controlled spelling error injection into English text
to simulate realistic input noise for the translation pipeline experiment.
"""

import random
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ErrorStats:
    """Statistics about injected errors."""
    total_words: int
    num_errors: int
    error_rate: float
    changes: List[Tuple[str, str]]  # (original, corrupted)


class ErrorInjector:
    """Inject controlled spelling errors into English text."""

    # Common keyboard adjacency for typo simulation
    KEYBOARD_MAP = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 'g', 'c', 'v'],
        'g': ['f', 't', 'h', 'v', 'b'],
        'h': ['g', 'y', 'j', 'b', 'n'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'k', 'n', 'm'],
        'k': ['j', 'i', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x'],
    }

    ERROR_TYPES = ['omit', 'substitute', 'duplicate', 'transpose']

    def __init__(self, seed: int = 42):
        """
        Initialize the error injector.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)

    def inject(self, text: str, error_rate: float) -> Tuple[str, ErrorStats]:
        """
        Inject spelling errors into text at specified rate.

        Args:
            text: Clean input text
            error_rate: Fraction of words to corrupt (0.0 to 1.0)

        Returns:
            Tuple of (corrupted_text, error_stats)
        """
        words = text.split()
        total_words = len(words)
        num_errors = max(1, int(total_words * error_rate))

        # Randomly select word indices to corrupt
        error_indices = random.sample(range(total_words), min(num_errors, total_words))

        changes = []
        for idx in error_indices:
            original = words[idx]
            corrupted = self._corrupt_word(original)
            words[idx] = corrupted
            changes.append((original, corrupted))

        corrupted_text = ' '.join(words)

        stats = ErrorStats(
            total_words=total_words,
            num_errors=len(changes),
            error_rate=len(changes) / total_words,
            changes=changes
        )

        return corrupted_text, stats

    def _corrupt_word(self, word: str) -> str:
        """
        Apply random corruption to a single word.

        Args:
            word: Original word

        Returns:
            Corrupted word
        """
        if len(word) <= 2:
            return word  # Don't corrupt very short words

        # Choose random error type
        error_type = random.choice(self.ERROR_TYPES)

        if error_type == 'omit':
            return self._omit_letter(word)
        elif error_type == 'substitute':
            return self._substitute_letter(word)
        elif error_type == 'duplicate':
            return self._duplicate_letter(word)
        else:  # transpose
            return self._transpose_letters(word)

    def _omit_letter(self, word: str) -> str:
        """Remove a random letter from the word."""
        if len(word) <= 3:
            return word
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx+1:]

    def _substitute_letter(self, word: str) -> str:
        """Replace a letter with an adjacent keyboard key."""
        idx = random.randint(0, len(word) - 1)
        char = word[idx].lower()

        if char in self.KEYBOARD_MAP:
            replacement = random.choice(self.KEYBOARD_MAP[char])
            # Preserve case
            if word[idx].isupper():
                replacement = replacement.upper()
            return word[:idx] + replacement + word[idx+1:]
        return word

    def _duplicate_letter(self, word: str) -> str:
        """Duplicate a random letter in the word."""
        idx = random.randint(0, len(word) - 1)
        return word[:idx+1] + word[idx] + word[idx+1:]

    def _transpose_letters(self, word: str) -> str:
        """Swap two adjacent letters."""
        if len(word) <= 2:
            return word
        idx = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return ''.join(chars)


def inject_errors_at_levels(
    text: str,
    error_levels: List[float],
    seed: int = 42
) -> Dict[float, Tuple[str, ErrorStats]]:
    """
    Inject errors at multiple levels for experimental analysis.

    Args:
        text: Clean input text
        error_levels: List of error rates to test (e.g., [0.0, 0.1, 0.25, 0.5])
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping error_rate -> (corrupted_text, stats)
    """
    results = {}

    for error_rate in error_levels:
        injector = ErrorInjector(seed=seed)
        corrupted, stats = injector.inject(text, error_rate)
        results[error_rate] = (corrupted, stats)

    return results


# Example usage
if __name__ == "__main__":
    # Test sentence
    clean_text = "Artificial intelligence is rapidly transforming the modern world by enabling machines to learn from data and make intelligent decisions"

    # Test at 25% error rate
    injector = ErrorInjector(seed=42)
    corrupted, stats = injector.inject(clean_text, 0.25)

    print(f"Original: {clean_text}")
    print(f"\nCorrupted: {corrupted}")
    print(f"\nStatistics:")
    print(f"  Total words: {stats.total_words}")
    print(f"  Errors injected: {stats.num_errors}")
    print(f"  Actual error rate: {stats.error_rate:.2%}")
    print(f"\nChanges made:")
    for original, corrupted_word in stats.changes:
        print(f"  {original} → {corrupted_word}")
