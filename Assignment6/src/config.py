"""
Configuration module for Ollama LLM integration.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for Ollama settings."""

    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    MODEL_NAME = os.getenv('MODEL_NAME', 'llama2')

    # Rate limiting
    REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '0.1'))

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    # Model parameters
    TEMPERATURE = 0.0  # Deterministic for reproducibility

    @classmethod
    def get_ollama_url(cls):
        """Get the full Ollama API URL."""
        return f"{cls.OLLAMA_HOST}/api/generate"

    @classmethod
    def get_model_name(cls):
        """Get the configured model name."""
        return cls.MODEL_NAME

    @classmethod
    def display_config(cls):
        """Display current configuration."""
        print("=== Ollama Configuration ===")
        print(f"Host: {cls.OLLAMA_HOST}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Request Delay: {cls.REQUEST_DELAY}s")
        print(f"Max Retries: {cls.MAX_RETRIES}")
        print("=" * 30)


if __name__ == "__main__":
    Config.display_config()
