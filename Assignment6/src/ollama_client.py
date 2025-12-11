"""
Ollama client wrapper for LLM API calls with retry logic.
"""

import time
import requests
from typing import Optional, Dict, Any
from config import Config


class OllamaClient:
    """Wrapper for Ollama API calls with error handling and retry logic."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model to use (default from config)
        """
        self.model_name = model_name or Config.get_model_name()
        self.host = Config.OLLAMA_HOST
        self.temperature = Config.TEMPERATURE
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY
        self.request_delay = Config.REQUEST_DELAY

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response from Ollama with retry logic.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context

        Returns:
            Dict containing 'response', 'success', 'error', and metadata
        """
        url = f"{self.host}/api/generate"

        # Combine system and user prompts if system prompt provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                result = response.json()

                # Add delay between requests
                time.sleep(self.request_delay)

                return {
                    "response": result.get("response", "").strip(),
                    "success": True,
                    "error": None,
                    "model": result.get("model"),
                    "total_duration": result.get("total_duration"),
                    "eval_count": result.get("eval_count"),
                    "prompt": full_prompt
                }

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "response": None,
                        "success": False,
                        "error": str(e),
                        "model": self.model_name,
                        "prompt": full_prompt
                    }

        return {
            "response": None,
            "success": False,
            "error": "Max retries exceeded",
            "model": self.model_name,
            "prompt": full_prompt
        }

    def classify_sentiment(self, text: str, prompt_template: str) -> Dict[str, Any]:
        """
        Classify sentiment of text using a prompt template.

        Args:
            text: Text to classify
            prompt_template: Template with {text} placeholder

        Returns:
            Dict with classification result and metadata
        """
        prompt = prompt_template.format(text=text)
        result = self.generate(prompt)

        if result["success"]:
            # Extract sentiment from response (positive/negative)
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

        return result

    def check_connection(self) -> bool:
        """
        Check if Ollama server is running and accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Ollama connection failed: {e}")
            print(f"Make sure Ollama is running at {self.host}")
            return False


if __name__ == "__main__":
    # Test the client
    client = OllamaClient()

    print("Testing Ollama connection...")
    if client.check_connection():
        print("✓ Connection successful!")

        # Test sentiment classification
        print("\nTesting sentiment classification...")
        test_text = "This product is amazing!"
        result = client.classify_sentiment(
            test_text,
            "Classify the sentiment of this text as 'positive' or 'negative': {text}"
        )

        print(f"Text: {test_text}")
        print(f"Sentiment: {result.get('sentiment')}")
        print(f"Response: {result.get('response')}")
    else:
        print("✗ Connection failed. Please start Ollama and try again.")
