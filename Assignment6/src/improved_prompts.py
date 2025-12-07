"""
Improved prompt variations for sentiment analysis.
Implements different prompting strategies to compare performance.
"""


class PromptVariations:
    """Collection of different prompt strategies for sentiment analysis."""

    @staticmethod
    def get_baseline():
        """
        Baseline prompt - simple, direct instruction.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "baseline"
        prompt_template = "Classify the sentiment of this text as 'positive' or 'negative': {text}"
        system_prompt = None
        return name, prompt_template, system_prompt

    @staticmethod
    def get_role_based():
        """
        Improved prompt with role definition.
        Establishes expertise and context for the model.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "role_based"
        prompt_template = "Classify the sentiment of this text as 'positive' or 'negative': {text}"
        system_prompt = """You are an expert sentiment analyst with years of experience in natural language processing and text analysis. Your task is to accurately determine whether text expresses positive or negative sentiment.

When analyzing text, consider:
- The overall emotional tone
- Context and implicit meanings
- Intensity of sentiment words
- Negations and qualifiers

Respond with ONLY the word 'positive' or 'negative'."""
        return name, prompt_template, system_prompt

    @staticmethod
    def get_few_shot():
        """
        Few-shot learning with examples.
        Provides 3 examples to guide the model.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "few_shot"
        prompt_template = """Here are some examples of sentiment classification:

Example 1:
Text: "This product exceeded all my expectations! Absolutely love it."
Sentiment: positive

Example 2:
Text: "Worst purchase I've ever made. Complete waste of money."
Sentiment: negative

Example 3:
Text: "The service was outstanding and the staff was incredibly helpful."
Sentiment: positive

Now classify this text:
Text: {text}
Sentiment:"""
        system_prompt = None
        return name, prompt_template, system_prompt

    @staticmethod
    def get_chain_of_thought():
        """
        Chain of thought prompting.
        Encourages step-by-step reasoning before final classification.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "chain_of_thought"
        prompt_template = """Analyze the sentiment of the following text step by step:

Text: {text}

Please follow these steps:
1. Identify key emotional words and phrases
2. Determine if they are positive, negative, or neutral
3. Consider the overall tone and context
4. Make your final classification

After your analysis, provide your final answer as either 'positive' or 'negative'."""
        system_prompt = None
        return name, prompt_template, system_prompt

    @staticmethod
    def get_structured_output():
        """
        Structured output prompting.
        Requests specific format to improve parsing reliability.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "structured_output"
        prompt_template = """Classify the sentiment of the following text.

Text: {text}

Provide your answer in this exact format:
Sentiment: [positive or negative]
Confidence: [high, medium, or low]

Answer:"""
        system_prompt = "You are a sentiment analysis assistant. Always respond in the exact format requested."
        return name, prompt_template, system_prompt

    @staticmethod
    def get_contrastive():
        """
        Contrastive prompting.
        Explicitly asks model to consider both possibilities.

        Returns:
            Tuple of (name, prompt_template, system_prompt)
        """
        name = "contrastive"
        prompt_template = """Analyze this text and determine if it is positive or negative:

Text: {text}

Consider:
- What makes this text positive? List positive indicators.
- What makes this text negative? List negative indicators.
- Which sentiment is stronger overall?

Final answer (positive or negative):"""
        system_prompt = None
        return name, prompt_template, system_prompt

    @staticmethod
    def get_all_variations():
        """
        Get all prompt variations.

        Returns:
            Dict mapping variation names to (prompt_template, system_prompt) tuples
        """
        variations = {
            "baseline": PromptVariations.get_baseline(),
            "role_based": PromptVariations.get_role_based(),
            "few_shot": PromptVariations.get_few_shot(),
            "chain_of_thought": PromptVariations.get_chain_of_thought(),
            "structured_output": PromptVariations.get_structured_output(),
            "contrastive": PromptVariations.get_contrastive(),
        }

        # Return as dict with name -> (prompt, system_prompt)
        return {
            name: (prompt, system)
            for name, prompt, system in variations.values()
        }

    @staticmethod
    def get_variation(name):
        """
        Get a specific prompt variation by name.

        Args:
            name: Name of the variation

        Returns:
            Tuple of (prompt_template, system_prompt)

        Raises:
            ValueError: If variation name not found
        """
        variations = PromptVariations.get_all_variations()
        if name not in variations:
            available = ", ".join(variations.keys())
            raise ValueError(f"Unknown variation '{name}'. Available: {available}")
        return variations[name]

    @staticmethod
    def describe_variation(name):
        """
        Get description of a prompt variation.

        Args:
            name: Name of the variation

        Returns:
            Description string
        """
        descriptions = {
            "baseline": "Simple, direct instruction with no additional context",
            "role_based": "Enhanced with expert role definition and guidelines",
            "few_shot": "Includes 3 examples to guide the model",
            "chain_of_thought": "Encourages step-by-step reasoning process",
            "structured_output": "Requests specific output format for reliability",
            "contrastive": "Asks model to consider both positive and negative aspects"
        }
        return descriptions.get(name, "No description available")


def main():
    """Demo the prompt variations."""
    print("=" * 60)
    print("Available Prompt Variations")
    print("=" * 60)
    print()

    variations = PromptVariations.get_all_variations()

    for name, (prompt, system) in variations.items():
        print(f"Variation: {name}")
        print(f"Description: {PromptVariations.describe_variation(name)}")
        print()

        if system:
            print("System Prompt:")
            print(f"  {system[:100]}..." if len(system) > 100 else f"  {system}")
            print()

        print("Prompt Template:")
        sample_text = "This is a test"
        formatted = prompt.format(text=sample_text)
        preview = formatted[:150] + "..." if len(formatted) > 150 else formatted
        print(f"  {preview}")
        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
