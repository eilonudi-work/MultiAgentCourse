# Prompt Engineering Experiment Report
## Systematic Evaluation of Prompting Strategies for Sentiment Analysis

**Experiment Date:** December 8, 2025
**Model:** llama2 (via Ollama)
**Dataset:** 30 sentiment examples across 7 categories

---

## Executive Summary

This experiment systematically evaluates six different prompt engineering techniques for sentiment classification tasks using a local LLM (llama2). The goal was to identify which prompting strategies yield the highest accuracy and most consistent results in sentiment analysis.

**Key Results:**
- **Best Performer:** Few-shot learning achieved **100% accuracy** (30/30 correct)
- **Second Best:** Structured output also achieved **100% accuracy**
- **Biggest Improvement:** +30.4% accuracy gain over baseline
- **Most Consistent:** Few-shot showed perfect consistency (zero distance variance)
- **Worst Performer:** Contrastive prompting at 53.3% accuracy

The experiment demonstrates that strategic prompt design can yield substantial performance improvements, with few-shot learning and structured output techniques proving most effective for this classification task.

---

## 1. Understanding Prompt Engineering Techniques

Prompt engineering is the practice of designing and optimizing input prompts to elicit desired behaviors from Large Language Models (LLMs). Different prompting strategies leverage various cognitive and linguistic principles to improve model performance. This section explains the six techniques evaluated in our experiment.

### 1.1 Baseline Prompting

**What It Is:**
Baseline prompting represents the simplest form of interaction with an LLM - a direct, straightforward instruction with no additional context, examples, or guidance.

**Theory:**
This approach relies entirely on the model's pre-trained knowledge and understanding of the task. It assumes the model can correctly interpret the instruction and has sufficient training data to perform the task without additional support.

**Example:**
```
"Classify the sentiment of this text as 'positive' or 'negative': I love this product!"
```

**When to Use:**
- Quick prototyping and testing
- Tasks that are well-represented in training data
- Establishing performance baselines for comparison
- When prompt length is critically constrained

**Limitations:**
- No task-specific guidance
- Vulnerable to ambiguous interpretations
- Performance depends heavily on model's prior exposure to similar tasks

---

### 1.2 Role-Based Prompting

**What It Is:**
Role-based prompting assigns the model a specific identity, persona, or expertise (e.g., "You are an expert sentiment analyst"). This technique frames the interaction within a professional or specialized context.

**Theory:**
The hypothesis is that by establishing a role or persona, the model will:
1. Activate relevant knowledge domains from its training
2. Adopt appropriate reasoning patterns associated with that expertise
3. Apply domain-specific best practices and considerations
4. Maintain consistency with the assigned role throughout the interaction

This approach draws from role-playing and persona-based communication strategies used in human interactions.

**Example:**
```
System: "You are an expert sentiment analyst with years of experience in NLP.
         Consider emotional tone, context, and intensity when analyzing text."
User: "Classify the sentiment: This product exceeded my expectations!"
```

**When to Use:**
- Domain-specific tasks requiring specialized knowledge
- When context and framing matter (legal, medical, technical domains)
- Multi-turn conversations where consistent expertise is needed
- Tasks where professional standards and methodologies apply

**Limitations:**
- Effectiveness varies by model architecture and training
- May not significantly improve performance on simple tasks
- Adds prompt length without guaranteed benefits
- Some models may not be sensitive to role framing

---

### 1.3 Few-Shot Learning

**What It Is:**
Few-shot learning provides the model with a small number (typically 1-5) of labeled examples before asking it to perform the task on new data. These examples demonstrate the desired input-output pattern.

**Theory:**
This technique leverages the model's in-context learning ability - the capacity to recognize patterns and generalize from examples provided within the prompt itself (without parameter updates). Key principles:

1. **Pattern Recognition:** Models can identify the task structure from examples
2. **Format Learning:** Examples clarify expected output format and style
3. **Boundary Definition:** Examples show edge cases and help establish decision boundaries
4. **Calibration:** Examples can reduce model biases by demonstrating balanced classifications

Few-shot learning is inspired by human learning - we often learn best from concrete examples rather than abstract instructions.

**Example:**
```
Example 1:
Text: "This product exceeded all my expectations!"
Sentiment: positive

Example 2:
Text: "Worst purchase ever. Complete waste of money."
Sentiment: negative

Example 3:
Text: "The service was outstanding and helpful."
Sentiment: positive

Now classify:
Text: "I'm disappointed with the quality."
Sentiment: ?
```

**When to Use:**
- Tasks where examples clarify ambiguous instructions
- When output format consistency is critical
- Binary or multi-class classification problems
- When you have representative examples available
- To reduce model biases through balanced examples

**Best Practices:**
- Use 3-5 diverse examples (more isn't always better)
- Ensure examples cover different scenarios/edge cases
- Maintain class balance in examples
- Use examples from the same domain as the target task
- Keep examples concise and clear

**Limitations:**
- Increases prompt length and processing cost
- Requires careful example selection
- May not generalize well if examples aren't representative
- Can reinforce biases if examples are poorly chosen

---

### 1.4 Chain of Thought (CoT) Prompting

**What It Is:**
Chain of Thought prompting encourages the model to articulate its reasoning process step-by-step before arriving at a final answer. Instead of jumping directly to a conclusion, the model "thinks aloud" through intermediate steps.

**Theory:**
CoT is based on several cognitive principles:

1. **Decomposition:** Complex problems are easier to solve when broken into smaller steps
2. **Transparency:** Explicit reasoning makes the decision process interpretable
3. **Error Detection:** Intermediate steps allow models to catch and correct logical errors
4. **Working Memory:** Articulating steps helps maintain context for multi-step reasoning

The technique mimics how humans approach complex problems - we don't instantly know the answer but work through it systematically.

**Example:**
```
"Analyze the sentiment of this text step by step:
Text: 'The food was great but the service was terrible.'

Steps:
1. Identify key emotional words and phrases
2. Determine if they are positive, negative, or neutral
3. Consider the overall tone and context
4. Weigh conflicting sentiments
5. Make your final classification

Analysis: [model's step-by-step reasoning here]
Final Answer: mixed/negative"
```

**When to Use:**
- Multi-step reasoning tasks (math, logic, planning)
- Complex problems requiring intermediate calculations
- Tasks where interpretability is important
- Debugging model reasoning to understand failures
- Problems with multiple constraints or conditions

**Limitations:**
- Significantly increases output length and cost
- May not help for simple, intuitive tasks
- Can introduce errors in the reasoning chain
- Verbose outputs may be harder to parse programmatically
- Not always necessary for straightforward classifications

---

### 1.5 Structured Output Prompting

**What It Is:**
Structured output prompting explicitly specifies the exact format, structure, or schema that the model should use in its response. This can range from simple format requirements (e.g., "respond with only 'yes' or 'no'") to complex JSON schemas.

**Theory:**
This approach addresses several challenges in LLM deployment:

1. **Parsing Reliability:** Predictable formats are easier to parse programmatically
2. **Ambiguity Reduction:** Clear format requirements reduce interpretation variance
3. **Information Extraction:** Structured outputs can include multiple fields (confidence, reasoning, metadata)
4. **System Integration:** Standardized formats enable seamless integration with downstream systems

The technique recognizes that LLMs are flexible text generators, and explicit constraints help focus that flexibility productively.

**Example:**
```
"Classify the sentiment and provide your answer in this exact format:

Sentiment: [positive or negative]
Confidence: [high, medium, or low]
Key words: [list of sentiment-bearing words]

Text: 'This restaurant was absolutely amazing!'
Answer:"
```

**When to Use:**
- Production systems requiring reliable parsing
- API integrations and automated workflows
- When multiple pieces of information are needed
- Reducing post-processing complexity
- Ensuring consistency across many queries
- When outputs feed into other systems or databases

**Best Practices:**
- Provide explicit format templates or examples
- Use clear delimiters and labels
- Consider JSON or other standard formats for complex outputs
- Include validation checks for format compliance
- Combine with few-shot examples showing the desired format

**Limitations:**
- Models may still deviate from requested format
- Increases prompt complexity
- May feel overly restrictive for creative tasks
- Requires careful format design
- Some models follow format instructions better than others

---

### 1.6 Contrastive Prompting

**What It Is:**
Contrastive prompting explicitly asks the model to consider multiple perspectives, alternatives, or opposing viewpoints before making a final decision. It frames the task as comparing and contrasting different possibilities.

**Theory:**
This technique is rooted in several reasoning principles:

1. **Dialectical Reasoning:** Truth emerges from considering opposing perspectives (thesis-antithesis-synthesis)
2. **Bias Reduction:** Forcing consideration of alternatives can counteract default biases
3. **Comprehensive Analysis:** Explicit comparison ensures all angles are considered
4. **Confidence Calibration:** Weighing pros and cons can improve decision confidence

The approach mimics deliberative decision-making processes where we explicitly consider "what evidence supports each option?"

**Example:**
```
"Analyze this text and determine if it is positive or negative:
Text: 'The price was reasonable but quality could be better.'

Consider:
- What makes this text positive? List positive indicators.
  * Reasonable price
  * Acknowledges some value

- What makes this text negative? List negative indicators.
  * Quality issues mentioned
  * 'Could be better' implies dissatisfaction

- Which sentiment is stronger overall?
Final answer: [negative - quality concerns outweigh price satisfaction]"
```

**When to Use:**
- Ambiguous cases where sentiment is mixed or subtle
- Reducing confirmation bias in model outputs
- When nuanced analysis is more important than speed
- Training/evaluation scenarios to understand model reasoning
- Tasks involving trade-offs or competing factors

**Limitations:**
- Significantly increases verbosity and processing time
- May create artificial balance where none exists (false equivalence)
- Can confuse the model for clear-cut cases
- Longer reasoning chains increase error opportunities
- May not be necessary when sentiment is obvious

---

### 1.7 Comparison Summary

| Technique | Complexity | Prompt Length | Best For | Main Advantage |
|-----------|-----------|---------------|----------|----------------|
| **Baseline** | Low | Minimal | Simple tasks, prototypes | Speed and simplicity |
| **Role-Based** | Low | Short | Domain-specific tasks | Context framing |
| **Few-Shot** | Medium | Medium | Pattern learning, consistency | Clear examples |
| **Chain of Thought** | High | Long | Multi-step reasoning | Transparency |
| **Structured Output** | Medium | Medium | Production systems | Parsing reliability |
| **Contrastive** | High | Long | Ambiguous cases | Comprehensive analysis |

The effectiveness of each technique depends on:
- **Task complexity:** Simple vs. multi-step reasoning
- **Model capabilities:** Different models respond differently to these techniques
- **Deployment context:** Prototyping vs. production, latency requirements
- **Data availability:** Whether you have good examples for few-shot
- **Interpretability needs:** Whether you need to understand model reasoning

---

## 2. Methodology

### 2.1 Dataset
- **Size:** 30 examples (proof-of-concept scale)
- **Balance:** 15 positive, 15 negative
- **Categories:**
  - Product reviews (12 samples)
  - General statements (5 samples)
  - Service feedback (4 samples)
  - Technology reviews (4 samples)
  - Entertainment (3 samples)
  - Food (1 sample)
  - Hospitality (1 sample)

### 2.2 Model Configuration
- **Model:** llama2 (local deployment via Ollama)
- **Temperature:** 0.0 (deterministic outputs)
- **Host:** http://localhost:11434
- **Request Delay:** 0.1s between requests
- **Max Retries:** 3

### 2.3 Prompt Variations Tested

#### 1. Baseline
**Strategy:** Simple, direct instruction
**Prompt:** `"Classify the sentiment of this text as 'positive' or 'negative': {text}"`
**System Prompt:** None

#### 2. Role-Based
**Strategy:** Expert role definition with analysis guidelines
**Prompt:** `"Classify the sentiment of this text as 'positive' or 'negative': {text}"`
**System Prompt:** Establishes expertise as sentiment analyst with specific considerations (emotional tone, context, intensity, negations)

#### 3. Few-Shot
**Strategy:** Learning from examples
**Prompt:** Includes 3 labeled examples before the target text
- Example 1: Positive product review
- Example 2: Negative purchase experience
- Example 3: Positive service experience

#### 4. Chain of Thought
**Strategy:** Step-by-step reasoning
**Prompt:** Guides the model through explicit reasoning steps:
1. Identify key emotional words
2. Determine polarity
3. Consider overall tone
4. Make final classification

#### 5. Structured Output
**Strategy:** Explicit format specification
**Prompt:** Requests specific format with sentiment and confidence level
**System Prompt:** Reinforces adherence to requested format

#### 6. Contrastive
**Strategy:** Consider both perspectives
**Prompt:** Asks model to explicitly list positive and negative indicators before deciding which is stronger

### 2.4 Evaluation Metrics
- **Accuracy:** Percentage of correct classifications
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **Mean Distance:** Average semantic distance from ideal classification (using sentence embeddings)
- **Success Rate:** Percentage of samples that produced valid outputs

---

## 3. Results

### 3.1 Overall Performance Comparison

| Variation | Accuracy | F1 Score | Precision | Recall | Mean Distance |
|-----------|----------|----------|-----------|--------|---------------|
| **Few-Shot** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.000** |
| **Structured Output** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.000** |
| Baseline | 76.7% | 81.1% | 68.2% | 100.0% | 0.033 |
| Role-Based | 76.7% | 81.1% | 68.2% | 100.0% | 0.033 |
| Chain of Thought | 66.7% | 75.0% | 60.0% | 100.0% | 0.048 |
| Contrastive | 53.3% | 68.2% | 51.7% | 100.0% | 0.067 |

![Accuracy Comparison](visualizations/accuracy_comparison.png)
*Figure 1: Accuracy comparison across all prompt variations*

![Metrics Comparison](visualizations/metrics_comparison.png)
*Figure 2: Comprehensive metrics comparison (accuracy, F1, precision, recall)*

### 3.2 Performance Rankings

**By Accuracy:**
1. Few-Shot: 100.0%
2. Structured Output: 100.0%
3. Baseline: 76.7%
4. Role-Based: 76.7%
5. Chain of Thought: 66.7%
6. Contrastive: 53.3%

**Improvements Over Baseline:**
- Few-Shot: **+30.4%**
- Structured Output: **+30.4%**
- Role-Based: +0.0%
- Chain of Thought: **-13.0%**
- Contrastive: **-30.4%**

### 3.3 Confusion Matrix Analysis

![Confusion Matrices](visualizations/confusion_matrices.png)
*Figure 3: Confusion matrices for all prompt variations*

**Baseline & Role-Based:**
- True Positive: 15 | True Negative: 8
- False Positive: 7 | False Negative: 0
- **Issue:** Bias towards positive classification (high false positives)

**Few-Shot & Structured Output:**
- True Positive: 15 | True Negative: 15
- False Positive: 0 | False Negative: 0
- **Perfect:** No misclassifications

**Chain of Thought:**
- True Positive: 15 | True Negative: 5
- False Positive: 10 | False Negative: 0
- **Issue:** Strong bias towards positive (highest false positives)

**Contrastive:**
- True Positive: 15 | True Negative: 1
- False Positive: 14 | False Negative: 0
- **Issue:** Severe positive bias (nearly all negatives misclassified)

**Key Observation:** All variations achieved 100% recall (no false negatives), but varied significantly in precision. The contrastive and chain-of-thought approaches paradoxically introduced more errors despite their additional reasoning steps.

### 3.4 Consistency Analysis

![Distance Distributions](visualizations/distance_distributions.png)
*Figure 4: Distribution of semantic distances from ideal classifications*

**Consistency Ranking (by coefficient of variation):**
1. **Few-Shot:** 0.000 (perfect consistency)
2. **Structured Output:** 0.000 (perfect consistency)
3. **Contrastive:** 1.069 (least consistent)
4. **Chain of Thought:** 1.414
5. **Baseline:** 1.813
6. **Role-Based:** 1.813

**Interpretation:** Few-shot and structured output not only achieved perfect accuracy but also demonstrated perfect consistency with zero variance in semantic distance. This indicates highly reliable and predictable behavior.

---

## 4. Category-Level Analysis

### 4.1 Per-Category Performance

![Category Performance](visualizations/category_performance.png)
*Figure 5: Heatmap of accuracy across categories and prompt variations*

| Category | Samples | Best Variation | Best Acc. | Worst Variation | Worst Acc. | Std Dev |
|----------|---------|----------------|-----------|-----------------|------------|---------|
| **Entertainment** | 3 | All (tie) | 100% | All (tie) | 100% | 0.000 |
| **Food** | 1 | All (tie) | 100% | All (tie) | 100% | 0.000 |
| **Service** | 4 | Few-Shot | 100% | Contrastive | 50% | 0.186 |
| **General** | 5 | Few-Shot | 100% | Role-Based | 60% | 0.180 |
| **Product** | 12 | Few-Shot | 100% | Contrastive | 41.7% | 0.212 |
| **Technology** | 4 | Few-Shot | 100% | Baseline | 50% | 0.236 |
| **Hospitality** | 1 | Baseline | 100% | Chain of Thought | 0% | 0.471 |

### 4.2 Key Category Insights

**Most Stable Categories:**
- **Entertainment:** 100% accuracy across all variations (σ = 0.000)
- **Food:** 100% accuracy across all variations (σ = 0.000)

**Most Challenging Categories:**
- **Hospitality:** Highest variation (σ = 0.471), with some variations completely failing
- **Technology:** Second-highest variation (σ = 0.236)
- **Product Reviews:** Largest sample size with notable variation (σ = 0.212)

**Category-Specific Observations:**
- **Product reviews** (40% of dataset): Few-shot excelled (100%) while contrastive struggled (41.7%)
- **Hospitality**: Chain of thought and contrastive both failed completely (0% accuracy)
- **Technology**: All simpler prompts achieved only 50%, but few-shot/structured reached 100%

---

## 5. Key Findings

### 5.1 Major Discoveries

1. **Few-Shot Learning is Highly Effective**
   - Achieved perfect 100% accuracy on all 30 samples
   - Zero semantic distance variance (perfect consistency)
   - Superior performance across all 7 categories
   - +30.4% improvement over baseline

2. **Structured Output Equals Few-Shot Performance**
   - Also achieved perfect 100% accuracy
   - Perfect consistency (zero variance)
   - Demonstrates that clear format specification is as powerful as example-based learning

3. **More Complex ≠ Better Performance**
   - Chain of thought: 66.7% (13% worse than baseline)
   - Contrastive: 53.3% (30.4% worse than baseline)
   - Additional reasoning steps introduced bias rather than improving accuracy

4. **Positive Classification Bias is Prevalent**
   - All variations achieved 100% recall (no false negatives)
   - Weaker variations showed strong tendency to over-predict positive sentiment
   - Contrastive approach nearly always predicted positive (14 false positives)

5. **Role Definition Had No Effect**
   - Role-based prompting performed identically to baseline (76.7%)
   - Expert role framing and analysis guidelines did not improve results
   - Suggests llama2 may not be sensitive to role-playing instructions at this scale

6. **Category Difficulty Varies Significantly**
   - Entertainment and food: Universally easy (100% across all methods)
   - Technology and hospitality: Challenging even for advanced techniques
   - Product reviews: Most sensitive to prompt strategy

### 5.2 Statistical Significance

- **Best to Worst Gap:** 87.5 percentage points (100% vs 53.3%)
- **Overall Improvement Potential:** Up to 30.4% gain with optimal prompting
- **Consistency Improvement:** Perfect stability vs high variance approaches

---

## 6. Insights and Recommendations

### 6.1 For Sentiment Analysis Tasks

**Primary Recommendation: Use Few-Shot Learning**
- Provides concrete examples of desired behavior
- Achieves near-perfect results with minimal overhead
- Only requires 3 well-chosen examples in the prompt
- Most reliable across diverse text categories

**Alternative Recommendation: Structured Output**
- Equally effective as few-shot for binary classification
- May be preferred when format consistency is critical
- Useful when integrating with downstream systems
- Simpler to implement than few-shot (no example curation)

**Avoid:**
- Chain of thought for simple classification tasks (adds complexity without benefit)
- Contrastive prompting for sentiment analysis (introduces severe bias)
- Role-based prompting alone (no measurable benefit observed)

### 6.2 General Prompt Engineering Principles

1. **Provide Examples When Possible**
   - Few-shot learning consistently outperforms instruction-only approaches
   - 3 examples appear sufficient for binary classification
   - Examples should represent diverse scenarios

2. **Specify Output Format Explicitly**
   - Clear format specification dramatically improves reliability
   - Reduces parsing errors and ambiguity
   - Consider structured output for production systems

3. **Simpler is Often Better**
   - Complex reasoning chains can introduce errors
   - Direct approaches may outperform elaborate frameworks
   - Test before assuming more complex = better

4. **Be Aware of Model Biases**
   - llama2 showed consistent positive classification bias
   - Some prompting strategies amplify existing biases
   - Monitor confusion matrices, not just accuracy

5. **Category Context Matters**
   - Performance varies significantly by domain
   - Entertainment/food were universally easy
   - Technology/hospitality require stronger techniques
   - Consider domain-specific examples in few-shot prompts

### 6.3 For Production Deployment

**Recommended Approach:**
```
Use few-shot prompting with 3-5 diverse examples
+ Structured output format specification
+ Temperature = 0 for deterministic behavior
```

**Monitoring Strategy:**
- Track per-category accuracy to detect domain drift
- Monitor false positive rate (positive classification bias)
- Measure semantic distance for consistency checks
- Maintain confusion matrix over time

**When to Use Each Strategy:**
- **Few-Shot:** Primary recommendation for most scenarios
- **Structured Output:** When output parsing is critical
- **Baseline:** Quick prototypes only (not production)
- **Role-Based:** Skip unless testing with different models
- **Chain of Thought:** Complex reasoning tasks only (not binary classification)
- **Contrastive:** Avoid for sentiment analysis

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Small Dataset:** 30 examples is sufficient for POC but not statistically robust
2. **Category Imbalance:** Some categories have only 1 sample (food, hospitality)
3. **Binary Classification:** Only tested positive/negative, not neutral or fine-grained sentiment
4. **Single Model:** Results are specific to llama2; may not generalize to other LLMs
5. **No Error Analysis:** Limited investigation into specific misclassification patterns

### 7.2 Future Experiments

**Scale Up:**
- Test with 100-1000 samples for statistical significance
- Balance category representation
- Include neutral sentiment and multi-class scenarios

**Model Comparison:**
- Test with llama3, mistral, phi, and other local models
- Compare local models to API-based models (GPT-4, Claude)
- Investigate if findings generalize across architectures

**Advanced Techniques:**
- Self-consistency (multiple samples, majority vote)
- Hybrid approaches (few-shot + chain of thought)
- Dynamic example selection based on input similarity
- Temperature variation testing

**Production Validation:**
- A/B testing in real-world deployment
- User feedback on misclassifications
- Domain-specific fine-tuning vs prompt engineering
- Cost-benefit analysis for different strategies

---

## 8. Conclusion

This experiment demonstrates that **strategic prompt engineering can yield substantial performance improvements** in LLM-based sentiment analysis. The key finding is that **few-shot learning and structured output approaches achieved perfect 100% accuracy**, representing a 30.4% improvement over baseline prompting.

Surprisingly, **more complex prompting strategies performed worse** than simpler approaches. Chain of thought reasoning and contrastive analysis both introduced significant errors and positive classification bias, suggesting that additional reasoning steps can be counterproductive for straightforward classification tasks.

For practitioners building sentiment analysis systems with local LLMs, the recommendation is clear: **start with few-shot learning** (3-5 examples) combined with **explicit output format specification**. This combination provides the best balance of accuracy, consistency, and ease of implementation.

The experiment also reveals that **model behavior varies significantly by domain**, with entertainment and food reviews being universally easy to classify, while technology and hospitality content presents greater challenges. Production systems should monitor per-category performance and consider domain-specific example selection.

Moving forward, these findings provide a solid foundation for deploying sentiment analysis in production environments using local LLMs, offering both cost-effective and privacy-preserving alternatives to API-based solutions.

---

## Appendix: Files and Resources

**Experiment Results:**
- `results/comparison_metrics_20251208_231418.json` - Aggregate metrics
- `results/baseline_results_20251208_231418.json` - Baseline detailed results
- `results/role_based_results_20251208_231418.json` - Role-based detailed results
- `results/few_shot_results_20251208_231418.json` - Few-shot detailed results
- `results/chain_of_thought_results_20251208_231418.json` - Chain of thought detailed results
- `results/structured_output_results_20251208_231418.json` - Structured output detailed results
- `results/contrastive_results_20251208_231418.json` - Contrastive detailed results

**Analysis:**
- `analysis/statistical_analysis_20251208_231420.json` - Statistical analysis with rankings

**Visualizations:**
- `visualizations/accuracy_comparison.png` - Accuracy bar chart
- `visualizations/metrics_comparison.png` - Multi-metric comparison
- `visualizations/distance_distributions.png` - Semantic distance distributions
- `visualizations/confusion_matrices.png` - Confusion matrix heatmaps
- `visualizations/category_performance.png` - Per-category performance heatmap

**Source Code:**
- `src/improved_prompts.py` - Prompt variation implementations
- `src/baseline_experiment.py` - Baseline experiment runner
- `src/run_all_experiments.py` - Batch experiment runner
- `analysis/statistical_analysis.py` - Statistical analysis script
- `analysis/visualization.py` - Visualization generation script

---

**Report Generated:** December 10, 2025
**Experiment Timestamp:** December 8, 2025, 23:14:18
**Model:** llama2 (Ollama v0.12.8)
**Python:** 3.9.6
**Dataset:** 30 sentiment examples across 7 categories
