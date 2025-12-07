# Project Plan: Prompt Engineering Assignment Implementation

## Executive Summary

This project implements a systematic study of prompt engineering techniques using sentiment analysis as the domain. The plan delivers a complete end-to-end solution demonstrating how different prompting strategies affect model performance, with quantitative measurements and visual comparisons.

**Estimated Duration:** 3-4 days
**Complexity Level:** Low-Medium
**Primary Deliverable:** Working implementation with comparison graphs and analysis report

---

## 1. Project Scope & Approach

### 1.1 Domain Selection: Sentiment Analysis

**Rationale:**
- **Simple & Clear**: Binary classification (positive/negative) with obvious ground truth
- **Token Efficient**: Short sentences (10-20 words) minimize API costs
- **Easy Validation**: Straightforward distance metrics for answer comparison
- **Real-world Relevance**: Common NLP task with practical applications

### 1.2 Dataset Specifications

**Dataset Size:** 100 question-answer pairs
- 50 positive sentiment examples
- 50 negative sentiment examples

**Format:**
```json
{
  "text": "The movie was absolutely fantastic!",
  "ground_truth": "positive",
  "category": "entertainment"
}
```

**Sources:**
- Product reviews (Amazon, Yelp)
- Movie reviews (IMDb)
- Social media posts
- Manual curation for quality

### 1.3 Prompt Strategies to Test

1. **Baseline**: Simple, direct instruction
2. **Standard Improvement**: Enhanced system prompt with role definition
3. **Few-Shot Learning**: 2-3 examples included in prompt
4. **Chain of Thought**: Step-by-step reasoning instruction
5. **(Optional) ReAct**: External sentiment lexicon tool integration

---

## 2. Implementation Phases

### Phase 1: Foundation Setup (Day 1, ~4 hours)

#### Tasks:
1. **Environment Setup**
   - Create project directory structure
   - Install required Python packages
   - Configure API keys (OpenAI/Anthropic)
   - Set up version control

2. **Dataset Creation**
   - Collect 100 sentiment examples
   - Format as JSON with standardized structure
   - Validate data quality (no duplicates, clear sentiment)
   - Split into categories for analysis

#### Deliverables:
- `/data/sentiment_dataset.json` - Complete dataset
- `/config/api_config.py` - API configuration
- `requirements.txt` - Python dependencies
- `README.md` - Setup instructions

#### Success Criteria:
- [ ] Dataset contains exactly 100 valid entries
- [ ] All entries have clear, unambiguous sentiment
- [ ] API connection successfully established
- [ ] Environment reproducible via requirements.txt

---

### Phase 2: Baseline Implementation (Day 1-2, ~6 hours)

#### Tasks:
1. **Baseline Prompt Creation**
   ```
   Simple prompt: "Classify the sentiment of this text as positive or negative: {text}"
   ```

2. **Execution Pipeline**
   - Build API call handler with retry logic
   - Implement response parsing (extract positive/negative)
   - Add error handling for malformed responses
   - Create progress tracking (important for 100 API calls)

3. **Distance Metric Implementation**
   - Text similarity using sentence embeddings (sentence-transformers)
   - Binary match score (1.0 for exact match, 0.0 for mismatch)
   - Confidence score extraction from model responses

4. **Results Storage**
   - Save raw responses to JSON
   - Store distance metrics in structured format
   - Log API call metadata (tokens, latency)

#### Deliverables:
- `/src/baseline_experiment.py` - Baseline runner
- `/src/metrics.py` - Distance calculation functions
- `/results/baseline_results.json` - Raw results
- `/results/baseline_metrics.csv` - Computed metrics

#### Success Criteria:
- [ ] All 100 examples processed successfully
- [ ] Distance metrics calculated for each response
- [ ] Mean accuracy > 70% (sanity check)
- [ ] Results reproducible

---

### Phase 3: Prompt Variations (Day 2, ~6 hours)

#### Tasks:

**Variation 1: Standard Improvement**
```python
system_prompt = """You are an expert sentiment analyst with years of experience
in natural language processing. Your task is to accurately determine whether
text expresses positive or negative sentiment. Consider context, tone, and
implicit meanings. Respond with only 'positive' or 'negative'."""
```

**Variation 2: Few-Shot Learning**
```python
examples = """
Example 1:
Text: "This product exceeded all my expectations!"
Sentiment: positive

Example 2:
Text: "Worst purchase I've ever made, total waste of money."
Sentiment: negative

Example 3:
Text: "The service was outstanding and the staff was very helpful."
Sentiment: positive

Now classify this text: {text}
"""
```

**Variation 3: Chain of Thought**
```python
cot_prompt = """Analyze the sentiment of the following text step by step:

1. Identify key emotional words
2. Determine the overall tone
3. Consider contextual factors
4. Make final classification

Text: {text}

Think through each step, then provide your final answer as either 'positive' or 'negative'."""
```

**Variation 4 (Optional): ReAct with Sentiment Lexicon**
```python
tools = {
    "sentiment_lexicon": {
        "description": "Look up sentiment score of a word (-1 to +1)",
        "implementation": "Load VADER or AFINN lexicon"
    }
}
```

#### Deliverables:
- `/src/improved_prompts.py` - All prompt variations
- `/src/run_experiments.py` - Batch experiment runner
- `/results/variation_1_results.json`
- `/results/variation_2_results.json`
- `/results/variation_3_results.json`
- `/results/variation_4_results.json` (if implemented)

#### Success Criteria:
- [ ] Each variation processes all 100 examples
- [ ] Consistent response format across variations
- [ ] Results comparable using same metrics
- [ ] At least one variation shows improvement

---

### Phase 4: Analysis & Visualization (Day 3, ~5 hours)

#### Tasks:

1. **Statistical Analysis**
   - Calculate mean accuracy for each variation
   - Compute variance and standard deviation
   - Create confusion matrices
   - Identify patterns in failures

2. **Histogram Creation**
   ```python
   # Distance distribution for each prompt variation
   # X-axis: Distance bins (0.0-0.2, 0.2-0.4, etc.)
   # Y-axis: Frequency count
   ```

3. **Comparison Graphs**
   - Bar chart: Mean accuracy across variations
   - Line plot: Performance distribution
   - Heatmap: Success rate by category
   - Box plots: Variance comparison

4. **Insight Documentation**
   - Why did certain prompts perform better?
   - Which types of texts benefited most from few-shot?
   - Did CoT help with complex/ambiguous cases?
   - Token cost vs. accuracy trade-offs

#### Deliverables:
- `/analysis/statistical_summary.py` - Metric calculations
- `/visualizations/histograms.png` - Distance distributions
- `/visualizations/comparison_chart.png` - Performance comparison
- `/analysis/insights.md` - Written analysis
- `/analysis/token_analysis.csv` - Cost breakdown

#### Success Criteria:
- [ ] Clear visual comparison showing improvements/degradations
- [ ] Statistical significance documented
- [ ] Insights explain why changes helped/hurt
- [ ] Token costs quantified

---

### Phase 5: Documentation & Presentation (Day 4, ~3 hours)

#### Tasks:

1. **Code Documentation**
   - Add docstrings to all functions
   - Create usage examples
   - Document API rate limiting approach
   - Add inline comments for complex logic

2. **Results Report**
   - Executive summary of findings
   - Methodology explanation
   - Results presentation with graphs
   - Conclusions and recommendations

3. **Reproducibility Package**
   - Step-by-step replication guide
   - Sample output files
   - Troubleshooting section
   - Environment specifications

#### Deliverables:
- `REPORT.md` - Complete analysis report
- `USAGE.md` - How to run the code
- `/examples/sample_output/` - Example results
- Presentation-ready slides (optional)

#### Success Criteria:
- [ ] Another person can reproduce results
- [ ] Graphs clearly show improvement/degradation
- [ ] Insights are actionable and specific
- [ ] Assignment requirements fully addressed

---

## 3. Technical Requirements

### 3.1 Core Technologies

**Programming Language:** Python 3.9+

**Essential Libraries:**
```
openai==1.12.0              # API access
anthropic==0.18.1           # Alternative LLM option
sentence-transformers==2.5.1  # Embeddings for distance
numpy==1.24.3               # Numerical operations
pandas==2.0.2               # Data manipulation
matplotlib==3.7.1           # Visualization
seaborn==0.12.2             # Enhanced plots
scikit-learn==1.3.0         # Metrics utilities
tqdm==4.65.0                # Progress tracking
python-dotenv==1.0.0        # Environment variables
```

**Optional Tools:**
```
vaderSentiment==3.3.2       # For ReAct variation
requests==2.31.0            # API calls
jupyter==1.0.0              # Interactive analysis
```

### 3.2 API Requirements

**LLM Access:**
- OpenAI API key (GPT-4 or GPT-3.5-turbo)
- OR Anthropic API key (Claude)
- Estimated cost: $5-15 for all experiments

**Rate Limiting:**
- Implement 1-second delay between calls
- Batch processing with progress saving
- Automatic retry on failures (max 3 attempts)

### 3.3 Infrastructure

**Storage:**
- Local file system sufficient
- ~50MB total storage needed
- JSON for results, CSV for metrics

**Compute:**
- Standard laptop/desktop
- No GPU required
- 8GB RAM recommended

---

## 4. Success Criteria

### 4.1 Functional Requirements

**Must Have:**
- [ ] 100-item dataset created and validated
- [ ] Baseline experiment completed with metrics
- [ ] At least 3 prompt variations implemented
- [ ] Distance metrics calculated consistently
- [ ] Histograms showing distribution per variation
- [ ] Comparison graph demonstrating improvement/degradation
- [ ] Written analysis explaining results

**Should Have:**
- [ ] Statistical significance testing
- [ ] Category-based performance breakdown
- [ ] Token cost analysis
- [ ] Failure case analysis

**Nice to Have:**
- [ ] ReAct implementation with tools
- [ ] Interactive visualization dashboard
- [ ] Cross-validation across multiple datasets
- [ ] Comparison across different LLM models

### 4.2 Quality Metrics

**Accuracy Improvement:**
- Target: At least one variation shows >10% improvement over baseline
- Minimum baseline accuracy: 70%

**Reproducibility:**
- Same results within ±2% when re-run
- Clear documentation allows independent reproduction

**Analysis Depth:**
- Identifies at least 3 specific insights
- Explains why changes helped/hurt
- Discusses trade-offs (accuracy vs. tokens)

---

## 5. Timeline Estimate

### Suggested Sequence

**Day 1 (8 hours):**
- Morning: Environment setup, dataset creation (4h)
- Afternoon: Baseline implementation and first run (4h)

**Day 2 (8 hours):**
- Morning: Implement prompt variations 1-2 (4h)
- Afternoon: Implement variations 3-4, run all experiments (4h)

**Day 3 (6 hours):**
- Morning: Statistical analysis and metrics (3h)
- Afternoon: Create visualizations and graphs (3h)

**Day 4 (3 hours):**
- Morning: Write analysis report (2h)
- Afternoon: Final documentation and cleanup (1h)

**Total: ~25 hours over 3-4 days**

### Parallel Processing Opportunities

- Dataset creation can happen while environment sets up
- Multiple prompt variations can be tested overnight
- Visualization and analysis can overlap

---

## 6. Risk Considerations

### 6.1 Technical Risks

**API Rate Limits**
- **Risk:** Hitting request limits during batch processing
- **Mitigation:** Implement exponential backoff, process in smaller batches
- **Backup:** Switch to alternative LLM provider if needed

**Model Inconsistency**
- **Risk:** Same prompt produces different results across runs
- **Mitigation:** Set temperature=0 for deterministic outputs, run multiple iterations
- **Backup:** Document variance as part of findings

**Token Costs**
- **Risk:** Exceeding budget with long prompts
- **Mitigation:** Use GPT-3.5-turbo instead of GPT-4, limit few-shot examples
- **Backup:** Reduce dataset size to 50 examples if needed

### 6.2 Data Quality Risks

**Ambiguous Examples**
- **Risk:** Ground truth not clearly positive/negative
- **Mitigation:** Manual review of dataset, exclude borderline cases
- **Backup:** Add "neutral" category or remove unclear examples

**Dataset Bias**
- **Risk:** Unbalanced representation skewing results
- **Mitigation:** Equal split of positive/negative, diverse sources
- **Backup:** Weight metrics by class frequency

### 6.3 Project Execution Risks

**Scope Creep**
- **Risk:** Adding too many variations delays completion
- **Mitigation:** Stick to 3-4 core variations, keep ReAct optional
- **Backup:** Prioritize baseline + one improvement

**Time Pressure**
- **Risk:** Insufficient time for thorough analysis
- **Mitigation:** Automate metric calculation, use templates
- **Backup:** Focus on core deliverables, defer nice-to-haves

**Environment Issues**
- **Risk:** Package conflicts or API changes
- **Mitigation:** Use virtual environment, pin versions
- **Backup:** Docker container specification included

---

## 7. Deliverables Checklist

### Code Artifacts
- [ ] `/data/sentiment_dataset.json` - 100 examples
- [ ] `/src/baseline_experiment.py` - Baseline runner
- [ ] `/src/improved_prompts.py` - Prompt variations
- [ ] `/src/metrics.py` - Distance calculations
- [ ] `/src/run_experiments.py` - Batch processor
- [ ] `/analysis/statistical_summary.py` - Analytics
- [ ] `requirements.txt` - Dependencies
- [ ] `.env.example` - Configuration template

### Results & Analysis
- [ ] `/results/baseline_results.json` - Raw baseline data
- [ ] `/results/variation_*_results.json` - Each variation
- [ ] `/visualizations/histograms.png` - Distance distributions
- [ ] `/visualizations/comparison_chart.png` - Performance graph
- [ ] `/analysis/insights.md` - Written analysis
- [ ] `REPORT.md` - Complete findings report

### Documentation
- [ ] `README.md` - Project overview and setup
- [ ] `USAGE.md` - How to run experiments
- [ ] Code comments and docstrings
- [ ] Sample output files in `/examples/`

---

## 8. Implementation Notes for AI Engineer

### Getting Started

1. **Clone and Setup**
   ```bash
   git clone <repo>
   cd prompt-engineering-assignment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

3. **Create Dataset**
   ```bash
   python src/create_dataset.py --size 100 --output data/sentiment_dataset.json
   ```

4. **Run Baseline**
   ```bash
   python src/baseline_experiment.py --dataset data/sentiment_dataset.json
   ```

5. **Run All Variations**
   ```bash
   python src/run_experiments.py --all
   ```

6. **Generate Analysis**
   ```bash
   python analysis/statistical_summary.py
   python analysis/create_visualizations.py
   ```

### Key Design Decisions

**Why Sentiment Analysis?**
- Simplest to validate (binary output)
- Minimal token usage per example
- Clear ground truth
- Real-world applicability

**Why 100 Examples?**
- Large enough for statistical significance
- Small enough to iterate quickly
- Manageable API costs (~$10-15 total)
- Reasonable processing time (2-3 hours)

**Why These Metrics?**
- Binary accuracy: Easy to interpret
- Embedding distance: Captures semantic similarity
- Token count: Practical cost consideration
- Variance: Shows consistency

### Customization Points

**Easy to Adjust:**
- Dataset size (50-200 examples)
- Number of few-shot examples (1-5)
- LLM model (GPT-3.5, GPT-4, Claude)
- Distance metric (cosine, Euclidean, exact match)

**Domain Substitution:**
If sentiment analysis isn't suitable, alternative domains:
1. **Math Problems**: Arithmetic questions with numeric answers
2. **Fact Checking**: True/false statements
3. **Translation**: Simple phrase translation
4. **Summarization**: Short text to headline

---

## 9. Expected Outcomes

### Quantitative Results

**Baseline Performance:**
- Expected accuracy: 75-85%
- Average tokens per query: ~50

**Improved Prompts:**
- Standard improvement: +5-10%
- Few-shot learning: +10-15%
- Chain of thought: +5-12% (varies by complexity)

**Token Trade-offs:**
- Few-shot: 2-3x more tokens
- CoT: 1.5-2x more tokens
- ReAct: 3-5x more tokens

### Qualitative Insights

**Expected Findings:**
1. Few-shot works best for edge cases
2. CoT helps with ambiguous sentiment
3. Clear role definition improves consistency
4. Simple prompts sufficient for obvious cases

**Graph Characteristics:**
- Clear performance ranking visible
- Some variations show variance trade-offs
- Token cost correlates with accuracy improvement
- Diminishing returns after certain complexity

---

## 10. Post-Implementation Extensions

### If Time Permits

1. **Cross-Model Comparison**
   - Test same prompts on GPT-4 vs. Claude vs. Llama
   - Compare cost-effectiveness

2. **Prompt Optimization**
   - Genetic algorithm for prompt engineering
   - A/B testing different phrasings

3. **Advanced Metrics**
   - F1 score per category
   - Precision-recall curves
   - Confidence calibration

4. **Interactive Dashboard**
   - Streamlit/Gradio interface
   - Real-time prompt testing
   - Visual comparison tool

### Research Questions to Explore

- How does prompt length correlate with accuracy?
- What's the optimal number of few-shot examples?
- Do certain sentiment categories benefit more from CoT?
- Is there a point where complexity hurts performance?

---

## Conclusion

This project plan provides a complete, actionable roadmap for implementing the prompt engineering assignment. The sentiment analysis approach balances simplicity with meaningful insights, while the phased implementation ensures steady progress toward measurable outcomes.

**Key Success Factors:**
1. Simple, well-defined domain (sentiment analysis)
2. Manageable dataset size (100 examples)
3. Clear metrics for comparison
4. Systematic testing of prompt variations
5. Visual presentation of results

The plan is designed to be completed in 3-4 days with clear deliverables at each phase, while remaining flexible enough to accommodate experimentation and creative insights.

**Next Step:** AI Engineer can begin with Phase 1 (Foundation Setup) and proceed sequentially, with clear success criteria at each stage to validate progress.
