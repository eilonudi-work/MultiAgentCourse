# Context Windows Lab - Comprehensive Project Plan

**Course:** LLMs in Multi-Agent Environments
**Assignment:** Assignment 5 - Context Windows in Practice
**Version:** 1.0
**Date:** November 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Stack & Architecture](#technical-stack--architecture)
3. [Development Phases Overview](#development-phases-overview)
4. [Experiment 1: Needle in Haystack](#experiment-1-needle-in-haystack)
5. [Experiment 2: Context Window Size Impact](#experiment-2-context-window-size-impact)
6. [Experiment 3: RAG Impact](#experiment-3-rag-impact)
7. [Experiment 4: Context Engineering Strategies](#experiment-4-context-engineering-strategies)
8. [Cross-Cutting Concerns](#cross-cutting-concerns)
9. [Quality Assurance & Testing Strategy](#quality-assurance--testing-strategy)
10. [Documentation & Deliverables](#documentation--deliverables)
11. [Timeline & Milestones](#timeline--milestones)

---

## Project Overview

### Objectives
1. Demonstrate **Lost in the Middle** phenomenon
2. Analyze **Context Window Size Impact** on accuracy and performance
3. Compare **RAG vs Full Context** retrieval strategies
4. Evaluate **Context Engineering Strategies** (Select, Compress, Write, Isolate)

### Success Criteria
- All 4 experiments completed with statistical validity (multiple runs)
- Visual analysis (graphs/tables) for all findings
- Code coverage: 70%-85%
- Professional documentation (README as user manual)
- Clean, modular codebase with SOC principles

---

## Technical Stack & Architecture

### Core Technologies
- **LLM Framework:** Ollama (local deployment)
- **Orchestration:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** Nomic Embed Text
- **Programming Language:** Python 3.10+
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Testing:** pytest, pytest-cov
- **Configuration:** python-dotenv, PyYAML

### Project Structure
```
Assignment5/
├── src/
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── experiment_1_needle.py
│   │   ├── experiment_2_context_size.py
│   │   ├── experiment_3_rag.py
│   │   └── experiment_4_strategies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── document_generator.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── config_loader.py
│   └── strategies/
│       ├── __init__.py
│       ├── select_strategy.py
│       ├── compress_strategy.py
│       └── write_strategy.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── config/
│   ├── config.yaml
│   └── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── results/
│   ├── experiment_1/
│   ├── experiment_2/
│   ├── experiment_3/
│   └── experiment_4/
├── notebooks/
│   └── analysis.ipynb
├── docs/
│   ├── architecture.md
│   └── api_reference.md
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Development Phases Overview

Each experiment follows a consistent 5-phase development cycle:

1. **Phase 1: Foundation & Setup**
2. **Phase 2: Core Implementation**
3. **Phase 3: Testing & Validation**
4. **Phase 4: Analysis & Visualization**
5. **Phase 5: Documentation & Integration**

---

## Experiment 1: Needle in Haystack

**Goal:** Demonstrate Lost in the Middle phenomenon
**Duration:** ~15 minutes execution, 2-3 days development
**Difficulty:** Basic

### Phase 1: Foundation & Setup (Day 1)

#### Tasks
1. **Project Infrastructure**
   - [ ] Create project directory structure
   - [ ] Initialize Git repository
   - [ ] Setup virtual environment
   - [ ] Create `requirements.txt` with initial dependencies
   - [ ] Configure `.gitignore` (exclude `.env`, `__pycache__`, etc.)

2. **Configuration Management**
   - [ ] Create `config/config.yaml` for experiment parameters
   - [ ] Create `.env.example` template
   - [ ] Implement `ConfigLoader` class in `utils/config_loader.py`
   - [ ] Document all configuration options

3. **LLM Setup**
   - [ ] Install and configure Ollama
   - [ ] Pull required model (e.g., llama2, mistral)
   - [ ] Create `LLMClient` wrapper class
   - [ ] Implement connection testing and health checks

#### Deliverables
- Working project structure
- Configuration system in place
- Ollama operational with test query

---

### Phase 2: Core Implementation (Day 2-3)

#### Tasks
1. **Document Generation Module**
   - [ ] Implement `DocumentGenerator` class
   - [ ] Create filler text generation (Lorem Ipsum variant)
   - [ ] Implement fact embedding at positions: start/middle/end
   - [ ] Add configurable document parameters (word count, fact types)
   - [ ] Create fact templates (CEO name, founding date, product info, etc.)

2. **Experiment Core Logic**
   - [ ] Implement `Experiment1` class
   - [ ] Create document generation with fact positioning
   - [ ] Implement LLM query mechanism
   - [ ] Add response parsing and fact extraction
   - [ ] Implement accuracy scoring algorithm

3. **Data Structures**
   - [ ] Define `ExperimentResult` dataclass
   - [ ] Define `DocumentMetadata` dataclass
   - [ ] Create result serialization (JSON/CSV)

#### Deliverables
- Functional document generator
- Experiment execution engine
- Structured data output

---

### Phase 3: Testing & Validation (Day 3-4)

#### Tasks
1. **Unit Tests**
   - [ ] Test document generation (word count, fact placement)
   - [ ] Test fact embedding logic
   - [ ] Test LLM client mocking
   - [ ] Test accuracy calculation
   - [ ] Target: 75% code coverage

2. **Integration Tests**
   - [ ] Test end-to-end experiment execution
   - [ ] Test with actual Ollama instance
   - [ ] Validate result data structure

3. **Error Handling**
   - [ ] Handle Ollama connection failures
   - [ ] Handle malformed responses
   - [ ] Add retry logic with exponential backoff
   - [ ] Implement comprehensive logging

#### Deliverables
- pytest suite with 75%+ coverage
- Error handling implementation
- Test report

---

### Phase 4: Analysis & Visualization (Day 4-5)

#### Tasks
1. **Metrics Collection**
   - [ ] Run experiment 10 times per position
   - [ ] Collect accuracy scores by position
   - [ ] Calculate statistical measures (mean, std, confidence intervals)
   - [ ] Store results in structured format

2. **Visualization**
   - [ ] Create bar chart: accuracy by position
   - [ ] Add error bars (standard deviation)
   - [ ] Create box plot for distribution analysis
   - [ ] Generate heatmap if multiple fact types tested
   - [ ] Export high-resolution graphs (300 DPI)

3. **Statistical Analysis**
   - [ ] Perform t-tests (start vs middle, middle vs end)
   - [ ] Calculate effect sizes
   - [ ] Determine statistical significance (p < 0.05)

#### Deliverables
- Professional graphs (PNG/SVG)
- Statistical analysis report
- Results CSV/JSON

---

### Phase 5: Documentation & Integration (Day 5)

#### Tasks
1. **Code Documentation**
   - [ ] Add docstrings (Google style)
   - [ ] Create inline comments for complex logic
   - [ ] Generate API documentation

2. **Experiment Report**
   - [ ] Write methodology section
   - [ ] Document findings and interpretations
   - [ ] Add limitations and future work

3. **README Section**
   - [ ] Add Experiment 1 installation instructions
   - [ ] Add usage examples
   - [ ] Document expected outputs

#### Deliverables
- Documented codebase
- Experiment 1 section in README
- Individual experiment report

---

## Experiment 2: Context Window Size Impact

**Goal:** Analyze how context window size affects accuracy and performance
**Duration:** ~20 minutes execution, 3-4 days development
**Difficulty:** Medium

### Phase 1: Foundation & Setup (Day 6)

#### Tasks
1. **Module Setup**
   - [ ] Create `experiment_2_context_size.py`
   - [ ] Extend configuration for variable document counts
   - [ ] Add timing utilities

2. **Performance Monitoring**
   - [ ] Implement token counter using tiktoken
   - [ ] Create latency measurement decorator
   - [ ] Add memory profiling (optional)

3. **LangChain Integration**
   - [ ] Setup LangChain with Ollama
   - [ ] Create prompt templates
   - [ ] Implement document concatenation logic

#### Deliverables
- LangChain integration working
- Performance monitoring in place

---

### Phase 2: Core Implementation (Day 6-7)

#### Tasks
1. **Variable Context Handler**
   - [ ] Implement dynamic document loader
   - [ ] Create context concatenation with separators
   - [ ] Add token counting per context size
   - [ ] Implement query templating

2. **Experiment Execution**
   - [ ] Loop over doc counts: [2, 5, 10, 20, 50]
   - [ ] Measure latency for each size
   - [ ] Track accuracy degradation
   - [ ] Store tokens used per query
   - [ ] Implement progress tracking

3. **Result Storage**
   - [ ] Create structured result format
   - [ ] Add metadata (model, timestamp, config)
   - [ ] Implement incremental result saving

#### Deliverables
- Multi-size experiment engine
- Performance metrics collection

---

### Phase 3: Testing & Validation (Day 7-8)

#### Tasks
1. **Unit Tests**
   - [ ] Test token counting accuracy
   - [ ] Test document concatenation
   - [ ] Test timing decorator
   - [ ] Mock LLM responses for speed

2. **Integration Tests**
   - [ ] Test full experiment run (subset: 2, 5, 10 docs)
   - [ ] Validate result structure
   - [ ] Test error recovery (OOM scenarios)

3. **Performance Tests**
   - [ ] Benchmark token counting overhead
   - [ ] Test timeout handling for large contexts

#### Deliverables
- Test suite with 75%+ coverage
- Performance benchmarks

---

### Phase 4: Analysis & Visualization (Day 8-9)

#### Tasks
1. **Multi-Dimensional Analysis**
   - [ ] Run experiment 5 times per size
   - [ ] Analyze accuracy vs context size
   - [ ] Analyze latency vs context size
   - [ ] Analyze tokens vs document count
   - [ ] Calculate correlation coefficients

2. **Visualization Suite**
   - [ ] Line plot: accuracy degradation
   - [ ] Line plot: latency growth
   - [ ] Scatter plot: accuracy vs tokens
   - [ ] Combined plot: accuracy + latency (dual y-axis)
   - [ ] Add trend lines and annotations

3. **Cost Analysis**
   - [ ] Calculate token costs (if using API)
   - [ ] Project costs for scaling scenarios

#### Deliverables
- Multi-faceted visualizations
- Statistical correlation analysis
- Cost projection report

---

### Phase 5: Documentation & Integration (Day 9)

#### Tasks
1. **Documentation**
   - [ ] Document experiment parameters
   - [ ] Add usage examples
   - [ ] Update README

2. **Comparative Analysis**
   - [ ] Compare findings with Experiment 1
   - [ ] Identify patterns and insights

#### Deliverables
- Experiment 2 documentation
- Comparative insights report

---

## Experiment 3: RAG Impact

**Goal:** Compare RAG vs Full Context retrieval strategies
**Duration:** ~25 minutes execution, 4-5 days development
**Difficulty:** Medium+

### Phase 1: Foundation & Setup (Day 10-11)

#### Tasks
1. **Vector Store Setup**
   - [ ] Install ChromaDB
   - [ ] Configure persistent storage
   - [ ] Create VectorStore wrapper class
   - [ ] Test CRUD operations

2. **Embeddings Integration**
   - [ ] Setup Nomic Embed Text
   - [ ] Create Embeddings service class
   - [ ] Implement batch embedding
   - [ ] Add caching mechanism

3. **Hebrew Document Collection**
   - [ ] Source/create 20 Hebrew documents
   - [ ] Topics: Technology, Law, Medicine
   - [ ] Validate encoding (UTF-8)
   - [ ] Store in `data/raw/hebrew_docs/`

#### Deliverables
- ChromaDB operational
- Hebrew document corpus ready
- Embeddings service functional

---

### Phase 2: Core Implementation (Day 11-12)

#### Tasks
1. **Document Processing Pipeline**
   - [ ] Implement text chunking (500 tokens)
   - [ ] Add overlap strategy (50 tokens)
   - [ ] Create chunk metadata (source, position)
   - [ ] Implement chunk validator

2. **RAG Implementation**
   - [ ] Create RAG retriever class
   - [ ] Implement similarity search (top-k)
   - [ ] Add re-ranking logic (optional)
   - [ ] Create prompt constructor with context

3. **Comparison Framework**
   - [ ] Implement full-context query mode
   - [ ] Implement RAG query mode
   - [ ] Create side-by-side evaluation
   - [ ] Add query templates in Hebrew

4. **Ground Truth Setup**
   - [ ] Create test queries with known answers
   - [ ] Define evaluation criteria
   - [ ] Implement answer similarity scoring

#### Deliverables
- Full RAG pipeline operational
- Comparison framework ready
- Hebrew query templates

---

### Phase 3: Testing & Validation (Day 12-13)

#### Tasks
1. **Unit Tests**
   - [ ] Test chunking algorithm
   - [ ] Test embedding generation
   - [ ] Test vector store operations
   - [ ] Test similarity search

2. **Integration Tests**
   - [ ] Test end-to-end RAG pipeline
   - [ ] Test with Hebrew text
   - [ ] Validate retrieval quality
   - [ ] Test full-context mode

3. **Quality Assurance**
   - [ ] Verify encoding issues don't occur
   - [ ] Test with edge cases (very long/short docs)
   - [ ] Validate chunk boundaries

#### Deliverables
- Comprehensive test suite
- Hebrew text handling validation

---

### Phase 4: Analysis & Visualization (Day 13-14)

#### Tasks
1. **Experimental Runs**
   - [ ] Run 10 queries in each mode
   - [ ] Measure accuracy (against ground truth)
   - [ ] Measure latency
   - [ ] Measure token usage
   - [ ] Record retrieval precision

2. **Comparative Analysis**
   - [ ] RAG vs Full Context accuracy
   - [ ] Latency comparison
   - [ ] Token efficiency
   - [ ] Calculate precision@k and recall@k

3. **Visualization**
   - [ ] Bar chart: accuracy comparison
   - [ ] Bar chart: latency comparison
   - [ ] Scatter: retrieval precision vs query complexity
   - [ ] Create confusion matrix for answer quality

#### Deliverables
- Comparative metrics report
- Professional visualizations
- RAG effectiveness analysis

---

### Phase 5: Documentation & Integration (Day 14)

#### Tasks
1. **Documentation**
   - [ ] Document RAG architecture
   - [ ] Add ChromaDB setup guide
   - [ ] Document Hebrew corpus structure

2. **Insights Report**
   - [ ] Synthesize RAG advantages/limitations
   - [ ] Provide recommendations

#### Deliverables
- RAG architecture documentation
- Experiment 3 insights report

---

## Experiment 4: Context Engineering Strategies

**Goal:** Evaluate Select, Compress, Write, Isolate strategies
**Duration:** ~30 minutes execution, 5-6 days development
**Difficulty:** Advanced

### Phase 1: Foundation & Setup (Day 15-16)

#### Tasks
1. **Strategy Framework**
   - [ ] Create abstract `ContextStrategy` base class
   - [ ] Define strategy interface
   - [ ] Create strategy factory pattern

2. **Multi-Step Agent Simulation**
   - [ ] Design 10-step agent workflow
   - [ ] Create action templates
   - [ ] Implement history accumulation
   - [ ] Add configurable action complexity

3. **Memory Systems**
   - [ ] Implement external scratchpad (SQLite/Redis)
   - [ ] Create key-value store interface
   - [ ] Add TTL and capacity limits

#### Deliverables
- Strategy architecture defined
- Agent simulation framework
- Memory systems operational

---

### Phase 2: Core Implementation (Day 16-18)

#### Tasks
1. **SELECT Strategy (RAG-based)**
   - [ ] Extend Experiment 3 RAG logic
   - [ ] Implement history as vector store
   - [ ] Add top-k retrieval from history
   - [ ] Tune k parameter

2. **COMPRESS Strategy (Summarization)**
   - [ ] Implement token threshold detection
   - [ ] Create summarization prompt
   - [ ] Use LLM for recursive summarization
   - [ ] Preserve key facts during compression
   - [ ] Add compression ratio monitoring

3. **WRITE Strategy (External Memory)**
   - [ ] Implement fact extraction from history
   - [ ] Store facts in scratchpad with keys
   - [ ] Create retrieval query mechanism
   - [ ] Implement fact relevance scoring

4. **ISOLATE Strategy (Contextual Partitioning)**
   - [ ] Implement context windowing
   - [ ] Create sliding window logic
   - [ ] Add context switching mechanism
   - [ ] Implement boundary management

5. **Baseline Strategy (Full History)**
   - [ ] Implement naive append-all approach
   - [ ] Track context overflow

#### Deliverables
- All 4 strategies implemented
- Strategy benchmarking framework

---

### Phase 3: Testing & Validation (Day 18-19)

#### Tasks
1. **Strategy Unit Tests**
   - [ ] Test each strategy in isolation
   - [ ] Mock LLM calls for speed
   - [ ] Validate compression ratio
   - [ ] Test scratchpad CRUD

2. **Integration Tests**
   - [ ] Test 10-step agent with each strategy
   - [ ] Validate context size limits
   - [ ] Test strategy switching

3. **Edge Case Testing**
   - [ ] Test with minimal history
   - [ ] Test with massive history (100+ actions)
   - [ ] Test compression failure scenarios

#### Deliverables
- Strategy test suite
- Edge case validation

---

### Phase 4: Analysis & Visualization (Day 19-20)

#### Tasks
1. **Benchmark Execution**
   - [ ] Run 10-action workflow 5 times per strategy
   - [ ] Measure accuracy at each step
   - [ ] Measure latency per action
   - [ ] Track memory usage
   - [ ] Monitor context size evolution

2. **Metrics Collection**
   - [ ] Calculate accuracy degradation over time
   - [ ] Measure compression efficiency
   - [ ] Track retrieval precision (SELECT)
   - [ ] Measure scratchpad hit rate (WRITE)

3. **Visualization Suite**
   - [ ] Line plot: accuracy over 10 steps (all strategies)
   - [ ] Line plot: latency over steps
   - [ ] Line plot: context size growth
   - [ ] Table: strategy comparison summary
   - [ ] Radar chart: multi-dimensional strategy comparison

4. **Statistical Analysis**
   - [ ] ANOVA across strategies
   - [ ] Post-hoc tests (Tukey HSD)
   - [ ] Effect size calculation

#### Deliverables
- Comprehensive strategy comparison
- Multi-dimensional visualizations
- Statistical significance report

---

### Phase 5: Documentation & Integration (Day 20-21)

#### Tasks
1. **Architecture Documentation**
   - [ ] Document strategy design patterns
   - [ ] Create sequence diagrams
   - [ ] Add strategy selection guide

2. **Final Synthesis**
   - [ ] Synthesize findings across all 4 experiments
   - [ ] Provide strategy recommendations
   - [ ] Discuss trade-offs

#### Deliverables
- Strategy architecture docs
- Cross-experiment synthesis

---

## Cross-Cutting Concerns

### Security & Configuration

#### Tasks (Throughout Development)
- [ ] Never commit API keys or secrets
- [ ] Use environment variables for all sensitive data
- [ ] Validate all external inputs
- [ ] Implement rate limiting for LLM calls
- [ ] Add request/response sanitization

### Logging & Monitoring

#### Tasks (Throughout Development)
- [ ] Setup structured logging (JSON format)
- [ ] Create log levels: DEBUG, INFO, WARNING, ERROR
- [ ] Add request/response logging
- [ ] Implement experiment audit trail
- [ ] Create monitoring dashboard (optional)

### Performance Optimization

#### Tasks (Final Week)
- [ ] Profile code for bottlenecks
- [ ] Implement caching (embeddings, LLM responses)
- [ ] Add batch processing where applicable
- [ ] Optimize database queries
- [ ] Document optimization decisions

---

## Quality Assurance & Testing Strategy

### Testing Pyramid

1. **Unit Tests (70% of tests)**
   - Core logic functions
   - Data transformations
   - Utility functions
   - Target: 85% coverage

2. **Integration Tests (25% of tests)**
   - Module interactions
   - Database operations
   - LLM client integration
   - Target: 70% coverage

3. **End-to-End Tests (5% of tests)**
   - Full experiment workflows
   - Real LLM calls (limited)
   - Target: Key paths covered

### Testing Standards

- [ ] All tests must pass before commits
- [ ] Use pytest fixtures for common setups
- [ ] Mock external dependencies (LLM, DB)
- [ ] Parametrize tests for multiple scenarios
- [ ] Add integration tests with real Ollama (CI/CD)

### Coverage Goals

- **Minimum:** 70%
- **Target:** 75-85%
- **Focus:** Business logic, not boilerplate

---

## Documentation & Deliverables

### README.md Structure

1. **Project Overview**
   - Purpose and objectives
   - Key findings summary

2. **Installation**
   - Prerequisites (Python, Ollama, ChromaDB)
   - Step-by-step setup instructions
   - Environment configuration

3. **Usage**
   - Running individual experiments
   - Configuration options
   - Output interpretation

4. **Architecture**
   - High-level system design
   - Module descriptions
   - Data flow diagrams

5. **Results**
   - Summary of findings
   - Links to detailed reports

6. **Troubleshooting**
   - Common issues and solutions
   - FAQ

### Analysis Notebook

- [ ] Create `notebooks/analysis.ipynb`
- [ ] Include mathematical formulas (LaTeX)
- [ ] Show data exploration
- [ ] Present all visualizations
- [ ] Add interpretations and insights

### Additional Documentation

- [ ] `docs/architecture.md` - System design
- [ ] `docs/api_reference.md` - Code documentation
- [ ] `EXPERIMENTS.md` - Detailed methodology
- [ ] `RESULTS.md` - Findings and analysis
- [ ] `COSTS.md` - Cost analysis and optimization

---

## Timeline & Milestones

### Week 1: Foundation & Experiments 1-2
- **Day 1-5:** Experiment 1 (Needle in Haystack)
- **Day 6-9:** Experiment 2 (Context Size)
- **Milestone:** 2/4 experiments complete with visualizations

### Week 2: Advanced Experiments 3-4
- **Day 10-14:** Experiment 3 (RAG Impact)
- **Day 15-20:** Experiment 4 (Context Strategies)
- **Milestone:** All experiments complete

### Week 3: Integration & Quality
- **Day 21-22:** Code cleanup and refactoring
- **Day 23-24:** Testing to 75%+ coverage
- **Day 25:** Performance optimization

### Week 4: Documentation & Finalization
- **Day 26-27:** Complete README and docs
- **Day 28:** Analysis notebook
- **Day 29:** Final review and testing
- **Day 30:** Submission preparation

---

## Risk Management

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ollama installation issues | High | Provide detailed setup guide, Docker option |
| LLM response inconsistency | Medium | Multiple runs, statistical averaging |
| Hebrew encoding problems | Medium | UTF-8 validation, encoding tests |
| ChromaDB performance | Low | Batch operations, index optimization |
| Memory overflow (large contexts) | Medium | Chunk processing, streaming where possible |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Experiment takes longer than estimated | Medium | Parallel development, MVP approach |
| Testing coverage below target | High | TDD approach, continuous testing |
| Documentation incomplete | High | Document as you go, templates |

---

## Success Metrics

### Code Quality
- [ ] Code coverage: 75-85%
- [ ] No hardcoded values
- [ ] All secrets in environment variables
- [ ] Modular, reusable code
- [ ] Comprehensive error handling

### Experiments
- [ ] All 4 experiments successfully executed
- [ ] Statistical validity (multiple runs)
- [ ] Professional visualizations
- [ ] Clear, actionable insights

### Documentation
- [ ] README functions as complete user manual
- [ ] All code documented (docstrings)
- [ ] Architecture clearly explained
- [ ] Reproducible results

### Analysis
- [ ] Deep parameter sensitivity analysis
- [ ] Statistical significance tests
- [ ] Cost and optimization analysis
- [ ] Jupyter notebook with LaTeX formulas

---

## Appendix: Configuration Examples

### config/config.yaml
```yaml
experiments:
  experiment_1:
    num_documents: 5
    words_per_document: 200
    num_runs: 10
    fact_positions: ['start', 'middle', 'end']

  experiment_2:
    document_counts: [2, 5, 10, 20, 50]
    num_runs: 5

  experiment_3:
    num_documents: 20
    chunk_size: 500
    chunk_overlap: 50
    top_k: 3
    num_queries: 10

  experiment_4:
    num_actions: 10
    num_runs: 5
    strategies: ['select', 'compress', 'write', 'isolate']
    max_context_tokens: 2000

llm:
  model: "llama2"
  temperature: 0.7
  max_tokens: 500
  timeout: 30

vectorstore:
  persist_directory: "./data/chromadb"
  collection_name: "context_experiments"

logging:
  level: "INFO"
  format: "json"
  file: "./logs/experiments.log"
```

---

## Notes

- This plan follows **agile principles**: each experiment is a separate sprint
- **Modularity** is key: share code between experiments
- **Test early and often**: TDD approach recommended
- **Document as you go**: Don't leave docs to the end
- **Iterate based on findings**: If Experiment 1 reveals insights, apply them to later experiments

**Remember:** Quality over speed. A well-executed subset is better than a rushed complete set.
