# Product Requirements Document: LSTM Signal Extraction System

## Document Information

| Field | Value |
|-------|-------|
| **Project Name** | LSTM Signal Extraction System |
| **Author** | Product Manager - AI/ML Systems |
| **Creation Date** | 2025-11-11 |
| **Version** | 1.0 |
| **Status** | Draft - Ready for Review |
| **Document Owner** | Product Management |
| **Last Updated** | 2025-11-11 |

---

## Executive Summary

The LSTM Signal Extraction System is a machine learning solution designed to extract individual pure sinusoidal components from composite noisy signals. This system addresses the fundamental signal processing challenge of isolating specific frequency components from mixed signals in the presence of noise.

**Key Objectives:**
- Develop an LSTM-based neural network capable of conditional signal extraction
- Achieve training and test MSE within acceptable tolerance (MSE_test ≈ MSE_train)
- Process 10,000 samples per frequency with real-time inference capability
- Demonstrate generalization across different signal amplitudes and phases

**Target Users:**
- Signal processing researchers
- ML engineering teams
- Data scientists working on time-series problems
- Academic researchers in deep learning applications

---

## 1. Introduction and Problem Statement

### 1.1 Project Goal

**Primary Goal:** Develop a Long Short-Term Memory (LSTM) neural network that can accurately extract individual pure sinusoidal components from a mixed, noisy signal containing multiple frequencies, given a frequency selector as input.

### 1.2 User Problem Description

**Problem Context:**

Signal processing applications frequently encounter scenarios where multiple sinusoidal signals are mixed together with added noise. Traditional signal processing methods (like Fourier transforms or bandpass filters) have limitations when:
- Signal amplitudes vary over time
- Phase relationships are unknown
- Real-time adaptation is required
- Signals overlap in frequency domain

**User Pain Points:**
1. **Complexity of Manual Signal Separation:** Engineers spend significant time tuning filters and algorithms for each specific case
2. **Poor Performance in Noisy Environments:** Traditional methods struggle when signal-to-noise ratio is low
3. **Limited Adaptability:** Fixed algorithms cannot adapt to varying signal characteristics
4. **Time-Domain Processing Challenges:** Need for solutions that work in time domain without frequency domain transformation

**Current Workarounds:**
- Manual filter tuning for each frequency band
- Multiple passes with different filter configurations
- Pre-processing with aggressive noise reduction (which may damage signal)
- Frequency domain methods with windowing and averaging

### 1.3 Context and Background

**Technical Context:**

The system operates on composite signals defined as:

```
S(t) = (1/4) * Σ[i=1 to 4] Sinus_i^noisy(t)
```

Where each noisy sinusoid component is:

```
Sinus_i^noisy(t) = A_i(t) * sin(2π * f_i * t + φ_i(t))
```

With:
- Frequencies: f₁ = 1 Hz, f₂ = 3 Hz, f₃ = 5 Hz, f₄ = 7 Hz
- Amplitude: A_i(t) ~ Uniform(0.8, 1.2)
- Phase: φ_i(t) ~ Uniform(0, 2π)

**Market Context:**

This solution addresses applications in:
- Audio signal processing and music source separation
- Telecommunications and carrier signal extraction
- Biomedical signal processing (ECG, EEG analysis)
- Vibration analysis in mechanical systems
- Radar and sonar signal processing

### 1.4 Measurable Objectives and Success Metrics

#### Primary KPIs (Key Performance Indicators)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Training MSE** | < 0.01 | Mean Squared Error on 40,000 training samples |
| **Test MSE** | < 0.01 | Mean Squared Error on 40,000 test samples |
| **MSE Ratio** | 0.9 < (MSE_test / MSE_train) < 1.1 | Generalization indicator |
| **Inference Time** | < 1ms per sample | Average processing time per time step |
| **Model Size** | < 10 MB | Trained model file size |

#### Secondary Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| **Peak Signal-to-Noise Ratio (PSNR)** | > 40 dB | Signal quality measure |
| **Correlation Coefficient** | > 0.95 | Output vs. target similarity |
| **Training Time** | < 2 hours | Development iteration speed |
| **Memory Usage** | < 2 GB | Resource efficiency |

#### Success Criteria

**Must Have (P0):**
1. MSE_test ≈ MSE_train (within 10% difference)
2. Successful extraction of all four frequency components
3. Visual validation: Output closely matches clean target signal
4. System processes 10,000 samples without errors

**Should Have (P1):**
5. Training converges within 100 epochs
6. Inference time suitable for real-time applications
7. Reproducible results with fixed random seeds

**Nice to Have (P2):**
8. Model interpretability through attention weights
9. Confidence scores for predictions
10. Support for additional frequency components

### 1.5 Guardrail Metrics

These metrics ensure the solution doesn't create negative side effects:

| Guardrail | Threshold | Purpose |
|-----------|-----------|---------|
| **Overfitting Indicator** | MSE_test / MSE_train < 1.5 | Prevent memorization |
| **Training Stability** | Loss variance < 0.001 | Ensure convergence |
| **Computational Cost** | GPU memory < 4 GB | Resource constraints |
| **Latency P95** | < 5ms per sample | Worst-case performance |
| **Model Degradation** | MSE increase < 5% after retraining | Consistency over time |

---

## 2. Project Requirements

### 2.1 Functional Requirements

#### FR-001: Dataset Generation System

**User Story:** As a data scientist, I need to generate training and test datasets with controlled signal characteristics so that I can train and evaluate the LSTM model.

**Description:**
The system shall generate synthetic datasets consisting of composite noisy signals and their corresponding pure sinusoidal components.

**Acceptance Criteria:**
- [ ] Generate signals for time range 0–10 seconds
- [ ] Sample at Fs = 1000 Hz (10,000 samples per signal)
- [ ] Create 4 sinusoids with frequencies f₁=1Hz, f₂=3Hz, f₃=5Hz, f₄=7Hz
- [ ] Apply random amplitude A_i(t) ~ Uniform(0.8, 1.2)
- [ ] Apply random phase φ_i(t) ~ Uniform(0, 2π)
- [ ] Mix signals: S(t) = (1/4) * Σ Sinus_i^noisy(t)
- [ ] Generate 40,000 training samples (4 frequencies × 10,000 samples)
- [ ] Generate 40,000 test samples with different random seed
- [ ] Store datasets in efficient format (HDF5 or NumPy arrays)
- [ ] Include metadata: sampling rate, frequencies, time range

**Dependencies:** NumPy, random number generation libraries

**Priority:** P0 (Must Have)

---

#### FR-002: Input Data Formatting

**User Story:** As an ML engineer, I need the input data properly formatted with signal values and frequency selectors so that the LSTM can perform conditional extraction.

**Description:**
Transform raw signal data into LSTM-compatible input vectors combining signal amplitude and one-hot encoded frequency selection.

**Acceptance Criteria:**
- [ ] Create input vector of size 5: [S[t], C1, C2, C3, C4]
- [ ] S[t] represents the noisy mixed signal value at time t
- [ ] C1, C2, C3, C4 form one-hot encoding (exactly one is 1, others are 0)
- [ ] C1=1 selects f₁=1Hz, C2=1 selects f₂=3Hz, etc.
- [ ] Normalize signal values to appropriate range
- [ ] Generate all combinations: each time step × each frequency
- [ ] Validate input vector dimensions before training
- [ ] Support batch processing for training efficiency

**Dependencies:** FR-001 (Dataset Generation)

**Priority:** P0 (Must Have)

---

#### FR-003: LSTM Network Architecture

**User Story:** As an ML engineer, I need an LSTM architecture that can process sequential data with proper state management so that the model learns temporal dependencies.

**Description:**
Implement LSTM neural network with specific architectural constraints for time-series signal extraction.

**Acceptance Criteria:**
- [ ] Use sequence length L = 1 (process one time step at a time)
- [ ] Implement proper LSTM cell with gates (input, forget, output)
- [ ] Maintain hidden state (h_t) and cell state (c_t) between consecutive time steps
- [ ] Reset internal states between different samples (not between t and t+1)
- [ ] Input layer accepts 5-dimensional vectors
- [ ] Output layer produces single scalar value (extracted signal)
- [ ] Support configurable hidden layer size
- [ ] Include at least one LSTM layer (can stack multiple)
- [ ] Add dense output layer for regression
- [ ] Initialize weights using appropriate method (Xavier/He initialization)

**Technical Specifications:**
```python
Input: [batch_size, sequence_length=1, features=5]
LSTM Layer(s): configurable hidden_size (recommend 32-128 units)
Output: [batch_size, 1] (single value prediction)
```

**Dependencies:** PyTorch/TensorFlow, FR-002 (Input Formatting)

**Priority:** P0 (Must Have)

---

#### FR-004: Model Training Pipeline

**User Story:** As a data scientist, I need a robust training pipeline that optimizes the LSTM model so that it learns to extract signals accurately.

**Description:**
Implement end-to-end training process with proper loss computation, optimization, and state management.

**Acceptance Criteria:**
- [ ] Use Mean Squared Error (MSE) as loss function
- [ ] Implement backpropagation through time (BPTT)
- [ ] Support mini-batch training (configurable batch size)
- [ ] Reset LSTM hidden/cell states between samples
- [ ] Do not reset states when moving from t to t+1 within a sequence
- [ ] Save model checkpoints during training
- [ ] Implement early stopping based on validation loss
- [ ] Track training metrics per epoch
- [ ] Support resuming training from checkpoints
- [ ] Use appropriate optimizer (Adam recommended)
- [ ] Implement learning rate scheduling
- [ ] Add gradient clipping to prevent exploding gradients

**Training Configuration:**
```yaml
optimizer: Adam
learning_rate: 0.001 (initial)
batch_size: 32-128
max_epochs: 100-200
early_stopping_patience: 10
gradient_clip: 1.0
```

**Dependencies:** FR-003 (LSTM Architecture), FR-002 (Input Data)

**Priority:** P0 (Must Have)

---

#### FR-005: Model Evaluation System

**User Story:** As a researcher, I need comprehensive evaluation metrics and visualizations so that I can assess model performance and quality.

**Description:**
Implement evaluation pipeline that computes metrics and generates required visualizations.

**Acceptance Criteria:**
- [ ] Compute Training MSE: (1/40000) Σ (LSTM(S_train[t], C) - Target[t])²
- [ ] Compute Test MSE: (1/40000) Σ (LSTM(S_test[t], C) - Target[t])²
- [ ] Calculate MSE for each frequency independently
- [ ] Generate per-frequency performance reports
- [ ] Support evaluation on unseen data (different random seeds)
- [ ] Measure inference time per sample
- [ ] Track memory usage during inference
- [ ] Export metrics to structured format (JSON/CSV)

**Dependencies:** FR-004 (Training Pipeline)

**Priority:** P0 (Must Have)

---

#### FR-006: Visualization and Reporting

**User Story:** As a researcher, I need clear visualizations of model outputs so that I can validate signal extraction quality.

**Description:**
Generate comprehensive plots comparing target signals, noisy inputs, and LSTM outputs.

**Acceptance Criteria:**
- [ ] **Graph 1 (Frequency f₂ Detailed):**
  - Plot clean target sinusoid for f₂=3Hz
  - Plot noisy input signal S(t)
  - Plot LSTM output for f₂ extraction
  - Include legend, axis labels, title
  - Show time range: 0-10 seconds
  - Use different colors for each signal
  - Save as high-resolution image

- [ ] **Graph 2 (All Frequencies):**
  - Create 4 subplots (one per frequency)
  - Each subplot: target vs. LSTM output
  - Show MSE value on each subplot
  - Synchronized x-axes (time)
  - Proper labeling for each frequency
  - Save as high-resolution image

- [ ] **Additional Visualizations:**
  - Training/validation loss curves
  - MSE comparison bar chart (train vs. test)
  - Error distribution histogram
  - Residual plots

**Technical Specifications:**
```python
Figure size: 12x8 inches minimum
Resolution: 300 DPI
Format: PNG and/or PDF
Color scheme: ColorBrewer or similar professional palette
```

**Dependencies:** FR-005 (Evaluation System)

**Priority:** P0 (Must Have)

---

#### FR-007: Model Persistence and Loading

**User Story:** As an ML engineer, I need to save and load trained models so that I can deploy them without retraining.

**Description:**
Implement model serialization and deserialization with all necessary metadata.

**Acceptance Criteria:**
- [ ] Save trained model weights/parameters
- [ ] Save model architecture configuration
- [ ] Save training hyperparameters
- [ ] Save normalization parameters (mean, std)
- [ ] Include model version and timestamp
- [ ] Support loading model for inference
- [ ] Validate loaded model matches architecture
- [ ] Include training/test MSE in metadata
- [ ] Support export to ONNX format (optional)
- [ ] Compress saved models (optional)

**File Format:**
```
model_checkpoint.pt (or .h5)
├── model_state_dict
├── optimizer_state_dict
├── hyperparameters
├── normalization_params
└── metadata (version, timestamp, metrics)
```

**Dependencies:** FR-004 (Training Pipeline)

**Priority:** P0 (Must Have)

---

#### FR-008: Inference Pipeline

**User Story:** As a user, I need to extract specific frequency components from new signal data so that I can apply the model to production scenarios.

**Description:**
Provide inference interface for applying trained model to new signal data.

**Acceptance Criteria:**
- [ ] Accept new signal data S(t) and frequency selector C
- [ ] Load pre-trained model
- [ ] Properly manage LSTM state during inference
- [ ] Process entire time series maintaining temporal continuity
- [ ] Return extracted signal values
- [ ] Support batch inference for efficiency
- [ ] Handle edge cases (empty input, invalid frequency)
- [ ] Provide prediction confidence (optional)
- [ ] Support real-time streaming inference
- [ ] Include preprocessing steps (normalization)

**API Specification:**
```python
def extract_signal(
    signal: np.ndarray,        # Shape: [n_samples]
    frequency_id: int,         # 0, 1, 2, or 3
    model: LSTMModel,
    return_confidence: bool = False
) -> np.ndarray:               # Shape: [n_samples]
    """Extract specified frequency component from mixed signal."""
    pass
```

**Dependencies:** FR-007 (Model Loading), FR-003 (LSTM Architecture)

**Priority:** P0 (Must Have)

---

#### FR-009: Data Validation and Quality Checks

**User Story:** As a data scientist, I need automated validation of input data quality so that I can catch issues before training.

**Description:**
Implement comprehensive data validation for both generation and training processes.

**Acceptance Criteria:**
- [ ] Validate signal time range (0-10 seconds)
- [ ] Verify sampling rate (1000 Hz)
- [ ] Check number of samples (10,000)
- [ ] Validate amplitude range (0.8-1.2 after mixing)
- [ ] Verify one-hot encoding correctness
- [ ] Check for NaN or infinite values
- [ ] Validate data shapes match expected dimensions
- [ ] Verify train/test split uses different random seeds
- [ ] Check signal statistics (mean, std, frequency content)
- [ ] Validate target signals are pure sinusoids

**Priority:** P1 (Should Have)

---

#### FR-010: Experiment Tracking and Reproducibility

**User Story:** As a researcher, I need to track all experiments with full reproducibility so that I can compare different approaches.

**Description:**
Implement experiment tracking system capturing all relevant parameters and results.

**Acceptance Criteria:**
- [ ] Log all hyperparameters used
- [ ] Record random seeds for reproducibility
- [ ] Track metrics for each epoch
- [ ] Save model checkpoints at intervals
- [ ] Store dataset generation parameters
- [ ] Version control for code and data
- [ ] Generate experiment summary reports
- [ ] Support comparing multiple experiments
- [ ] Export experiment data for analysis
- [ ] Include system information (hardware, library versions)

**Dependencies:** All training and evaluation components

**Priority:** P1 (Should Have)

---

### 2.2 Non-Functional Requirements

#### NFR-001: Performance

**Requirements:**

1. **Training Performance:**
   - Training shall complete within 2 hours on standard GPU (NVIDIA RTX 3080 or equivalent)
   - Each epoch shall process 40,000 samples within 5 minutes
   - Support for multi-GPU training (optional enhancement)

2. **Inference Performance:**
   - Average inference time: < 1ms per sample on CPU
   - P95 inference latency: < 5ms per sample
   - Support batch inference at 1000+ samples/second
   - Memory usage during inference: < 500 MB

3. **Scalability:**
   - System shall handle datasets up to 1M samples
   - Support concurrent inference requests (10+ simultaneous)
   - Efficient memory management for large batches

**Measurement:** Performance profiling with cProfile, time tracking, memory monitoring

**Priority:** P0 (Must Have)

---

#### NFR-002: Accuracy and Quality

**Requirements:**

1. **Model Accuracy:**
   - Training MSE < 0.01
   - Test MSE < 0.01
   - MSE_test / MSE_train within range [0.9, 1.1]
   - Per-frequency MSE consistent across all four frequencies

2. **Signal Quality:**
   - Peak Signal-to-Noise Ratio (PSNR) > 40 dB
   - Correlation coefficient between output and target > 0.95
   - Visual inspection: output should closely match clean target

3. **Robustness:**
   - Stable performance across different random initializations
   - Consistent results with different batch sizes
   - Graceful degradation with increased noise levels

**Measurement:** Automated test suite, statistical analysis of multiple runs

**Priority:** P0 (Must Have)

---

#### NFR-003: Maintainability

**Requirements:**

1. **Code Quality:**
   - Follow PEP 8 style guidelines (Python)
   - Type hints for all public functions
   - Comprehensive docstrings (Google or NumPy style)
   - Unit test coverage > 80%
   - Integration tests for end-to-end workflows

2. **Modularity:**
   - Separate modules for: data generation, model architecture, training, evaluation
   - Clear interfaces between components
   - Easy to swap different LSTM implementations
   - Configuration-driven design (YAML/JSON config files)

3. **Documentation:**
   - README with setup and usage instructions
   - API documentation (auto-generated)
   - Architecture diagrams
   - Example notebooks for common use cases
   - Troubleshooting guide

**Measurement:** Code review checklist, automated linting, test coverage reports

**Priority:** P1 (Should Have)

---

#### NFR-004: Usability

**Requirements:**

1. **Ease of Use:**
   - Simple command-line interface for training and inference
   - Sensible default parameters (minimal configuration required)
   - Clear error messages with actionable guidance
   - Progress bars for long-running operations
   - Automatic visualization generation

2. **Developer Experience:**
   - Quick start guide (< 15 minutes to first results)
   - Example scripts for common workflows
   - Jupyter notebook tutorials
   - Pre-trained model available for testing

3. **Configuration:**
   - Single configuration file for all parameters
   - Environment variable support for key settings
   - Validation of configuration on startup

**Example CLI Interface:**
```bash
# Training
python train_lstm.py --config config.yaml --output-dir ./models/

# Inference
python extract_signal.py --model models/best_model.pt --signal data/test_signal.npy --frequency 3Hz

# Evaluation
python evaluate.py --model models/best_model.pt --test-data data/test_set.npy
```

**Priority:** P1 (Should Have)

---

#### NFR-005: Portability

**Requirements:**

1. **Platform Support:**
   - Support Linux, macOS, Windows
   - Python 3.8+ compatibility
   - Both CPU and GPU execution modes
   - Docker containerization (optional)

2. **Dependencies:**
   - Minimize external dependencies
   - Use standard scientific Python stack (NumPy, SciPy, Matplotlib)
   - Support both PyTorch and TensorFlow (or choose one)
   - Pin dependency versions for reproducibility

3. **Deployment:**
   - Pip installable package
   - Conda environment file provided
   - Docker image for production deployment (optional)
   - Export to ONNX for cross-platform inference

**Priority:** P1 (Should Have)

---

#### NFR-006: Security (Academic Context)

**Requirements:**

1. **Data Security:**
   - No external data transmission required
   - Local storage of all artifacts
   - No collection of user data

2. **Code Security:**
   - Input validation for all user-provided data
   - Safe handling of file paths (prevent traversal)
   - No execution of arbitrary code

3. **Reproducibility:**
   - Deterministic behavior with fixed random seeds
   - Version control for code and data
   - Audit trail of experiments

**Priority:** P2 (Nice to Have)

---

#### NFR-007: Reliability

**Requirements:**

1. **Error Handling:**
   - Graceful handling of invalid inputs
   - Automatic recovery from minor errors
   - Clear error messages with context
   - Logging of all errors with timestamps

2. **Training Stability:**
   - Automatic checkpoint saving every N epochs
   - Resume capability from checkpoints
   - Early stopping to prevent overfitting
   - Gradient clipping to prevent instability

3. **Testing:**
   - Unit tests for all core functions
   - Integration tests for workflows
   - Regression tests for model performance
   - Continuous integration (optional)

**Priority:** P1 (Should Have)

---

## 3. Planning and Management

### 3.1 Dependencies

#### External Dependencies

**Python Libraries:**
- **PyTorch** (v2.0+) or **TensorFlow** (v2.12+): Deep learning framework
- **NumPy** (v1.23+): Numerical computations
- **Matplotlib** (v3.7+): Visualization
- **SciPy** (v1.10+): Signal processing utilities (optional)
- **Pandas** (v2.0+): Data management (optional)
- **TensorBoard** or **Weights & Biases**: Experiment tracking (optional)

**System Dependencies:**
- Python 3.8 or higher
- CUDA 11.8+ and cuDNN (for GPU acceleration)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM recommended (but not required)

#### Internal Dependencies

**Component Dependencies:**
```
FR-001 (Dataset Generation)
    └── FR-002 (Input Formatting)
            └── FR-003 (LSTM Architecture)
                    └── FR-004 (Training Pipeline)
                            ├── FR-005 (Evaluation)
                            │       └── FR-006 (Visualization)
                            └── FR-007 (Model Persistence)
                                    └── FR-008 (Inference)
```

### 3.2 Assumptions

**Technical Assumptions:**
1. Training will be performed on a machine with at least one GPU (fallback to CPU acceptable with longer training time)
2. Random number generation is sufficient for signal synthesis (no need for real-world signal data)
3. LSTM architecture is appropriate for this time-series problem
4. 40,000 samples provide sufficient data for training
5. Four frequencies are sufficiently separated (1, 3, 5, 7 Hz) for extraction

**Project Assumptions:**
1. This is an academic/research project (not production deployment)
2. Users have basic Python and ML knowledge
3. Development will be done by 1-2 engineers
4. No real-time constraints beyond inference performance
5. Single-node training is sufficient (no distributed training needed)

**Data Assumptions:**
1. Synthetic data adequately represents the signal extraction problem
2. Uniform distributions for amplitude and phase are appropriate
3. The mixing formula (simple averaging) represents realistic scenarios
4. No missing data or temporal gaps in signals

### 3.3 Constraints

**Technical Constraints:**
1. **LSTM Constraints:**
   - Must use sequence length L = 1
   - Must reset state between samples
   - Must not reset state between consecutive time steps (t to t+1)

2. **Architecture Constraints:**
   - Fixed input dimension: 5 (signal + 4 one-hot values)
   - Fixed output dimension: 1 (scalar value)
   - Must support standard LSTM formulation

3. **Data Constraints:**
   - Fixed sampling rate: 1000 Hz
   - Fixed time range: 0-10 seconds
   - Fixed frequencies: 1, 3, 5, 7 Hz
   - Fixed dataset size: 40,000 samples

**Resource Constraints:**
1. Development time: Limited by academic semester timeline
2. Computational resources: Personal/lab GPU access
3. Storage: Local machine storage (no cloud services required)
4. Team size: Typically 1-2 students

**Academic Constraints:**
1. Must demonstrate understanding of LSTM internals
2. Must include detailed evaluation and analysis
3. Must produce publication-quality figures
4. Code must be readable and well-documented for grading

### 3.4 Timeline and Milestones

#### Project Phases

**Phase 1: Foundation (Week 1)**
- **Milestone M1:** Dataset Generation Complete
  - Deliverables:
    - Working data generation script
    - Train and test datasets created
    - Data validation tests passing
  - Acceptance: 40,000 train + 40,000 test samples generated successfully

**Phase 2: Model Development (Week 2)**
- **Milestone M2:** LSTM Architecture Implemented
  - Deliverables:
    - LSTM model class implemented
    - Input/output interfaces defined
    - Unit tests for model components
  - Acceptance: Model can process sample inputs without errors

**Phase 3: Training Pipeline (Week 3)**
- **Milestone M3:** Training Pipeline Functional
  - Deliverables:
    - Training loop implemented
    - State management working correctly
    - Checkpoint saving functional
    - Basic metrics logging
  - Acceptance: Model trains for at least 10 epochs without errors

**Phase 4: Optimization (Week 4)**
- **Milestone M4:** Model Performance Achieved
  - Deliverables:
    - Hyperparameter tuning completed
    - MSE targets achieved
    - Training/test convergence validated
  - Acceptance: MSE_test ≈ MSE_train and both < 0.01

**Phase 5: Evaluation & Visualization (Week 5)**
- **Milestone M5:** Evaluation Complete
  - Deliverables:
    - All required visualizations generated
    - Comprehensive evaluation metrics
    - Performance analysis report
  - Acceptance: All graphs meet requirements, clear signal extraction demonstrated

**Phase 6: Documentation & Delivery (Week 6)**
- **Milestone M6:** Project Complete
  - Deliverables:
    - Complete documentation
    - Clean, commented code
    - README with instructions
    - Final report/presentation
  - Acceptance: All deliverables submitted, code runs successfully

#### Detailed Schedule

```
Week 1: Dataset Generation & Validation
Week 2: LSTM Implementation & Testing
Week 3: Training Pipeline Development
Week 4: Hyperparameter Tuning & Optimization
Week 5: Evaluation & Visualization
Week 6: Documentation & Final Delivery
```

**Critical Path:**
1. Dataset Generation (blocks everything)
2. LSTM Implementation (blocks training)
3. Training Pipeline (blocks evaluation)
4. Model Performance Achievement (blocks final delivery)

**Contingency Planning:**
- **If training doesn't converge:** Allocate extra time in Week 4 for debugging
- **If MSE targets not met:** Review architecture, try different hyperparameters
- **If implementation delayed:** Simplify non-critical features (FR-009, FR-010)

---

## 4. Architecture Documentation

### 4.1 System Architecture Overview

The LSTM Signal Extraction System follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (CLI, Scripts, Notebooks)                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                 Application Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │  Evaluation  │  │  Inference   │      │
│  │  Orchestrator│  │   Pipeline   │  │   Service    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                   Core Components Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Dataset    │  │     LSTM     │  │     Loss     │      │
│  │  Generator   │  │    Model     │  │  Functions   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ State Manager│  │  Optimizer   │  │Visualization │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  Infrastructure Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     File     │  │    Model     │  │   Metrics    │      │
│  │     I/O      │  │  Checkpoint  │  │   Logging    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 C4 Model Architecture Diagrams

#### Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│   ┌───────────────┐                                          │
│   │               │                                          │
│   │  Researcher   │────────┐                                │
│   │  Data Scientist│        │                                │
│   └───────────────┘        │                                │
│                            │                                │
│                            ▼                                │
│              ┌──────────────────────────┐                   │
│              │                          │                   │
│              │   LSTM Signal            │                   │
│              │   Extraction System      │                   │
│              │                          │                   │
│              │  - Generate Datasets     │                   │
│              │  - Train LSTM Model      │                   │
│              │  - Extract Signals       │                   │
│              │  - Evaluate Performance  │                   │
│              │                          │                   │
│              └────────┬─────────────────┘                   │
│                       │                                     │
│                       │ Uses                                │
│                       ▼                                     │
│              ┌──────────────────┐                           │
│              │  PyTorch/TF      │                           │
│              │  Deep Learning   │                           │
│              │  Framework       │                           │
│              └──────────────────┘                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**System Purpose:** Extract individual pure sinusoidal components from mixed noisy signals using LSTM neural networks.

**Users:** Researchers, data scientists, ML engineers working on signal processing problems.

**External Dependencies:** Deep learning framework (PyTorch/TensorFlow), scientific Python stack.

---

#### Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                LSTM Signal Extraction System                     │
│                                                                   │
│   ┌─────────────────────┐         ┌─────────────────────┐      │
│   │                     │         │                     │      │
│   │  Data Generation    │────────▶│   LSTM Training     │      │
│   │     Container       │         │     Container       │      │
│   │                     │         │                     │      │
│   │ - Generate signals  │         │ - Train model       │      │
│   │ - Create datasets   │         │ - Optimize weights  │      │
│   │ - Validate data     │         │ - Save checkpoints  │      │
│   └─────────────────────┘         └──────────┬──────────┘      │
│                                               │                  │
│                                               │                  │
│   ┌─────────────────────┐         ┌──────────▼──────────┐      │
│   │                     │         │                     │      │
│   │  Visualization      │◀────────│   Evaluation        │      │
│   │    Container        │         │    Container        │      │
│   │                     │         │                     │      │
│   │ - Generate plots    │         │ - Compute metrics   │      │
│   │ - Create reports    │         │ - Analyze results   │      │
│   └─────────────────────┘         └──────────┬──────────┘      │
│                                               │                  │
│                                               │                  │
│                                    ┌──────────▼──────────┐      │
│                                    │                     │      │
│                                    │   Inference         │      │
│                                    │   Container         │      │
│                                    │                     │      │
│                                    │ - Load model        │      │
│                                    │ - Extract signals   │      │
│                                    │ - Serve predictions │      │
│                                    └─────────────────────┘      │
│                                                                   │
│   ┌────────────────────────────────────────────────────┐        │
│   │         Persistence Layer                          │        │
│   │  - Model Checkpoints                               │        │
│   │  - Dataset Files (HDF5/NPY)                        │        │
│   │  - Configuration Files (YAML)                      │        │
│   │  - Metrics and Logs (JSON/CSV)                     │        │
│   └────────────────────────────────────────────────────┘        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Containers:**

1. **Data Generation Container:** Python module responsible for synthetic signal creation
2. **Training Container:** LSTM model training and optimization
3. **Evaluation Container:** Performance measurement and analysis
4. **Visualization Container:** Plot generation and reporting
5. **Inference Container:** Production signal extraction service
6. **Persistence Layer:** File-based storage for models, data, and results

---

#### Level 3: Component Diagram - Training Container

```
┌─────────────────────────────────────────────────────────────────┐
│              LSTM Training Container                             │
│                                                                   │
│   ┌─────────────────────┐         ┌─────────────────────┐      │
│   │                     │         │                     │      │
│   │  DataLoader         │────────▶│   TrainingLoop      │      │
│   │  Component          │         │   Component         │      │
│   │                     │         │                     │      │
│   │ - Load datasets     │         │ - Epoch iteration   │      │
│   │ - Batch creation    │         │ - Forward pass      │      │
│   │ - Data shuffling    │         │ - Backward pass     │      │
│   │ - State reset logic │         │ - Weight updates    │      │
│   └─────────────────────┘         └──────────┬──────────┘      │
│                                               │                  │
│                                               │                  │
│   ┌─────────────────────┐         ┌──────────▼──────────┐      │
│   │                     │         │                     │      │
│   │  LSTMModel          │◀────────│   Optimizer         │      │
│   │  Component          │         │   Component         │      │
│   │                     │         │                     │      │
│   │ - LSTM layers       │         │ - Adam optimizer    │      │
│   │ - Forward method    │         │ - Learning rate     │      │
│   │ - State management  │         │ - Gradient clip     │      │
│   │ - Weight init       │         └─────────────────────┘      │
│   └─────────┬───────────┘                                       │
│             │                                                    │
│             │                     ┌─────────────────────┐       │
│             │                     │                     │       │
│             └────────────────────▶│   LossFunction      │       │
│                                   │   Component         │       │
│                                   │                     │       │
│                                   │ - MSE computation   │       │
│                                   │ - Batch reduction   │       │
│                                   └─────────────────────┘       │
│                                                                   │
│   ┌─────────────────────┐         ┌─────────────────────┐      │
│   │                     │         │                     │      │
│   │  CheckpointManager  │         │   MetricsLogger     │      │
│   │  Component          │         │   Component         │      │
│   │                     │         │                     │      │
│   │ - Save models       │         │ - Track loss        │      │
│   │ - Load models       │         │ - Log metrics       │      │
│   │ - Best model logic  │         │ - Export results    │      │
│   └─────────────────────┘         └─────────────────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **DataLoader:** Manages data loading, batching, and state reset logic
2. **TrainingLoop:** Orchestrates the training process
3. **LSTMModel:** The neural network architecture
4. **Optimizer:** Weight update mechanism
5. **LossFunction:** MSE computation
6. **CheckpointManager:** Model persistence
7. **MetricsLogger:** Training metrics tracking

---

#### Level 4: Code-Level Component Detail - LSTM Model

```python
class LSTMSignalExtractor(nn.Module):
    """
    LSTM-based signal extraction model.

    Architecture:
        Input Layer (5) → LSTM Layer(s) → Dense Output (1)

    Input: [batch_size, sequence_length=1, features=5]
        - features[0]: Mixed signal value S(t)
        - features[1-4]: One-hot frequency selector C

    Output: [batch_size, 1]
        - Extracted pure sinusoid value
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # State storage
        self.hidden_state = None
        self.cell_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with state management.

        Args:
            x: Input tensor [batch_size, 1, 5]

        Returns:
            output: Predicted signal [batch_size, 1]
        """
        # Use stored states if available, else initialize
        if self.hidden_state is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(
                x,
                (self.hidden_state, self.cell_state)
            )

        # Store states for next time step
        self.hidden_state = h_n.detach()
        self.cell_state = c_n.detach()

        # Output layer
        output = self.fc(lstm_out[:, -1, :])

        return output

    def reset_states(self):
        """Reset hidden and cell states (call between samples)."""
        self.hidden_state = None
        self.cell_state = None
```

---

### 4.3 Operational Architecture

#### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Local Development Environment                 │
│                                                               │
│   ┌─────────────────────────────────────────────────┐       │
│   │              Python Environment                  │       │
│   │  (virtualenv / conda)                            │       │
│   │                                                   │       │
│   │   ┌─────────────┐    ┌─────────────┐            │       │
│   │   │   Training  │    │  Inference  │            │       │
│   │   │   Scripts   │    │   Scripts   │            │       │
│   │   └──────┬──────┘    └──────┬──────┘            │       │
│   │          │                   │                   │       │
│   │          └─────────┬─────────┘                   │       │
│   │                    │                             │       │
│   │          ┌─────────▼─────────┐                   │       │
│   │          │  LSTM Package     │                   │       │
│   │          │                   │                   │       │
│   │          │  - Models         │                   │       │
│   │          │  - Data           │                   │       │
│   │          │  - Training       │                   │       │
│   │          │  - Evaluation     │                   │       │
│   │          └─────────┬─────────┘                   │       │
│   │                    │                             │       │
│   │          ┌─────────▼─────────┐                   │       │
│   │          │  PyTorch / TF     │                   │       │
│   │          └───────────────────┘                   │       │
│   └─────────────────────────────────────────────────┘       │
│                                                               │
│   ┌─────────────────────────────────────────────────┐       │
│   │            Hardware Resources                    │       │
│   │                                                   │       │
│   │  CPU: Intel/AMD x64  GPU: NVIDIA (optional)     │       │
│   │  RAM: 8GB+           VRAM: 4GB+ (if GPU)        │       │
│   │  Storage: 10GB+                                  │       │
│   └─────────────────────────────────────────────────┘       │
│                                                               │
│   ┌─────────────────────────────────────────────────┐       │
│   │            File System Storage                   │       │
│   │                                                   │       │
│   │  ./data/              - Training & test datasets │       │
│   │  ./models/            - Saved model checkpoints  │       │
│   │  ./results/           - Evaluation outputs       │       │
│   │  ./plots/             - Generated visualizations │       │
│   │  ./logs/              - Training logs            │       │
│   │  ./configs/           - Configuration files      │       │
│   └─────────────────────────────────────────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Deployment Notes:**
- Single-machine deployment (no distributed system)
- Local file system for all storage
- GPU acceleration optional but recommended
- No external services or APIs required

---

#### Scalability Considerations

**Current Scale:**
- Dataset: 80,000 total samples (40K train + 40K test)
- Model size: ~1-5 MB (depending on architecture)
- Training time: 1-2 hours
- Inference: 1000+ samples/second

**Future Scaling (if needed):**

1. **Data Scaling:**
   - HDF5 for larger-than-memory datasets
   - Data loading pipeline optimization
   - Prefetching and caching strategies

2. **Model Scaling:**
   - Larger LSTM hidden sizes (128, 256 units)
   - Deeper architectures (3-4 LSTM layers)
   - Multi-GPU training with DataParallel

3. **Inference Scaling:**
   - Batch processing optimization
   - Model quantization (INT8)
   - ONNX export for production deployment

---

#### Monitoring and Logging

**Training Monitoring:**

```python
# Metrics logged per epoch
{
    "epoch": 42,
    "train_loss": 0.0087,
    "val_loss": 0.0092,
    "train_mse": 0.0087,
    "val_mse": 0.0092,
    "learning_rate": 0.0005,
    "epoch_time": 245.3,
    "timestamp": "2025-11-11T14:32:15"
}
```

**Monitoring Components:**

1. **TensorBoard Integration:**
   - Loss curves (train/validation)
   - Learning rate schedule
   - Gradient histograms
   - Model graph visualization

2. **Console Logging:**
   - Progress bars for epochs
   - Real-time loss updates
   - ETA for training completion

3. **File Logging:**
   - Structured JSON logs
   - Error stack traces
   - System resource usage

**Log Structure:**
```
logs/
├── training_YYYYMMDD_HHMMSS.log
├── evaluation_YYYYMMDD_HHMMSS.log
└── errors_YYYYMMDD_HHMMSS.log
```

---

### 4.4 Data Flow Architecture

#### Training Data Flow

```
┌─────────────────┐
│  Raw Signal     │
│  Parameters     │
│  (f, A, φ)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Signal Generator           │
│  - Generate sinusoids       │
│  - Add noise                │
│  - Mix signals              │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dataset Formatter          │
│  - Create input vectors     │
│  - One-hot encode frequency │
│  - Normalize values         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  DataLoader                 │
│  - Batch creation           │
│  - Shuffle (train only)     │
│  - State reset markers      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  LSTM Model                 │
│  - Forward pass             │
│  - State management         │
│  - Prediction generation    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Loss Computation           │
│  - MSE calculation          │
│  - Backpropagation          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Optimizer                  │
│  - Gradient computation     │
│  - Weight updates           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Checkpoint Save            │
│  (every N epochs)           │
└─────────────────────────────┘
```

#### Inference Data Flow

```
┌─────────────────┐
│  New Signal     │
│  S(t)           │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Input Preparation          │
│  - Format signal            │
│  - Add frequency selector   │
│  - Normalize                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Load Trained Model         │
│  - Load weights             │
│  - Initialize states        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Sequential Processing      │
│  For each time step t:      │
│    1. Forward pass          │
│    2. Store output          │
│    3. Update states         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Extracted Signal           │
│  - Denormalize              │
│  - Format output            │
│  - Return results           │
└─────────────────────────────┘
```

---

### 4.5 Architectural Decision Records (ADRs)

#### ADR-001: Choice of LSTM over Other RNN Architectures

**Status:** Accepted

**Context:**
Need to select a recurrent neural network architecture for time-series signal extraction. Options include:
- Simple RNN
- LSTM
- GRU
- Transformer

**Decision:**
Use LSTM (Long Short-Term Memory) architecture.

**Rationale:**
1. **Gradient Stability:** LSTM's gating mechanisms prevent vanishing/exploding gradients better than simple RNNs
2. **Long-term Dependencies:** Cell state allows learning dependencies across entire 10-second signal
3. **Academic Requirement:** Assignment specifically requires LSTM implementation
4. **Proven Effectiveness:** LSTMs have strong track record in time-series problems
5. **Simplicity vs. Power:** More powerful than simple RNN, simpler than Transformer

**Consequences:**
- **Positive:**
  - Stable training with long sequences
  - Well-supported by frameworks
  - Extensive documentation available
- **Negative:**
  - More parameters than simple RNN
  - Slower training than GRU
  - More complex than simpler alternatives

**Alternatives Considered:**
- **GRU:** Simpler, faster, but less expressive
- **Transformer:** More powerful, but overkill for this problem and computationally expensive
- **Simple RNN:** Too simple, gradient problems

---

#### ADR-002: Sequence Length L = 1 with State Persistence

**Status:** Accepted

**Context:**
Need to decide how to handle temporal sequences in LSTM processing. Options:
1. Long sequences (L = 100-1000) with truncated BPTT
2. Short sequences (L = 1) with state persistence
3. Full sequences (L = 10000) processed at once

**Decision:**
Use sequence length L = 1 with state persistence between consecutive time steps within a sample.

**Rationale:**
1. **Assignment Requirement:** Explicitly required by specifications
2. **State Management Control:** Clear distinction between sample boundaries and time steps
3. **Memory Efficiency:** Avoids storing long sequences in memory
4. **Flexibility:** Easier to implement sliding window inference
5. **Debugging:** Simpler to debug state-related issues

**Consequences:**
- **Positive:**
  - Clear state management semantics
  - Lower memory footprint
  - Easier to implement sample boundaries
- **Negative:**
  - More function calls per epoch
  - Cannot leverage sequence parallelization
  - Slightly slower training

**Implementation Details:**
```python
# Reset states between samples
model.reset_states()

# Process time steps t=0 to t=9999
for t in range(10000):
    output = model(input[t])  # States maintained automatically
```

---

#### ADR-003: State Reset Strategy

**Status:** Accepted

**Context:**
Need to determine when to reset LSTM internal states (h_t, c_t). Critical for preventing information leakage between independent samples.

**Decision:**
- **MUST reset** states between different training samples
- **MUST NOT reset** states when moving from t to t+1 within the same sample

**Rationale:**
1. **Independence:** Different samples have different amplitudes and phases (independent)
2. **Temporal Continuity:** Within a sample, consecutive time steps are causally related
3. **Learning Quality:** State persistence helps learn temporal patterns
4. **Generalization:** Resetting between samples prevents overfitting to specific sequences

**Consequences:**
- **Positive:**
  - Proper temporal modeling
  - No information leakage
  - Better generalization
- **Negative:**
  - Must carefully track sample boundaries
  - More complex data loading logic

**Implementation:**
```python
class SignalDataLoader:
    def __iter__(self):
        for sample_id in self.samples:
            # Reset before each sample
            yield ResetMarker()

            for t in range(self.seq_length):
                yield self.data[sample_id][t]
```

---

#### ADR-004: MSE as Primary Loss Function

**Status:** Accepted

**Context:**
Need to select appropriate loss function for regression problem. Options:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss
- Custom loss function

**Decision:**
Use Mean Squared Error (MSE) as the sole training objective.

**Rationale:**
1. **Assignment Requirement:** MSE explicitly specified for evaluation
2. **Sensitivity to Large Errors:** MSE penalizes large deviations more than MAE
3. **Smooth Gradient:** MSE provides smooth gradients for optimization
4. **Standard Practice:** MSE is standard for continuous regression
5. **Interpretability:** Clear meaning (average squared deviation)

**Formula:**
```
MSE = (1/N) * Σ(predicted - target)²
```

**Consequences:**
- **Positive:**
  - Aligns training and evaluation metrics
  - Well-behaved optimization
  - Easy to interpret
- **Negative:**
  - Sensitive to outliers
  - Squared units (less intuitive than MAE)

---

#### ADR-005: PyTorch as Deep Learning Framework

**Status:** Accepted

**Context:**
Need to select deep learning framework. Major options:
- PyTorch
- TensorFlow/Keras
- JAX

**Decision:**
Use PyTorch as the primary framework (allow TensorFlow as alternative).

**Rationale:**
1. **Academic Preference:** PyTorch dominant in research settings
2. **Pythonic API:** More intuitive for researchers
3. **Dynamic Graphs:** Easier debugging and experimentation
4. **LSTM Support:** Excellent built-in LSTM implementations
5. **Community:** Large user base, extensive tutorials

**Consequences:**
- **Positive:**
  - Faster development
  - Better debugging experience
  - Extensive documentation
- **Negative:**
  - Production deployment less mature than TensorFlow
  - Slightly larger model files

**Alternative Allowed:**
TensorFlow 2.x with Keras API is acceptable alternative if user prefers it.

---

#### ADR-006: File-Based Persistence (No Database)

**Status:** Accepted

**Context:**
Need to store datasets, models, and results. Options:
- File system (NumPy, HDF5, PyTorch files)
- SQLite database
- Remote storage (cloud)

**Decision:**
Use local file system with structured directory organization.

**Rationale:**
1. **Simplicity:** No additional dependencies
2. **Academic Context:** No need for multi-user access
3. **Performance:** Fast local I/O
4. **Portability:** Easy to share and archive
5. **Tools Support:** NumPy/PyTorch handle file I/O well

**File Structure:**
```
project/
├── data/
│   ├── train_signals.npy
│   ├── test_signals.npy
│   └── metadata.json
├── models/
│   ├── checkpoint_epoch_50.pt
│   └── best_model.pt
├── results/
│   ├── metrics.json
│   └── evaluation_report.txt
└── plots/
    ├── frequency_2_detailed.png
    └── all_frequencies.png
```

**Consequences:**
- **Positive:**
  - Simple implementation
  - No setup required
  - Easy backup and version control
- **Negative:**
  - No concurrent access support
  - Manual metadata management
  - No query capabilities

---

#### ADR-007: One-Hot Encoding for Frequency Selection

**Status:** Accepted

**Context:**
Need to encode which frequency to extract. Options:
1. One-hot vector [0,1,0,0] for f₂
2. Single integer index (0, 1, 2, 3)
3. Actual frequency value (1, 3, 5, 7)

**Decision:**
Use one-hot encoding as part of the input vector.

**Rationale:**
1. **Assignment Requirement:** Explicitly specified
2. **Neural Network Compatibility:** NNs handle one-hot well
3. **No Ordinal Assumption:** Frequencies are categorical, not ordinal
4. **Clear Semantics:** Each frequency has its own "channel"
5. **Flexibility:** Easy to extend to more frequencies

**Format:**
```python
Input vector: [S(t), C1, C2, C3, C4]
Example for f₂: [0.523, 0, 1, 0, 0]
```

**Consequences:**
- **Positive:**
  - Clear, unambiguous encoding
  - No artificial ordering
  - Each frequency learned independently
- **Negative:**
  - Larger input vector (5 vs. 2 dimensions)
  - Slight redundancy (4 values encode 2 bits)

---

### 4.6 API and Interface Documentation

#### Dataset Generation API

```python
def generate_signal_dataset(
    frequencies: List[float],
    duration: float,
    sampling_rate: float,
    amplitude_range: Tuple[float, float],
    phase_range: Tuple[float, float],
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic mixed signal dataset.

    Args:
        frequencies: List of sinusoid frequencies in Hz (e.g., [1, 3, 5, 7])
        duration: Signal duration in seconds (e.g., 10.0)
        sampling_rate: Sampling rate in Hz (e.g., 1000)
        amplitude_range: (min, max) for amplitude uniform distribution
        phase_range: (min, max) for phase uniform distribution in radians
        random_seed: Seed for reproducibility (None for random)

    Returns:
        mixed_signal: Mixed noisy signal S(t), shape [n_samples]
        target_signals: Pure sinusoids, shape [n_frequencies, n_samples]
        metadata: Dictionary with generation parameters

    Example:
        >>> mixed, targets, meta = generate_signal_dataset(
        ...     frequencies=[1, 3, 5, 7],
        ...     duration=10.0,
        ...     sampling_rate=1000,
        ...     amplitude_range=(0.8, 1.2),
        ...     phase_range=(0, 2*np.pi),
        ...     random_seed=42
        ... )
        >>> mixed.shape
        (10000,)
        >>> targets.shape
        (4, 10000)
    """
```

---

#### Model Training API

```python
def train_lstm_model(
    model: nn.Module,
    train_data: SignalDataset,
    val_data: Optional[SignalDataset],
    config: TrainingConfig
) -> TrainingResults:
    """
    Train LSTM signal extraction model.

    Args:
        model: LSTMSignalExtractor instance
        train_data: Training dataset
        val_data: Validation dataset (optional)
        config: Training configuration

    Returns:
        TrainingResults object containing:
            - best_model_path: Path to saved best model
            - train_history: Dict of training metrics
            - val_history: Dict of validation metrics
            - final_train_mse: Final training MSE
            - final_val_mse: Final validation MSE

    Example:
        >>> config = TrainingConfig(
        ...     batch_size=64,
        ...     num_epochs=100,
        ...     learning_rate=0.001,
        ...     early_stopping_patience=10
        ... )
        >>> results = train_lstm_model(model, train_data, val_data, config)
        >>> print(f"Best MSE: {results.final_val_mse:.6f}")
    """
```

---

#### Inference API

```python
def extract_signal(
    model: nn.Module,
    mixed_signal: np.ndarray,
    frequency_id: int,
    sampling_rate: float = 1000
) -> np.ndarray:
    """
    Extract specific frequency component from mixed signal.

    Args:
        model: Trained LSTMSignalExtractor
        mixed_signal: Mixed noisy signal S(t), shape [n_samples]
        frequency_id: Which frequency to extract (0=1Hz, 1=3Hz, 2=5Hz, 3=7Hz)
        sampling_rate: Sampling rate in Hz

    Returns:
        extracted_signal: Extracted pure sinusoid, shape [n_samples]

    Example:
        >>> model = load_model('best_model.pt')
        >>> signal = np.load('test_signal.npy')
        >>> extracted_3hz = extract_signal(model, signal, frequency_id=1)
        >>> plt.plot(extracted_3hz)

    Raises:
        ValueError: If frequency_id not in [0, 1, 2, 3]
        RuntimeError: If model not in evaluation mode
    """
```

---

#### Evaluation API

```python
def evaluate_model(
    model: nn.Module,
    test_data: SignalDataset,
    device: str = 'cuda'
) -> EvaluationResults:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained LSTMSignalExtractor
        test_data: Test dataset
        device: Computation device ('cuda' or 'cpu')

    Returns:
        EvaluationResults containing:
            - overall_mse: Overall test MSE
            - per_frequency_mse: Dict mapping frequency to MSE
            - correlation_coefficients: Per-frequency correlation
            - psnr_values: Per-frequency PSNR
            - inference_time: Average inference time per sample

    Example:
        >>> results = evaluate_model(model, test_dataset)
        >>> print(f"Overall MSE: {results.overall_mse:.6f}")
        >>> for freq, mse in results.per_frequency_mse.items():
        ...     print(f"  {freq}Hz: MSE = {mse:.6f}")
    """
```

---

#### Visualization API

```python
def plot_extraction_results(
    time: np.ndarray,
    target: np.ndarray,
    noisy_input: np.ndarray,
    predicted: np.ndarray,
    frequency: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create detailed plot showing signal extraction quality.

    Args:
        time: Time array in seconds, shape [n_samples]
        target: Clean target sinusoid, shape [n_samples]
        noisy_input: Noisy mixed signal, shape [n_samples]
        predicted: LSTM prediction, shape [n_samples]
        frequency: Frequency in Hz (for title)
        save_path: Path to save figure (None to display only)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_extraction_results(
        ...     time=np.linspace(0, 10, 10000),
        ...     target=clean_signal,
        ...     noisy_input=mixed_signal,
        ...     predicted=lstm_output,
        ...     frequency=3.0,
        ...     save_path='results/freq_3hz.png'
        ... )
    """
```

---

## 5. Risk Assessment and Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Training doesn't converge** | Medium | High | - Try different learning rates<br>- Adjust architecture (hidden size, layers)<br>- Implement learning rate scheduling<br>- Add gradient clipping |
| **MSE_test >> MSE_train (overfitting)** | Medium | High | - Add dropout layers<br>- Increase dataset size<br>- Implement early stopping<br>- L2 regularization |
| **Exploding/vanishing gradients** | Low | Medium | - Gradient clipping (threshold=1.0)<br>- Proper weight initialization<br>- Monitor gradient norms |
| **Memory overflow during training** | Low | Medium | - Reduce batch size<br>- Use gradient accumulation<br>- Clear CUDA cache regularly |
| **State management bugs** | Medium | High | - Comprehensive unit tests<br>- Visualize state behavior<br>- Debug with small sequences first |

### Project Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Timeline delays** | Medium | Medium | - Prioritize P0 requirements<br>- Start with simple baseline<br>- Parallel work on visualization |
| **Insufficient computational resources** | Low | Medium | - Optimize for CPU training<br>- Use cloud resources if needed<br>- Reduce model size if necessary |
| **Requirement misunderstanding** | Low | High | - Clarify specifications early<br>- Validate with instructor<br>- Document assumptions |
| **Code quality issues** | Medium | Low | - Follow coding standards<br>- Peer review<br>- Automated testing |

### Quality Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Poor signal extraction quality** | Medium | High | - Hyperparameter tuning<br>- Architecture search<br>- Data augmentation<br>- Ensemble methods |
| **Inconsistent results** | Low | Medium | - Fix all random seeds<br>- Document environment<br>- Version control dependencies |
| **Visualization not meeting standards** | Low | Low | - Use professional plotting libraries<br>- Follow publication guidelines<br>- Get early feedback |

---

## 6. Success Criteria and Acceptance Testing

### Definition of Done

A feature is considered "done" when:

1. **Code Complete:**
   - Implementation matches specification
   - Code follows style guidelines
   - No critical bugs or errors
   - Error handling implemented

2. **Tested:**
   - Unit tests written and passing
   - Integration tests passing
   - Manual testing completed
   - Edge cases handled

3. **Documented:**
   - Docstrings complete
   - README updated
   - Examples provided
   - Known limitations documented

4. **Reviewed:**
   - Code review completed
   - Feedback addressed
   - Approved by stakeholder

### Acceptance Test Plan

#### Test Suite 1: Dataset Generation

```python
def test_dataset_generation():
    """Validate dataset generation meets requirements."""

    # Test 1: Correct dimensions
    mixed, targets, meta = generate_signal_dataset(...)
    assert mixed.shape == (10000,), "Wrong number of samples"
    assert targets.shape == (4, 10000), "Wrong target dimensions"

    # Test 2: Frequency content
    fft_result = np.fft.fft(targets[0])
    peak_freq = np.argmax(np.abs(fft_result))
    assert peak_freq == 1, "Incorrect frequency in target"

    # Test 3: Amplitude range
    assert np.all(targets >= -1.2) and np.all(targets <= 1.2)

    # Test 4: Reproducibility
    data1 = generate_signal_dataset(..., random_seed=42)
    data2 = generate_signal_dataset(..., random_seed=42)
    assert np.allclose(data1[0], data2[0]), "Not reproducible"

    print("✓ Dataset generation tests passed")
```

#### Test Suite 2: LSTM Architecture

```python
def test_lstm_architecture():
    """Validate LSTM model structure."""

    model = LSTMSignalExtractor(input_size=5, hidden_size=64)

    # Test 1: Input/output shapes
    input_tensor = torch.randn(32, 1, 5)  # [batch, seq, features]
    output = model(input_tensor)
    assert output.shape == (32, 1), "Incorrect output shape"

    # Test 2: State persistence
    model.reset_states()
    out1 = model(input_tensor)
    out2 = model(input_tensor)
    assert not torch.allclose(out1, out2), "States not persisting"

    # Test 3: State reset
    model.reset_states()
    out3 = model(input_tensor)
    assert torch.allclose(out1, out3), "State reset not working"

    # Test 4: Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 1_000_000, "Model too large"

    print("✓ LSTM architecture tests passed")
```

#### Test Suite 3: Training Process

```python
def test_training_process():
    """Validate training functionality."""

    # Test 1: Training loop executes
    try:
        results = train_lstm_model(model, train_data, val_data, config)
        assert results is not None
    except Exception as e:
        pytest.fail(f"Training failed: {e}")

    # Test 2: Loss decreases
    losses = results.train_history['loss']
    assert losses[-1] < losses[0], "Loss not decreasing"

    # Test 3: Checkpoints saved
    assert os.path.exists(results.best_model_path)

    # Test 4: Reproducibility with seed
    results1 = train_lstm_model(..., random_seed=42)
    results2 = train_lstm_model(..., random_seed=42)
    assert np.isclose(results1.final_train_mse, results2.final_train_mse)

    print("✓ Training process tests passed")
```

#### Test Suite 4: Performance Requirements

```python
def test_performance_requirements():
    """Validate performance metrics meet targets."""

    # Test 1: Training MSE < 0.01
    assert results.final_train_mse < 0.01, f"Train MSE too high: {results.final_train_mse}"

    # Test 2: Test MSE < 0.01
    assert results.final_test_mse < 0.01, f"Test MSE too high: {results.final_test_mse}"

    # Test 3: MSE ratio in acceptable range
    ratio = results.final_test_mse / results.final_train_mse
    assert 0.9 <= ratio <= 1.1, f"MSE ratio out of range: {ratio}"

    # Test 4: Inference speed
    import time
    start = time.time()
    _ = model(test_input)
    elapsed = time.time() - start
    assert elapsed < 0.001, f"Inference too slow: {elapsed}s"

    print("✓ Performance requirements met")
```

#### Test Suite 5: Visualization Quality

```python
def test_visualization_quality():
    """Validate generated plots meet requirements."""

    # Test 1: All required plots generated
    assert os.path.exists('plots/frequency_2_detailed.png')
    assert os.path.exists('plots/all_frequencies.png')

    # Test 2: Image resolution
    img = plt.imread('plots/frequency_2_detailed.png')
    assert img.shape[0] >= 2400, "DPI too low"  # 8 inches * 300 DPI

    # Test 3: Contains all required elements
    # (This would require image processing or manual verification)

    print("✓ Visualization quality tests passed")
```

### Final Acceptance Criteria

The project is considered complete and successful when:

- [ ] All test suites pass
- [ ] Training MSE < 0.01
- [ ] Test MSE < 0.01
- [ ] 0.9 < (MSE_test / MSE_train) < 1.1
- [ ] All four frequencies extracted successfully
- [ ] Both required visualizations generated and meeting quality standards
- [ ] Code is clean, documented, and follows style guidelines
- [ ] README provides clear setup and usage instructions
- [ ] Training completes within 2 hours
- [ ] Inference time < 1ms per sample
- [ ] No critical bugs or errors
- [ ] All deliverables submitted on time

---

## 7. Glossary and Terminology

| Term | Definition |
|------|------------|
| **LSTM** | Long Short-Term Memory, a type of recurrent neural network architecture |
| **MSE** | Mean Squared Error, loss function measuring average squared difference |
| **Hidden State (h_t)** | LSTM's short-term memory, passed between time steps |
| **Cell State (c_t)** | LSTM's long-term memory, preserved across time steps |
| **One-Hot Encoding** | Binary vector representation with single 1 and rest 0s |
| **Sinusoid** | Periodic wave function: A*sin(2πft + φ) |
| **Sampling Rate (Fs)** | Number of samples per second (1000 Hz = 1000 samples/second) |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | Subset of training data processed together |
| **Sequence Length (L)** | Number of time steps processed in one forward pass |
| **Forward Pass** | Computing output from input through network layers |
| **Backward Pass** | Computing gradients via backpropagation |
| **Gradient Clipping** | Limiting gradient magnitude to prevent instability |
| **Early Stopping** | Halting training when validation performance stops improving |
| **Checkpoint** | Saved model state at a specific training point |
| **PSNR** | Peak Signal-to-Noise Ratio, quality metric in dB |
| **BPTT** | Backpropagation Through Time, training algorithm for RNNs |

---

## 8. Appendices

### Appendix A: Mathematical Formulations

#### Signal Generation Formula

Individual noisy sinusoid:
```
Sinus_i^noisy(t) = A_i(t) * sin(2π * f_i * t + φ_i(t))

where:
  A_i(t) ~ Uniform(0.8, 1.2)
  φ_i(t) ~ Uniform(0, 2π)
  f_i ∈ {1, 3, 5, 7} Hz
```

Mixed signal:
```
S(t) = (1/4) * Σ[i=1 to 4] Sinus_i^noisy(t)
```

#### LSTM Equations

**Input Gate:**
```
i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
```

**Forget Gate:**
```
f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
```

**Cell Update:**
```
g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

**Output Gate:**
```
o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
h_t = o_t ⊙ tanh(c_t)
```

#### Loss Function

Mean Squared Error:
```
MSE = (1/N) * Σ[i=1 to N] (predicted_i - target_i)²
```

### Appendix B: Configuration File Example

```yaml
# config.yaml - LSTM Signal Extraction Configuration

# Dataset parameters
dataset:
  frequencies: [1, 3, 5, 7]  # Hz
  duration: 10.0              # seconds
  sampling_rate: 1000         # Hz
  amplitude_range: [0.8, 1.2]
  phase_range: [0, 6.283185]  # [0, 2π]
  train_seed: 42
  test_seed: 123

# Model architecture
model:
  input_size: 5
  hidden_size: 64
  num_layers: 2
  output_size: 1
  dropout: 0.1

# Training parameters
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  optimizer: 'adam'
  lr_scheduler: 'step'
  lr_step_size: 30
  lr_gamma: 0.1
  gradient_clip: 1.0
  early_stopping_patience: 10
  checkpoint_interval: 10

# Evaluation parameters
evaluation:
  metrics: ['mse', 'correlation', 'psnr']
  save_predictions: true

# Visualization parameters
visualization:
  dpi: 300
  figure_size: [12, 8]
  color_scheme: 'tab10'
  save_format: 'png'

# Paths
paths:
  data_dir: './data'
  model_dir: './models'
  results_dir: './results'
  plots_dir: './plots'
  logs_dir: './logs'

# Reproducibility
random_seed: 42

# Device configuration
device: 'cuda'  # 'cuda' or 'cpu'
```

### Appendix C: Project Directory Structure

```
lstm-signal-extraction/
│
├── README.md
├── requirements.txt
├── config.yaml
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py        # Dataset generation
│   │   ├── loader.py            # Data loading utilities
│   │   └── preprocessor.py     # Data preprocessing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py              # LSTM architecture
│   │   └── losses.py            # Loss functions
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── optimizer.py         # Optimizer configuration
│   │   └── checkpointing.py    # Model saving/loading
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── evaluator.py         # Evaluation pipeline
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plotter.py           # Plotting functions
│   │   └── report.py            # Report generation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logging_utils.py     # Logging setup
│       └── reproducibility.py   # Seed setting
│
├── scripts/
│   ├── generate_data.py         # Generate datasets
│   ├── train.py                 # Train model
│   ├── evaluate.py              # Evaluate model
│   └── extract_signal.py        # Inference script
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── data/                        # Generated at runtime
│   ├── train/
│   └── test/
│
├── models/                      # Generated at runtime
│   └── checkpoints/
│
├── results/                     # Generated at runtime
│   ├── metrics/
│   └── predictions/
│
├── plots/                       # Generated at runtime
│
└── logs/                        # Generated at runtime
```

### Appendix D: Sample Command-Line Usage

```bash
# Step 1: Generate datasets
python scripts/generate_data.py \
    --config config.yaml \
    --output-dir ./data

# Step 2: Train model
python scripts/train.py \
    --config config.yaml \
    --data-dir ./data \
    --output-dir ./models \
    --device cuda

# Step 3: Evaluate model
python scripts/evaluate.py \
    --model ./models/best_model.pt \
    --test-data ./data/test \
    --output-dir ./results

# Step 4: Generate visualizations
python scripts/visualize.py \
    --results ./results/predictions.npy \
    --output-dir ./plots

# Optional: Extract specific signal
python scripts/extract_signal.py \
    --model ./models/best_model.pt \
    --signal ./data/custom_signal.npy \
    --frequency 3 \
    --output extracted_3hz.npy
```

### Appendix E: Performance Benchmarks

Expected performance on reference hardware:

**Training Performance (NVIDIA RTX 3080, 10GB VRAM):**
- Dataset generation: ~30 seconds
- Training (100 epochs): ~45 minutes
- Evaluation: ~5 minutes
- Visualization: ~10 seconds

**Training Performance (CPU only - Intel i7-10700K):**
- Dataset generation: ~30 seconds
- Training (100 epochs): ~2 hours
- Evaluation: ~10 minutes
- Visualization: ~10 seconds

**Memory Usage:**
- Training (batch_size=64): ~2 GB GPU memory
- Inference: ~500 MB system memory
- Dataset storage: ~320 MB (80K samples)

**Model Size:**
- Hidden size 64, 2 layers: ~850 KB
- Hidden size 128, 2 layers: ~3.2 MB

---

## Document Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-11 | Product Manager | Initial PRD creation |

---

## Document Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | [To be filled] | [Date] | [Signature] |
| Technical Lead | [To be filled] | [Date] | [Signature] |
| Instructor/Advisor | [To be filled] | [Date] | [Signature] |

---

**End of Document**

*This PRD is a living document and will be updated as the project progresses and requirements evolve.*
