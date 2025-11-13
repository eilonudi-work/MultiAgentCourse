# Data Science Project Planning Prompt

## Primary Task
Read and analyze `../sources/task.md` - this document contains the core project requirements and is the foundation for your entire project plan.

## Your Mission
Create a comprehensive project plan for the task described in task.md. Your plan must demonstrate mastery of transformer architectures, positional encodings, vector embeddings, and semantic analysis.

## Required Sources Review
Before planning, thoroughly review these materials in `../sources/`:

1. **task.md** - The core project requirements (read this first)
2. **Basic-Transformer-Book.pdf** - Reference for transformer architecture decisions
3. **sin-cos-positions-book.pdf** - Reference for positional encoding strategies
4. **software_submission_guidelines.pdf** - Code quality and project structure standards

## Project Plan Structure

### 1. Task Analysis
- Summarize the core requirements from task.md
- Identify the main NLP/ML challenges
- Define success criteria and metrics
- List computational and data requirements

### 2. Technical Approach
- **Architecture**: Which transformer variant (encoder/decoder/both) and why
- **Positional Encoding**: Sinusoidal vs learned, justify your choice
- **Embeddings**: Vector dimensions, similarity metrics, retrieval strategy
- **Training**: Pre-trained model selection, fine-tuning approach, optimization strategy

### 3. Implementation Roadmap
Break down into phases with concrete milestones:
- Phase 1: Data preparation and preprocessing
- Phase 2: Model architecture implementation
- Phase 3: Training and hyperparameter tuning
- Phase 4: Evaluation and analysis
- Phase 5: Deployment preparation

Each phase should include: timeline, deliverables, success criteria

### 4. Code Organization
Following software_submission_guidelines.pdf, define:
- Project structure with key directories
- Module organization (models, training, inference, utils)
- Testing strategy (unit, integration)
- Documentation approach

### 5. Risk Mitigation
- Technical risks and mitigation strategies
- Fallback approaches if primary method fails
- Resource constraints and solutions

## Key Deliverables
Your plan should produce:
1. Detailed project plan document (referencing task.md requirements)
2. Architecture diagrams for transformer components
3. Implementation timeline with milestones
4. Code structure template
5. Testing and evaluation strategy

## Quality Standards
- Be specific: actual numbers, configurations, frameworks
- Justify all major decisions with references to source materials
- Include both theoretical understanding and practical implementation details
- Think production-ready from the start

Create a plan that another senior data scientist could execute successfully based solely on your documentation.
