#!/bin/bash

################################################################################
# Run Script for Prompt Engineering Assignment
# This script will:
# 1. Check if setup is complete
# 2. Ensure Ollama is running
# 3. Activate virtual environment
# 4. Run the baseline experiment
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "Starting Experiment Runner"

# Check 1: Ollama installed
print_info "Checking prerequisites..."

if ! command_exists ollama; then
    print_error "Ollama is not installed"
    print_info "Please run './setup.sh' first"
    exit 1
fi
print_success "Ollama installed"

# Check 2: Ollama running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_warning "Ollama service is not running. Starting it..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/Applications/Ollama.app" ]; then
            open -a Ollama
        else
            nohup ollama serve > /dev/null 2>&1 &
        fi
    else
        nohup ollama serve > /dev/null 2>&1 &
    fi

    # Wait for service
    print_info "Waiting for Ollama to start..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama service started"
            break
        fi
        sleep 1
    done

    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Failed to start Ollama"
        print_info "Try running 'ollama serve' manually"
        exit 1
    fi
else
    print_success "Ollama service running"
fi

# Check 3: Virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found"
    print_info "Please run './setup.sh' first"
    exit 1
fi
print_success "Virtual environment found"

# Check 4: Dataset exists
if [ ! -f "data/sentiment_dataset.json" ]; then
    print_error "Dataset not found"
    print_info "Please run './setup.sh' first"
    exit 1
fi
print_success "Dataset found"

# Check 5: Source files exist
if [ ! -f "src/baseline_experiment.py" ]; then
    print_error "Experiment files not found"
    exit 1
fi
print_success "Experiment files found"

################################################################################
# Activate Virtual Environment
################################################################################

print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

################################################################################
# Parse Arguments
################################################################################

EXPERIMENT_TYPE="baseline"
ADDITIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --model $2"
            shift 2
            ;;
        --show-errors)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --show-errors"
            shift
            ;;
        --no-save)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --no-save"
            shift
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Use specific Ollama model"
            echo "  --show-errors        Show misclassified examples"
            echo "  --no-save           Don't save results to file"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                          # Run with default settings"
            echo "  ./run.sh --model mistral          # Use mistral model"
            echo "  ./run.sh --show-errors            # Show error cases"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Run Experiment
################################################################################

print_header "Running Baseline Experiment"

cd src

# Load model name from .env if not specified
if [ -f "../.env" ]; then
    source ../.env
    print_info "Model: $MODEL_NAME"
    print_info "Ollama Host: $OLLAMA_HOST"
fi

# Run the experiment
python baseline_experiment.py $ADDITIONAL_ARGS

if [ $? -eq 0 ]; then
    print_header "Experiment Complete! 🎉"

    echo -e "${GREEN}Results saved to 'results/' directory${NC}\n"

    # Show results directory
    if [ -d "../results" ] && [ "$(ls -A ../results)" ]; then
        echo -e "${BLUE}Recent results:${NC}"
        ls -lht ../results | head -n 6
    fi

    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "  • View results: ${YELLOW}ls -lh results/${NC}"
    echo -e "  • Run with different model: ${YELLOW}./run.sh --model mistral${NC}"
    echo -e "  • Show error cases: ${YELLOW}./run.sh --show-errors${NC}\n"
else
    print_error "Experiment failed"
    exit 1
fi
