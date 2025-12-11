#!/bin/bash

################################################################################
# Run All Experiments Script
# Runs all prompt variations and generates comparison
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "Running All Prompt Variations"

# Check virtual environment
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found"
    print_info "Please run './setup.sh' first"
    exit 1
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_error "Ollama service is not running"
    print_info "Please start Ollama: ollama serve"
    exit 1
fi

print_success "Pre-flight checks passed"

################################################################################
# Activate Virtual Environment
################################################################################

source venv/bin/activate
print_success "Virtual environment activated"

################################################################################
# Parse Arguments
################################################################################

ADDITIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --model $2"
            shift 2
            ;;
        --variations)
            shift
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --variations"
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"
                shift
            done
            ;;
        --no-save)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --no-save"
            shift
            ;;
        --help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Use specific Ollama model"
            echo "  --variations VAR...     Run only specific variations"
            echo "  --no-save              Don't save results to file"
            echo "  --help                 Show this help message"
            echo ""
            echo "Available variations:"
            echo "  baseline, role_based, few_shot, chain_of_thought,"
            echo "  structured_output, contrastive"
            echo ""
            echo "Examples:"
            echo "  ./run_all.sh                                    # Run all variations"
            echo "  ./run_all.sh --model mistral                    # Use mistral model"
            echo "  ./run_all.sh --variations baseline few_shot     # Run only 2 variations"
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
# Run All Experiments
################################################################################

print_header "Starting Batch Experiments"

cd src

python run_all_experiments.py $ADDITIONAL_ARGS

if [ $? -eq 0 ]; then
    print_header "All Experiments Complete! 🎉"

    echo -e "${GREEN}Results saved to 'results/' directory${NC}\n"

    # Show results
    if [ -d "../results" ] && [ "$(ls -A ../results)" ]; then
        echo -e "${BLUE}Recent results:${NC}"
        ls -lht ../results | grep "$(date +%Y%m%d)" | head -n 10
    fi

    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "  • View comparison: ${YELLOW}cat results/comparison_metrics_*.json | jq${NC}"
    echo -e "  • Analyze results: Move to Phase 4 (visualizations)${NC}\n"
else
    print_error "Experiments failed"
    exit 1
fi
