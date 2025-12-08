#!/bin/bash

################################################################################
# Analysis Script - Phase 4
# Generates visualizations, statistical analysis, and final report
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
# Main
################################################################################

print_header "Phase 4: Analysis & Visualization"

# Check virtual environment
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found"
    print_info "Please run './setup.sh' first"
    exit 1
fi

# Activate venv
source venv/bin/activate
print_success "Virtual environment activated"

# Check for results
if [ ! -d "results" ] || [ -z "$(ls -A results/comparison_metrics_*.json 2>/dev/null)" ]; then
    print_error "No experiment results found"
    print_info "Please run experiments first: ./run_all.sh"
    exit 1
fi

# Find latest comparison file
COMPARISON_FILE=$(ls -t results/comparison_metrics_*.json 2>/dev/null | head -n1)

if [ -z "$COMPARISON_FILE" ]; then
    print_error "No comparison file found"
    exit 1
fi

print_success "Found results: $COMPARISON_FILE"

################################################################################
# Run Analysis
################################################################################

print_header "Step 1/3: Statistical Analysis"

cd analysis
python statistical_analysis.py --latest

if [ $? -ne 0 ]; then
    print_error "Statistical analysis failed"
    exit 1
fi

print_header "Step 2/3: Generating Visualizations"

python visualization.py --latest

if [ $? -ne 0 ]; then
    print_error "Visualization generation failed"
    exit 1
fi

print_header "Step 3/3: Generating Final Report"

python generate_report.py --latest

if [ $? -ne 0 ]; then
    print_error "Report generation failed"
    exit 1
fi

cd ..

################################################################################
# Summary
################################################################################

print_header "Analysis Complete! 🎉"

echo -e "${GREEN}All analysis artifacts generated successfully!${NC}\n"

echo -e "${BLUE}Generated Files:${NC}"
echo ""

echo -e "${YELLOW}Visualizations:${NC}"
if [ -d "visualizations" ]; then
    ls -lh visualizations/*.png 2>/dev/null | awk '{print "  - "$9}' || echo "  (none)"
else
    echo "  (none)"
fi

echo ""
echo -e "${YELLOW}Analysis:${NC}"
if [ -d "analysis" ]; then
    ls -lht analysis/statistical_analysis_*.json 2>/dev/null | head -n1 | awk '{print "  - "$9}'
else
    echo "  (none)"
fi

echo ""
echo -e "${YELLOW}Report:${NC}"
if [ -f "EXPERIMENT_REPORT.md" ]; then
    echo "  - EXPERIMENT_REPORT.md"
else
    echo "  (none)"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  • View visualizations: ${YELLOW}open visualizations/${NC}"
echo -e "  • Read report: ${YELLOW}cat EXPERIMENT_REPORT.md${NC}"
echo -e "  • View analysis: ${YELLOW}cat analysis/statistical_analysis_*.json | jq${NC}"

echo ""
