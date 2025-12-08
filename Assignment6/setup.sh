#!/bin/bash

################################################################################
# Automated Setup Script for Prompt Engineering Assignment
# This script will:
# 1. Check for Ollama installation
# 2. Install Ollama if not present
# 3. Pull the required model
# 4. Set up Python virtual environment
# 5. Install Python dependencies
# 6. Configure environment variables
# 7. Run setup tests
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model
DEFAULT_MODEL="llama2"
MODEL_NAME="${1:-$DEFAULT_MODEL}"

# Helper functions
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Main Setup Process
################################################################################

print_header "Prompt Engineering Assignment - Automated Setup"

# Step 1: Check for Ollama
print_info "Step 1/7: Checking for Ollama installation..."

if command_exists ollama; then
    print_success "Ollama is already installed"
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    print_info "Version: $OLLAMA_VERSION"
else
    print_warning "Ollama not found. Installing Ollama..."

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_info "Detected macOS. Installing Ollama..."
        if command_exists brew; then
            brew install ollama
        else
            # Manual install
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        print_info "Detected Linux. Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        print_error "Unsupported OS: $OSTYPE"
        print_info "Please install Ollama manually from https://ollama.ai"
        exit 1
    fi

    if command_exists ollama; then
        print_success "Ollama installed successfully"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
fi

# Step 2: Start Ollama service
print_info "Step 2/7: Starting Ollama service..."

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_success "Ollama service is already running"
else
    print_info "Starting Ollama service in background..."

    # Start Ollama in background
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use open if it's an app, otherwise use ollama serve
        if [ -d "/Applications/Ollama.app" ]; then
            open -a Ollama
        else
            nohup ollama serve > /dev/null 2>&1 &
        fi
    else
        # Linux
        nohup ollama serve > /dev/null 2>&1 &
    fi

    # Wait for service to start
    print_info "Waiting for Ollama service to start..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama service started successfully"
            break
        fi
        sleep 1
    done

    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Failed to start Ollama service"
        print_info "Try running 'ollama serve' manually in another terminal"
        exit 1
    fi
fi

# Step 3: Pull model
print_info "Step 3/7: Checking for model '$MODEL_NAME'..."

if ollama list | grep -q "$MODEL_NAME"; then
    print_success "Model '$MODEL_NAME' is already available"
else
    print_info "Pulling model '$MODEL_NAME'... (this may take a few minutes)"
    ollama pull "$MODEL_NAME"

    if ollama list | grep -q "$MODEL_NAME"; then
        print_success "Model '$MODEL_NAME' pulled successfully"
    else
        print_error "Failed to pull model '$MODEL_NAME'"
        exit 1
    fi
fi

# Step 4: Check Python version
print_info "Step 4/7: Checking Python installation..."

if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 is installed (version: $PYTHON_VERSION)"
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_success "Python is installed (version: $PYTHON_VERSION)"
    PYTHON_CMD="python"
else
    print_error "Python is not installed"
    print_info "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Step 5: Create virtual environment
print_info "Step 5/7: Setting up Python virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment recreated"
    else
        print_info "Using existing virtual environment"
    fi
else
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Step 6: Install dependencies
print_info "Step 6/7: Installing Python dependencies..."

pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Python dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Step 7: Configure environment
print_info "Step 7/7: Configuring environment..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    # Update model name in .env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" .env
    else
        sed -i "s/MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" .env
    fi
    print_success "Environment file created and configured"
else
    print_info ".env file already exists (not overwriting)"
fi

# Run setup test
print_header "Running Setup Tests"

$PYTHON_CMD test_setup.py

if [ $? -eq 0 ]; then
    print_header "Setup Complete! 🎉"
    echo -e "${GREEN}Everything is ready to run experiments!${NC}\n"

    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Activate virtual environment: ${YELLOW}source venv/bin/activate${NC}"
    echo -e "  2. Run baseline experiment:     ${YELLOW}./run.sh${NC}"
    echo -e "  or manually:                    ${YELLOW}cd src && python baseline_experiment.py${NC}\n"

    echo -e "${BLUE}Installed components:${NC}"
    echo -e "  • Ollama: $(ollama --version 2>/dev/null | head -n1 || echo 'installed')"
    echo -e "  • Model:  $MODEL_NAME"
    echo -e "  • Python: $PYTHON_VERSION"
    echo -e "  • Dataset: 30 sentiment examples\n"
else
    print_error "Setup tests failed"
    print_info "Please check the errors above and try again"
    exit 1
fi
