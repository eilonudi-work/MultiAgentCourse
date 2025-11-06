#!/bin/bash

# Ollama Web GUI - Development Startup Script
# This script starts both the backend and frontend in development mode

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  🦙 Ollama Web GUI - Development Startup              ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down services..."

    # Kill background processes
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    print_info "Services stopped"
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

# Main script starts here
print_header

# 1. Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

# Check Node.js
if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi
print_success "Node.js found: $(node --version)"

# Check npm
if ! command_exists npm; then
    print_error "npm is not installed. Please install npm."
    exit 1
fi
print_success "npm found: $(npm --version)"

# Check Ollama
if ! command_exists ollama; then
    print_error "Ollama is not installed. Please install Ollama first."
    echo ""
    print_info "Installation instructions:"
    print_info "  macOS: brew install ollama"
    print_info "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    print_info "  Or download from: https://ollama.ai/download"
    exit 1
fi
print_success "Ollama found: $(ollama --version 2>&1 | head -n 1)"

# 2. Check if ports are available
print_info "Checking ports..."

if port_in_use 8000; then
    print_error "Port 8000 is already in use. Please free the port or stop the existing backend."
    exit 1
fi
print_success "Port 8000 is available (backend)"

if port_in_use 5173; then
    print_error "Port 5173 is already in use. Please free the port or stop the existing frontend."
    exit 1
fi
print_success "Port 5173 is available (frontend)"

# 3. Check if Ollama is running and start if needed
print_info "Checking Ollama service..."

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_success "Ollama is running on port 11434"
else
    print_warning "Ollama service is not running. Starting Ollama..."

    # Start Ollama in background
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use brew services or start manually
        if command_exists brew; then
            brew services start ollama 2>/dev/null || ollama serve > /dev/null 2>&1 &
        else
            ollama serve > /dev/null 2>&1 &
        fi
    else
        # Linux - start manually
        ollama serve > /dev/null 2>&1 &
    fi

    # Wait for Ollama to start (max 10 seconds)
    print_info "Waiting for Ollama to start..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama started successfully"
            break
        fi
        sleep 1
    done

    # Check if it started
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Failed to start Ollama service"
        print_info "Please start manually: ollama serve"
        exit 1
    fi
fi

# 3.1 Check and pull required model
REQUIRED_MODEL="llama3.2:1b"
print_info "Checking for required model: $REQUIRED_MODEL..."

# Get list of models
if curl -s http://localhost:11434/api/tags | grep -q "\"name\":\"$REQUIRED_MODEL\""; then
    print_success "Model $REQUIRED_MODEL is already available"
else
    print_warning "Model $REQUIRED_MODEL not found."
    echo ""
    print_info "The recommended model is $REQUIRED_MODEL (~1.3GB download)."
    echo -n "Would you like to download it now? (y/n): "
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Pulling model $REQUIRED_MODEL..."
        print_info "This may take several minutes depending on your internet connection..."

        if ollama pull "$REQUIRED_MODEL"; then
            print_success "Model $REQUIRED_MODEL pulled successfully"
        else
            print_error "Failed to pull model $REQUIRED_MODEL"
            print_info "You can pull it manually later: ollama pull $REQUIRED_MODEL"
            print_warning "Continuing without the model (chat may not work until model is available)"
        fi
    else
        print_warning "Skipping model download."
        print_info "Make sure you have another model available. To list models: ollama list"
        print_info "To pull a model manually: ollama pull <model-name>"
        print_info "Popular options: llama2, mistral, codellama"
    fi
fi

# 4. Setup backend
print_info "Setting up backend..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in backend directory!"
    print_error "Cannot install backend dependencies."
    exit 1
fi

# Check if dependencies are installed
if [ ! -f "venv/installed" ]; then
    print_info "Installing backend dependencies (this may take a few minutes)..."
    echo "  Upgrading pip..."
    pip install --upgrade pip --quiet

    echo "  Installing requirements..."
    if pip install -r requirements.txt --quiet; then
        touch venv/installed
        print_success "Backend dependencies installed"
    else
        print_error "Failed to install backend dependencies"
        print_error "Try running manually: cd backend && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
else
    print_success "Backend dependencies already installed"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file from .env.example..."
    cp .env.example .env
    print_success ".env file created"
else
    print_success ".env file exists"
fi

# Initialize database
if [ ! -f "ollama_web.db" ]; then
    print_info "Initializing database..."
    python -c "from app.database import init_db; init_db()" 2>/dev/null || true
    print_success "Database initialized"
fi

cd ..

# 5. Setup frontend
print_info "Setting up frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_info "Installing frontend dependencies (this may take a few minutes)..."
    if npm install --silent; then
        print_success "Frontend dependencies installed"
    else
        print_error "Failed to install frontend dependencies"
        print_error "Try running manually: cd frontend && npm install"
        exit 1
    fi
else
    print_success "Frontend dependencies already installed"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file from .env.example..."
    cp .env.example .env
    print_success ".env file created"
else
    print_success ".env file exists"
fi

cd ..

# Create logs directory
mkdir -p logs

# 6. Start services
echo ""
print_header
print_info "Starting services..."
echo ""

# Start backend
print_info "Starting backend on http://localhost:8000..."
cd backend
source venv/bin/activate
python run.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3

# Check if backend started successfully
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    print_success "Backend started successfully"
    print_info "  API: http://localhost:8000"
    print_info "  Docs: http://localhost:8000/docs"
    print_info "  Health: http://localhost:8000/health"
else
    print_error "Backend failed to start. Check logs/backend.log for details."
    echo ""
    print_info "Last 20 lines of backend log:"
    tail -n 20 logs/backend.log 2>/dev/null || echo "  (no log file found)"
    cleanup
    exit 1
fi

# Start frontend
print_info "Starting frontend on http://localhost:5173..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 5

# Check if frontend started successfully
if curl -s http://localhost:5173 >/dev/null 2>&1; then
    print_success "Frontend started successfully"
    print_info "  URL: http://localhost:5173"
else
    print_error "Frontend failed to start. Check logs/frontend.log for details."
    echo ""
    print_info "Last 20 lines of frontend log:"
    tail -n 20 logs/frontend.log 2>/dev/null || echo "  (no log file found)"
    cleanup
    exit 1
fi

# 7. Display status
echo ""
print_header
print_success "All services started successfully!"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Service Status                                          ${GREEN}║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  Backend:   ${BLUE}http://localhost:8000${NC}                       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Frontend:  ${BLUE}http://localhost:5173${NC}                       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  API Docs:  ${BLUE}http://localhost:8000/docs${NC}                  ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
print_info "Logs are available in:"
print_info "  Backend:  logs/backend.log"
print_info "  Frontend: logs/frontend.log"
echo ""
print_info "Press Ctrl+C to stop all services"
echo ""

# Keep script running and tail logs
tail -f logs/backend.log logs/frontend.log 2>/dev/null &
TAIL_PID=$!

# Wait for user interrupt
wait
