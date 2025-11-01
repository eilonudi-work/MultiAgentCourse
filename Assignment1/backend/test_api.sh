#!/bin/bash
# Test script for Ollama Web GUI Backend API

echo "==================================="
echo "Testing Ollama Web GUI Backend API"
echo "==================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"
API_KEY="test-api-key-12345"

# Test 1: Health Check
echo "1. Testing Health Check..."
response=$(curl -s ${BASE_URL}/health)
if [[ $response == *"healthy"* ]]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi
echo ""

# Test 2: Root endpoint
echo "2. Testing Root Endpoint..."
response=$(curl -s ${BASE_URL}/)
if [[ $response == *"Ollama Web GUI API"* ]]; then
    echo -e "${GREEN}✓ Root endpoint passed${NC}"
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
fi
echo ""

# Test 3: Setup API Key
echo "3. Testing API Key Setup..."
response=$(curl -s -X POST "${BASE_URL}/api/auth/setup" \
  -H "Content-Type: application/json" \
  -d "{\"api_key\":\"${API_KEY}\",\"ollama_url\":\"http://localhost:11434\"}")
if [[ $response == *"success"*true* ]]; then
    echo -e "${GREEN}✓ API key setup passed${NC}"
else
    echo -e "${RED}✗ API key setup failed${NC}"
fi
echo ""

# Test 4: Verify API Key
echo "4. Testing API Key Verification..."
response=$(curl -s -X POST "${BASE_URL}/api/auth/verify" \
  -H "Content-Type: application/json" \
  -d "{\"api_key\":\"${API_KEY}\"}")
if [[ $response == *"valid"*true* ]]; then
    echo -e "${GREEN}✓ API key verification passed${NC}"
else
    echo -e "${RED}✗ API key verification failed${NC}"
fi
echo ""

# Test 5: List Models (with auth)
echo "5. Testing Model Listing (with authentication)..."
response=$(curl -s -X GET "${BASE_URL}/api/models/list" \
  -H "Authorization: Bearer ${API_KEY}")
if [[ $response == *"models"* ]] && [[ $response == *"llama3.2"* ]]; then
    echo -e "${GREEN}✓ Model listing passed${NC}"
else
    echo -e "${RED}✗ Model listing failed${NC}"
fi
echo ""

# Test 6: Save Configuration
echo "6. Testing Configuration Save..."
response=$(curl -s -X POST "${BASE_URL}/api/config/save" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"ollama_url":"http://localhost:11434","settings":{"theme":"dark","default_model":"llama3.2:1b"}}')
if [[ $response == *"success"*true* ]]; then
    echo -e "${GREEN}✓ Configuration save passed${NC}"
else
    echo -e "${RED}✗ Configuration save failed${NC}"
fi
echo ""

# Test 7: Get Configuration
echo "7. Testing Configuration Retrieval..."
response=$(curl -s -X GET "${BASE_URL}/api/config/get" \
  -H "Authorization: Bearer ${API_KEY}")
if [[ $response == *"ollama_url"* ]] && [[ $response == *"settings"* ]]; then
    echo -e "${GREEN}✓ Configuration retrieval passed${NC}"
else
    echo -e "${RED}✗ Configuration retrieval failed${NC}"
fi
echo ""

# Test 8: Test Invalid API Key
echo "8. Testing Invalid API Key (should fail)..."
response=$(curl -s -X GET "${BASE_URL}/api/models/list" \
  -H "Authorization: Bearer wrong-key")
if [[ $response == *"Invalid API key"* ]]; then
    echo -e "${GREEN}✓ Invalid API key rejection passed${NC}"
else
    echo -e "${RED}✗ Invalid API key rejection failed${NC}"
fi
echo ""

# Test 9: Test Missing Authentication
echo "9. Testing Missing Authentication (should fail)..."
response=$(curl -s -X GET "${BASE_URL}/api/config/get")
if [[ $response == *"API key is required"* ]]; then
    echo -e "${GREEN}✓ Missing auth rejection passed${NC}"
else
    echo -e "${RED}✗ Missing auth rejection failed${NC}"
fi
echo ""

echo "==================================="
echo "All tests completed!"
echo "==================================="
echo ""
echo "Swagger UI: ${BASE_URL}/docs"
echo "ReDoc: ${BASE_URL}/redoc"
