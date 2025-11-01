# Ollama Web GUI Backend - Quick Start Guide

## 5-Minute Setup

### Prerequisites

- Python 3.9+
- UV package manager installed
- Ollama running on localhost:11434

### Step 1: Install Dependencies

```bash
cd backend
uv sync
```

### Step 2: Start the Server

```bash
source .venv/bin/activate  # Linux/Mac
python run.py
```

Server will start on http://localhost:8000

### Step 3: Test the API

Run the test script:
```bash
./test_api.sh
```

Or open Swagger UI in your browser:
```
http://localhost:8000/docs
```

### Step 4: Setup Your API Key

```bash
curl -X POST 'http://localhost:8000/api/auth/setup' \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"your-secure-key","ollama_url":"http://localhost:11434"}'
```

### Step 5: List Available Models

```bash
curl -X GET 'http://localhost:8000/api/models/list' \
  -H 'Authorization: Bearer your-secure-key'
```

## Common Commands

### Start Server
```bash
cd backend
source .venv/bin/activate
python run.py
```

### Run Tests
```bash
./test_api.sh
```

### Check Logs
```bash
tail -f logs/app.log
```

### Reset Database
```bash
rm ollama_web.db
python run.py  # Will recreate database
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/auth/setup` - Setup API key
- `POST /api/auth/verify` - Verify API key
- `GET /api/models/list` - List models (requires auth)
- `POST /api/config/save` - Save config (requires auth)
- `GET /api/config/get` - Get config (requires auth)

## Documentation

- Full API docs: http://localhost:8000/docs
- README: backend/README.md
- Implementation summary: backend/IMPLEMENTATION_SUMMARY.md

## Troubleshooting

**Port 8000 in use?**
```bash
lsof -i :8000
kill -9 <PID>
```

**Ollama not connecting?**
```bash
curl http://localhost:11434/api/tags
# If fails, start Ollama: ollama serve
```

**Database issues?**
```bash
rm ollama_web.db
python run.py
```

## Next Steps

- Integrate with frontend (http://localhost:5173)
- Configure .env file for production
- Implement Phase 2 features (streaming chat)

---

**Need help?** Check the full README.md or IMPLEMENTATION_SUMMARY.md
