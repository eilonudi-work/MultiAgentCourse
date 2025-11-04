# 🔗 Frontend-Backend Integration Guide

## Overview

This guide explains how the Ollama Web GUI frontend and backend are integrated, how to run them together, and how to verify the integration is working correctly.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Port 5173)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              React Frontend (Vite)                     │  │
│  │  - Components (React)                                  │  │
│  │  - State Management (Zustand)                          │  │
│  │  - API Client (Axios)                                  │  │
│  │  - SSE Client (EventSource)                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↕ HTTP/SSE
┌─────────────────────────────────────────────────────────────┐
│              Backend API (Port 8000)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              FastAPI Server                            │  │
│  │  - REST API Endpoints                                  │  │
│  │  - SSE Streaming                                       │  │
│  │  - Authentication Middleware                           │  │
│  │  - Rate Limiting                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↕                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              SQLite Database                           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↕ HTTP
┌─────────────────────────────────────────────────────────────┐
│              Ollama API (Port 11434)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. API Base URL Configuration

**Frontend Configuration:**
- File: `frontend/.env`
- Variable: `VITE_API_BASE_URL=http://localhost:8000`
- Used by: `frontend/src/services/api.js`

**Backend Configuration:**
- File: `backend/.env`
- Variable: `CORS_ORIGINS=http://localhost:5173,http://localhost:3000`
- Ensures frontend can make requests to backend

### 2. Authentication Flow

**Setup Flow:**
```
1. User visits http://localhost:5173
2. Frontend redirects to /setup if not authenticated
3. User enters API key and Ollama URL
4. Frontend → POST /api/auth/setup → Backend
5. Backend validates, hashes API key, saves to SQLite
6. Backend tests Ollama connection
7. Backend returns success response
8. Frontend stores API key in localStorage
9. Frontend redirects to /chat
```

**Request Authentication:**
```
1. Frontend reads API key from localStorage
2. Axios interceptor adds header: Authorization: Bearer {api_key}
3. Backend middleware validates API key
4. Request proceeds or returns 401 Unauthorized
```

### 3. Chat Streaming Integration

**Streaming Flow:**
```
1. User types message in chat input
2. Frontend → POST /api/chat/stream (with query params) → Backend
3. Backend establishes SSE connection
4. Backend streams request to Ollama
5. Ollama streams tokens back to Backend
6. Backend proxies tokens via SSE to Frontend
7. Frontend EventSource receives events:
   - type: "token" → Append to message
   - type: "done" → Complete message, save to DB
   - type: "error" → Show error to user
8. Frontend updates UI in real-time
```

**Implementation:**
- Frontend: `frontend/src/services/chatService.js`
- Backend: `backend/app/routes/chat.py`

### 4. Conversation Management

**CRUD Operations:**
- Create: `POST /api/conversations`
- Read List: `GET /api/conversations`
- Read Single: `GET /api/conversations/{id}`
- Update: `PUT /api/conversations/{id}`
- Delete: `DELETE /api/conversations/{id}`

**Frontend Implementation:**
- Service: `frontend/src/services/conversationsService.js`
- Store: `frontend/src/store/conversationStore.js`
- Component: `frontend/src/components/ConversationSidebar.jsx`

**Backend Implementation:**
- Routes: `backend/app/routes/conversations.py`
- Models: `backend/app/models/conversation.py`
- Schemas: `backend/app/schemas/conversation.py`

### 5. Model Selection

**Flow:**
```
1. User clicks "Select Model" button
2. Frontend → GET /api/models/list → Backend
3. Backend → GET /api/tags → Ollama
4. Backend caches response (5 min TTL)
5. Backend returns models to Frontend
6. Frontend displays in modal
7. User selects model
8. Frontend updates state and conversation
```

**Implementation:**
- Frontend: `frontend/src/components/ModelSelectorModal.jsx`
- Backend: `backend/app/routes/models.py`

### 6. Export/Import

**Export Flow:**
```
1. User clicks Export button
2. Frontend → GET /api/conversations/{id}/export/json → Backend
3. Backend retrieves conversation with messages
4. Backend formats as JSON/Markdown
5. Frontend receives file
6. Frontend triggers browser download
```

**Import Flow:**
```
1. User uploads file
2. Frontend validates file format
3. Frontend → POST /api/conversations/import → Backend
4. Backend validates, sanitizes data
5. Backend creates conversation and messages
6. Frontend reloads conversation list
```

**Implementation:**
- Frontend: `frontend/src/components/ExportImportModal.jsx`
- Backend: `backend/app/routes/export.py`

### 7. System Prompts

**Flow:**
```
1. User opens Settings modal
2. Frontend → GET /api/prompts/templates → Backend
3. Backend returns 15 predefined templates
4. User selects or custom enters prompt
5. Frontend saves to configStore
6. Prompt included in chat requests
```

**Implementation:**
- Frontend: `frontend/src/components/SettingsModal.jsx`
- Backend: `backend/app/routes/prompts.py`

---

## Environment Configuration

### Backend Environment Variables

**File:** `backend/.env`

```env
# Database
DATABASE_URL=sqlite:///./ollama_web.db

# Ollama
OLLAMA_URL=http://localhost:11434

# Security
SECRET_KEY=your-secret-key-change-this-in-production

# CORS (IMPORTANT: Must include frontend URL)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=false

# Sessions
SESSION_TIMEOUT_MINUTES=60
API_KEY_EXPIRY_DAYS=0

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100

# Security Features
CSRF_PROTECTION_ENABLED=true
SECURITY_HEADERS_ENABLED=true
```

### Frontend Environment Variables

**File:** `frontend/.env`

```env
# Backend API URL (IMPORTANT: Must match backend port)
VITE_API_BASE_URL=http://localhost:8000

# Default Ollama URL (shown in setup form)
VITE_OLLAMA_DEFAULT_URL=http://localhost:11434
```

---

## Running the Integrated Application

### Prerequisites

1. **Ollama must be running:**
   ```bash
   ollama serve
   # Or if installed via app, it should be running in background
   ```

2. **Backend dependencies installed:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend dependencies installed:**
   ```bash
   cd frontend
   npm install
   ```

### Option 1: Run Services Separately

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python run.py
# Backend will run on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# Frontend will run on http://localhost:5173
```

### Option 2: Use Startup Script (Recommended)

We'll create a startup script next that runs both services.

---

## Verification Checklist

### 1. Backend Health Check

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Ollama Web GUI API",
  "version": "1.0.0"
}
```

### 2. Ollama Connection Check

```bash
curl http://localhost:8000/api/models/list \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Expected Response:**
```json
{
  "models": [...],
  "count": 5
}
```

### 3. Frontend Access

1. Open browser to http://localhost:5173
2. Should see setup screen if first time
3. After setup, should see chat interface

### 4. Complete Integration Test

**Step 1: Initial Setup**
1. Navigate to http://localhost:5173
2. Enter API key: `test-api-key-123`
3. Ollama URL: `http://localhost:11434`
4. Click "Test Connection"
5. Should see "Connection successful"
6. Click "Save Configuration"

**Step 2: Chat Interface**
1. Should redirect to /chat
2. Should see empty chat area
3. Should see "New Chat" button in sidebar

**Step 3: Send Message**
1. Click "New Chat"
2. Type message: "Hello, tell me about yourself"
3. Press Enter or click Send
4. Should see:
   - User message appears immediately
   - Assistant message starts streaming
   - Tokens appear one by one
   - Message completes

**Step 4: Conversation Management**
1. Check sidebar - should see new conversation
2. Click "New Chat" again
3. Send another message
4. Should have 2 conversations in sidebar
5. Click first conversation - should load previous messages

**Step 5: Model Selection**
1. Click "Select Model" button
2. Should see list of available models
3. Select a different model
4. Send a message with new model

**Step 6: Export/Import**
1. Click conversation menu (three dots)
2. Click "Export as JSON"
3. File should download
4. Click "Import Conversation"
5. Upload the downloaded file
6. Should create duplicate conversation

---

## API Endpoints Summary

### Authentication
- `POST /api/auth/setup` - Initial API key setup
- `POST /api/auth/verify` - Verify API key

### Configuration
- `POST /api/config/save` - Save configuration
- `GET /api/config/get` - Get configuration

### Models
- `GET /api/models/list` - List available models (cached)
- `GET /api/models/{name}/info` - Get model details
- `POST /api/models/cache/clear` - Clear model cache

### Conversations
- `POST /api/conversations` - Create conversation
- `GET /api/conversations` - List conversations (with pagination)
- `GET /api/conversations/{id}` - Get single conversation
- `PUT /api/conversations/{id}` - Update conversation
- `DELETE /api/conversations/{id}` - Delete conversation

### Chat
- `POST /api/chat/stream` - Stream chat response (SSE)
- `POST /api/chat/search` - Search messages

### Prompts
- `GET /api/prompts/templates` - Get prompt templates

### Export/Import
- `GET /api/conversations/{id}/export/json` - Export as JSON
- `GET /api/conversations/{id}/export/markdown` - Export as Markdown
- `POST /api/conversations/import` - Import conversation

### Health
- `GET /health` - Health check
- `GET /api/health` - Detailed health check

---

## Common Integration Issues

### Issue 1: CORS Errors

**Symptom:**
```
Access to XMLHttpRequest at 'http://localhost:8000/api/...' from origin
'http://localhost:5173' has been blocked by CORS policy
```

**Solution:**
Check `backend/.env`:
```env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Issue 2: 401 Unauthorized

**Symptom:**
All API requests return 401 status

**Solutions:**
1. Verify API key is stored in localStorage: Open DevTools → Application → LocalStorage → `ollama_api_key`
2. Check backend logs for authentication errors
3. Try re-running setup flow

### Issue 3: Ollama Connection Failed

**Symptom:**
"Ollama service is not available" error

**Solutions:**
1. Verify Ollama is running: `ollama list`
2. Check Ollama URL in settings
3. Try: `curl http://localhost:11434/api/tags`

### Issue 4: Streaming Not Working

**Symptom:**
Messages don't stream, entire response appears at once

**Solutions:**
1. Verify browser supports EventSource
2. Check Network tab for SSE connection
3. Verify backend is sending SSE headers
4. Check `chatService.js` EventSource implementation

### Issue 5: Messages Not Persisting

**Symptom:**
Messages disappear after page reload

**Solutions:**
1. Check SQLite database exists: `backend/ollama_web.db`
2. Verify database writes in backend logs
3. Check conversation_id is being passed correctly
4. Verify SQLite WAL mode is enabled

---

## Security Considerations

### 1. API Key Storage
- **Frontend:** Stored in localStorage (not secure for sensitive data)
- **Backend:** Hashed with bcrypt before storing in database
- **Production:** Consider using HTTP-only cookies or more secure storage

### 2. HTTPS in Production
- Always use HTTPS in production
- Update CORS_ORIGINS to use https://
- Use secure WebSocket (wss://) for streaming

### 3. Rate Limiting
- Enabled by default (100 requests/minute)
- Configurable in `backend/.env`
- Applied per IP and per API key

### 4. Input Sanitization
- All inputs validated on backend
- XSS prevention in export/import
- SQL injection prevented by ORM

---

## Performance Optimization

### Frontend Optimizations
1. **Code Splitting:** Implemented via Vite lazy loading
2. **Caching:** API responses cached where appropriate
3. **Virtual Scrolling:** For large message lists
4. **Debouncing:** For search and streaming updates

### Backend Optimizations
1. **Database Indexes:** All foreign keys indexed
2. **Query Optimization:** Eager loading for relationships
3. **Caching:** Model list cached for 5 minutes
4. **Connection Pooling:** SQLite connection pooling enabled

---

## Monitoring Integration

### Frontend Metrics
- Network tab in DevTools
- React DevTools for component rendering
- Console logs (development mode only)

### Backend Metrics
- Endpoint: `GET /api/health`
- Logs in `backend/logs/`
- Metrics stored in memory (configurable)

### Integration Monitoring
```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:5173

# Check backend logs
tail -f backend/logs/app.log

# Monitor SSE connection
# In browser DevTools → Network → Filter: EventStream
```

---

## Development Workflow

### 1. Making Changes

**Frontend Changes:**
1. Edit files in `frontend/src/`
2. Vite hot-reload updates browser automatically
3. No restart needed

**Backend Changes:**
1. Edit files in `backend/app/`
2. FastAPI auto-reload detects changes
3. Server restarts automatically

### 2. Adding New Endpoint

**Backend Steps:**
1. Create route in `backend/app/routes/`
2. Create schema in `backend/app/schemas/`
3. Add to router in `backend/app/main.py`

**Frontend Steps:**
1. Add service function in `frontend/src/services/`
2. Use in component or store
3. Handle loading/error states

### 3. Testing Integration

**Unit Tests:**
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

**Integration Tests:**
```bash
# Run integration test script (created next)
./scripts/test-integration.sh
```

---

## Deployment Considerations

### Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| Backend URL | http://localhost:8000 | https://api.yourdomain.com |
| Frontend URL | http://localhost:5173 | https://yourdomain.com |
| CORS | localhost:5173 | https://yourdomain.com |
| Logging | INFO, console | WARNING, JSON files |
| Rate Limiting | 100/min | 50/min |
| API Keys | Test keys | Secure rotation |

### Docker Deployment

Both frontend and backend include Dockerfiles and docker-compose.yml for easy deployment.

```bash
# Build and run both services
docker-compose up -d

# Frontend: http://localhost:80
# Backend: http://localhost:8000
```

---

## Troubleshooting Commands

```bash
# Check if ports are in use
lsof -i :5173  # Frontend
lsof -i :8000  # Backend
lsof -i :11434 # Ollama

# Kill processes on ports
kill -9 $(lsof -ti:5173)
kill -9 $(lsof -ti:8000)

# Check Ollama status
ollama list
ollama ps

# View backend logs
tail -f backend/logs/app.log

# Check database
sqlite3 backend/ollama_web.db ".tables"

# Test backend endpoints
curl -X POST http://localhost:8000/api/auth/setup \
  -H "Content-Type: application/json" \
  -d '{"api_key":"test123","ollama_url":"http://localhost:11434"}'
```

---

## Next Steps

1. ✅ Backend fully implemented (Phases 1-3)
2. ✅ Frontend fully implemented (Phases 1-3)
3. ✅ Integration configured
4. ⏭️ Run integration tests (see below)
5. ⏭️ Deploy to production (see DEPLOYMENT.md)

---

## Support

For issues with:
- **Backend:** Check `backend/PHASE3_IMPLEMENTATION.md`
- **Frontend:** Check `frontend/PHASE1_SUMMARY.md`
- **Security:** Check `backend/SECURITY.md`
- **Deployment:** Check `backend/DEPLOYMENT.md`

---

**Document Version:** 1.0
**Last Updated:** 2025-01-04
**Status:** Integration Complete ✅
