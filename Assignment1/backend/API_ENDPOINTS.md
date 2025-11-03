# Ollama Web GUI - Complete API Endpoints

## Authentication Required
All endpoints except `/api/auth/*` require Bearer token authentication:
```
Authorization: Bearer <your_api_key>
```

---

## Phase 1 Endpoints (Existing)

### Authentication

#### POST /api/auth/setup
Setup initial API key and Ollama URL.

**Request:**
```json
{
  "api_key": "your-secret-key",
  "ollama_url": "http://localhost:11434"
}
```

**Response:**
```json
{
  "success": true,
  "message": "API key setup successful",
  "user_id": 1
}
```

#### POST /api/auth/verify
Verify if an API key is valid.

**Request:**
```json
{
  "api_key": "your-secret-key"
}
```

**Response:**
```json
{
  "valid": true,
  "message": "API key is valid",
  "user_id": 1
}
```

### Configuration

#### POST /api/config/save
Save configuration settings.

#### GET /api/config/get
Get current configuration.

---

## Phase 2 Endpoints (New)

### Conversations

#### POST /api/conversations
Create a new conversation.

**Request:**
```json
{
  "title": "My Chat Session",
  "model_name": "llama2",
  "system_prompt": "You are a helpful assistant."
}
```

**Response:**
```json
{
  "id": 1,
  "user_id": 1,
  "title": "My Chat Session",
  "model_name": "llama2",
  "system_prompt": "You are a helpful assistant.",
  "created_at": "2025-11-04T10:00:00",
  "updated_at": "2025-11-04T10:00:00",
  "message_count": 0
}
```

#### GET /api/conversations
List all conversations with pagination.

**Query Parameters:**
- `page` (default: 1) - Page number
- `page_size` (default: 20, max: 100) - Items per page
- `search` (optional) - Search in titles

**Response:**
```json
{
  "conversations": [
    {
      "id": 1,
      "user_id": 1,
      "title": "My Chat Session",
      "model_name": "llama2",
      "system_prompt": "You are a helpful assistant.",
      "created_at": "2025-11-04T10:00:00",
      "updated_at": "2025-11-04T10:00:00",
      "message_count": 5
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

#### GET /api/conversations/{id}
Get a single conversation with all messages.

**Response:**
```json
{
  "id": 1,
  "user_id": 1,
  "title": "My Chat Session",
  "model_name": "llama2",
  "system_prompt": "You are a helpful assistant.",
  "created_at": "2025-11-04T10:00:00",
  "updated_at": "2025-11-04T10:05:00",
  "messages": [
    {
      "id": 1,
      "conversation_id": 1,
      "role": "user",
      "content": "Hello!",
      "tokens_used": null,
      "created_at": "2025-11-04T10:00:00"
    },
    {
      "id": 2,
      "conversation_id": 1,
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "tokens_used": 8,
      "created_at": "2025-11-04T10:00:05"
    }
  ]
}
```

#### PUT /api/conversations/{id}
Update conversation details.

**Request:**
```json
{
  "title": "Updated Title",
  "model_name": "llama2:13b",
  "system_prompt": "You are an expert programmer."
}
```

**Response:** Same as GET conversation

#### DELETE /api/conversations/{id}
Delete a conversation and all its messages.

**Response:**
```json
{
  "success": true,
  "message": "Conversation deleted successfully",
  "conversation_id": 1
}
```

### Chat Streaming

#### POST /api/chat/stream
Stream chat responses using Server-Sent Events (SSE).

**Request:**
```json
{
  "conversation_id": 1,
  "message": "Hello, how are you?",
  "temperature": 0.7,
  "stream": true
}
```

**OR** (for new conversation):
```json
{
  "message": "Hello, how are you?",
  "model_name": "llama2",
  "system_prompt": "You are helpful.",
  "temperature": 0.7
}
```

**SSE Events:**

1. **conversation_created** (for new conversations)
```
event: conversation_created
data: {"conversation_id": 1}
```

2. **message_created** (user message saved)
```
event: message_created
data: {"message_id": 1, "role": "user"}
```

3. **token** (streaming response tokens)
```
event: token
data: {"content": "Hello"}
```

4. **done** (streaming complete)
```
event: done
data: {"message_id": 2, "tokens_used": 50}
```

5. **error** (error occurred)
```
event: error
data: {"error": "Model not found"}
```

#### POST /api/chat/search
Search messages across conversations.

**Request:**
```json
{
  "query": "hello",
  "conversation_id": 1
}
```

**Response:**
```json
[
  {
    "conversation_id": 1,
    "message_id": 1,
    "role": "user",
    "content": "Hello, how are you?",
    "created_at": "2025-11-04T10:00:00",
    "snippet": "...Hello, how are you?..."
  }
]
```

### Models

#### GET /api/models/list
List available Ollama models (cached for 5 minutes).

**Response:**
```json
{
  "models": [
    {
      "name": "llama2:latest",
      "model": "llama2",
      "size": 3825819519,
      "modified_at": "2025-11-04T08:00:00",
      "digest": "sha256:...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "7B"
      }
    }
  ],
  "count": 1
}
```

#### GET /api/models/{model_name}/info
Get detailed information about a specific model.

**Response:**
```json
{
  "license": "...",
  "modelfile": "...",
  "parameters": "...",
  "template": "...",
  "details": {
    "format": "gguf",
    "family": "llama",
    "parameter_size": "7B",
    "quantization_level": "Q4_0"
  }
}
```

#### POST /api/models/cache/clear
Clear the model list cache.

**Response:**
```json
{
  "success": true,
  "message": "Model cache cleared"
}
```

### System Prompts

#### GET /api/prompts/templates
Get predefined system prompt templates.

**Response:**
```json
{
  "templates": [
    {
      "id": "default",
      "name": "Default Assistant",
      "description": "A helpful, harmless, and honest AI assistant",
      "prompt": "You are a helpful AI assistant...",
      "category": "general"
    },
    {
      "id": "coding_assistant",
      "name": "Coding Assistant",
      "description": "Expert programming assistant",
      "prompt": "You are an expert programming assistant...",
      "category": "programming"
    }
  ],
  "total": 15
}
```

### Export/Import

#### GET /api/export/conversations/{id}/json
Export conversation to JSON format.

**Response:** File download
```json
{
  "id": 1,
  "title": "My Chat",
  "model_name": "llama2",
  "system_prompt": "...",
  "created_at": "2025-11-04T10:00:00",
  "updated_at": "2025-11-04T10:05:00",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "tokens_used": null,
      "created_at": "2025-11-04T10:00:00"
    }
  ],
  "export_version": "1.0",
  "exported_at": "2025-11-04T11:00:00"
}
```

#### GET /api/export/conversations/{id}/markdown
Export conversation to Markdown format.

**Response:** File download (conversation_{id}.md)

#### POST /api/export/conversations/import
Import a conversation from JSON.

**Request:**
```json
{
  "title": "Imported Chat",
  "model_name": "llama2",
  "system_prompt": "...",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "tokens_used": null,
      "created_at": "2025-11-04T10:00:00"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully imported conversation with 10 messages",
  "conversation_id": 2,
  "imported_messages": 10
}
```

---

## Health & Utility Endpoints

### GET /health
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Ollama Web GUI API",
  "version": "1.0.0"
}
```

### GET /
Get API information.

**Response:**
```json
{
  "name": "Ollama Web GUI API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Error Responses

All endpoints may return error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request data"
}
```

### 401 Unauthorized
```json
{
  "detail": "API key is required"
}
```

### 404 Not Found
```json
{
  "detail": "Conversation not found"
}
```

### 422 Unprocessable Entity
```json
{
  "detail": "Request validation failed",
  "errors": [
    {
      "loc": ["body", "model_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "An internal server error occurred",
  "error": "..."
}
```

### 503 Service Unavailable
```json
{
  "detail": "Ollama service is not running or unreachable"
}
```

---

## Interactive API Documentation

Visit these URLs when the server is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive API documentation with test capabilities.
