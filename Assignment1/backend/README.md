# Ollama Web GUI - Backend API

FastAPI backend for the Ollama Web GUI project. Provides secure API endpoints for authentication, configuration management, and Ollama model integration.

## Features

- API Key authentication with bcrypt hashing
- SQLite database with SQLAlchemy ORM
- Ollama API integration with connection pooling
- Configuration persistence
- Comprehensive error handling and logging
- CORS support for frontend integration
- OpenAPI documentation (Swagger UI)

## Tech Stack

- **Framework:** FastAPI 0.120+
- **Database:** SQLite 3.40+ with SQLAlchemy 2.0
- **Authentication:** bcrypt
- **HTTP Client:** httpx (async)
- **Package Manager:** UV

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database setup
│   ├── models/              # SQLAlchemy models
│   │   ├── user.py
│   │   ├── conversation.py
│   │   ├── message.py
│   │   └── setting.py
│   ├── routes/              # API endpoints
│   │   ├── auth.py          # Authentication routes
│   │   ├── config.py        # Configuration routes
│   │   └── models.py        # Model management routes
│   ├── schemas/             # Pydantic schemas
│   │   ├── auth.py
│   │   ├── config.py
│   │   └── models.py
│   ├── middleware/          # Middleware components
│   │   └── auth.py          # Authentication middleware
│   ├── services/            # Business logic
│   │   └── ollama_client.py # Ollama API client
│   └── utils/               # Utility functions
│       ├── auth.py          # Auth utilities
│       ├── logging.py       # Logging setup
│       └── exceptions.py    # Custom exceptions
├── .env                     # Environment variables
├── .gitignore
├── pyproject.toml           # UV project config
├── run.py                   # Run script
└── README.md
```

## Prerequisites

- Python 3.9+
- UV package manager
- Ollama running on localhost:11434
- SQLite 3.40+

## Installation

1. **Install dependencies with UV:**

```bash
cd backend
uv sync
```

2. **Configure environment variables:**

Edit `.env` file with your settings:

```env
DATABASE_URL=sqlite:///./ollama_web.db
OLLAMA_URL=http://localhost:11434
SECRET_KEY=your-secret-key-change-this-in-production
CORS_ORIGINS=http://localhost:5173
LOG_LEVEL=INFO
```

3. **Ensure Ollama is running:**

```bash
# Check if Ollama is accessible
curl http://localhost:11434/api/tags
```

## Running the Server

### Development Mode (with auto-reload)

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Run the server
python run.py
```

Or use uvicorn directly:

```bash
uvicorn app.main:app --reload --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check

- **GET** `/health` - Health check endpoint

### Authentication

- **POST** `/api/auth/setup` - Setup API key (first-time)
- **POST** `/api/auth/verify` - Verify API key

### Configuration

- **POST** `/api/config/save` - Save user configuration (requires auth)
- **GET** `/api/config/get` - Get user configuration (requires auth)

### Models

- **GET** `/api/models/list` - List available Ollama models (requires auth)

## API Documentation

Once the server is running, access the interactive API documentation at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Testing the API

### 1. Setup API Key

```bash
curl -X POST "http://localhost:8000/api/auth/setup" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "my-secure-api-key-123",
    "ollama_url": "http://localhost:11434"
  }'
```

### 2. Verify API Key

```bash
curl -X POST "http://localhost:8000/api/auth/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "my-secure-api-key-123"
  }'
```

### 3. List Models (requires authentication)

```bash
curl -X GET "http://localhost:8000/api/models/list" \
  -H "Authorization: Bearer my-secure-api-key-123"
```

### 4. Save Configuration

```bash
curl -X POST "http://localhost:8000/api/config/save" \
  -H "Authorization: Bearer my-secure-api-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "ollama_url": "http://localhost:11434",
    "settings": {
      "theme": "dark",
      "default_model": "llama3.2:1b"
    }
  }'
```

### 5. Get Configuration

```bash
curl -X GET "http://localhost:8000/api/config/get" \
  -H "Authorization: Bearer my-secure-api-key-123"
```

## Database

The application uses SQLite with the following tables:

- **users** - User accounts with hashed API keys
- **conversations** - Chat conversations
- **messages** - Individual chat messages
- **settings** - User-specific settings

Database file location: `./ollama_web.db`

### SQLite WAL Mode

The database is configured to use Write-Ahead Logging (WAL) mode for better concurrency:

- Improved read/write concurrency
- Better performance for multiple connections
- Automatic checkpointing

## Logging

Logs are written to:

- **Console:** Standard output with INFO level
- **File:** `logs/app.log` - All logs
- **File:** `logs/error.log` - Error logs only

Log level can be configured via `LOG_LEVEL` environment variable.

## Security

- API keys are hashed using bcrypt before storage
- API keys are never logged or displayed in plain text
- CORS is configured to only allow specified origins
- Input validation using Pydantic schemas
- SQL injection protection via SQLAlchemy ORM

## Error Handling

The API returns standardized error responses:

```json
{
  "detail": "Error message",
  "error": "Technical details (in development mode)"
}
```

HTTP Status Codes:

- **200** - Success
- **401** - Unauthorized (invalid API key)
- **422** - Validation error
- **500** - Internal server error
- **503** - Service unavailable (Ollama not running)

## Frontend Integration

The backend is configured for the frontend running on `http://localhost:5173`.

CORS is enabled for:

- All HTTP methods
- All headers
- Credentials support

## Troubleshooting

### Database Issues

If you encounter database issues:

```bash
# Remove the database file and restart
rm ollama_web.db
python run.py
```

### Ollama Connection Issues

Ensure Ollama is running:

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Port Already in Use

If port 8000 is in use:

```bash
# Change port in run.py or use:
uvicorn app.main:app --reload --port 8001
```

## Development

### Adding New Routes

1. Create route file in `app/routes/`
2. Define Pydantic schemas in `app/schemas/`
3. Include router in `app/main.py`

### Database Migrations

For schema changes, use Alembic:

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## Phase 1 Completion

Phase 1 backend implementation includes:

- ✅ Project setup with UV
- ✅ SQLite database with schema
- ✅ API key authentication
- ✅ Ollama client integration
- ✅ Configuration persistence
- ✅ Error handling and logging
- ✅ All Phase 1 endpoints

## Next Steps (Phase 2)

- Conversation CRUD endpoints
- Streaming chat endpoint with SSE
- Message persistence
- System prompt handling
- Export/import functionality

## License

MIT License

## Support

For issues and questions, please refer to the project documentation or open an issue.
