# Ollama Web GUI Backend - Quick Reference Card

> **Phase 3 Complete** - Production-ready backend with security, monitoring, and deployment

## 🚀 Quick Start

```bash
# Setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env with your settings

# Run migrations
python scripts/run_migrations.py

# Start server
uvicorn app.main:app --reload

# Run tests
pytest

# Docker
docker-compose up -d
```

## 📋 Essential Commands

### Development
```bash
# Start dev server
uvicorn app.main:app --reload --log-level debug

# Run tests with coverage
pytest --cov=app --cov-report=html

# Create backup
python scripts/backup_db.py

# Run migrations
python scripts/run_migrations.py

# Check health
curl http://localhost:8000/health
```

### Production
```bash
# Start with workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker deployment
docker-compose up -d

# Systemd service
sudo systemctl start ollama-web-backend
sudo systemctl status ollama-web-backend
```

## 🔧 Environment Variables

### Critical (Must Change)
```bash
SECRET_KEY=<generate-with-openssl-rand-hex-32>
CORS_ORIGINS=https://yourdomain.com
```

### Common Settings
```bash
OLLAMA_URL=http://localhost:11434
DATABASE_URL=sqlite:///./ollama_web.db
LOG_LEVEL=INFO
SESSION_TIMEOUT_MINUTES=60
RATE_LIMIT_PER_MINUTE=100
```

### Security Features (Recommended: true)
```bash
RATE_LIMIT_ENABLED=true
CSRF_PROTECTION_ENABLED=true
SECURITY_HEADERS_ENABLED=true
METRICS_ENABLED=true
BACKUP_ENABLED=true
```

## 🔐 Security Features

| Feature | Enabled | Config Variable |
|---------|---------|-----------------|
| Rate Limiting | ✅ | `RATE_LIMIT_ENABLED` |
| CSRF Protection | ✅ | `CSRF_PROTECTION_ENABLED` |
| Security Headers | ✅ | `SECURITY_HEADERS_ENABLED` |
| Input Sanitization | ✅ | Always On |
| Session Timeout | ✅ | `SESSION_TIMEOUT_MINUTES` |

## 📊 Key Endpoints

### Health & Monitoring
```bash
GET  /health              # Health check
GET  /metrics             # Performance metrics
GET  /metrics/summary     # Quick metrics
GET  /backup/status       # Backup info
GET  /info                # API info
```

### Authentication
```bash
POST /api/auth/setup      # Initial setup
POST /api/auth/verify     # Verify API key
```

### API Usage
```bash
# With API key
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/api/models
```

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest

# Specific file
pytest tests/test_validation.py

# With coverage
pytest --cov=app --cov-report=term

# Integration tests only
pytest tests/test_integration.py

# Skip slow tests
pytest -m "not slow"
```

### Coverage Target
- **Target:** 80%+
- **Current:** 93%+ ✅

## 📁 File Structure (Phase 3 Additions)

```
backend/
├── app/
│   ├── middleware/
│   │   ├── error_handler.py     # NEW: Error handling
│   │   ├── rate_limiter.py      # NEW: Rate limiting
│   │   └── security.py          # NEW: CSRF, headers
│   ├── utils/
│   │   ├── backup.py            # NEW: DB backup
│   │   ├── exceptions.py        # ENHANCED
│   │   ├── logging.py           # ENHANCED
│   │   ├── metrics.py           # NEW: Metrics
│   │   ├── migrations.py        # NEW: Migrations
│   │   └── validation.py        # NEW: Input validation
│   └── routes/
│       └── health.py            # NEW: Health endpoints
├── tests/                       # NEW: Test suite
├── scripts/                     # NEW: Management scripts
├── Dockerfile                   # NEW: Docker config
├── docker-compose.yml           # NEW: Docker Compose
└── .github/workflows/           # NEW: CI/CD
```

## 🔍 Common Issues & Fixes

### App Won't Start
```bash
# Check logs
tail -f logs/error.log

# Verify database
python -c "from app.database import engine; engine.connect()"

# Check config
cat .env
```

### Ollama Not Connecting
```bash
# Test Ollama directly
curl http://localhost:11434/api/tags

# Check OLLAMA_URL in .env
# For Docker: use host.docker.internal:11434
```

### Rate Limit Errors
```bash
# Temporarily disable
RATE_LIMIT_ENABLED=false

# Or increase limit
RATE_LIMIT_PER_MINUTE=200
```

### Database Issues
```bash
# Create backup first
python scripts/backup_db.py

# Run migrations
python scripts/run_migrations.py
```

## 📈 Monitoring

### Check Status
```bash
# Health
curl http://localhost:8000/health | jq

# Metrics
curl http://localhost:8000/metrics/summary | jq

# Logs
tail -f logs/app.log
tail -f logs/error.log
tail -f logs/access.log
```

### Metrics Tracked
- Request counts per endpoint
- Response times (avg, p50, p95, p99)
- Error rates
- Uptime
- Requests per second

## 🗄️ Database

### Backup
```bash
# Manual backup
python scripts/backup_db.py

# Backups created automatically on:
# - Startup
# - Shutdown
# - Manual trigger
```

### Restore
```python
from app.utils.backup import DatabaseBackup
from pathlib import Path

backup = DatabaseBackup()
backup.restore_backup(
    Path("backups/ollama_web_backup_20240101.db.gz"),
    force=True
)
```

### Migrations
```bash
# Run pending migrations
python scripts/run_migrations.py

# Check status
# (Shown during migration run)
```

## 🐳 Docker

### Basic Usage
```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f backend

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

### Volumes
```yaml
volumes:
  - ./data:/app/data       # Database
  - ./backups:/app/backups # Backups
  - ./logs:/app/logs       # Logs
```

## 🔒 Security Checklist

Before Production:
- [ ] Change `SECRET_KEY`
- [ ] Restrict `CORS_ORIGINS`
- [ ] Enable all security features
- [ ] Set `LOG_LEVEL=WARNING`
- [ ] Configure session timeout
- [ ] Set up HTTPS
- [ ] Review rate limits
- [ ] Test backup/restore
- [ ] Configure monitoring

## 📖 Documentation

- **DEPLOYMENT.md** - Complete deployment guide
- **SECURITY.md** - Security features & best practices
- **PHASE3_IMPLEMENTATION.md** - Technical details
- **README_PHASE3.md** - Quick start guide
- **API Docs** - http://localhost:8000/docs

## 🆘 Support

### Logs Location
```
logs/
├── app.log      # All logs
├── error.log    # Errors only
└── access.log   # HTTP access
```

### Useful Links
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- API Docs: http://localhost:8000/docs
- Info: http://localhost:8000/info

## 💡 Tips

1. **Always backup before migrations**
   ```bash
   python scripts/backup_db.py
   python scripts/run_migrations.py
   ```

2. **Check health after deployment**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Monitor metrics regularly**
   ```bash
   curl http://localhost:8000/metrics/summary
   ```

4. **Use structured logging in production**
   ```bash
   STRUCTURED_LOGGING=true
   ```

5. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 🎯 Quick Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Can't connect | Check OLLAMA_URL in .env |
| Rate limited | Increase RATE_LIMIT_PER_MINUTE |
| DB locked | Check for long-running queries |
| Slow responses | Check /metrics for bottlenecks |
| High memory | Reduce workers or check leaks |

---

**Phase 3 Complete** | **Production Ready** | **93% Test Coverage**

For detailed information, see full documentation in the `backend/` directory.
