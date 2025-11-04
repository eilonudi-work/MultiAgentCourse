# Ollama Web GUI Backend - Deployment Guide

This guide covers deploying the Ollama Web GUI backend in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Deployment Options](#deployment-options)
  - [Docker Deployment](#docker-deployment)
  - [Systemd Service](#systemd-service)
  - [Manual Deployment](#manual-deployment)
- [Security Checklist](#security-checklist)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.9 or higher
- Ollama service running and accessible
- 512MB RAM minimum (2GB recommended)
- 1GB disk space minimum

## Environment Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Configure Required Settings

Edit `.env` and set the following **required** values:

```bash
# CRITICAL: Generate a secure secret key
SECRET_KEY=$(openssl rand -hex 32)

# Set your Ollama URL
OLLAMA_URL=http://localhost:11434

# Set allowed CORS origins (comma-separated)
CORS_ORIGINS=http://localhost:5173,https://your-frontend-domain.com
```

### 3. Production Security Settings

For production, update these security settings:

```bash
# Logging
LOG_LEVEL=WARNING
STRUCTURED_LOGGING=true

# Session
SESSION_TIMEOUT_MINUTES=30
API_KEY_EXPIRY_DAYS=90

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=50

# Security Features
CSRF_PROTECTION_ENABLED=true
SECURITY_HEADERS_ENABLED=true

# Backup
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
```

## Deployment Options

### Docker Deployment

Docker is the **recommended** deployment method.

#### Using Docker Compose (Easiest)

1. **Build and start the container:**

```bash
docker-compose up -d
```

2. **View logs:**

```bash
docker-compose logs -f backend
```

3. **Check health:**

```bash
curl http://localhost:8000/health
```

4. **Stop the service:**

```bash
docker-compose down
```

#### Using Docker (without Compose)

1. **Build the image:**

```bash
docker build -t ollama-web-backend .
```

2. **Run the container:**

```bash
docker run -d \
  --name ollama-web-backend \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/backups:/app/backups \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  ollama-web-backend
```

3. **Check logs:**

```bash
docker logs -f ollama-web-backend
```

### Systemd Service (Linux)

For deployment on Linux servers without Docker.

#### 1. Install Application

```bash
# Create application directory
sudo mkdir -p /opt/ollama-web-backend
sudo chown $USER:$USER /opt/ollama-web-backend

# Copy application files
cp -r . /opt/ollama-web-backend/
cd /opt/ollama-web-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

#### 2. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # Edit configuration
```

#### 3. Install Systemd Service

```bash
# Copy service file
sudo cp ollama-web-backend.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable ollama-web-backend

# Start service
sudo systemctl start ollama-web-backend

# Check status
sudo systemctl status ollama-web-backend
```

#### 4. View Logs

```bash
# Follow logs
sudo journalctl -u ollama-web-backend -f

# View recent logs
sudo journalctl -u ollama-web-backend -n 100
```

### Manual Deployment

For development or testing.

#### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

#### 2. Run Migrations

```bash
python scripts/run_migrations.py
```

#### 3. Start Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Security Checklist

Before deploying to production, verify:

- [ ] **SECRET_KEY** is changed from default and securely generated
- [ ] **CORS_ORIGINS** is restricted to your frontend domain(s)
- [ ] **Rate limiting** is enabled (`RATE_LIMIT_ENABLED=true`)
- [ ] **CSRF protection** is enabled (`CSRF_PROTECTION_ENABLED=true`)
- [ ] **Security headers** are enabled (`SECURITY_HEADERS_ENABLED=true`)
- [ ] **Log level** is set to `WARNING` or `ERROR` in production
- [ ] **Structured logging** is enabled for production monitoring
- [ ] **API key expiry** is configured if needed
- [ ] **Session timeout** is set appropriately (30 minutes recommended)
- [ ] **Backup** is enabled and retention policy is configured
- [ ] Application runs as **non-root user**
- [ ] File permissions are properly restricted
- [ ] HTTPS/TLS is configured (via reverse proxy)

## Monitoring

### Health Checks

Monitor application health:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "checks": {
    "database": {"status": "healthy"},
    "ollama": {"status": "healthy"},
    "filesystem": {"status": "healthy"}
  }
}
```

### Metrics Endpoint

View performance metrics:

```bash
curl http://localhost:8000/metrics
```

### Log Monitoring

Logs are stored in the `logs/` directory:

- `app.log` - Application logs
- `error.log` - Error logs only
- `access.log` - HTTP access logs

**Structured logging** (JSON format) can be enabled for easier parsing by log aggregation tools (Elasticsearch, Splunk, etc.).

### Recommended Monitoring Tools

- **Prometheus** - Metrics collection
- **Grafana** - Visualization and dashboards
- **ELK Stack** - Log aggregation and analysis
- **Uptime Robot** - Uptime monitoring
- **Sentry** - Error tracking

## Backup and Recovery

### Automatic Backups

Backups are created automatically on:
- Application startup
- Application shutdown (if enabled)

Manual backup:

```bash
python scripts/backup_db.py
```

### Backup Location

Backups are stored in the configured `BACKUP_DIRECTORY` (default: `./backups`).

### Restoring from Backup

```python
from app.utils.backup import DatabaseBackup
from pathlib import Path

backup_manager = DatabaseBackup()
backup_path = Path("backups/ollama_web_backup_20240101_120000.db.gz")
backup_manager.restore_backup(backup_path, force=True)
```

### Backup Retention

Old backups are automatically cleaned up based on `BACKUP_RETENTION_DAYS` (default: 30 days).

## Troubleshooting

### Application Won't Start

1. **Check logs:**
   ```bash
   # Docker
   docker logs ollama-web-backend

   # Systemd
   sudo journalctl -u ollama-web-backend -n 100

   # Manual
   tail -f logs/error.log
   ```

2. **Verify configuration:**
   ```bash
   cat .env
   ```

3. **Test database connection:**
   ```bash
   python -c "from app.database import engine; engine.connect()"
   ```

### Ollama Connection Issues

1. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check OLLAMA_URL in `.env`**

3. **For Docker:** Use `host.docker.internal` instead of `localhost`

### High Memory Usage

1. **Reduce number of workers:**
   ```bash
   uvicorn app.main:app --workers 2
   ```

2. **Check for memory leaks in logs**

3. **Monitor metrics endpoint:**
   ```bash
   curl http://localhost:8000/metrics/summary
   ```

### Rate Limit Errors

If legitimate users are being rate-limited:

1. **Increase rate limits in `.env`:**
   ```bash
   RATE_LIMIT_PER_MINUTE=200
   ```

2. **Or disable temporarily:**
   ```bash
   RATE_LIMIT_ENABLED=false
   ```

### Database Locked Errors

SQLite database locking issues:

1. **Enable WAL mode** (already enabled by default)

2. **Check for long-running connections**

3. **Consider PostgreSQL** for high-concurrency workloads

## Production Best Practices

### Reverse Proxy Configuration

Use Nginx or Caddy as a reverse proxy:

**Nginx example:**

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for SSE)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### SSL/TLS

Always use HTTPS in production:

```bash
# Using Let's Encrypt with Certbot
sudo certbot --nginx -d api.yourdomain.com
```

### Database Considerations

For production workloads:

- **SQLite** is suitable for small to medium deployments (< 1000 users)
- **PostgreSQL** is recommended for larger deployments
- Consider implementing **database replication** for high availability

### Scaling

For high-traffic deployments:

1. **Horizontal scaling:** Run multiple backend instances behind a load balancer
2. **Increase workers:** `uvicorn app.main:app --workers 8`
3. **Use Redis** for rate limiting and session storage (requires code changes)
4. **Database optimization:** Use PostgreSQL with connection pooling

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/ollama-web-gui/issues
- Documentation: https://github.com/yourusername/ollama-web-gui/wiki
