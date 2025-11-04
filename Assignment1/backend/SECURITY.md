# Security Guide - Ollama Web GUI Backend

This document outlines the security features, best practices, and configuration options for the Ollama Web GUI backend.

## Security Features

### 1. Input Validation and Sanitization

All user inputs are validated and sanitized to prevent injection attacks.

**Protected Against:**
- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- Command Injection
- Buffer Overflow (via length limits)

**Implementation:**
- `app/utils/validation.py` - Comprehensive validation functions
- Pattern-based detection of malicious inputs
- Maximum length enforcement
- Type validation

**Example:**
```python
from app.utils.validation import sanitize_string

# Safely sanitize user input
safe_input = sanitize_string(user_input, "field_name", max_length=200)
```

### 2. Rate Limiting

Protection against brute force attacks and API abuse.

**Rate Limits (default):**
- Authentication endpoints: 5 requests/minute
- Chat endpoints: 20 requests/minute
- General API: 100 requests/minute
- Health endpoints: 300 requests/minute

**Features:**
- Per-IP rate limiting
- Per-API-key rate limiting
- Automatic token refill
- Configurable limits per endpoint

**Configuration:**
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
```

### 3. CSRF Protection

Protection against Cross-Site Request Forgery attacks.

**Features:**
- Token-based CSRF protection
- Automatic token generation
- State-changing endpoint protection (POST, PUT, PATCH, DELETE)
- Exempt paths for authentication

**Usage:**
Clients must include CSRF token in `X-CSRF-Token` header for state-changing requests.

**Configuration:**
```bash
CSRF_PROTECTION_ENABLED=true
```

### 4. Security Headers

Comprehensive security headers to protect against common web vulnerabilities.

**Headers Implemented:**
- **Content-Security-Policy** - Prevents XSS and data injection
- **X-Frame-Options: DENY** - Prevents clickjacking
- **X-Content-Type-Options: nosniff** - Prevents MIME sniffing
- **X-XSS-Protection: 1; mode=block** - Enables XSS filter
- **Strict-Transport-Security** - Forces HTTPS (when using HTTPS)
- **Referrer-Policy: strict-origin-when-cross-origin** - Controls referrer information
- **Permissions-Policy** - Controls browser features

**Configuration:**
```bash
SECURITY_HEADERS_ENABLED=true
```

### 5. Session Management

Secure session handling with configurable timeouts.

**Features:**
- Configurable session timeout
- Activity-based session extension
- Session revocation
- Session validation on each request

**Configuration:**
```bash
SESSION_TIMEOUT_MINUTES=60  # 60 minutes default
```

**Session Lifecycle:**
1. User authenticates with API key
2. Session created with expiration time
3. Each request updates `last_activity`
4. Session extended if within timeout
5. Session invalidated after timeout or manual revocation

### 6. API Key Management

Secure API key storage and management.

**Features:**
- API keys hashed using bcrypt
- Optional key expiration
- Key rotation support with grace period
- Admin user designation

**Security Measures:**
- Keys never stored in plain text
- Constant-time comparison for validation
- Secure random key generation
- Key rotation with overlap period

**Configuration:**
```bash
API_KEY_EXPIRY_DAYS=90  # 0 = never expires
```

### 7. Error Handling

Secure error responses that don't leak sensitive information.

**Features:**
- User-friendly error messages
- Detailed logging for debugging (not exposed to users)
- Structured error codes
- No stack traces in production

**Error Response Format:**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid input provided",
  "details": {
    "field": "email",
    "expected": "valid email format"
  }
}
```

### 8. Authentication

API key-based authentication with secure validation.

**Features:**
- Bearer token authentication
- Hashed key storage
- Automatic session management
- Admin role support

**Usage:**
```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/api/models
```

---

## Security Best Practices

### Production Deployment

#### 1. Generate Secure Secret Key

```bash
# Generate a secure secret key
openssl rand -hex 32

# Set in .env file
SECRET_KEY=your_generated_key_here
```

#### 2. Use HTTPS

Always use HTTPS in production. Configure a reverse proxy (Nginx/Caddy) with SSL/TLS.

**Nginx Example:**
```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 3. Restrict CORS Origins

Only allow trusted frontend domains:

```bash
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

#### 4. Enable All Security Features

```bash
RATE_LIMIT_ENABLED=true
CSRF_PROTECTION_ENABLED=true
SECURITY_HEADERS_ENABLED=true
```

#### 5. Configure Appropriate Timeouts

```bash
SESSION_TIMEOUT_MINUTES=30  # Shorter for sensitive applications
API_KEY_EXPIRY_DAYS=90      # Require periodic key rotation
```

#### 6. Use Structured Logging

```bash
LOG_LEVEL=WARNING
STRUCTURED_LOGGING=true
```

#### 7. Run as Non-Root User

Docker and systemd configurations already implement this.

#### 8. Keep Dependencies Updated

Regularly update dependencies to patch security vulnerabilities:

```bash
pip install --upgrade -r requirements.txt
```

### Input Validation

Always validate and sanitize inputs:

```python
from app.utils.validation import sanitize_string, validate_temperature

# Validate all inputs
title = sanitize_conversation_title(request.title)
temp = validate_temperature(request.temperature)
```

### Rate Limiting

Adjust rate limits based on your needs:

```python
# In route handler
@router.post("/expensive-operation")
async def expensive_operation():
    # This endpoint will be rate-limited automatically
    pass
```

### Error Handling

Use custom exceptions for structured error handling:

```python
from app.utils.exceptions import ValidationError, ModelNotFoundError

if not model_exists:
    raise ModelNotFoundError(model_name=model_name)
```

---

## Security Checklist

Use this checklist before deploying to production:

### Configuration
- [ ] Changed `SECRET_KEY` from default
- [ ] Restricted `CORS_ORIGINS` to trusted domains
- [ ] Set appropriate `SESSION_TIMEOUT_MINUTES`
- [ ] Configured `API_KEY_EXPIRY_DAYS` if needed
- [ ] Set `LOG_LEVEL` to `WARNING` or `ERROR`
- [ ] Enabled `STRUCTURED_LOGGING`

### Security Features
- [ ] Enabled rate limiting (`RATE_LIMIT_ENABLED=true`)
- [ ] Enabled CSRF protection (`CSRF_PROTECTION_ENABLED=true`)
- [ ] Enabled security headers (`SECURITY_HEADERS_ENABLED=true`)
- [ ] Configured appropriate rate limits

### Infrastructure
- [ ] Using HTTPS/TLS in production
- [ ] Running behind reverse proxy
- [ ] Application runs as non-root user
- [ ] File permissions properly restricted
- [ ] Firewall configured appropriately

### Monitoring
- [ ] Health checks configured
- [ ] Metrics collection enabled
- [ ] Log monitoring set up
- [ ] Alerting configured for errors
- [ ] Backup verification working

### Testing
- [ ] Security tests passing
- [ ] Input validation tests passing
- [ ] Rate limiting tests passing
- [ ] Authentication tests passing

### Backup
- [ ] Backup enabled and tested
- [ ] Restore procedure tested
- [ ] Backup retention configured
- [ ] Off-site backup storage configured

---

## Vulnerability Reporting

If you discover a security vulnerability, please email: security@yourdomain.com

**Please do not:**
- Open public issues for security vulnerabilities
- Discuss vulnerabilities in public forums
- Exploit vulnerabilities for malicious purposes

**Please include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and provide updates as we work on a fix.

---

## Security Updates

Stay informed about security updates:

1. Watch the GitHub repository for security announcements
2. Subscribe to security mailing list (if available)
3. Regularly check for dependency updates
4. Review security advisories for dependencies

### Checking for Vulnerabilities

```bash
# Check Python dependencies
pip install safety
safety check

# Security linting
pip install bandit
bandit -r app
```

---

## Common Security Scenarios

### Scenario 1: Suspected Brute Force Attack

**Symptoms:**
- High rate of authentication failures
- Same IP making many requests

**Response:**
1. Check rate limit logs
2. Verify rate limiting is enabled
3. Lower rate limits temporarily
4. Block IP at firewall level if needed

### Scenario 2: SQL Injection Attempt

**Symptoms:**
- SQL patterns in logs
- Input sanitization errors

**Response:**
1. Input validation already blocks this
2. Review logs for attack patterns
3. Verify all inputs are validated
4. Consider additional WAF rules

### Scenario 3: API Key Compromised

**Response:**
1. Revoke user session: `user.revoke_session()`
2. Generate new API key
3. Review access logs for suspicious activity
4. Notify user if necessary

### Scenario 4: DDoS Attack

**Response:**
1. Enable/tighten rate limits
2. Use reverse proxy rate limiting
3. Enable DDoS protection (Cloudflare, etc.)
4. Scale horizontally if needed

---

## Security Monitoring

### Metrics to Monitor

1. **Authentication Failures**
   - High rate indicates brute force attempt

2. **Rate Limit Hits**
   - Track which endpoints are being hit
   - Identify abusive clients

3. **Input Validation Errors**
   - May indicate attack attempts
   - Review patterns in logs

4. **Error Rates**
   - Sudden spikes may indicate issues
   - Review error logs for patterns

5. **Response Times**
   - Unusual slowness may indicate attack
   - Check slow request logs

### Log Analysis

```bash
# Check for authentication failures
grep "AUTH_FAILED" logs/error.log

# Check for rate limit violations
grep "RATE_LIMIT_EXCEEDED" logs/error.log

# Check for input sanitization errors
grep "INPUT_SANITIZATION_ERROR" logs/error.log
```

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)

---

**Last Updated:** November 2024
**Security Version:** 1.0.0
