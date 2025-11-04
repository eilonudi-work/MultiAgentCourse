# Deployment Guide - Ollama Web GUI Frontend

## Overview

This guide provides instructions for deploying the Ollama Web GUI frontend in various environments.

---

## Prerequisites

- Node.js 18+ 
- npm 9+ or pnpm
- Docker (for containerized deployment)
- nginx (for production server deployment)

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Backend API URL
VITE_API_BASE_URL=http://localhost:8000

# Default Ollama URL (optional)
VITE_OLLAMA_DEFAULT_URL=http://localhost:11434
```

**Production:**
```env
VITE_API_BASE_URL=https://api.your domain.com
VITE_OLLAMA_DEFAULT_URL=http://localhost:11434
```

---

## Development Deployment

### Local Development Server

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Access at: `http://localhost:5173`

---

## Production Deployment

### Option 1: Docker Deployment (Recommended)

#### Build Docker Image

```bash
# From frontend directory
docker build -t ollama-web-gui-frontend .
```

#### Run Container

```bash
docker run -d \
  --name ollama-frontend \
  -p 80:80 \
  -e VITE_API_BASE_URL=http://localhost:8000 \
  ollama-web-gui-frontend
```

#### Using Docker Compose

From project root:

```bash
docker-compose up -d
```

This starts both frontend and backend services.

### Option 2: Static File Deployment

#### Build Production Bundle

```bash
# Install dependencies
npm install

# Build for production
npm run build
```

Output directory: `dist/`

#### Deploy to Web Server

##### nginx Configuration

Create `/etc/nginx/sites-available/ollama-web-gui`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/ollama-web-gui/dist;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/javascript application/json;

    # SPA routing - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # API proxy (if backend on same server)
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/ollama-web-gui /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

##### Apache Configuration

```apache
<VirtualHost *:80>
    ServerName your-domain.com
    DocumentRoot /var/www/ollama-web-gui/dist

    <Directory /var/www/ollama-web-gui/dist>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted

        # SPA routing
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>

    # Enable compression
    <IfModule mod_deflate.c>
        AddOutputFilterByType DEFLATE text/html text/plain text/xml text/css text/javascript application/javascript application/json
    </IfModule>
</VirtualHost>
```

### Option 3: Cloud Platform Deployment

#### Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

**vercel.json:**
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "env": {
    "VITE_API_BASE_URL": "https://api.your-domain.com"
  }
}
```

#### Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --prod
```

**netlify.toml:**
```toml
[build]
  command = "npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[build.environment]
  VITE_API_BASE_URL = "https://api.your-domain.com"
```

#### AWS S3 + CloudFront

1. Build production bundle: `npm run build`
2. Upload `dist/` to S3 bucket
3. Configure S3 for static website hosting
4. Create CloudFront distribution
5. Configure error pages to serve `index.html` (for SPA routing)

---

## SSL/HTTPS Configuration

### Using Let's Encrypt (Certbot)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### nginx HTTPS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

---

## Performance Optimization

### Build Optimization

The production build is already optimized with:
- ✅ Code splitting
- ✅ Tree shaking
- ✅ Minification
- ✅ Gzip compression
- ✅ Asset optimization

### CDN Configuration

For better performance, serve static assets via CDN:

1. Upload `/dist/assets/*` to CDN
2. Update `vite.config.js` base URL:

```js
export default defineConfig({
  base: 'https://cdn.your-domain.com/',
  // ...
})
```

---

## Monitoring & Analytics

### Add Google Analytics (Optional)

```html
<!-- In index.html -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Error Tracking (Sentry)

```bash
npm install @sentry/react
```

```js
// In main.jsx
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: "YOUR_SENTRY_DSN",
  environment: import.meta.env.MODE,
});
```

---

## Health Checks

### Application Health Endpoint

Create `public/health.html`:

```html
<!DOCTYPE html>
<html>
<head><title>Health Check</title></head>
<body>OK</body>
</html>
```

Access at: `https://your-domain.com/health.html`

---

## Troubleshooting

### Issue: 404 on page refresh

**Solution:** Configure server to serve `index.html` for all routes (SPA routing)

### Issue: API calls failing in production

**Solution:** Check CORS settings and API_BASE_URL environment variable

### Issue: Blank page after deployment

**Solution:** 
1. Check browser console for errors
2. Verify environment variables are set correctly
3. Ensure `base` path in `vite.config.js` matches deployment path

### Issue: Assets not loading

**Solution:** Check `base` path in `vite.config.js` and asset paths in HTML

---

## Rollback Procedure

### Docker Deployment

```bash
# List images
docker images

# Run previous version
docker run -d --name ollama-frontend -p 80:80 ollama-web-gui-frontend:previous-tag
```

### Static File Deployment

```bash
# Keep backup of previous build
cp -r dist dist-backup-$(date +%Y%m%d)

# Restore if needed
rm -rf dist
mv dist-backup-YYYYMMDD dist
```

---

## Security Checklist

- [ ] HTTPS enabled
- [ ] Security headers configured
- [ ] No sensitive data in frontend code
- [ ] API keys not exposed
- [ ] Dependencies updated (no vulnerabilities)
- [ ] CORS properly configured
- [ ] CSP headers set (if applicable)

---

## Maintenance

### Update Dependencies

```bash
# Check for updates
npm outdated

# Update dependencies
npm update

# Audit for vulnerabilities
npm audit
npm audit fix
```

### Rebuild and Redeploy

```bash
npm run build
# Then deploy using chosen method
```

---

## Support

For deployment issues:
1. Check application logs
2. Check nginx/Apache error logs
3. Verify environment variables
4. Test locally with production build (`npm run preview`)

---

**Last Updated:** November 2025
**Version:** 1.0.0
