# 🚀 Quick Start Guide

## What Was Fixed

The `start-dev.sh` script was stopping at "Installing backend dependencies..." because:

1. ❌ **Missing `backend/requirements.txt`** - The Python dependencies file was missing
2. ❌ **Silent failures** - Errors were hidden by redirecting output to `/dev/null`
3. ❌ **No error messages** - The script didn't show why installations failed

### ✅ Fixes Applied

1. ✅ Created `backend/requirements.txt` with all required Python dependencies
2. ✅ Updated `start-dev.sh` to show installation progress
3. ✅ Added proper error checking and helpful error messages
4. ✅ Shows last 20 lines of logs if services fail to start

---

## How to Start the Application

### Option 1: Automated Script (Recommended)

```bash
cd "/Users/eilonudi/Desktop/HW/LLMs in multiagent env/MultiAgentCourse/Assignment1"
./start-dev.sh
```

**What the script does:**
1. ✅ Checks prerequisites (Python, Node.js, Ollama)
2. ✅ Creates Python virtual environment
3. ✅ Installs backend dependencies (~2-3 minutes first time)
4. ✅ Installs frontend dependencies (~2-3 minutes first time)
5. ✅ Creates environment files
6. ✅ Starts backend on http://localhost:8000
7. ✅ Starts frontend on http://localhost:5173

### Option 2: Manual Start

**Backend:**
```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend
python run.py
```

**Frontend (in a new terminal):**
```bash
cd frontend

# Install dependencies
npm install

# Start frontend
npm run dev
```

---

## Expected Output

When you run `./start-dev.sh`, you should see:

```
╔══════════════════════════════════════════════════════════╗
║  🦙 Ollama Web GUI - Development Startup              ║
╚══════════════════════════════════════════════════════════╝

ℹ Checking prerequisites...
✓ Python 3 found: Python 3.x.x
✓ Node.js found: v18.x.x
✓ npm found: 9.x.x
✓ Ollama found: ollama version x.x.x

ℹ Checking ports...
✓ Port 8000 is available (backend)
✓ Port 5173 is available (frontend)

ℹ Checking Ollama service...
✓ Ollama is running on port 11434

ℹ Setting up backend...
ℹ Creating Python virtual environment...
✓ Virtual environment created
ℹ Activating virtual environment...
ℹ Installing backend dependencies (this may take a few minutes)...
  Upgrading pip...
  Installing requirements...
✓ Backend dependencies installed
✓ .env file exists
ℹ Initializing database...
✓ Database initialized

ℹ Setting up frontend...
ℹ Installing frontend dependencies (this may take a few minutes)...
✓ Frontend dependencies installed
✓ .env file exists

╔══════════════════════════════════════════════════════════╗
║  🦙 Ollama Web GUI - Development Startup              ║
╚══════════════════════════════════════════════════════════╝

ℹ Starting services...

ℹ Starting backend on http://localhost:8000...
✓ Backend started successfully
ℹ   API: http://localhost:8000
ℹ   Docs: http://localhost:8000/docs
ℹ   Health: http://localhost:8000/health

ℹ Starting frontend on http://localhost:5173...
✓ Frontend started successfully
ℹ   URL: http://localhost:5173

╔══════════════════════════════════════════════════════════╗
║  Service Status                                          ║
╠══════════════════════════════════════════════════════════╣
║  Backend:   http://localhost:8000                       ║
║  Frontend:  http://localhost:5173                       ║
║  API Docs:  http://localhost:8000/docs                  ║
╚══════════════════════════════════════════════════════════╝

ℹ Logs are available in:
ℹ   Backend:  logs/backend.log
ℹ   Frontend: logs/frontend.log

ℹ Press Ctrl+C to stop all services
```

---

## First Time Setup

Once the application is running:

1. **Open your browser** to http://localhost:5173

2. **You'll see the setup screen**
   - Enter an API key (any string, e.g., `my-secret-key-123`)
   - Enter Ollama URL: `http://localhost:11434`

3. **Click "Test Connection"**
   - Should show "Connection successful" if Ollama is running

4. **Click "Save Configuration"**
   - You'll be redirected to the chat interface

5. **Start chatting!**
   - Select a model from the dropdown
   - Type a message and press Enter
   - Watch the response stream in real-time

---

## Troubleshooting

### Issue 1: "Python 3 is not installed"

**Solution:** Install Python 3.10 or higher
```bash
# macOS (using Homebrew)
brew install python@3.10

# Or download from python.org
```

### Issue 2: "Node.js is not installed"

**Solution:** Install Node.js 18 or higher
```bash
# macOS (using Homebrew)
brew install node

# Or download from nodejs.org
```

### Issue 3: "Ollama doesn't seem to be running"

**Solution:** Start Ollama
```bash
# If installed via CLI
ollama serve

# If installed via app, it should be running in background
# Check: curl http://localhost:11434/api/tags
```

### Issue 4: "Port 8000 is already in use"

**Solution:** Kill the process using port 8000
```bash
lsof -ti:8000 | xargs kill -9
```

### Issue 5: "Port 5173 is already in use"

**Solution:** Kill the process using port 5173
```bash
lsof -ti:5173 | xargs kill -9
```

### Issue 6: "Failed to install backend dependencies"

**Solution:** Install manually and check for errors
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
# Look at any error messages
```

### Issue 7: "Failed to install frontend dependencies"

**Solution:** Install manually and check for errors
```bash
cd frontend
npm install
# Look at any error messages
```

### Issue 8: "Backend failed to start"

**Check the logs:**
```bash
cat logs/backend.log
```

**Common causes:**
- Missing dependencies
- Port already in use
- Database initialization failed
- Environment file issues

### Issue 9: "Frontend failed to start"

**Check the logs:**
```bash
cat logs/frontend.log
```

**Common causes:**
- Missing dependencies
- Port already in use
- Vite configuration issues

### Issue 10: Script hangs during installation

**Solution:**
1. Press Ctrl+C to stop the script
2. Delete the `venv/installed` marker file:
   ```bash
   rm backend/venv/installed
   ```
3. Run the script again:
   ```bash
   ./start-dev.sh
   ```

---

## Stopping the Application

Press **Ctrl+C** in the terminal where `start-dev.sh` is running.

The script will automatically:
- Stop the backend server
- Stop the frontend server
- Clean up background processes

---

## Resetting Everything

If you want to start fresh:

```bash
# Remove virtual environment
rm -rf backend/venv

# Remove node modules
rm -rf frontend/node_modules

# Remove database
rm backend/ollama_web.db

# Remove logs
rm -rf logs

# Run the script again
./start-dev.sh
```

---

## Next Steps

1. ✅ Run `./start-dev.sh`
2. ✅ Wait for "All services started successfully!"
3. ✅ Open http://localhost:5173
4. ✅ Complete setup (API key + Ollama URL)
5. ✅ Start chatting with your local models!

---

## Testing the Integration

After starting the application, you can run integration tests:

```bash
# In a new terminal
cd "/Users/eilonudi/Desktop/HW/LLMs in multiagent env/MultiAgentCourse/Assignment1"
./test-integration.sh
```

Expected result: **18/18 tests passing** ✅

---

## Documentation

For more information, see:
- **`README.md`** - Complete project documentation
- **`INTEGRATION_GUIDE.md`** - Detailed integration guide
- **`INTEGRATION_STATUS.md`** - Integration verification
- **`backend/API_ENDPOINTS.md`** - API reference
- **`backend/DEPLOYMENT.md`** - Production deployment

---

## Support

If you encounter issues:
1. Check this quickstart guide
2. Look at the logs in `logs/` directory
3. Review the detailed documentation
4. Check `INTEGRATION_GUIDE.md` troubleshooting section

---

**Ready? Let's start!** 🚀

```bash
./start-dev.sh
```
