#!/usr/bin/env bash
# Startup script for RoadZone AI: Launches both FastAPI backend and Streamlit frontend

set -e

echo "=========================================================="
echo "🛣️  Starting RoadZone AI (Google Gemma 4 Civil Defense)"
echo "=========================================================="

# 1. Initialize SQLite database schema
python3 -c "import database; database.init_db()"

# 2. Launch FastAPI Backend on port 8000 in background
echo "🚀 Starting FastAPI REST backend on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap signals to gracefully terminate both processes
trap "kill -TERM $BACKEND_PID 2>/dev/null || true; exit" SIGINT SIGTERM EXIT

# 3. Launch Streamlit UI Dashboard on port 8501 in foreground
echo "🌟 Starting Streamlit visual dashboard on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
