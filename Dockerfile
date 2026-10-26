# Production Dockerfile for RoadZone AI
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (OpenCV headless & image support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Expose ports: 8000 (FastAPI backend) & 8501 (Streamlit frontend)
EXPOSE 8000 8501

# Run both backend and frontend via start script
CMD ["./start.sh"]
