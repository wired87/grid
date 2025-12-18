# Use a lightweight python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for pip deps if needed, build stuff)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path to allow imports from current dir
ENV PYTHONPATH=/app

# Default command: run the test script
CMD ["python", "test_gnn_run.py"]