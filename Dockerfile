# Use a slim Python image
FROM python:3.10-slim AS base

# Set build argument for GitHub Access Key (Passed at build time)

# Set a dedicated working directory (not root)
WORKDIR /usr/src/app

### 🔹 SUBMODULE HANDLING (Using HTTPS with Token)

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


### 🔹 MAIN BUILD

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential  \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching dependencies
COPY r.txt ./

# Install dependencies with no cache
RUN pip install -r r.txt

# Copy the rest of the project files into the working directory
COPY . .

# Set environment variables
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH

# Copy the startup script
COPY startup.sh .

# Make the script executable
RUN chmod +x startup.sh

# Expose the application port
EXPOSE 8080

# Run Django using Gunicorn
CMD ["python main.py"]