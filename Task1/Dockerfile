# Use a more lightweight base image
FROM python:3.9-slim-buster

# Install system dependencies and clean up in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Set the working directory
WORKDIR /opt/app

# Copy only the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy mednext and install it
COPY ./mednext ./mednext
RUN pip install --no-cache-dir -e ./mednext

# Copy only necessary application files
COPY inference.py .
COPY models ./models

# Set environment variables
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /opt/app

# Switch to the non-root user
USER appuser

# Set the entry point for the container
ENTRYPOINT ["python3", "inference.py"]