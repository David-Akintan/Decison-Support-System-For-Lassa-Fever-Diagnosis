# Dockerfile for Lassa Fever Diagnosis System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_production.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements_production.txt

# Copy application code
COPY . /app/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "app_production:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120"]