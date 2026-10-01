FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer separately
# If requirements.txt hasn't changed, pip install is skipped on rebuild
# This makes rebuilds much faster during development
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose port 8000
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]