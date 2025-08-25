# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /workspace

# 3. Install system dependencies (for kiwipiepy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first to leverage Docker cache
COPY ./app/requirements.txt /workspace/requirements.txt

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY ./app /workspace/app
COPY ./pipeline /workspace/pipeline

# 7. Expose port and define command
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
