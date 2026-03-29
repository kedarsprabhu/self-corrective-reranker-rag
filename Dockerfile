# Use Python 3.11 slim for a lean image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install OS-level deps (for psycopg, pdfmupdf, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server on the HF-required port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
