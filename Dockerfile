FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate data
RUN python src/data_generator.py

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"] 