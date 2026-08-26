# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies (ffmpeg is required for video slicing/yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install dependencies (plus sentence-transformers)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sentence-transformers

# Create writeable directories for Hugging Face user
RUN mkdir -p /app/data && chmod -R 777 /app

# Copy the application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY media/ ./media/
COPY data/ ./data/

# Expose port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "backend/app.py"]
