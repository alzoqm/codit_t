# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Hugging Face (can be overridden at runtime)
# These are primarily for downloading models if they are private.
# For public models, tokens might not be strictly necessary for download.
ENV HF_HUB_TOKEN_READ=""
ENV HF_MODEL_ID=""
ENV BASE_MODEL_ID="Helsinki-NLP/opus-mt-ko-en"
# Papago (optional, if Papago endpoint is used from within container)
ENV PAPAGO_CLIENT_ID=""
ENV PAPAGO_CLIENT_SECRET=""

# Install system dependencies that might be needed (e.g., for onnxruntime or other C extensions)
# Add any other dependencies like git if you were cloning something
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
# Consider torch CPU version if not using GPU in Docker:
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file (make sure it only contains non-sensitive defaults or use Docker secrets/env vars for production)
# Or, better, rely on environment variables passed during `docker run`
COPY .env .

# Copy the rest of the application's code into the container at /app/src
COPY src/ /app/src

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application using Uvicorn
# This will run the FastAPI app defined in src/inference/main.py
# Ensure HF_HUB_TOKEN_READ and HF_MODEL_ID are passed as env vars during `docker run`
# if models need to be downloaded from a private hub repo by the app.
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]