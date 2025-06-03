# Start with a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Hugging Face (can be overridden at runtime)
# It's better to pass sensitive tokens as build args or runtime env vars
ARG HF_HUB_TOKEN_READ
ENV HF_HUB_TOKEN_READ=${HF_HUB_TOKEN_READ}
ENV HF_MODEL_ID="" # To be set at runtime or as another ARG
ENV BASE_MODEL_ID="Helsinki-NLP/opus-mt-ko-en" # Default, can be overridden
ENV PAPAGO_CLIENT_ID=""
ENV PAPAGO_CLIENT_SECRET=""
ENV USE_ONNX_FOR_API="true" # Default to true for API

# For ONNX Runtime with GPU, you might need a different base image or extra installations
# For CPU, onnxruntime is usually sufficient.
# Ensure transformers, optimum[onnxruntime] are in requirements for ONNX
# Add PyTorch if not using ONNX or for ONNX model generation by Optimum.
# CUDA specific dependencies for PyTorch (if not using ONNX on CPU)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# If using CUDAExecutionProvider for ONNX, ensure onnxruntime-gpu is installed
# RUN pip install onnxruntime-gpu # instead of onnxruntime in requirements.txt

# Install system dependencies that might be needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # Add other system dependencies if required by your packages (e.g., build-essential for C extensions)
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Consider creating a virtual environment if preferred
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application using Uvicorn
# Using 0.0.0.0 to be accessible from outside the container
# The number of workers can be configured based on your server's CPU cores
# For production, you'd typically use Gunicorn with Uvicorn workers:
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "api.main:app"]
# For simplicity in this example, we'll use Uvicorn directly.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]