#!/bin/bash

# Exit on any error
set -e

# Load environment variables if .env file exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# --- Configuration ---
# Ensure these are set in your .env file or directly here
# Required: HF_HUB_TOKEN_WRITE, HF_MODEL_ID, DATASET_ID, TRAIN_DATASET_FILE, etc.
# PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET (for evaluation)

# Check for critical environment variables
if [ -z "$HF_HUB_TOKEN_WRITE" ] || [ -z "$HF_MODEL_ID" ] || [ -z "$DATASET_ID" ]; then
  echo "Error: HF_HUB_TOKEN_WRITE, HF_MODEL_ID, and DATASET_ID must be set in your .env file or environment."
  exit 1
fi
# HF_HUB_TOKEN_READ is also important, often same as write token or a separate read-only token.
if [ -z "$HF_HUB_TOKEN_READ" ]; then
    echo "Warning: HF_HUB_TOKEN_READ is not set. It might be needed for downloading models/datasets."
    # Set it to write token if not provided, assuming write token has read access
    export HF_HUB_TOKEN_READ=$HF_HUB_TOKEN_WRITE
fi


echo "---------- 1. Running Training (Fine-tuning) Script ----------"
# Ensure you are in the root directory of the project
# cd /path/to/your/project
python train.py

echo "\n---------- 2. Running Evaluation Script ----------"
# The evaluation script will use HF_MODEL_ID to fetch the fine-tuned models from the Hub
python evaluate.py

# --- Docker Operations (Optional) ---
# These are examples and might need adjustment based on your Docker setup and needs.

DOCKER_IMAGE_NAME="ko-en-translator-app"
DOCKER_IMAGE_TAG="latest"

echo "\n---------- 3. Building Docker Image ----------"
# Pass HF_HUB_TOKEN_READ as a build argument if needed by Dockerfile to download models during build
# However, it's generally better to download models at runtime if they change frequently or are large,
# or to bake them into the image if they are stable.
# The current Dockerfile expects HF_MODEL_ID to be set at runtime.
docker build \
  --build-arg HF_HUB_TOKEN_READ=${HF_HUB_TOKEN_READ} \
  -t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} .

echo "\n---------- 4. Running Docker Container (Example) ----------"
# Ensure HF_MODEL_ID is correctly passed for the API to load the right model.
# Also pass PAPAGO credentials if your API or any startup script needs them.
# The API in api/main.py uses HF_MODEL_ID to load the ONNX or Merged model.
# USE_ONNX_FOR_API can also be passed if you want to control it.
echo "To run the Docker container (example):"
echo "docker run -d -p 8000:8000 \\"
echo "  -e HF_MODEL_ID=\"${HF_MODEL_ID}\" \\"
echo "  -e HF_HUB_TOKEN_READ=\"${HF_HUB_TOKEN_READ}\" \\"
echo "  -e USE_ONNX_FOR_API=\"true\" \\" # or false
echo "  --name ${DOCKER_IMAGE_NAME}-container \\"
echo "  ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo ""
echo "You can then access the API at http://localhost:8000/docs"

# Example: Run container in foreground for quick test (Ctrl+C to stop)
# docker run --rm -p 8000:8000 \
#   -e HF_MODEL_ID="${HF_MODEL_ID}" \
#   -e HF_HUB_TOKEN_READ="${HF_HUB_TOKEN_READ}" \
#   -e USE_ONNX_FOR_API="true" \
#   --name ${DOCKER_IMAGE_NAME}-dev \
#   ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}

echo "\n---------- MLOps Pipeline Script Finished ----------"
echo "Next steps might include pushing the Docker image to a registry and deploying it."