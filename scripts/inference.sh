#!/bin/bash
set -e

IMAGE_NAME="ko-en-translator"
CONTAINER_NAME="ko-en-translator-app"

# Load .env file for Docker build args or runtime envs if needed
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    # This makes them available to the script, not directly to docker build --build-arg unless specified.
    export $(grep -v '^#' .env | xargs) 
else
    echo "Warning: .env file not found. Ensure required environment variables are set for Docker."
fi

# Check for required environment variables for inference service
if [ -z "$HF_HUB_TOKEN_READ" ] || [ -z "$HF_MODEL_ID" ]; then
    echo "Error: HF_HUB_TOKEN_READ and HF_MODEL_ID must be set for the Docker container to download models."
    echo "These can be set in your .env file or passed directly to 'docker run -e VAR=value ...'"
    # exit 1 # Commenting out to allow build even if vars are not set locally (they might be set in CI/CD)
fi


echo "Building Docker image: $IMAGE_NAME..."
# Pass build arguments if your Dockerfile uses ARG to set ENV defaults
# For example, if Dockerfile has `ARG HF_TOKEN_ARG` and `ENV HF_HUB_TOKEN_READ=$HF_TOKEN_ARG`
# docker build \
#   --build-arg HF_TOKEN_ARG="$HF_HUB_TOKEN_READ" \
#   -t $IMAGE_NAME .

docker build -t $IMAGE_NAME .

echo "Docker image built."

# Stop and remove existing container with the same name, if any
if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Stopping and removing existing container: $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "Running Docker container: $CONTAINER_NAME..."
# Pass runtime environment variables to the container
# These will be used by the FastAPI app inside the container (e.g., by config.py)
docker run -d \
    -p 8000:8000 \
    -e HF_HUB_TOKEN_READ="$HF_HUB_TOKEN_READ" \
    -e HF_MODEL_ID="$HF_MODEL_ID" \
    -e BASE_MODEL_ID="${BASE_MODEL_ID:-Helsinki-NLP/opus-mt-ko-en}" \
    -e PAPAGO_CLIENT_ID="$PAPAGO_CLIENT_ID" \
    -e PAPAGO_CLIENT_SECRET="$PAPAGO_CLIENT_SECRET" \
    --name $CONTAINER_NAME \
    $IMAGE_NAME

echo "Container $CONTAINER_NAME started on port 8000."
echo "API documentation will be available at http://localhost:8000/docs"
echo "To see logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"