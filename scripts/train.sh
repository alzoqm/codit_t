#!/bin/bash

# Ensure the script exits if any command fails
set -e

echo "Starting training script..."

# Ensure .env file exists and is sourced if needed, or variables are already in environment
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Relying on pre-set environment variables."
fi

# Check for required environment variables for training
if [ -z "$HF_HUB_TOKEN_WRITE" ] || [ -z "$HF_MODEL_ID" ]; then
    echo "Error: HF_HUB_TOKEN_WRITE and HF_MODEL_ID must be set in your environment or .env file."
    exit 1
fi

echo "HF_MODEL_ID is set to: $HF_MODEL_ID"
echo "BASE_MODEL_ID is set to: $BASE_MODEL_ID"
echo "DATASET_ID is set to: $DATASET_ID"

# Login to Hugging Face CLI (optional, if src/train_model.py handles login internally)
# echo "Attempting Hugging Face CLI login (if not already logged in)..."
# huggingface-cli login --token $HF_HUB_TOKEN_WRITE || echo "CLI login failed or already logged in. Continuing..."


# Set PYTHONPATH to include the project root, so src.module imports work
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"


echo "Running the training Python script (src/train_model.py)..."
python src/train_model.py

echo "Training script finished."