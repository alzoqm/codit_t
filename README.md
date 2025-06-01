# Korean-to-English Translation Model Project

This project fine-tunes a Hugging Face Korean-to-English translation model using qLoRA, provides an inference API via FastAPI, and includes MLOps automation with GitHub Actions.

## Project Structure

- **`.env`**: Configuration file for API keys and model IDs (must be created from template).
- **`Dockerfile`**: For building the FastAPI application Docker image.
- **`requirements.txt`**: Python dependencies.
- **`scripts/`**: Shell scripts for common tasks.
  - `train.sh`: Runs the model training and ONNX conversion pipeline.
  - `inference.sh`: Builds and runs the Dockerized FastAPI inference server.
- **`src/`**: Source code.
  - `config.py`: Loads environment variables and project configurations.
  - `data_load.py`: Loads and splits the translation dataset.
  - `preprocess.py`: Tokenizes and preprocesses data.
  - `train_model.py`: Core training script (qLoRA fine-tuning, ONNX conversion, Hub upload).
  - `inference/`: FastAPI application for serving translations.
    - `main.py`: FastAPI app definition and endpoints.
    - `model_loader.py`: Loads models (base, fine-tuned, ONNX) from Hugging Face Hub.
    - `papago_translator.py`: Utility for Papago API translation.
  - `evaluation/`: Scripts for evaluating model performance.
    - `bleu_evaluator.py`: Calculates BLEU scores against Papago and base model.
    - `speed_evaluator.py`: Measures translation speed.
- **`.github/workflows/`**: GitHub Actions for MLOps.
  - `main.yml`: Workflow for training, model upload, and optional Docker image push.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and configure `.env` file:**
    Copy the `.env.template` (if provided) or create `.env` and fill in your Hugging Face tokens, Papago API keys, and desired Hugging Face Model ID.
    ```
    HF_HUB_TOKEN_WRITE="your_hf_write_token"
    HF_HUB_TOKEN_READ="your_hf_read_token"
    HF_MODEL_ID="your-hf-username/your-model-name"
    # BASE_MODEL_ID="Helsinki-NLP/opus-mt-ko-en" (default)
    # DATASET_ID="klei22/korean-english-jamon-parallel-corpora" (default)
    # DATASET_FILE="ko_ja_en.parquet" (default)
    PAPAGO_CLIENT_ID="your_papago_id"
    PAPAGO_CLIENT_SECRET="your_papago_secret"
    # Optional: training params like NUM_TRAIN_EPOCHS, MAX_SAMPLES_DATASET etc.
    ```

3.  **Install dependencies:**
    (It's recommended to use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Hugging Face Login (CLI):**
    You might need to log in to Hugging Face Hub via the CLI, especially for operations like pushing to the hub if the script's internal login has issues.
    ```bash
    huggingface-cli login
    ```
    Enter your Hugging Face token (preferably a write token).

## Training

To train the model, run the training script:
```bash
chmod +x scripts/train.sh
./scripts/train.sh