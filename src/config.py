import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Hub Configuration
HF_HUB_TOKEN_WRITE = os.getenv("HF_HUB_TOKEN_WRITE")
HF_HUB_TOKEN_READ = os.getenv("HF_HUB_TOKEN_READ") # Used by inference for downloading
HF_MODEL_ID = os.getenv("HF_MODEL_ID") # Your target model ID on the Hub
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Helsinki-NLP/opus-mt-ko-en")
DATASET_ID = os.getenv("DATASET_ID", "klei22/korean-english-jamon-parallel-corpora")
DATASET_FILE = os.getenv("DATASET_FILE", "ko_ja_en.parquet")


# Papago API Configuration
PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET")

# Training Configuration
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 16))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", 16))
MAX_SAMPLES_DATASET = os.getenv("MAX_SAMPLES_DATASET")
if MAX_SAMPLES_DATASET:
    try:
        MAX_SAMPLES_DATASET = int(MAX_SAMPLES_DATASET)
    except ValueError:
        MAX_SAMPLES_DATASET = None # Set to None if "None" or invalid int
else:
    MAX_SAMPLES_DATASET = None


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results_training")
QLORA_R = int(os.getenv("QLORA_R", 8))
QLORA_ALPHA = int(os.getenv("QLORA_ALPHA", 32))
QLORA_DROPOUT = float(os.getenv("QLORA_DROPOUT", 0.05))

# Derived configurations
HF_DATASET_PATH = f"hf://datasets/{DATASET_ID}/{DATASET_FILE}"
LOCAL_ADAPTERS_PATH = "peft_lora_adapters"
LOCAL_ONNX_MODEL_PATH = "onnx_model"

# Ensure tokens are set if required actions are performed
def check_tokens():
    if not HF_MODEL_ID:
        raise ValueError("HF_MODEL_ID environment variable is not set.")
    # Further checks can be added here depending on the operation
    # e.g., for training, HF_HUB_TOKEN_WRITE is essential.
    # for inference from private repo, HF_HUB_TOKEN_READ is essential.

print(f"Configuration Loaded:")
print(f"  Base Model ID: {BASE_MODEL_ID}")
print(f"  Target HF Model ID: {HF_MODEL_ID}")
print(f"  Dataset Path: {HF_DATASET_PATH}")
print(f"  Max Samples: {MAX_SAMPLES_DATASET if MAX_SAMPLES_DATASET is not None else 'All'}")