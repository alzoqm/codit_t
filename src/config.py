import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Hub Configuration
HF_HUB_TOKEN_WRITE = os.getenv("HF_HUB_TOKEN_WRITE")
HF_HUB_TOKEN_READ = os.getenv("HF_HUB_TOKEN_READ") # Used by inference for downloading
HF_MODEL_ID = os.getenv("HF_MODEL_ID") # Your target model ID on the Hub
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Helsinki-NLP/opus-mt-ko-en")
DATASET_ID = os.getenv("DATASET_ID") # The HF Hub ID of your dataset repo

# Specific file paths for pre-split data within the DATASET_ID repo
TRAIN_DATASET_FILE = os.getenv("TRAIN_DATASET_FILE")
VALID_DATASET_FILE = os.getenv("VALID_DATASET_FILE")
TEST_DATASET_FILE = os.getenv("TEST_DATASET_FILE")

# Papago API Configuration
PAPAGO_CLIENT_ID = os.getenv("PAPAGO_CLIENT_ID")
PAPAGO_CLIENT_SECRET = os.getenv("PAPAGO_CLIENT_SECRET")

# Training Configuration
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 16))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", 16))
MAX_SAMPLES_DATASET = os.getenv("MAX_SAMPLES_DATASET") # Max samples PER SPLIT for quick testing
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

# Derived configurations for dataset paths
if DATASET_ID and TRAIN_DATASET_FILE:
    HF_TRAIN_DATASET_PATH = f"hf://datasets/{DATASET_ID}/{TRAIN_DATASET_FILE}"
else:
    HF_TRAIN_DATASET_PATH = None

if DATASET_ID and VALID_DATASET_FILE:
    HF_VALID_DATASET_PATH = f"hf://datasets/{DATASET_ID}/{VALID_DATASET_FILE}"
else:
    HF_VALID_DATASET_PATH = None

if DATASET_ID and TEST_DATASET_FILE:
    HF_TEST_DATASET_PATH = f"hf://datasets/{DATASET_ID}/{TEST_DATASET_FILE}"
else:
    HF_TEST_DATASET_PATH = None

LOCAL_ADAPTERS_PATH = "peft_lora_adapters"
LOCAL_ONNX_MODEL_PATH = "onnx_model"

# Ensure tokens are set if required actions are performed
def check_tokens():
    if not HF_MODEL_ID:
        raise ValueError("HF_MODEL_ID environment variable is not set.")
    # Further checks can be added here depending on the operation

print(f"Configuration Loaded:")
print(f"  Base Model ID: {BASE_MODEL_ID}")
print(f"  Target HF Model ID: {HF_MODEL_ID}")
print(f"  Dataset Repo ID: {DATASET_ID}")
print(f"  Train Dataset File Path: {HF_TRAIN_DATASET_PATH}")
print(f"  Validation Dataset File Path: {HF_VALID_DATASET_PATH}")
print(f"  Test Dataset File Path: {HF_TEST_DATASET_PATH}")
print(f"  Max Samples per split: {MAX_SAMPLES_DATASET if MAX_SAMPLES_DATASET is not None else 'All'}")