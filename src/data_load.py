import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import HF_DATASET_PATH, MAX_SAMPLES_DATASET

def load_and_split_data(dataset_path: str = HF_DATASET_PATH, max_samples: int = None):
    """
    Loads data from a Hugging Face dataset URL using pandas and splits it.
    The dataset is expected to have 'ko' and 'en' columns.
    """
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you are logged in to huggingface_hub if the dataset is private,")
        print("or that the dataset path is correct for public datasets.")
        print("You might need to run: huggingface-cli login")
        raise

    print(f"Original dataset columns: {df.columns.tolist()}")
    if "ko" not in df.columns or "en" not in df.columns:
        raise ValueError("Dataset must contain 'ko' and 'en' columns.")

    df = df[["ko", "en"]] # Ensure only necessary columns are kept
    df = df.dropna().drop_duplicates() # Basic cleaning

    if max_samples is not None and max_samples > 0:
        print(f"Using a subset of {max_samples} samples for training/evaluation.")
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    if len(df) == 0:
        raise ValueError("No data after loading and initial filtering. Check dataset or max_samples.")

    print(f"Dataset loaded with {len(df)} samples after initial processing.")

    hf_dataset = Dataset.from_pandas(df)

    # Split: 80% train, 10% validation, 10% test
    train_test_split = hf_dataset.train_test_split(test_size=0.20, seed=42)
    test_valid_split = train_test_split["test"].train_test_split(test_size=0.50, seed=42)

    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    })

    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")

    return dataset_dict

if __name__ == "__main__":
    # Example usage:
    # Ensure your .env file is configured or HF_DATASET_PATH is valid
    # And you are logged in: huggingface-cli login
    try:
        data_dict = load_and_split_data(max_samples=MAX_SAMPLES_DATASET or 100) # test with 100 samples
        print("\nSample data from train split:")
        print(data_dict["train"][0])
    except Exception as e:
        print(f"An error occurred during data loading: {e}")