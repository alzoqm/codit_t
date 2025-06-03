import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import (
    HF_TRAIN_DATASET_PATH,
    HF_VALID_DATASET_PATH,
    HF_TEST_DATASET_PATH,
    MAX_SAMPLES_DATASET
)

def _load_single_dataset_from_path(dataset_path: str, split_name: str, max_samples: int = None) -> Dataset:
    """
    Loads a single dataset from a Hugging Face dataset URL (parquet file).
    Applies cleaning and sampling.
    """
    if not dataset_path:
        print(f"Dataset path for {split_name} is not configured. Skipping.")
        return None

    print(f"Loading {split_name} dataset from: {dataset_path}")
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error loading {split_name} dataset from {dataset_path}: {e}")
        print("Please ensure you are logged in to huggingface_hub if the dataset is private,")
        print("or that the dataset path is correct for public datasets.")
        print("You might need to run: huggingface-cli login")
        raise

    print(f"Original {split_name} dataset columns: {df.columns.tolist()}")
    if "ko" not in df.columns or "en" not in df.columns:
        raise ValueError(f"{split_name.capitalize()} dataset at {dataset_path} must contain 'ko' and 'en' columns.")

    df = df[["ko", "en"]]  # Ensure only necessary columns are kept
    df = df.dropna().drop_duplicates()  # Basic cleaning

    if max_samples is not None and max_samples > 0:
        num_to_sample = min(max_samples, len(df))
        if num_to_sample < len(df):
            print(f"Sampling {num_to_sample} samples from {split_name} dataset (originally {len(df)} samples).")
            df = df.sample(n=num_to_sample, random_state=42)
        else:
            print(f"Using all {len(df)} samples from {split_name} dataset (max_samples set to {max_samples}).")


    if len(df) == 0:
        raise ValueError(f"No data in {split_name} dataset after loading and initial filtering from {dataset_path}. Check dataset or max_samples.")

    print(f"{split_name.capitalize()} dataset loaded with {len(df)} samples after processing.")
    return Dataset.from_pandas(df)

def load_pre_split_data(
    train_path: str = HF_TRAIN_DATASET_PATH,
    valid_path: str = HF_VALID_DATASET_PATH,
    test_path: str = HF_TEST_DATASET_PATH,
    max_samples_per_split: int = MAX_SAMPLES_DATASET
):
    """
    Loads pre-split train, validation, and test data from specified Hugging Face dataset URLs.
    Each dataset is expected to be a parquet file with 'ko' and 'en' columns.
    """
    dataset_dict_content = {}

    # Load training data
    train_dataset = _load_single_dataset_from_path(train_path, "train", max_samples_per_split)
    if train_dataset:
        dataset_dict_content["train"] = train_dataset

    # Load validation data
    valid_dataset = _load_single_dataset_from_path(valid_path, "validation", max_samples_per_split)
    if valid_dataset:
        dataset_dict_content["validation"] = valid_dataset
    
    # Load test data
    test_dataset = _load_single_dataset_from_path(test_path, "test", max_samples_per_split)
    if test_dataset:
        dataset_dict_content["test"] = test_dataset

    if not dataset_dict_content:
        raise ValueError("No datasets were loaded. Check configurations and file paths.")
    
    final_dataset_dict = DatasetDict(dataset_dict_content)

    print("\nFinal DatasetDict structure:")
    for split_name, data in final_dataset_dict.items():
        print(f"  {split_name}: {len(data)} samples")

    return final_dataset_dict

if __name__ == "__main__":
    # Example usage:
    # Ensure .env file is configured with DATASET_ID, TRAIN_DATASET_FILE, etc.
    # And you are logged in: huggingface-cli login
    try:
        # MAX_SAMPLES_DATASET from config will be used by default for max_samples_per_split
        # You can override it here for testing, e.g., max_samples_per_split=50
        data_dict = load_pre_split_data() 
        
        if "train" in data_dict:
            print("\nSample data from train split:")
            print(data_dict["train"][0])
        if "validation" in data_dict:
            print("\nSample data from validation split:")
            print(data_dict["validation"][0])
        if "test" in data_dict:
            print("\nSample data from test split:")
            print(data_dict["test"][0])
            
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()