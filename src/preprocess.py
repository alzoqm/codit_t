from transformers import AutoTokenizer
from datasets import DatasetDict
from src.config import BASE_MODEL_ID

# Global tokenizer (initialized once)
_tokenizer = None

def get_tokenizer(model_name_or_path: str = BASE_MODEL_ID):
    global _tokenizer
    if _tokenizer is None:
        print(f"Loading tokenizer for: {model_name_or_path}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Tokenizer loaded.")
    return _tokenizer

def preprocess_function(examples, tokenizer):
    inputs = examples["ko"]
    targets = examples["en"]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)

    # Tokenize targets
    # Temporarily set the tokenizer to target mode if it supports it (some models need this)
    # For Helsinki-NLP models, this is usually not strictly necessary but good practice.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_datasets(dataset_dict: DatasetDict, tokenizer_name_or_path: str = BASE_MODEL_ID):
    tokenizer = get_tokenizer(tokenizer_name_or_path)
    
    print("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["ko", "en"], # Remove original text columns
        desc="Running tokenizer on dataset"
    )
    print("Tokenization complete.")
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    from src.data_load import load_pre_split_data
    from src.config import MAX_SAMPLES_DATASET

    # Example usage:
    try:
        raw_datasets = load_pre_split_data(max_samples=MAX_SAMPLES_DATASET or 100) # test with 100 samples
        tokenized_data, tokenizer_instance = tokenize_datasets(raw_datasets)
        
        print("\nTokenized train dataset features:")
        print(tokenized_data["train"].features)
        print("\nSample tokenized input:")
        print(tokenized_data["train"][0])
        print("\nDecoded sample input_ids:")
        print(tokenizer_instance.decode(tokenized_data["train"][0]['input_ids']))
        print("\nDecoded sample labels:")
        print(tokenizer_instance.decode(tokenized_data["train"][0]['labels']))

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")