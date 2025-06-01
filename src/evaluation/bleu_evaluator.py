import sacrebleu
import requests
import time
from typing import List, Dict, Callable
from src.data_load import load_and_split_data
from src.config import MAX_SAMPLES_DATASET

# FastAPI server URL (assuming it's running locally)
API_BASE_URL = "http://localhost:8000/translate"

def get_translation_from_api(text: str, model_type: str) -> str | None:
    """Fetches translation from the running FastAPI service."""
    endpoint_map = {
        "base": f"{API_BASE_URL}/base",
        "finetuned": f"{API_BASE_URL}/finetuned",
        "onnx": f"{API_BASE_URL}/onnx",
        "papago": f"{API_BASE_URL}/papago",
    }
    if model_type not in endpoint_map:
        print(f"Unknown model type for API: {model_type}")
        return None

    try:
        payload = {"text": text, "source_lang": "ko", "target_lang": "en"}
        response = requests.post(endpoint_map[model_type], json=payload, timeout=30) # Increased timeout
        response.raise_for_status()
        return response.json()["translated_text"]
    except requests.RequestException as e:
        print(f"Error calling translation API for {model_type} for text '{text[:30]}...': {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"API Response: {e.response.text}")
        return None
    except KeyError:
        print(f"Error parsing API response for {model_type}. Response: {response.text}")
        return None


def evaluate_bleu_on_test_set(max_eval_samples: int = 100):
    """
    Loads the test set, gets translations from different models via API,
    and calculates BLEU scores.
    """
    print(f"Starting BLEU evaluation on up to {max_eval_samples} samples...")
    
    # 1. Load test data
    # Use a small subset of MAX_SAMPLES_DATASET for loading if it's very large,
    # then further limit by max_eval_samples for actual evaluation.
    dataset_load_limit = (MAX_SAMPLES_DATASET or 500) if MAX_SAMPLES_DATASET else 500
    all_data = load_and_split_data(max_samples=dataset_load_limit)
    test_data = all_data["test"]

    if len(test_data) == 0:
        print("No test data found. Skipping BLEU evaluation.")
        return

    if len(test_data) > max_eval_samples:
        print(f"Using a subset of {max_eval_samples} samples from the test set for BLEU evaluation.")
        test_data = test_data.select(range(max_eval_samples))

    references = [[item["en"]] for item in test_data]  # List of lists for sacreBLEU
    korean_sources = [item["ko"] for item in test_data]

    model_predictions: Dict[str, List[str]] = {
        "base": [],
        "finetuned": [],
        "onnx": [],
        "papago": []
    }

    # 2. Get translations for each model
    for model_name in model_predictions.keys():
        print(f"\nGetting translations for model: {model_name}...")
        for i, source_text in enumerate(korean_sources):
            if (i + 1) % 10 == 0:
                print(f"  Translated {i+1}/{len(korean_sources)} for {model_name}")
            
            # Add a small delay to avoid overwhelming the server if it's struggling
            time.sleep(0.1) 
            
            translated = get_translation_from_api(source_text, model_name)
            if translated:
                model_predictions[model_name].append(translated)
            else:
                model_predictions[model_name].append("") # Append empty if translation failed
                print(f"Warning: Failed to get translation for a sample using {model_name}.")
        
        if len(model_predictions[model_name]) != len(references):
            print(f"Warning: Mismatch in translation count for {model_name}. Expected {len(references)}, got {len(model_predictions[model_name])}")


    # 3. Calculate BLEU scores
    print("\n--- BLEU Score Results ---")
    for model_name, hypotheses in model_predictions.items():
        if not hypotheses or len(hypotheses) != len(references):
            print(f"Skipping BLEU for {model_name} due to missing or mismatched predictions.")
            continue
        
        # Filter out cases where hypothesis might be empty due to API errors, if any.
        # And corresponding reference. This is a simple way, might need refinement.
        filtered_refs = [ref for i, ref in enumerate(references) if hypotheses[i]]
        filtered_hyps = [hyp for hyp in hypotheses if hyp]

        if not filtered_hyps or not filtered_refs:
            print(f"No valid hypothesis/reference pairs for {model_name} after filtering empty.")
            bleu_score = None
        else:
            bleu_score = sacrebleu.corpus_bleu(filtered_hyps, filtered_refs)
        
        print(f"Model: {model_name:<15} | BLEU Score: {bleu_score.score if bleu_score else 'N/A'}")
        if bleu_score:
             print(f"  (Details: {str(bleu_score)})")


if __name__ == "__main__":
    print("Ensure the FastAPI server is running on http://localhost:8000 before starting BLEU evaluation.")
    # Prompt user to confirm server is running
    # input("Press Enter to start BLEU evaluation once the server is running...")
    
    try:
        evaluate_bleu_on_test_set(max_eval_samples=50) # Evaluate on 50 samples for a quick test
    except Exception as e:
        print(f"An error occurred during BLEU evaluation: {e}")
        import traceback
        traceback.print_exc()