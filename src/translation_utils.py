import time
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from sacrebleu.metrics import BLEU
from datasets import Dataset
from tqdm import tqdm

from src.config import (
    BASE_MODEL_ID,
    HF_MODEL_ID,
    HF_HUB_TOKEN_READ,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET
)

# --- Hugging Face Model Translation ---

def get_hf_model_and_tokenizer(model_id_or_path, tokenizer_id_or_path=None, token=None, is_peft_adapter=False, adapter_subfolder=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads a Hugging Face model and tokenizer."""
    if tokenizer_id_or_path is None:
        tokenizer_id_or_path = model_id_or_path

    print(f"Loading tokenizer from: {tokenizer_id_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_path, token=token)

    print(f"Loading model from: {model_id_or_path}")
    if is_peft_adapter:
        # Load base model first
        base_model_for_peft = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID, # Base model for PEFT
            token=token,
            torch_dtype=torch.float16, # Using float16 for inference if possible
            low_cpu_mem_usage=True
        ).to(device)
        print(f"Loading PEFT adapter from: {model_id_or_path}, subfolder: {adapter_subfolder}")
        model = PeftModel.from_pretrained(
            base_model_for_peft,
            model_id_or_path, # This should be HF_MODEL_ID
            subfolder=adapter_subfolder, # e.g., "peft_lora"
            token=token
        ).to(device)
        model = model.merge_and_unload() # Merge for faster inference, optional if you want to keep adapters separate
        print("PEFT model loaded and merged.")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id_or_path,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        print("Standard model loaded.")
    
    model.eval() # Set to evaluation mode
    return model, tokenizer

def translate_texts_hf(texts, model, tokenizer, batch_size=16, max_length=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Translates a list of texts using a Hugging Face model."""
    translations = []
    print(f"Translating {len(texts)} texts using Hugging Face model on {device} with batch size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- ONNX Model Translation ---

def get_onnx_model_and_tokenizer(model_repo_id, model_subfolder, tokenizer_repo_id=None, token=None, provider="CPUExecutionProvider"):
    """Loads an ONNX model and tokenizer from Hugging Face Hub."""
    model_path = f"{model_repo_id}/{model_subfolder}" # e.g., "alzoqm/test_model_2/onnx_merged"
    if tokenizer_repo_id is None:
        tokenizer_repo_id = model_repo_id # Assume tokenizer is in the root of the model repo or same as ONNX folder
    
    print(f"Loading ONNX tokenizer from: {tokenizer_repo_id} (potentially from subfolder: {model_subfolder})")
    try:
        # Try loading tokenizer from the specific ONNX subfolder first
        tokenizer = AutoTokenizer.from_pretrained(f"{model_repo_id}/{model_subfolder}", token=token)
        print(f"Loaded tokenizer from ONNX subfolder: {model_repo_id}/{model_subfolder}")
    except Exception:
        print(f"Tokenizer not found in {model_repo_id}/{model_subfolder}, trying base {tokenizer_repo_id}")
        # Fallback to the tokenizer from the base model or merged model path if not in ONNX folder
        # Typically, the tokenizer for onnx_merged should be co-located or same as merged.
        tokenizer = AutoTokenizer.from_pretrained(f"{model_repo_id}/merged", token=token) # Try merged folder
        if not tokenizer: # Fallback to base model tokenizer if still not found
             tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)
        print(f"Loaded tokenizer from: {tokenizer.name_or_path}")


    print(f"Loading ONNX model from Hugging Face Hub: {model_path} using provider: {provider}")
    # ORTModelForSeq2SeqLM.from_pretrained expects the model name and optionally config for the tokenizer
    # It will download the specified subfolder if the main repo_id is given and subfolder has model files.
    # Or directly point to the hub path if supported (less common, usually downloads to cache then loads)

    # Forcing download from a subfolder:
    from huggingface_hub import snapshot_download
    local_onnx_model_dir = snapshot_download(repo_id=model_repo_id, allow_patterns=[f"{model_subfolder}/*"], token=token, repo_type="model")
    
    model_load_path = f"{local_onnx_model_dir}/{model_subfolder}"

    print(f"Loading ONNX model from local snapshot: {model_load_path}")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_load_path, # Path to the directory containing onnx files and config
        provider=provider,
        # The tokenizer_name_or_path is usually derived if config.json is present
    )
    print("ONNX model loaded.")
    model.eval()
    return model, tokenizer

def translate_texts_onnx(texts, model, tokenizer, batch_size=16, max_length=128):
    """Translates a list of texts using an ONNX model."""
    translations = []
    print(f"Translating {len(texts)} texts using ONNX model with batch size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        # ONNX model might not need inputs on a specific device like "cuda" unless using CUDAExecutionProvider
        # and the model itself is not moved to device. `inputs` are on CPU by default here.
        with torch.no_grad(): # Good practice, though ONNX runtime handles its own graph
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- Papago API Translation ---

def translate_texts_papago(texts, client_id, client_secret, batch_size=1): # Papago typically one by one
    """Translates a list of texts using Papago API."""
    if not client_id or not client_secret:
        print("Papago API credentials not found. Skipping Papago translation.")
        return ["Papago API not configured."] * len(texts)

    url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
    headers = {
        "Content-Type": "application/json",
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret
    }
    translations = []
    print(f"Translating {len(texts)} texts using Papago API...")
    
    for text_to_translate in tqdm(texts):
        payload = {
            "source": "ko",
            "target": "en",
            "text": text_to_translate
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors
            res_data = response.json()
            if "message" in res_data and "result" in res_data["message"]:
                translations.append(res_data["message"]["result"]["translatedText"])
            else:
                print(f"Warning: Papago API response format unexpected for text: '{text_to_translate}'. Response: {res_data}")
                translations.append(f"[Papago Error: Unexpected response format: {response.status_code}]")

        except requests.exceptions.RequestException as e:
            print(f"Error during Papago API request for text '{text_to_translate}': {e}")
            translations.append(f"[Papago Error: {e}]")
        except json.JSONDecodeError:
            print(f"Error decoding Papago API JSON response for text '{text_to_translate}'. Status: {response.status_code}, Text: {response.text}")
            translations.append(f"[Papago Error: JSON decode error {response.status_code}]")
        time.sleep(0.1) # Basic rate limiting

    return translations

# --- Evaluation Metrics ---

def calculate_bleu(hypotheses, references):
    """
    Calculates BLEU score.
    :param hypotheses: A list of translated sentences.
    :param references: A list of lists of reference sentences (or a list of single reference sentences).
    """
    if not hypotheses or not references:
        return 0.0
    # Ensure references are in the format [ [ref1_sent1, ref2_sent1,...], [ref1_sent2, ...], ...]
    # If references is List[str], convert to List[List[str]]
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    bleu = BLEU()
    # corpus_bleu expects hypotheses as list of strings, and references as list of lists of strings
    score = bleu.corpus_score(hypotheses, references)
    return score.score


def measure_translation_speed(translation_function, model_data, texts, num_warmup=2, num_repeats=5, batch_size_override=None):
    """
    Measures translation speed.
    :param translation_function: The function to call for translation (e.g., translate_texts_hf).
    :param model_data: Tuple containing (model, tokenizer) or None if not applicable.
    :param texts: List of texts to translate.
    :param num_warmup: Number of warmup runs.
    :param num_repeats: Number of actual timing runs.
    :param batch_size_override: Optional batch size for HF/ONNX models during speed test.
    :return: Average time per batch/request, total time for all texts.
    """
    if not texts:
        return 0.0, 0.0

    # Warmup runs
    print(f"Speed test: Warming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        if model_data: # HF or ONNX
            _ = translation_function(texts[:5], model_data[0], model_data[1], batch_size=batch_size_override or 16)
        else: # Papago
            _ = translation_function(texts[:1], PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET) # Papago takes individual texts

    # Actual timing
    start_times = []
    print(f"Speed test: Running {num_repeats} timed iterations...")
    for _ in tqdm(range(num_repeats)):
        start_time = time.time()
        if model_data:
            _ = translation_function(texts, model_data[0], model_data[1], batch_size=batch_size_override or len(texts)) # translate all at once or in batches
        else:
            _ = translation_function(texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
        end_time = time.time()
        start_times.append(end_time - start_time)

    avg_total_time = sum(start_times) / num_repeats
    # For HF/ONNX, time per batch depends on how translation_function is implemented.
    # Here we report total time for the given number of texts.
    # If we want time per 100 words, we need to count words.
    # For simplicity, we report total time for N texts and average time per text.

    avg_time_per_text = avg_total_time / len(texts) if texts else 0

    print(f"Average total time for {len(texts)} texts over {num_repeats} runs: {avg_total_time:.4f} seconds.")
    print(f"Average time per text: {avg_time_per_text:.6f} seconds.")
    return avg_total_time, avg_time_per_text


if __name__ == '__main__':
    # Example Usage (Illustrative - requires .env setup and logged in to Hugging Face)
    print("Running translation_utils.py example...")

    # 0. Ensure HF_HUB_TOKEN_READ is available if needed for private/gated models
    # For public models, it might not be strictly necessary but good practice.
    
    # 1. Load Test Data (Simplified - replace with actual data loading for robust testing)
    sample_ko_texts = [
        "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "이것은 번역 모델을 테스트하기 위한 샘플 문장입니다.",
        "허깅페이스 트랜스포머 라이브러리는 매우 유용합니다."
    ]
    sample_en_refs = [ # Corresponding English references
        ["Hello, the weather is really nice today."],
        ["This is a sample sentence for testing the translation model."],
        ["The Hugging Face Transformers library is very useful."]
    ]

    # 2. Test Base Model Translation
    print("\n--- Testing Base Model (Helsinki-NLP/opus-mt-ko-en) ---")
    try:
        base_model, base_tokenizer = get_hf_model_and_tokenizer(BASE_MODEL_ID, token=HF_HUB_TOKEN_READ)
        base_translations = translate_texts_hf(sample_ko_texts, base_model, base_tokenizer, batch_size=2)
        print("Base Model Translations:", base_translations)
        if base_translations and sample_en_refs:
             base_bleu = calculate_bleu(base_translations, sample_en_refs)
             print(f"Base Model BLEU: {base_bleu:.2f}")
        
        # Speed Test for Base Model
        avg_total_time, avg_time_per_text = measure_translation_speed(
            translate_texts_hf, (base_model, base_tokenizer), sample_ko_texts * 10, # Repeat texts for more stable speed test
            num_warmup=1, num_repeats=2, batch_size_override=4
        )
        print(f"Base Model - Avg Total Time (for {len(sample_ko_texts*10)} texts): {avg_total_time:.4f}s, Avg Time/Text: {avg_time_per_text:.6f}s")

        del base_model, base_tokenizer # Clean up memory
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error testing base model: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test Fine-tuned Model (Merged version from Hub)
    print(f"\n--- Testing Fine-tuned Merged Model ({HF_MODEL_ID}/merged) ---")
    # Ensure HF_MODEL_ID is set in .env and the merged model exists on the Hub
    if HF_MODEL_ID:
        try:
            ft_merged_model_id = f"{HF_MODEL_ID}/merged"
            # Tokenizer for merged model is usually co-located or same as base.
            # get_hf_model_and_tokenizer will try model_id first for tokenizer.
            ft_model, ft_tokenizer = get_hf_model_and_tokenizer(ft_merged_model_id, token=HF_HUB_TOKEN_READ)
            ft_translations = translate_texts_hf(sample_ko_texts, ft_model, ft_tokenizer, batch_size=2)
            print("Fine-tuned Merged Model Translations:", ft_translations)
            if ft_translations and sample_en_refs:
                ft_bleu = calculate_bleu(ft_translations, sample_en_refs)
                print(f"Fine-tuned Merged Model BLEU: {ft_bleu:.2f}")

            avg_total_time_ft, avg_time_per_text_ft = measure_translation_speed(
                translate_texts_hf, (ft_model, ft_tokenizer), sample_ko_texts * 10,
                num_warmup=1, num_repeats=2, batch_size_override=4
            )
            print(f"Finetuned Model - Avg Total Time (for {len(sample_ko_texts*10)} texts): {avg_total_time_ft:.4f}s, Avg Time/Text: {avg_time_per_text_ft:.6f}s")

            del ft_model, ft_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing fine-tuned merged model from {ft_merged_model_id}: {e}")
            print("Ensure the 'merged' model is available on the Hub and HF_MODEL_ID is correct.")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping fine-tuned merged model test.")


    # 4. Test ONNX Model (from Hub)
    print(f"\n--- Testing ONNX Model ({HF_MODEL_ID}/onnx_merged) ---")
    if HF_MODEL_ID:
        try:
            # The provider can be 'CUDAExecutionProvider' if onnxruntime-gpu is installed and CUDA is available
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"Using ONNX provider: {onnx_provider}")
            
            onnx_model, onnx_tokenizer = get_onnx_model_and_tokenizer(
                HF_MODEL_ID, # Repo ID
                "onnx_merged", # Subfolder where ONNX model resides
                token=HF_HUB_TOKEN_READ,
                provider=onnx_provider
            )
            onnx_translations = translate_texts_onnx(sample_ko_texts, onnx_model, onnx_tokenizer, batch_size=2)
            print("ONNX Model Translations:", onnx_translations)
            if onnx_translations and sample_en_refs:
                onnx_bleu = calculate_bleu(onnx_translations, sample_en_refs)
                print(f"ONNX Model BLEU: {onnx_bleu:.2f}")

            avg_total_time_onnx, avg_time_per_text_onnx = measure_translation_speed(
                translate_texts_onnx, (onnx_model, onnx_tokenizer), sample_ko_texts * 10,
                num_warmup=1, num_repeats=2, batch_size_override=4
            )
            print(f"ONNX Model - Avg Total Time (for {len(sample_ko_texts*10)} texts): {avg_total_time_onnx:.4f}s, Avg Time/Text: {avg_time_per_text_onnx:.6f}s")
            
            del onnx_model, onnx_tokenizer
            # No explicit CUDA cache clear for ONNX unless model was on CUDA device via specific provider options.
        except Exception as e:
            print(f"Error testing ONNX model from {HF_MODEL_ID}/onnx_merged: {e}")
            print("Ensure the 'onnx_merged' model is available on the Hub, HF_MODEL_ID is correct, and onnxruntime is installed.")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping ONNX model test.")


    # 5. Test Papago Translation
    print("\n--- Testing Papago API Translation ---")
    if PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET:
        try:
            papago_translations = translate_texts_papago(sample_ko_texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
            print("Papago Translations:", papago_translations)
            # Filter out error messages before calculating BLEU
            valid_papago_translations = [t for t in papago_translations if not t.startswith("[Papago Error")]
            if valid_papago_translations and len(valid_papago_translations) == len(sample_en_refs):
                 papago_bleu = calculate_bleu(valid_papago_translations, sample_en_refs)
                 print(f"Papago BLEU: {papago_bleu:.2f}")
            elif valid_papago_translations:
                 print(f"Could not calculate Papago BLEU due to partial success (got {len(valid_papago_translations)} translations for {len(sample_en_refs)} refs).")

            # Speed test for Papago is less about raw compute and more about API latency.
            # For fairness in comparing local models vs API, speed test sample size should be managed.
            # Papago is usually slower due to network & per-request processing.
            avg_total_time_papago, avg_time_per_text_papago = measure_translation_speed(
                translate_texts_papago, None, sample_ko_texts[:3], # Test with fewer samples due to API calls
                num_warmup=0, num_repeats=1
            )
            print(f"Papago - Avg Total Time (for {len(sample_ko_texts[:3])} texts): {avg_total_time_papago:.4f}s, Avg Time/Text: {avg_time_per_text_papago:.6f}s")

        except Exception as e:
            print(f"Error testing Papago API: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Papago API credentials (PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET) not found in .env. Skipping Papago test.")

    print("\nTranslation utils example finished.")