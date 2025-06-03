import time
import requests
import json
import torch
import os # For os.path.join
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from sacrebleu.metrics import BLEU
# from datasets import Dataset # Not used directly in this version of the file
from tqdm import tqdm
from huggingface_hub import snapshot_download # Moved import to top

from src.config import (
    BASE_MODEL_ID,
    HF_MODEL_ID, # Used as a default or fallback, specific repo_ids passed as args
    HF_HUB_TOKEN_READ,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET
)

# --- Hugging Face Model Translation ---

def get_hf_model_and_tokenizer(
    repo_id: str,
    model_subfolder: str = None,
    tokenizer_repo_id: str = None,
    tokenizer_subfolder: str = None,
    token: str = None,
    is_peft_adapter: bool = False,
    base_model_for_peft_id: str = BASE_MODEL_ID, # Default from config
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads a Hugging Face model and tokenizer.
    - repo_id: Main repository ID for the model (e.g., "Helsinki-NLP/opus-mt-ko-en" or "alzoqm/test_model_2").
    - model_subfolder: Subfolder for model files within repo_id (e.g., "merged", or "peft_lora" if is_peft_adapter).
    - tokenizer_repo_id: Repo ID for tokenizer. Defaults to repo_id.
    - tokenizer_subfolder: Subfolder for tokenizer. Defaults to model_subfolder.
    - is_peft_adapter: If True, loads repo_id/model_subfolder as PEFT adapters onto base_model_for_peft_id.
    """
    actual_tokenizer_repo_id = tokenizer_repo_id if tokenizer_repo_id else repo_id
    # If tokenizer_subfolder is explicitly None, it means root. If not given (is None by default), it mirrors model_subfolder.
    actual_tokenizer_subfolder = tokenizer_subfolder if tokenizer_subfolder is not None else model_subfolder

    print(f"Loading tokenizer from repo: {actual_tokenizer_repo_id}, subfolder: {actual_tokenizer_subfolder or 'root'}")
    tokenizer = AutoTokenizer.from_pretrained(
        actual_tokenizer_repo_id,
        subfolder=actual_tokenizer_subfolder,
        token=token
    )

    print(f"Loading model from repo: {repo_id}, subfolder: {model_subfolder or 'root'}")
    if is_peft_adapter:
        print(f"Loading base model for PEFT: {base_model_for_peft_id}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_for_peft_id,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device} # Ensure base model is on the correct device
        ) # .to(device) might be redundant if device_map is used effectively
        print(f"Loading PEFT adapter from repo: {repo_id}, subfolder: {model_subfolder}")
        model = PeftModel.from_pretrained(
            base_model,
            repo_id, # This is the repo where adapters are stored
            subfolder=model_subfolder, # This is the subfolder name like "peft_lora"
            token=token,
            device_map={"": device} # Try to map PEFT model to device too
        )
        # model = model.merge_and_unload() # Optional: merge for faster inference
        # print("PEFT model loaded and merged (if merge_and_unload is called).")
        print("PEFT model loaded.")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            repo_id,
            subfolder=model_subfolder,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device} # Map model to device
        ) # .to(device)
        print("Standard model loaded.")
    
    model.eval()
    return model, tokenizer

def translate_texts_hf(texts, model, tokenizer, batch_size=16, max_length=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Translates a list of texts using a Hugging Face model."""
    translations = []
    # Model should already be on the correct device from get_hf_model_and_tokenizer
    model_device = next(model.parameters()).device 
    print(f"Translating {len(texts)} texts using Hugging Face model on {model_device} with batch size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model_device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- ONNX Model Translation ---

def get_onnx_model_and_tokenizer(
    model_repo_id: str,
    onnx_model_subfolder: str,
    tokenizer_base_repo_id: str = None, # Defaults to model_repo_id
    token: str = None,
    provider: str = "CPUExecutionProvider"
):
    """Loads an ONNX model and tokenizer from Hugging Face Hub."""
    actual_tokenizer_base_repo_id = tokenizer_base_repo_id if tokenizer_base_repo_id else model_repo_id
    tokenizer = None

    # Attempt 1: Tokenizer from ONNX model's own subfolder
    print(f"Attempt 1: Loading ONNX tokenizer from repo: {actual_tokenizer_base_repo_id}, subfolder: {onnx_model_subfolder}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_base_repo_id, subfolder=onnx_model_subfolder, token=token)
        print(f"Loaded tokenizer from ONNX subfolder: {actual_tokenizer_base_repo_id}/{onnx_model_subfolder}")
    except Exception as e1:
        print(f"Tokenizer not found in {actual_tokenizer_base_repo_id}/{onnx_model_subfolder} (Error: {e1}).")
        # Attempt 2: Tokenizer from "merged" subfolder in the same repo
        print(f"Attempt 2: Loading tokenizer from repo: {actual_tokenizer_base_repo_id}, subfolder: merged")
        try:
            tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_base_repo_id, subfolder="merged", token=token)
            print(f"Loaded tokenizer from merged subfolder: {actual_tokenizer_base_repo_id}/merged")
        except Exception as e2:
            print(f"Tokenizer not found in {actual_tokenizer_base_repo_id}/merged (Error: {e2}).")
            # Attempt 3: Tokenizer from the root of the actual_tokenizer_base_repo_id
            print(f"Attempt 3: Loading tokenizer from repo: {actual_tokenizer_base_repo_id} (root)")
            try:
                tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_base_repo_id, token=token)
                print(f"Loaded tokenizer from root of: {actual_tokenizer_base_repo_id}")
            except Exception as e3:
                print(f"Tokenizer not found in root of {actual_tokenizer_base_repo_id} (Error: {e3}).")
                # Attempt 4: Fallback to BASE_MODEL_ID's tokenizer
                print(f"Attempt 4: Falling back to tokenizer from BASE_MODEL_ID: {BASE_MODEL_ID}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)
                    print(f"Loaded tokenizer from BASE_MODEL_ID: {BASE_MODEL_ID}")
                except Exception as e4:
                    raise RuntimeError(f"All attempts to load tokenizer failed. Last error (BASE_MODEL_ID): {e4}")
    
    if tokenizer is None:
        raise RuntimeError("Critical: Could not load tokenizer after multiple attempts.")

    print(f"Loading ONNX model from Hugging Face Hub repo: {model_repo_id}, subfolder: {onnx_model_subfolder}, provider: {provider}")
    
    onnx_files_pattern = f"{onnx_model_subfolder}/*"
    print(f"Downloading files from repo '{model_repo_id}' matching pattern '{onnx_files_pattern}'")
    
    # snapshot_download downloads the repo and returns the path to the local cache of the whole repo.
    # We then need to point to the subfolder within this downloaded cache.
    downloaded_repo_cache_path = snapshot_download(
        repo_id=model_repo_id,
        # allow_patterns=[onnx_files_pattern], # This filters what's downloaded from the repo if it's a large repo
        # For models saved with subfolders, often the config.json for ORTModel is expected at the level passed to from_pretrained
        token=token,
        repo_type="model"
    )
    
    model_load_path = os.path.join(downloaded_repo_cache_path, onnx_model_subfolder)

    if not os.path.exists(model_load_path) or not os.listdir(model_load_path):
        raise FileNotFoundError(f"ONNX model files not found or directory empty at expected local path: {model_load_path}. "
                                f"Check if subfolder '{onnx_model_subfolder}' exists in repo '{model_repo_id}' and was downloaded correctly.")

    print(f"Loading ONNX model from local snapshot directory: {model_load_path}")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_load_path, # This must be the directory containing model.onnx and config.json
        provider=provider,
        # use_io_binding=False # Consider if issues arise with specific providers/hardware
    )
    print("ONNX model loaded.")
    # model.eval() # ORT Models don't typically have an eval() method like PyTorch nn.Module
    return model, tokenizer

def translate_texts_onnx(texts, model, tokenizer, batch_size=16, max_length=128):
    """Translates a list of texts using an ONNX model."""
    translations = []
    print(f"Translating {len(texts)} texts using ONNX model with batch size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        # For ONNX with CPUExecutionProvider, inputs remain on CPU.
        # For CUDAExecutionProvider, Optimum handles moving data if model is on GPU.
        with torch.no_grad(): 
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- Papago API Translation ---
# (No changes to translate_texts_papago)
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
            response.raise_for_status() 
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
        time.sleep(0.1) 

    return translations

# --- Evaluation Metrics ---
# (No changes to calculate_bleu and measure_translation_speed)
def calculate_bleu(hypotheses, references):
    """
    Calculates BLEU score.
    :param hypotheses: A list of translated sentences.
    :param references: A list of lists of reference sentences (or a list of single reference sentences).
    """
    if not hypotheses or not references:
        return 0.0
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    bleu = BLEU()
    score = bleu.corpus_score(hypotheses, references)
    return score.score


def measure_translation_speed(translation_function, model_data, texts, num_warmup=2, num_repeats=5, batch_size_override=None):
    """
    Measures translation speed.
    """
    if not texts:
        return 0.0, 0.0

    # Warmup runs
    print(f"Speed test: Warming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        if model_data: # HF or ONNX
            _ = translation_function(texts[:5], model_data[0], model_data[1], batch_size=batch_size_override or 16)
        else: # Papago
            _ = translation_function(texts[:1], PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)

    # Actual timing
    start_times = []
    print(f"Speed test: Running {num_repeats} timed iterations...")
    for _ in tqdm(range(num_repeats)):
        start_time = time.time()
        if model_data:
            _ = translation_function(texts, model_data[0], model_data[1], batch_size=batch_size_override or len(texts))
        else:
            _ = translation_function(texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
        end_time = time.time()
        start_times.append(end_time - start_time)

    avg_total_time = sum(start_times) / num_repeats
    avg_time_per_text = avg_total_time / len(texts) if texts else 0

    print(f"Average total time for {len(texts)} texts over {num_repeats} runs: {avg_total_time:.4f} seconds.")
    print(f"Average time per text: {avg_time_per_text:.6f} seconds.")
    return avg_total_time, avg_time_per_text

if __name__ == '__main__':
    # Example Usage (Illustrative - requires .env setup and logged in to Hugging Face)
    print("Running translation_utils.py example...")
    
    sample_ko_texts = [
        "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "이것은 번역 모델을 테스트하기 위한 샘플 문장입니다.",
        "허깅페이스 트랜스포머 라이브러리는 매우 유용합니다."
    ]
    sample_en_refs = [
        ["Hello, the weather is really nice today."],
        ["This is a sample sentence for testing the translation model."],
        ["The Hugging Face Transformers library is very useful."]
    ]
    
    # Ensure HF_MODEL_ID is set in your environment or src.config for the example to run fully
    # For example: export HF_MODEL_ID="your_namespace/your_model_id"
    configured_hf_model_id = HF_MODEL_ID or "alzoqm/test_model_2" # Fallback for example if not in env

    # 2. Test Base Model Translation
    print("\n--- Testing Base Model (Helsinki-NLP/opus-mt-ko-en) ---")
    try:
        # For base model, repo_id is BASE_MODEL_ID, no subfolder
        base_model, base_tokenizer = get_hf_model_and_tokenizer(
            repo_id=BASE_MODEL_ID, 
            token=HF_HUB_TOKEN_READ
        )
        base_translations = translate_texts_hf(sample_ko_texts, base_model, base_tokenizer, batch_size=2)
        print("Base Model Translations:", base_translations)
        if base_translations and sample_en_refs:
             base_bleu = calculate_bleu(base_translations, sample_en_refs)
             print(f"Base Model BLEU: {base_bleu:.2f}")
        
        avg_total_time, avg_time_per_text = measure_translation_speed(
            translate_texts_hf, (base_model, base_tokenizer), sample_ko_texts * 10,
            num_warmup=1, num_repeats=2, batch_size_override=4
        )
        print(f"Base Model - Avg Total Time ({len(sample_ko_texts*10)} texts): {avg_total_time:.4f}s, Avg Time/Text: {avg_time_per_text:.6f}s")
        del base_model, base_tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error testing base model: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test Fine-tuned Model (Merged version from Hub)
    print(f"\n--- Testing Fine-tuned Merged Model ({configured_hf_model_id}/merged) ---")
    if configured_hf_model_id:
        try:
            ft_model, ft_tokenizer = get_hf_model_and_tokenizer(
                repo_id=configured_hf_model_id,
                model_subfolder="merged", # Specify the subfolder for model files
                tokenizer_subfolder="merged", # Assuming tokenizer is also in "merged"
                token=HF_HUB_TOKEN_READ
            )
            ft_translations = translate_texts_hf(sample_ko_texts, ft_model, ft_tokenizer, batch_size=2)
            print("Fine-tuned Merged Model Translations:", ft_translations)
            if ft_translations and sample_en_refs:
                ft_bleu = calculate_bleu(ft_translations, sample_en_refs)
                print(f"Fine-tuned Merged Model BLEU: {ft_bleu:.2f}")

            avg_total_time_ft, avg_time_per_text_ft = measure_translation_speed(
                translate_texts_hf, (ft_model, ft_tokenizer), sample_ko_texts * 10,
                num_warmup=1, num_repeats=2, batch_size_override=4
            )
            print(f"Finetuned Model - Avg Total Time ({len(sample_ko_texts*10)} texts): {avg_total_time_ft:.4f}s, Avg Time/Text: {avg_time_per_text_ft:.6f}s")
            del ft_model, ft_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing fine-tuned merged model from {configured_hf_model_id}/merged: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set in config. Skipping fine-tuned merged model test.")

    # 4. Test ONNX Model (from Hub)
    print(f"\n--- Testing ONNX Model ({configured_hf_model_id}/onnx_merged) ---")
    if configured_hf_model_id:
        try:
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"Using ONNX provider: {onnx_provider}")
            
            onnx_model, onnx_tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=configured_hf_model_id,
                onnx_model_subfolder="onnx_merged", # Specify the subfolder for ONNX files
                # tokenizer_base_repo_id will default to model_repo_id
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
            print(f"ONNX Model - Avg Total Time ({len(sample_ko_texts*10)} texts): {avg_total_time_onnx:.4f}s, Avg Time/Text: {avg_time_per_text_onnx:.6f}s")
            del onnx_model, onnx_tokenizer
        except Exception as e:
            print(f"Error testing ONNX model from {configured_hf_model_id}/onnx_merged: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set in config. Skipping ONNX model test.")

    # 5. Test Papago Translation
    # (This part remains the same)

    print("\nTranslation utils example finished.")