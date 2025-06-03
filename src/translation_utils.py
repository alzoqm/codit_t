import time
import requests
import json
import torch
import os
import onnxruntime as ort

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from huggingface_hub import snapshot_download

from src.config import (
    BASE_MODEL_ID,
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
    base_model_for_peft_id: str = BASE_MODEL_ID,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads a Hugging Face model and tokenizer.
    Handles None for subfolders correctly for from_pretrained.
    """
    actual_tokenizer_repo_id = tokenizer_repo_id if tokenizer_repo_id else repo_id
    
    # Prepare kwargs for tokenizer loading, handling None subfolder
    tokenizer_load_kwargs = {"token": token}
    if tokenizer_subfolder: # Only add subfolder if it's not None or empty
        tokenizer_load_kwargs["subfolder"] = tokenizer_subfolder
    
    print(f"Loading tokenizer from repo: {actual_tokenizer_repo_id}, args: {tokenizer_load_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_repo_id, **tokenizer_load_kwargs)

    if is_peft_adapter:
        print(f"Loading base model for PEFT: {base_model_for_peft_id}")
        # For PEFT, base model config should be loaded from base_model_for_peft_id
        base_config_kwargs = {"token": token}
        base_model_config = AutoConfig.from_pretrained(base_model_for_peft_id, **base_config_kwargs)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_for_peft_id,
            config=base_model_config,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device}
        )
        
        peft_model_load_path = repo_id # Adapters are in this repo
        peft_kwargs = {"token": token, "device_map": {"": device}}
        if model_subfolder: # If adapters are in a subfolder of 'repo_id'
            peft_kwargs["subfolder"] = model_subfolder
        
        print(f"Loading PEFT adapter from repo: {peft_model_load_path}, args: {peft_kwargs}")
        model = PeftModel.from_pretrained(base_model, peft_model_load_path, **peft_kwargs)
        # model = model.merge_and_unload() # Optional: for faster inference
        print("PEFT model loaded.")
    else:
        # Standard model loading (e.g., base model or fully merged fine-tuned model)
        config_load_kwargs = {"token": token}
        if model_subfolder:
            config_load_kwargs["subfolder"] = model_subfolder

        print(f"Loading config from repo: {repo_id}, args: {config_load_kwargs}")
        try:
            # Try to load config from the specified model path (could be a subfolder)
            model_config = AutoConfig.from_pretrained(repo_id, **config_load_kwargs)
            # Ensure model_type is present, especially for Marian models from Helsinki-NLP
            if not hasattr(model_config, 'model_type') or not model_config.model_type:
                if "helsinki-nlp" in repo_id.lower() and (not model_subfolder or model_subfolder == ""): # Base Helsinki model
                    print(f"Manually setting model_type to 'marian' for base Helsinki-NLP model: {repo_id}")
                    model_config.model_type = "marian"
                elif model_subfolder == "merged": # Specific check for our merged models
                     print(f"Warning: Config from {repo_id}/{model_subfolder} is missing model_type. Assuming 'marian' if base was Helsinki.")
                     # This might need to be set during the merge-and-save step in train.py
                     # For now, we can try to proceed or explicitly set it if we know the base type.
                     # If BASE_MODEL_ID is Helsinki, assume merged is also marian
                     if "helsinki-nlp" in BASE_MODEL_ID.lower():
                         model_config.model_type = "marian"


        except Exception as e_conf:
            print(f"Could not load config explicitly from {repo_id} with args {config_load_kwargs}. Error: {e_conf}")
            print("Proceeding to load model, allowing AutoModelForSeq2SeqLM to infer/load config.")
            model_config = None # Let from_pretrained handle config loading

        model_load_kwargs = {
            "token": token,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": {"": device},
        }
        if model_subfolder:
            model_load_kwargs["subfolder"] = model_subfolder
        if model_config: # Pass the explicitly loaded config if available and valid
             model_load_kwargs["config"] = model_config
        
        print(f"Loading standard model from repo: {repo_id}, args: {model_load_kwargs}")
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, **model_load_kwargs)
        print("Standard model loaded.")
    
    model.eval()
    return model, tokenizer

def translate_texts_hf(texts, model, tokenizer, batch_size=16, max_length=128, device=None):
    """Translates a list of texts using a Hugging Face model."""
    translations = []
    # Determine device from model if not specified
    model_device = device if device else next(model.parameters()).device
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
    tokenizer_base_repo_id: str = None,
    token: str = None,
    preferred_provider: str = "CPUExecutionProvider"
):
    """Loads an ONNX model and tokenizer from Hugging Face Hub."""
    actual_tokenizer_base_repo_id = tokenizer_base_repo_id if tokenizer_base_repo_id else model_repo_id
    tokenizer = None
    tokenizer_load_paths_tried = []

    def _try_load_tokenizer(repo_id_to_try, subfolder_name=None):
        nonlocal tokenizer
        nonlocal tokenizer_load_paths_tried
        path_key = f"{repo_id_to_try}" + (f"/{subfolder_name}" if subfolder_name else "/(root)")
        if path_key in tokenizer_load_paths_tried: return False
        tokenizer_load_paths_tried.append(path_key)

        load_kwargs = {"token": token}
        if subfolder_name: # Only add if not None or empty
            load_kwargs["subfolder"] = subfolder_name
        
        print(f"Attempting to load tokenizer from repo: {repo_id_to_try}, args: {load_kwargs}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id_to_try, **load_kwargs)
            print(f"Successfully loaded tokenizer from: {path_key}")
            return True
        except Exception as e:
            print(f"Failed to load tokenizer from {path_key}. Error: {e}")
            return False

    # Try loading tokenizer from various locations
    if not _try_load_tokenizer(actual_tokenizer_base_repo_id, onnx_model_subfolder):
        if not _try_load_tokenizer(actual_tokenizer_base_repo_id, "merged"):
            if not _try_load_tokenizer(actual_tokenizer_base_repo_id): # Root of actual_tokenizer_base_repo_id
                 if not _try_load_tokenizer(BASE_MODEL_ID): # Fallback to global BASE_MODEL_ID
                    raise RuntimeError(f"All attempts to load tokenizer failed. Tried: {tokenizer_load_paths_tried}")
    
    if tokenizer is None:
        raise RuntimeError("Critical: Could not load tokenizer after multiple attempts.")

    available_providers = ort.get_available_providers()
    print(f"Available ONNX Execution Providers: {available_providers}")
    
    final_provider = preferred_provider
    if preferred_provider not in available_providers:
        print(f"Preferred ONNX provider '{preferred_provider}' is not available.")
        if "CPUExecutionProvider" in available_providers:
            print("Falling back to CPUExecutionProvider.")
            final_provider = "CPUExecutionProvider"
        elif available_providers: # If CPU is not there but others are (unlikely for general purpose)
            final_provider = available_providers[0]
            print(f"Falling back to first available provider: {final_provider}")
        else: # No providers at all
            raise RuntimeError("No ONNX Execution Providers available. Please install onnxruntime or onnxruntime-gpu.")
    print(f"Using ONNX provider: {final_provider}")

    print(f"Downloading ONNX model files from Hugging Face Hub repo: {model_repo_id}, subfolder: {onnx_model_subfolder}")
    
    # snapshot_download downloads the entire repo by default unless allow_patterns is very specific.
    # The goal is to get the path to the subfolder containing the ONNX model.
    download_kwargs = {"token": token, "repo_type": "model"}
    # To optimize download, one might use allow_patterns for files inside the onnx_model_subfolder
    # e.g., allow_patterns=[f"{onnx_model_subfolder}/*.onnx", f"{onnx_model_subfolder}/*.json"]
    # However, if config.json for ORTModel is at the root of onnx_model_subfolder, that's fine.

    downloaded_repo_cache_path = snapshot_download(repo_id=model_repo_id, **download_kwargs)
    
    # Construct the full path to the subfolder containing the ONNX model files
    model_load_path = os.path.join(downloaded_repo_cache_path, onnx_model_subfolder)

    if not os.path.exists(model_load_path) or not os.path.isdir(model_load_path):
         raise FileNotFoundError(f"ONNX model directory not found at expected local path: {model_load_path}. "
                                 f"Ensure subfolder '{onnx_model_subfolder}' exists in repo '{model_repo_id}' and contains ONNX model files.")
    if not os.listdir(model_load_path): # Check if directory is empty
        raise FileNotFoundError(f"ONNX model directory '{model_load_path}' is empty. "
                                f"Ensure ONNX files were correctly saved to and pushed to subfolder '{onnx_model_subfolder}' in repo '{model_repo_id}'.")


    print(f"Loading ONNX model from local snapshot directory: {model_load_path}")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_load_path, # This path must point to the directory containing .onnx files and relevant configs
        provider=final_provider
    )
    print("ONNX model loaded.")
    return model, tokenizer

def translate_texts_onnx(texts, model, tokenizer, batch_size=16, max_length=128):
    """Translates a list of texts using an ONNX model."""
    translations = []
    print(f"Translating {len(texts)} texts using ONNX model with batch size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad(): 
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- Papago API Translation ---
def translate_texts_papago(texts, client_id, client_secret, batch_size=1):
    if not client_id or not client_secret:
        print("Papago API credentials not found. Skipping Papago translation.")
        return ["[Papago API not configured.]" * len(texts) for _ in texts] # Corrected list comprehension

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
                translations.append(f"[Papago Error: Unexpected response format {response.status_code}]")
        except requests.exceptions.RequestException as e:
            print(f"Error during Papago API request for text '{text_to_translate}': {e}")
            translations.append(f"[Papago Error: {e}]")
        except json.JSONDecodeError:
            print(f"Error decoding Papago API JSON response for text '{text_to_translate}'. Status: {response.status_code}, Text: {response.text}")
            translations.append(f"[Papago Error: JSON decode error {response.status_code}]")
        time.sleep(0.1) 
    return translations

# --- Evaluation Metrics ---
def calculate_bleu(hypotheses, references):
    if not hypotheses or not references: return 0.0
    if isinstance(references[0], str): references = [[ref] for ref in references]
    bleu_scorer = BLEU() # Renamed to avoid conflict with sacrebleu.BLEU
    score = bleu_scorer.corpus_score(hypotheses, references)
    return score.score

def measure_translation_speed(translation_function, model_data, texts, num_warmup=2, num_repeats=5, batch_size_override=None):
    if not texts: return 0.0, 0.0
    effective_batch_size = batch_size_override if batch_size_override is not None else (16 if model_data else 1)

    print(f"Speed test: Warming up for {num_warmup} iterations with batch size {effective_batch_size}...")
    for _ in range(num_warmup):
        if model_data:
            _ = translation_function(texts[:min(5, len(texts))], model_data[0], model_data[1], batch_size=effective_batch_size)
        else: # Papago
            _ = translation_function(texts[:min(1, len(texts))], PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET) # Papago one by one
    
    start_times = []
    print(f"Speed test: Running {num_repeats} timed iterations for {len(texts)} texts with batch size {effective_batch_size}...")
    for rep_idx in tqdm(range(num_repeats)):
        iter_start_time = time.time()
        if model_data:
            _ = translation_function(texts, model_data[0], model_data[1], batch_size=effective_batch_size)
        else:
            _ = translation_function(texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
        iter_end_time = time.time()
        start_times.append(iter_end_time - iter_start_time)

    avg_total_time = sum(start_times) / num_repeats if num_repeats > 0 else 0
    avg_time_per_text = avg_total_time / len(texts) if texts else 0
    
    print(f"Average total time for {len(texts)} texts over {num_repeats} runs: {avg_total_time:.4f} seconds.")
    print(f"Average time per text: {avg_time_per_text:.6f} seconds.")
    return avg_total_time, avg_time_per_text

if __name__ == '__main__':
    print("Running translation_utils.py example...")
    # Ensure .env is loaded for HF_HUB_TOKEN_READ, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET
    # HF_MODEL_ID should also be available from config or environment for these examples
    
    sample_ko_texts = [
        "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "이것은 번역 모델을 테스트하기 위한 샘플 문장입니다."
    ]
    sample_en_refs = [
        ["Hello, the weather is really nice today."],
        ["This is a sample sentence for testing the translation model."]
    ]
    
    # For example, HF_MODEL_ID needs to be configured (e.g., from src.config or os.getenv)
    # Fallback for testing if not set in environment for this direct script run
    configured_hf_model_id = os.getenv("HF_MODEL_ID", "alzoqm/test_model_2") 
    print(f"Using HF_MODEL_ID for examples: {configured_hf_model_id}")

    # Test Base Model
    print(f"\n--- Testing Base Model ({BASE_MODEL_ID}) ---")
    try:
        base_m, base_t = get_hf_model_and_tokenizer(repo_id=BASE_MODEL_ID, token=HF_HUB_TOKEN_READ)
        base_trans = translate_texts_hf(sample_ko_texts, base_m, base_t)
        print("Base Translations:", base_trans)
        if base_trans: print(f"Base BLEU: {calculate_bleu(base_trans, sample_en_refs):.2f}")
        del base_m, base_t
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e: print(f"Error (Base Model): {e}\n{traceback.format_exc()}")

    # Test Fine-tuned Merged Model
    if configured_hf_model_id:
        print(f"\n--- Testing Fine-tuned Merged Model ({configured_hf_model_id}/merged) ---")
        try:
            ft_m, ft_t = get_hf_model_and_tokenizer(
                repo_id=configured_hf_model_id, 
                model_subfolder="merged", 
                tokenizer_subfolder="merged", # Assuming tokenizer files are also in 'merged'
                token=HF_HUB_TOKEN_READ
            )
            ft_trans = translate_texts_hf(sample_ko_texts, ft_m, ft_t)
            print("FT Merged Translations:", ft_trans)
            if ft_trans: print(f"FT Merged BLEU: {calculate_bleu(ft_trans, sample_en_refs):.2f}")
            del ft_m, ft_t
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e: print(f"Error (FT Merged Model): {e}\n{traceback.format_exc()}")
    else: print("HF_MODEL_ID not configured, skipping FT Merged model test.")

    # Test ONNX Model
    if configured_hf_model_id:
        print(f"\n--- Testing ONNX Model ({configured_hf_model_id}/onnx_merged) ---")
        try:
            onnx_pref_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"Attempting ONNX with preferred provider: {onnx_pref_provider}")
            onnx_m, onnx_t = get_onnx_model_and_tokenizer(
                model_repo_id=configured_hf_model_id,
                onnx_model_subfolder="onnx_merged",
                token=HF_HUB_TOKEN_READ,
                preferred_provider=onnx_pref_provider
            )
            print(f"ONNX model loaded with provider(s): {onnx_m.providers}")
            onnx_trans = translate_texts_onnx(sample_ko_texts, onnx_m, onnx_t)
            print("ONNX Translations:", onnx_trans)
            if onnx_trans: print(f"ONNX BLEU: {calculate_bleu(onnx_trans, sample_en_refs):.2f}")
            del onnx_m, onnx_t
        except Exception as e: print(f"Error (ONNX Model): {e}\n{traceback.format_exc()}")
    else: print("HF_MODEL_ID not configured, skipping ONNX model test.")

    # Test Papago
    if PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET:
        print("\n--- Testing Papago API ---")
        try:
            papago_trans = translate_texts_papago(sample_ko_texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
            print("Papago Translations:", papago_trans)
            valid_papago_trans = [t for t in papago_trans if not t.startswith("[Papago Error")]
            if valid_papago_trans and len(valid_papago_trans) == len(sample_en_refs):
                 print(f"Papago BLEU: {calculate_bleu(valid_papago_trans, sample_en_refs):.2f}")
        except Exception as e: print(f"Error (Papago): {e}\n{traceback.format_exc()}")
    else: print("Papago credentials not set, skipping Papago test.")
    
    print("\nTranslation utils example finished.")