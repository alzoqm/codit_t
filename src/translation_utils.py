# src/translation_utils.py

import time
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PeftModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from sacrebleu.metrics import BLEU
from datasets import Dataset
from tqdm import tqdm
import os # For os.path.join
from huggingface_hub import snapshot_download # For ONNX model download

from src.config import (
    BASE_MODEL_ID,
    HF_MODEL_ID, # This is the main repo_id for fine-tuned models
    HF_HUB_TOKEN_READ,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET
)

# --- Hugging Face Model Translation ---

def get_hf_model_and_tokenizer(
    repo_id: str,
    model_subfolder: str = None,
    tokenizer_repo_id: str = None, # Use if tokenizer is in a different repo
    tokenizer_subfolder: str = None, # Use if tokenizer is in a subfolder of tokenizer_repo_id
    token: str = None,
    is_peft_adapter: bool = False, # True if loading LoRA adapters that need to be merged
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Loads a Hugging Face model and tokenizer, correctly handling subfolders."""

    actual_tokenizer_load_repo = tokenizer_repo_id if tokenizer_repo_id else repo_id
    actual_tokenizer_load_subfolder = tokenizer_subfolder

    tokenizer_display_path = f"{actual_tokenizer_load_repo}"
    if actual_tokenizer_load_subfolder:
        tokenizer_display_path += f" (subfolder: {actual_tokenizer_load_subfolder})"
    print(f"Loading tokenizer from: {tokenizer_display_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        actual_tokenizer_load_repo,
        subfolder=actual_tokenizer_load_subfolder,
        token=token
    )

    model_display_path = f"{repo_id}"
    if model_subfolder:
        model_display_path += f" (subfolder: {model_subfolder})"
    print(f"Loading model from: {model_display_path}")

    if is_peft_adapter:
        # For PEFT, 'repo_id' is the Hub location of the adapters (e.g., HF_MODEL_ID),
        # and 'model_subfolder' is where adapter files are (e.g., "peft_lora").
        # The base model (BASE_MODEL_ID) is loaded first.
        print(f"Loading base model for PEFT: {BASE_MODEL_ID}")
        base_model_for_peft = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID,
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, # float16 for GPU
            low_cpu_mem_usage=True
        ).to(device)
        
        print(f"Loading PEFT adapter from repo: {repo_id}, subfolder: {model_subfolder}")
        # repo_id here is HF_MODEL_ID (where adapters were pushed)
        model = PeftModel.from_pretrained(
            base_model_for_peft,
            model_id=repo_id, # The repo where adapters are stored
            subfolder=model_subfolder, # Subfolder within 'repo_id' for adapters
            token=token
        ).to(device)
        print("Merging PEFT adapters...")
        model = model.merge_and_unload() 
        print("PEFT model loaded and merged.")
    else:
        # For standard models or merged models residing in a subfolder of 'repo_id'
        model = AutoModelForSeq2SeqLM.from_pretrained(
            repo_id,
            subfolder=model_subfolder, # e.g., "merged"
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        print("Standard or Merged model loaded.")
    
    model.eval() # Set to evaluation mode
    return model, tokenizer

def translate_texts_hf(texts, model, tokenizer, batch_size=16, max_length=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Translates a list of texts using a Hugging Face model."""
    translations = []
    # Ensure model is on the correct device, inputs will be moved by .to(device)
    model.to(device)
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

def get_onnx_model_and_tokenizer(
    model_repo_id: str,       # e.g., "alzoqm/test_model_2"
    onnx_model_subfolder: str,# e.g., "onnx_merged"
    token: str = None,
    provider: str = "CPUExecutionProvider"
):
    """Loads an ONNX model and its tokenizer from Hugging Face Hub."""
    
    tokenizer = None
    # Try loading tokenizer from the specific ONNX subfolder first.
    # (train.py saves tokenizer files alongside onnx model files)
    print(f"Attempting to load tokenizer from: {model_repo_id} (subfolder: {onnx_model_subfolder})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_repo_id,
            subfolder=onnx_model_subfolder,
            token=token
        )
        print(f"Loaded tokenizer from ONNX subfolder: {model_repo_id}/{onnx_model_subfolder}")
    except Exception as e1:
        print(f"Tokenizer not found in {model_repo_id}/{onnx_model_subfolder} (Error: {e1}).")
        # Fallback to tokenizer from the "merged" subfolder
        # (train.py also saves tokenizer with merged model)
        merged_subfolder = "merged"
        print(f"Attempting to load tokenizer from: {model_repo_id} (subfolder: {merged_subfolder})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_repo_id,
                subfolder=merged_subfolder,
                token=token
            )
            print(f"Loaded tokenizer from merged subfolder: {model_repo_id}/{merged_subfolder}")
        except Exception as e2:
            print(f"Tokenizer not found in {model_repo_id}/{merged_subfolder} (Error: {e2}).")
            # Fallback to base model tokenizer if still not found
            print(f"Attempting to load tokenizer from base model: {BASE_MODEL_ID}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)
                print(f"Loaded tokenizer from base model: {BASE_MODEL_ID}")
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load tokenizer from {model_repo_id}/{onnx_model_subfolder}, "
                    f"nor from {model_repo_id}/{merged_subfolder}, nor from {BASE_MODEL_ID}. Errors: E1({e1}), E2({e2}), E3({e3})"
                )

    if not tokenizer: # Should be caught by the RuntimeError above, but as a safeguard.
        raise RuntimeError("Critical: Tokenizer could not be loaded.")

    print(f"Downloading ONNX model files from repo {model_repo_id}, subfolder {onnx_model_subfolder}...")
    # snapshot_download downloads the repo (or specified parts) and returns the path to the snapshot's root.
    # Files from the subfolder will be within that root at their relative path.
    snapshot_root_dir = snapshot_download(
        repo_id=model_repo_id,
        allow_patterns=[f"{onnx_model_subfolder}/*"], # Ensure this pattern correctly grabs all necessary files
        token=token,
        repo_type="model"
    )
    # The actual path to the ONNX model files will be snapshot_root_dir + onnx_model_subfolder
    model_load_path = os.path.join(snapshot_root_dir, onnx_model_subfolder)

    if not os.path.exists(os.path.join(model_load_path, "model.onnx")): # Check if main onnx file exists
        print(f"Warning: model.onnx not found at {os.path.join(model_load_path, 'model.onnx')}. Listing {model_load_path}: {os.listdir(model_load_path) if os.path.exists(model_load_path) else 'Path does not exist'}")
        # This could indicate an issue with how files were uploaded or how snapshot_download + subfolder pathing works.
        # If files are directly in snapshot_root_dir (e.g. if onnx_model_subfolder was 'root' or similar logic),
        # model_load_path might need to be just snapshot_root_dir.
        # For now, assume os.path.join is correct based on standard Hub behavior.
        # One common pattern for snapshot_download is that it might place the files matching the pattern
        # directly into the snapshot root if the pattern itself specifies the "directory".
        # Let's try checking if model_load_path exists and contains the model.
        if not os.path.isdir(model_load_path):
             raise FileNotFoundError(f"ONNX model directory {model_load_path} not found after snapshot_download. Snapshot root: {snapshot_root_dir}")


    print(f"Loading ONNX model from local path: {model_load_path} using provider: {provider}")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_load_path, # This must be the directory containing model.onnx, config.json etc.
        provider=provider,
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
        with torch.no_grad(): 
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        batch_translations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        translations.extend(batch_translations)
    return translations

# --- Papago API Translation --- (No changes needed for Papago part)
def translate_texts_papago(texts, client_id, client_secret, batch_size=1):
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

# --- Evaluation Metrics --- (No changes needed for calculate_bleu, measure_translation_speed)
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
    :return: Average total time for all texts, average time per text.
    """
    if not texts:
        return 0.0, 0.0

    print(f"Speed test: Warming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        if model_data: 
            _ = translation_function(texts[:5], model_data[0], model_data[1], batch_size=batch_size_override or 16)
        else: 
            _ = translation_function(texts[:1], PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)

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

    print("\n--- Testing Base Model (Helsinki-NLP/opus-mt-ko-en) ---")
    try:
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
        print(f"Base Model - Avg Total Time (for {len(sample_ko_texts*10)} texts): {avg_total_time:.4f}s, Avg Time/Text: {avg_time_per_text:.6f}s")
        del base_model, base_tokenizer 
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error testing base model: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n--- Testing Fine-tuned Merged Model ({HF_MODEL_ID}/merged) ---")
    if HF_MODEL_ID:
        try:
            # train.py saves tokenizer for merged model in the "merged" subfolder.
            ft_model, ft_tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,
                model_subfolder="merged",
                tokenizer_repo_id=HF_MODEL_ID, # Tokenizer is in the same repo
                tokenizer_subfolder="merged",  # In the 'merged' subfolder
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
            print(f"Finetuned Model - Avg Total Time (for {len(sample_ko_texts*10)} texts): {avg_total_time_ft:.4f}s, Avg Time/Text: {avg_time_per_text_ft:.6f}s")
            del ft_model, ft_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing fine-tuned merged model from {HF_MODEL_ID}/merged: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping fine-tuned merged model test.")

    print(f"\n--- Testing ONNX Model ({HF_MODEL_ID}/onnx_merged) ---")
    if HF_MODEL_ID:
        try:
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"Using ONNX provider: {onnx_provider}")
            
            onnx_model, onnx_tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=HF_MODEL_ID, 
                onnx_model_subfolder="onnx_merged", 
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
        except Exception as e:
            print(f"Error testing ONNX model from {HF_MODEL_ID}/onnx_merged: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping ONNX model test.")

    print("\n--- Testing Papago API Translation ---")
    # ... (Papago part remains the same) ...
    if PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET:
        try:
            papago_translations = translate_texts_papago(sample_ko_texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
            print("Papago Translations:", papago_translations)
            valid_papago_translations = [t for t in papago_translations if not t.startswith("[Papago Error")]
            if valid_papago_translations and len(valid_papago_translations) == len(sample_en_refs):
                 papago_bleu = calculate_bleu(valid_papago_translations, sample_en_refs)
                 print(f"Papago BLEU: {papago_bleu:.2f}")
            elif valid_papago_translations:
                 print(f"Could not calculate Papago BLEU due to partial success (got {len(valid_papago_translations)} translations for {len(sample_en_refs)} refs).")

            avg_total_time_papago, avg_time_per_text_papago = measure_translation_speed(
                translate_texts_papago, None, sample_ko_texts[:3], 
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