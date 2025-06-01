from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import os
from src.config import BASE_MODEL_ID, HF_MODEL_ID, HF_HUB_TOKEN_READ, LOCAL_ONNX_MODEL_PATH

# Global cache for models and tokenizers
_tokenizer_cache = {}
_model_cache = {}

def get_tokenizer(model_id_or_path: str):
    if model_id_or_path not in _tokenizer_cache:
        print(f"Loading tokenizer for: {model_id_or_path}")
        try:
            _tokenizer_cache[model_id_or_path] = AutoTokenizer.from_pretrained(
                model_id_or_path, token=HF_HUB_TOKEN_READ
            )
        except Exception as e:
            print(f"Failed to load tokenizer {model_id_or_path}: {e}")
            # Fallback to base model tokenizer if fine-tuned one fails (e.g. not pushed correctly)
            if model_id_or_path != BASE_MODEL_ID:
                print(f"Attempting to load base tokenizer: {BASE_MODEL_ID}")
                _tokenizer_cache[model_id_or_path] = AutoTokenizer.from_pretrained(
                    BASE_MODEL_ID, token=HF_HUB_TOKEN_READ
                )
            else:
                raise
        print(f"Tokenizer for {model_id_or_path} loaded.")
    return _tokenizer_cache[model_id_or_path]

def load_base_model():
    model_key = "base_model"
    if model_key not in _model_cache:
        print(f"Loading base model: {BASE_MODEL_ID}")
        tokenizer = get_tokenizer(BASE_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID,
            token=HF_HUB_TOKEN_READ,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        _model_cache[model_key] = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        print("Base model pipeline loaded.")
    return _model_cache[model_key]

def load_finetuned_model(use_qlora_config_for_loading=False):
    model_key = "finetuned_peft_model"
    if model_key not in _model_cache:
        print(f"Loading fine-tuned PEFT model from: {HF_MODEL_ID}")
        tokenizer = get_tokenizer(HF_MODEL_ID) # Tokenizer should be saved with the fine-tuned model

        # Load the base model first
        if use_qlora_config_for_loading and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, # Or 8bit if that was used and preferred for inference
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                token=HF_HUB_TOKEN_READ,
                device_map="auto"
            )
            print(f"Base model {BASE_MODEL_ID} loaded with qLoRA config for PEFT merging.")
        else:
             base_model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL_ID,
                token=HF_HUB_TOKEN_READ,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            print(f"Base model {BASE_MODEL_ID} loaded.")

        # Load the PeftModel by specifying the adapters path from the Hub
        # The adapters are expected to be at the root of HF_MODEL_ID or in an 'adapters' subfolder.
        # Check HF_MODEL_ID repo structure. If adapters are in 'adapters/' subfolder:
        # model_to_load_adapters_from = f"{HF_MODEL_ID}/adapters" # if adapters are in subfolder
        model_to_load_adapters_from = HF_MODEL_ID # if adapters are at the root

        try:
            peft_model = PeftModel.from_pretrained(
                base_model,
                model_to_load_adapters_from, # This should point to where adapter_config.json is
                token=HF_HUB_TOKEN_READ,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            print(f"PEFT adapters loaded from {model_to_load_adapters_from}.")
        except Exception as e:
            print(f"Failed to load PEFT model from {model_to_load_adapters_from}: {e}")
            print("This might happen if adapters are not at the root or not named as expected (e.g. 'adapter_model.bin').")
            print("Ensure 'adapter_config.json' and adapter weights are present in the HF_MODEL_ID repo.")
            # Fallback: Try to load HF_MODEL_ID directly, assuming it might be a fully merged model
            print(f"Attempting to load {HF_MODEL_ID} as a standard AutoModel...")
            try:
                peft_model = AutoModelForSeq2SeqLM.from_pretrained(
                    HF_MODEL_ID,
                    token=HF_HUB_TOKEN_READ,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                print(f"Successfully loaded {HF_MODEL_ID} as a standard AutoModel.")
            except Exception as e2:
                print(f"Failed to load {HF_MODEL_ID} as standard AutoModel: {e2}")
                print("Fine-tuned model could not be loaded. Check your HF_MODEL_ID repo.")
                raise e2
        
        # If you used qLoRA and want to merge for faster inference (and have enough RAM/VRAM)
        # if use_qlora_config_for_loading and hasattr(peft_model, 'merge_and_unload'):
        #     print("Merging PEFT adapters and unloading...")
        #     try:
        #         peft_model = peft_model.merge_and_unload()
        #         print("Adapters merged and unloaded.")
        #     except Exception as e:
        #         print(f"Could not merge and unload LoRA adapters: {e}. Using unmerged model.")

        _model_cache[model_key] = pipeline("translation", model=peft_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        print("Fine-tuned PEFT model pipeline loaded.")
    return _model_cache[model_key]


def load_onnx_model():
    model_key = "onnx_model"
    if model_key not in _model_cache:
        onnx_model_hub_path = f"{HF_MODEL_ID}/onnx" # Assuming ONNX model is in 'onnx' subfolder
        print(f"Loading ONNX model from Hugging Face Hub: {onnx_model_hub_path}")
        
        try:
            # Check if provider options are needed, e.g. for GPU
            provider = "CUDAExecutionProvider" if torch.cuda.is_available() and "onnxruntime-gpu" in os.popen('pip freeze').read() else "CPUExecutionProvider"
            print(f"Using ONNX provider: {provider}")
            
            # ORTModelForSeq2SeqLM.from_pretrained will download from the hub
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                HF_MODEL_ID, # Pass the main model ID
                file_name="onnx/model.onnx", # Specify the path to the ONNX file within the repo
                # Or if tokenizer etc are also in onnx subfolder: onnx_model_hub_path
                token=HF_HUB_TOKEN_READ,
                # provider=provider, # Comment out if causing issues, let it auto-detect
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            tokenizer = get_tokenizer(HF_MODEL_ID) # Use tokenizer from the main fine-tuned model ID
            
            _model_cache[model_key] = pipeline("translation", model=onnx_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            print("ONNX model pipeline loaded.")
        except Exception as e:
            print(f"Failed to load ONNX model from {onnx_model_hub_path}: {e}")
            print("Make sure the ONNX model and its configuration are correctly uploaded to the 'onnx' subfolder of your HF_MODEL_ID repo.")
            print("The subfolder should contain 'model.onnx' (or 'encoder_model.onnx', etc.) and 'config.json'.")
            _model_cache[model_key] = None # Indicate failure
            # raise # Optionally re-raise
    
    if _model_cache.get(model_key) is None:
         print("ONNX model could not be loaded. Returning None.")
    return _model_cache.get(model_key)


if __name__ == "__main__":
    # Example Usage (ensure .env is configured and HF_MODEL_ID exists with models)
    print("Testing model loaders (ensure .env is set and models are on Hub)...")
    
    korean_text = "안녕하세요, 오늘 날씨가 정말 좋네요."
    print(f"\nOriginal Korean: {korean_text}")

    try:
        base_translator = load_base_model()
        if base_translator:
            base_translation = base_translator(korean_text, src_lang="ko", tgt_lang="en")[0]['translation_text']
            print(f"Base Model Translation: {base_translation}")
    except Exception as e:
        print(f"Error loading/using base model: {e}")

    try:
        # Set use_qlora_config_for_loading=True if your base model for PEFT was loaded with BitsAndBytes
        # This might be needed if you didn't merge adapters before pushing.
        # If you pushed a fully merged model or if PeftModel handles it, False might be fine.
        finetuned_translator = load_finetuned_model(use_qlora_config_for_loading=False)
        if finetuned_translator:
            ft_translation = finetuned_translator(korean_text, src_lang="ko", tgt_lang="en")[0]['translation_text']
            print(f"Fine-tuned Model Translation: {ft_translation}")
    except Exception as e:
        print(f"Error loading/using fine-tuned model: {e}")
        import traceback
        traceback.print_exc()


    try:
        onnx_translator = load_onnx_model()
        if onnx_translator:
            onnx_translation = onnx_translator(korean_text, src_lang="ko", tgt_lang="en")[0]['translation_text']
            print(f"ONNX Model Translation: {onnx_translation}")
        else:
            print("ONNX model was not loaded, skipping ONNX translation test.")
    except Exception as e:
        print(f"Error loading/using ONNX model: {e}")
        import traceback
        traceback.print_exc()