from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
import os
from contextlib import asynccontextmanager

from src.config import (
    HF_MODEL_ID, 
    HF_HUB_TOKEN_READ, 
    BASE_MODEL_ID 
)
USE_ONNX_FOR_API = os.getenv("USE_ONNX_FOR_API", "true").lower() == "true"

# Conditional import for translation utilities
if USE_ONNX_FOR_API:
    try:
        from src.translation_utils import get_onnx_model_and_tokenizer, translate_texts_onnx
        print("API will attempt to use ONNX model.")
    except ImportError: # Should not happen if file is correct
        print("ONNX utilities not found. Falling back to Hugging Face model for API.")
        USE_ONNX_FOR_API = False
        from src.translation_utils import get_hf_model_and_tokenizer, translate_texts_hf
else:
    from src.translation_utils import get_hf_model_and_tokenizer, translate_texts_hf
    print("API will use Hugging Face (PyTorch) model.")


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "ko"
    target_lang: str = "en"

class TranslationResponse(BaseModel):
    translated_text: str
    model_type: str

model_artifacts = {"model": None, "tokenizer": None, "model_type": "None"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Loading translation model...")
    global model_artifacts
    try:
        if not HF_MODEL_ID: # HF_MODEL_ID from src.config
            raise ValueError("HF_MODEL_ID is not set. Cannot load fine-tuned model for API.")

        token = HF_HUB_TOKEN_READ # From src.config
        
        if USE_ONNX_FOR_API:
            print(f"Loading ONNX model from repo {HF_MODEL_ID}, subfolder onnx_merged for API.")
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            model, tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=HF_MODEL_ID,
                onnx_model_subfolder="onnx_merged",
                # tokenizer_base_repo_id will default to HF_MODEL_ID
                token=token,
                provider=onnx_provider
            )
            model_artifacts["model_type"] = f"ONNX ({onnx_provider})"
        else:
            print(f"Loading Merged PyTorch model from repo {HF_MODEL_ID}, subfolder merged for API.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,
                model_subfolder="merged",
                tokenizer_subfolder="merged", # Assuming tokenizer is also in "merged"
                token=token,
                device=device
            )
            model_artifacts["model_type"] = f"PyTorch ({device})"

        model_artifacts["model"] = model
        model_artifacts["tokenizer"] = tokenizer
        print(f"Model {model_artifacts['model_type']} loaded successfully for API.")
        
    except Exception as e:
        print(f"CRITICAL: Failed to load model at startup: {e}")
        model_artifacts["model_type"] = f"Error: {e}"
        import traceback
        traceback.print_exc()
    
    yield 

    print("Application shutdown: Cleaning up resources...")
    model_artifacts["model"] = None
    model_artifacts["tokenizer"] = None
    if torch.cuda.is_available() and not (USE_ONNX_FOR_API and model_artifacts["model_type"].startswith("ONNX (CUDAExecutionProvider)")):
        torch.cuda.empty_cache()
    print("Resources cleaned.")

app = FastAPI(
    title="Korean-to-English Translation API",
    description="Translates Korean text to English using a fine-tuned model.",
    version="1.0.0",
    lifespan=lifespan
)

# ... (rest of api/main.py remains the same) ...
@app.get("/health")
async def health_check():
    if model_artifacts["model"] and model_artifacts["tokenizer"]:
        return {"status": "ok", "message": "Translator API is running", "model_type": model_artifacts["model_type"]}
    else:
        # Provide more detail if model loading failed
        error_detail = model_artifacts["model_type"] if "Error" in model_artifacts["model_type"] else "Model not loaded"
        raise HTTPException(status_code=503, detail=f"Translator model not available. Status: {error_detail}")

@app.post("/translate", response_model=TranslationResponse)
async def translate_korean_to_english(request: TranslationRequest):
    if not model_artifacts["model"] or not model_artifacts["tokenizer"]:
        error_detail = model_artifacts["model_type"] if "Error" in model_artifacts["model_type"] else "Model not available"
        raise HTTPException(status_code=503, detail=f"Model not available: {error_detail}")

    if request.source_lang != "ko" or request.target_lang != "en":
        raise HTTPException(status_code=400, detail="Currently, only ko -> en translation is supported.")

    try:
        input_text = [request.text] 

        if USE_ONNX_FOR_API:
            translations = translate_texts_onnx(
                input_text,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1
            )
        else:
            # Determine device from loaded model if possible, or default
            model_device = next(model_artifacts["model"].parameters()).device if hasattr(model_artifacts["model"], 'parameters') else ("cuda" if torch.cuda.is_available() else "cpu")
            translations = translate_texts_hf(
                input_text,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1,
                device=str(model_device) # Ensure device is a string
            )
        
        translated_text = translations[0] if translations else "Error in translation"
        
        return TranslationResponse(translated_text=translated_text, model_type=model_artifacts["model_type"])

    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for local development...")
    print(f"API will try to use ONNX: {USE_ONNX_FOR_API}")
    print(f"HF_MODEL_ID from config: {HF_MODEL_ID}")
    uvicorn.run(app, host="0.0.0.0", port=8000)