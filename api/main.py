from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
import os
from contextlib import asynccontextmanager

from src.config import (
    HF_MODEL_ID, 
    HF_HUB_TOKEN_READ, 
    BASE_MODEL_ID # Fallback or for tokenizer if needed
)
# Prioritize ONNX for API if available and configured
# Fallback to Merged Finetuned model if ONNX is not setup or preferred
USE_ONNX_FOR_API = os.getenv("USE_ONNX_FOR_API", "true").lower() == "true"

# Conditional import for translation utilities
if USE_ONNX_FOR_API:
    try:
        from src.translation_utils import get_onnx_model_and_tokenizer, translate_texts_onnx
        print("API will attempt to use ONNX model.")
    except ImportError:
        print("ONNX utilities not found, or USE_ONNX_FOR_API is false. Falling back to Hugging Face model for API.")
        USE_ONNX_FOR_API = False
        from src.translation_utils import get_hf_model_and_tokenizer, translate_texts_hf
else:
    from src.translation_utils import get_hf_model_and_tokenizer, translate_texts_hf
    print("API will use Hugging Face (PyTorch) model.")


# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "ko" # Default Korean
    target_lang: str = "en" # Default English

class TranslationResponse(BaseModel):
    translated_text: str
    model_type: str

# --- Global variables for Model and Tokenizer ---
# These will be loaded at startup
model_artifacts = {"model": None, "tokenizer": None, "model_type": "None"}

# --- Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and tokenizer at startup
    print("Application startup: Loading translation model...")
    global model_artifacts
    try:
        if not HF_MODEL_ID:
            raise ValueError("HF_MODEL_ID is not set. Cannot load fine-tuned model for API.")

        token = HF_HUB_TOKEN_READ
        
        if USE_ONNX_FOR_API:
            print(f"Loading ONNX model from {HF_MODEL_ID}/onnx_merged for API.")
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            model, tokenizer = get_onnx_model_and_tokenizer(
                HF_MODEL_ID,
                "onnx_merged",
                token=token,
                provider=onnx_provider
            )
            model_artifacts["model_type"] = f"ONNX ({onnx_provider})"
        else:
            # Fallback to merged PyTorch model
            merged_model_id = f"{HF_MODEL_ID}/merged"
            print(f"Loading Merged PyTorch model from {merged_model_id} for API.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = get_hf_model_and_tokenizer(
                merged_model_id,
                token=token,
                device=device
            )
            model_artifacts["model_type"] = f"PyTorch ({device})"

        model_artifacts["model"] = model
        model_artifacts["tokenizer"] = tokenizer
        print(f"Model {model_artifacts['model_type']} loaded successfully for API.")
        
    except Exception as e:
        print(f"CRITICAL: Failed to load model at startup: {e}")
        # Optionally, prevent app startup or run in a degraded mode
        # For now, we'll let it start, but endpoints will fail.
        model_artifacts["model"] = None 
        model_artifacts["tokenizer"] = None
        model_artifacts["model_type"] = f"Error: {e}"
        import traceback
        traceback.print_exc()
    
    yield # Application runs here

    # Clean up resources at shutdown (optional, often handled by OS)
    print("Application shutdown: Cleaning up resources...")
    model_artifacts["model"] = None
    model_artifacts["tokenizer"] = None
    if torch.cuda.is_available() and not USE_ONNX_FOR_API: # Only PyTorch models explicitly moved to CUDA
        torch.cuda.empty_cache()
    print("Resources cleaned.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Korean-to-English Translation API",
    description="Translates Korean text to English using a fine-tuned model.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    if model_artifacts["model"] and model_artifacts["tokenizer"]:
        return {"status": "ok", "message": "Translator API is running", "model_type": model_artifacts["model_type"]}
    else:
        return {"status": "error", "message": "Translator model not loaded", "details": model_artifacts["model_type"]}

# --- Translation Endpoint ---
@app.post("/translate", response_model=TranslationResponse)
async def translate_korean_to_english(request: TranslationRequest):
    if not model_artifacts["model"] or not model_artifacts["tokenizer"]:
        raise HTTPException(status_code=503, detail=f"Model not available: {model_artifacts['model_type']}")

    if request.source_lang != "ko" or request.target_lang != "en":
        raise HTTPException(status_code=400, detail="Currently, only ko -> en translation is supported.")

    try:
        input_text = [request.text] # Expecting a list of texts for utility functions

        if USE_ONNX_FOR_API:
            translations = translate_texts_onnx(
                input_text,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1 # Single text for API endpoint usually
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            translations = translate_texts_hf(
                input_text,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1,
                device=device
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
    # This is for local development. For production, use a process manager like Gunicorn.
    print("Starting Uvicorn server for local development...")
    print(f"API will try to use ONNX: {USE_ONNX_FOR_API}")
    print(f"HF_MODEL_ID: {HF_MODEL_ID}")

    # For Uvicorn to pick up changes during development, use reload.
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # For this script, we'll run it directly without reload for simplicity.
    uvicorn.run(app, host="0.0.0.0", port=8000)