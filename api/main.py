from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import traceback
from contextlib import asynccontextmanager

from src.config import (
    HF_MODEL_ID, 
    HF_HUB_TOKEN_READ, 
    BASE_MODEL_ID # Used by translation_utils as fallback for tokenizer if needed
)
# Determine if API should use ONNX model based on environment variable
USE_ONNX_FOR_API = os.getenv("USE_ONNX_FOR_API", "true").lower() == "true"

# Dynamically import the correct utilities based on USE_ONNX_FOR_API
if USE_ONNX_FOR_API:
    try:
        from src.translation_utils import get_onnx_model_and_tokenizer, translate_texts_onnx
        print("API will attempt to use ONNX model.")
    except ImportError as e:
        print(f"Failed to import ONNX utilities: {e}. API cannot start with ONNX.")
        # Depending on desired behavior, could raise error or force fallback
        raise RuntimeError(f"API configured to use ONNX but failed to import utilities: {e}") from e
else:
    try:
        from src.translation_utils import get_hf_model_and_tokenizer, translate_texts_hf
        print("API will use Hugging Face (PyTorch) model.")
    except ImportError as e:
        print(f"Failed to import PyTorch utilities: {e}. API cannot start.")
        raise RuntimeError(f"API configured to use PyTorch but failed to import utilities: {e}") from e

# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "ko"
    target_lang: str = "en"

class TranslationResponse(BaseModel):
    translated_text: str
    model_type_used: str # Renamed for clarity

# --- Global variables for Model and Tokenizer ---
# These will be loaded at startup via the lifespan manager
model_artifacts = {"model": None, "tokenizer": None, "model_type_loaded": "None (Not Loaded)"}

# --- Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Loading translation model...")
    global model_artifacts # Allow modification of the global dict
    try:
        if not HF_MODEL_ID:
            raise ValueError("HF_MODEL_ID environment variable is not set. Cannot determine which fine-tuned model to load for the API.")

        token_to_use = HF_HUB_TOKEN_READ
        
        if USE_ONNX_FOR_API:
            # Determine preferred ONNX provider (e.g., CUDA if available, else CPU)
            onnx_api_preferred_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            print(f"Attempting to load ONNX model from repo: {HF_MODEL_ID}, subfolder: onnx_merged. Preferred provider: {onnx_api_preferred_provider}")
            
            model, tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=HF_MODEL_ID,
                onnx_model_subfolder="onnx_merged",
                tokenizer_base_repo_id=HF_MODEL_ID, # Assuming tokenizer is related to this model repo
                token=token_to_use,
                preferred_provider=onnx_api_preferred_provider
            )
            # Get the actual provider that ONNX Runtime decided to use
            actual_provider_in_use = model.providers[0] if hasattr(model, 'providers') and model.providers else "Unknown ONNX Provider"
            model_artifacts["model_type_loaded"] = f"ONNX ({actual_provider_in_use})"
        else:
            # Fallback or explicit choice to use merged PyTorch model
            pytorch_api_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Attempting to load Merged PyTorch model from repo: {HF_MODEL_ID}, subfolder: merged. Device: {pytorch_api_device}")
            
            model, tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,
                model_subfolder="merged",
                tokenizer_repo_id=HF_MODEL_ID, # Assuming tokenizer is with the merged model
                tokenizer_subfolder="merged",
                token=token_to_use,
                device=pytorch_api_device
            )
            model_artifacts["model_type_loaded"] = f"PyTorch ({pytorch_api_device})"

        model_artifacts["model"] = model
        model_artifacts["tokenizer"] = tokenizer
        print(f"Model '{model_artifacts['model_type_loaded']}' loaded successfully for API.")
        
    except Exception as e:
        error_msg = f"CRITICAL: Failed to load model at startup: {e}"
        print(error_msg)
        print(traceback.format_exc())
        model_artifacts["model_type_loaded"] = f"Error during model loading: {e}" # Store error for health check
        # Optionally, raise the error to prevent FastAPI from starting if model is critical
        # raise RuntimeError(error_msg) from e 
    
    yield # Application runs after this point

    # Clean up resources at shutdown (optional, often handled by OS or container orchestrator)
    print("Application shutdown: Cleaning up model resources...")
    model_artifacts["model"] = None
    model_artifacts["tokenizer"] = None
    if torch.cuda.is_available():
        # Clear cache if PyTorch was using CUDA and not an ONNX CUDA model (which manages its own resources)
        if not (USE_ONNX_FOR_API and "CUDAExecutionProvider" in model_artifacts["model_type_loaded"]):
            torch.cuda.empty_cache()
            print("PyTorch CUDA cache cleared.")
    print("Resources cleaned.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Korean-to-English Translation API",
    description="Translates Korean text to English using a fine-tuned model (ONNX or PyTorch).",
    version="1.0.1", # Incremented version
    lifespan=lifespan # Use the lifespan context manager for startup/shutdown logic
)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    if model_artifacts["model"] and model_artifacts["tokenizer"]:
        return {
            "status": "ok",
            "message": "Translator API is running and model is loaded.",
            "model_type_loaded": model_artifacts["model_type_loaded"]
        }
    else:
        # If model loading failed, model_type_loaded will contain the error message.
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Translator model is not available. Last status: {model_artifacts['model_type_loaded']}"
        )

# --- Translation Endpoint ---
@app.post("/translate", response_model=TranslationResponse)
async def translate_korean_to_english(request: TranslationRequest):
    if not model_artifacts["model"] or not model_artifacts["tokenizer"]:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Model not available for translation. Status: {model_artifacts['model_type_loaded']}"
        )

    if request.source_lang != "ko" or request.target_lang != "en":
        raise HTTPException(status_code=400, detail="Invalid language pair. Currently, only 'ko' to 'en' translation is supported.")

    try:
        input_text_list = [request.text] # Utility functions expect a list

        if USE_ONNX_FOR_API:
            # ONNX model translation
            translations = translate_texts_onnx(
                input_text_list,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1 # Typically, API requests handle one item at a time
            )
        else:
            # PyTorch model translation
            # Determine device from loaded PyTorch model, if applicable
            pytorch_model_device = "cpu" # Default
            if hasattr(model_artifacts["model"], 'device'):
                pytorch_model_device = str(model_artifacts["model"].device)
            elif hasattr(model_artifacts["model"], 'parameters'):
                try:
                    pytorch_model_device = str(next(model_artifacts["model"].parameters()).device)
                except StopIteration: # Model has no parameters
                    pass

            translations = translate_texts_hf(
                input_text_list,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1,
                device=pytorch_model_device 
            )
        
        translated_text_result = translations[0] if translations else "Error: No translation result."
        
        return TranslationResponse(
            translated_text=translated_text_result,
            model_type_used=model_artifacts["model_type_loaded"] # Report the model type that was loaded
        )

    except Exception as e:
        print(f"Error during translation request: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error during translation: {e}")

if __name__ == "__main__":
    import uvicorn
    # This is for local development. For production, use a process manager like Gunicorn + Uvicorn workers.
    print("Starting Uvicorn server for local development...")
    print(f"API Configuration - USE_ONNX_FOR_API: {USE_ONNX_FOR_API}")
    print(f"API Configuration - HF_MODEL_ID (from config): {HF_MODEL_ID}")
    
    # For Uvicorn to pick up changes during development, use `reload=True`.
    # For this script, running directly:
    uvicorn.run(app, host="0.0.0.0", port=8000)