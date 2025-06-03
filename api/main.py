# api/main.py
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

# Import correct functions
from src.translation_utils import (
    get_hf_model_and_tokenizer, 
    translate_texts_hf,
    get_onnx_model_and_tokenizer,
    translate_texts_onnx
)

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
        if not HF_MODEL_ID:
            raise ValueError("HF_MODEL_ID is not set. Cannot load fine-tuned model for API.")

        token = HF_HUB_TOKEN_READ
        
        if USE_ONNX_FOR_API:
            print(f"Loading ONNX model from {HF_MODEL_ID}/onnx_merged for API.")
            onnx_provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            # train.py saves ONNX model and its tokenizer in the "onnx_merged" subfolder
            model, tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=HF_MODEL_ID,
                onnx_model_subfolder="onnx_merged",
                token=token,
                provider=onnx_provider
            )
            model_artifacts["model_type"] = f"ONNX ({onnx_provider})"
        else:
            print(f"Loading Merged PyTorch model from {HF_MODEL_ID}/merged for API.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # train.py saves merged model and its tokenizer in the "merged" subfolder
            model, tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,
                model_subfolder="merged",
                tokenizer_repo_id=HF_MODEL_ID, # Tokenizer files are in the same repo
                tokenizer_subfolder="merged",  # and in the "merged" subfolder
                token=token,
                device=device
            )
            model_artifacts["model_type"] = f"PyTorch ({device})"

        model_artifacts["model"] = model
        model_artifacts["tokenizer"] = tokenizer
        print(f"Model {model_artifacts['model_type']} loaded successfully for API.")
        
    except Exception as e:
        print(f"CRITICAL: Failed to load model at startup: {e}")
        model_artifacts["model"] = None 
        model_artifacts["tokenizer"] = None
        model_artifacts["model_type"] = f"Error: {e}"
        import traceback
        traceback.print_exc()
    
    yield 

    print("Application shutdown: Cleaning up resources...")
    model_artifacts["model"] = None
    model_artifacts["tokenizer"] = None
    if torch.cuda.is_available() and model_artifacts["model_type"].startswith("PyTorch"):
        torch.cuda.empty_cache()
    print("Resources cleaned.")

app = FastAPI(
    title="Korean-to-English Translation API",
    description="Translates Korean text to English using a fine-tuned model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
# ... (health check 동일)
async def health_check():
    if model_artifacts["model"] and model_artifacts["tokenizer"]:
        return {"status": "ok", "message": "Translator API is running", "model_type": model_artifacts["model_type"]}
    else:
        return {"status": "error", "message": "Translator model not loaded", "details": model_artifacts["model_type"]}

@app.post("/translate", response_model=TranslationResponse)
# ... (translate 엔드포인트 로직 동일, 내부적으로 수정된 translate_texts_... 함수 사용)
async def translate_korean_to_english(request: TranslationRequest):
    if not model_artifacts["model"] or not model_artifacts["tokenizer"]:
        raise HTTPException(status_code=503, detail=f"Model not available: {model_artifacts['model_type']}")

    if request.source_lang != "ko" or request.target_lang != "en":
        raise HTTPException(status_code=400, detail="Currently, only ko -> en translation is supported.")

    try:
        input_text = [request.text] 

        if USE_ONNX_FOR_API: # model_artifacts["model_type"].startswith("ONNX")
            translations = translate_texts_onnx(
                input_text,
                model_artifacts["model"],
                model_artifacts["tokenizer"],
                batch_size=1 
            )
        else: # Assuming PyTorch model
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
    print("Starting Uvicorn server for local development...")
    print(f"API will try to use ONNX: {USE_ONNX_FOR_API}")
    print(f"HF_MODEL_ID: {HF_MODEL_ID}")
    uvicorn.run(app, host="0.0.0.0", port=8000)