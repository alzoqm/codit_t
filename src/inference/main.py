from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model loaders and Papago translator
from src.inference.model_loader import load_base_model, load_finetuned_model, load_onnx_model
from src.inference.papago_translator import translate_with_papago
from src.config import HF_MODEL_ID # To confirm model loading

app = FastAPI(title="Korean-to-English Translation API")

# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "ko"
    target_lang: str = "en"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    model_type: str
    processing_time_seconds: float

# --- Global variables for models ---
# These will be loaded at startup
base_translator_pipeline = None
finetuned_translator_pipeline = None
onnx_translator_pipeline = None

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global base_translator_pipeline, finetuned_translator_pipeline, onnx_translator_pipeline
    logger.info("FastAPI application startup...")
    
    logger.info("Loading base model...")
    try:
        base_translator_pipeline = load_base_model()
        if base_translator_pipeline:
            logger.info("Base model loaded successfully.")
        else:
            logger.warning("Base model pipeline could not be loaded.")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}", exc_info=True)

    logger.info(f"Loading fine-tuned model ({HF_MODEL_ID})...")
    try:
        # Set use_qlora_config_for_loading based on how your model was trained/saved
        finetuned_translator_pipeline = load_finetuned_model(use_qlora_config_for_loading=False)
        if finetuned_translator_pipeline:
            logger.info("Fine-tuned model loaded successfully.")
        else:
            logger.warning("Fine-tuned model pipeline could not be loaded.")
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}", exc_info=True)

    logger.info(f"Loading ONNX model ({HF_MODEL_ID}/onnx)...")
    try:
        onnx_translator_pipeline = load_onnx_model()
        if onnx_translator_pipeline:
            logger.info("ONNX model loaded successfully.")
        else:
            logger.warning("ONNX model pipeline could not be loaded.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
    
    logger.info("Startup model loading complete.")

# --- API Endpoints ---
@app.post("/translate/base", response_model=TranslationResponse)
async def translate_base_model(request: TranslationRequest):
    if base_translator_pipeline is None:
        raise HTTPException(status_code=503, detail="Base model is not available.")
    
    start_time = time.time()
    try:
        # The pipeline expects src_lang and tgt_lang if model is multilingual
        result = base_translator_pipeline(
            request.text,
            src_lang=request.source_lang,
            tgt_lang=request.target_lang
        )
        translated_text = result[0]['translation_text']
    except Exception as e:
        logger.error(f"Error during base model translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation error with base model: {str(e)}")
    
    end_time = time.time()
    return TranslationResponse(
        original_text=request.text,
        translated_text=translated_text,
        model_type="base_pretrained",
        processing_time_seconds=round(end_time - start_time, 4)
    )

@app.post("/translate/finetuned", response_model=TranslationResponse)
async def translate_finetuned_model(request: TranslationRequest):
    if finetuned_translator_pipeline is None:
        raise HTTPException(status_code=503, detail="Fine-tuned model is not available.")
    
    start_time = time.time()
    try:
        result = finetuned_translator_pipeline(
            request.text,
            src_lang=request.source_lang,
            tgt_lang=request.target_lang
        )
        translated_text = result[0]['translation_text']
    except Exception as e:
        logger.error(f"Error during fine-tuned model translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation error with fine-tuned model: {str(e)}")
        
    end_time = time.time()
    return TranslationResponse(
        original_text=request.text,
        translated_text=translated_text,
        model_type="fine_tuned_peft",
        processing_time_seconds=round(end_time - start_time, 4)
    )

@app.post("/translate/onnx", response_model=TranslationResponse)
async def translate_onnx_model(request: TranslationRequest):
    if onnx_translator_pipeline is None:
        raise HTTPException(status_code=503, detail="ONNX model is not available.")
        
    start_time = time.time()
    try:
        result = onnx_translator_pipeline(
            request.text,
            src_lang=request.source_lang,
            tgt_lang=request.target_lang
        )
        translated_text = result[0]['translation_text']
    except Exception as e:
        logger.error(f"Error during ONNX model translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation error with ONNX model: {str(e)}")
        
    end_time = time.time()
    return TranslationResponse(
        original_text=request.text,
        translated_text=translated_text,
        model_type="fine_tuned_onnx",
        processing_time_seconds=round(end_time - start_time, 4)
    )

@app.post("/translate/papago", response_model=TranslationResponse)
async def translate_papago_external(request: TranslationRequest):
    start_time = time.time()
    translated_text = translate_with_papago(request.text, request.source_lang, request.target_lang)
    if translated_text is None:
        raise HTTPException(status_code=500, detail="Papago translation failed or credentials missing.")
    end_time = time.time()
    return TranslationResponse(
        original_text=request.text,
        translated_text=translated_text,
        model_type="papago_api",
        processing_time_seconds=round(end_time - start_time, 4)
    )

@app.get("/")
async def root():
    return {
        "message": "Translation API is running.",
        "available_models": {
            "base_model_loaded": base_translator_pipeline is not None,
            "finetuned_model_loaded": finetuned_translator_pipeline is not None,
            "onnx_model_loaded": onnx_translator_pipeline is not None,
            "papago_available": True # Assuming Papago credentials might be there
        },
        "endpoints": [
            "/translate/base",
            "/translate/finetuned",
            "/translate/onnx",
            "/translate/papago",
            "/docs"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # This is for local development. For production, use a process manager like Gunicorn.
    # Ensure .env is readable from where this script is run (e.g., project root)
    # uvicorn.run("src.inference.main:app", host="0.0.0.0", port=8000, reload=True)
    # If running directly, make sure PYTHONPATH includes the project root or use `python -m src.inference.main`
    logger.warning("Running Uvicorn directly. For production, use a command like:")
    logger.warning("uvicorn src.inference.main:app --host 0.0.0.0 --port 8000")
    # uvicorn.run(app, host="0.0.0.0", port=8000) # Simpler call if app is defined in this scope