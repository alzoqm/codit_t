import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel # Added PeftModel
from huggingface_hub import HfApi, login
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os
import shutil

from src.config import (
    BASE_MODEL_ID, HF_MODEL_ID, HF_HUB_TOKEN_WRITE, HF_HUB_TOKEN_READ,
    NUM_TRAIN_EPOCHS, LEARNING_RATE, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, OUTPUT_DIR,
    QLORA_R, QLORA_ALPHA, QLORA_DROPOUT,
    LOCAL_ADAPTERS_PATH, LOCAL_ONNX_MODEL_PATH, MAX_SAMPLES_DATASET
)
from src.data_load import load_pre_split_data
from src.preprocess import tokenize_datasets

def train():
    print("Starting training process...")

    if not HF_HUB_TOKEN_WRITE:
        raise ValueError("HF_HUB_TOKEN_WRITE is not set. Cannot push to Hub.")
    if not HF_MODEL_ID:
        raise ValueError("HF_MODEL_ID is not set. Cannot determine push location.")

    # Login to Hugging Face Hub
    print(f"Logging in to Hugging Face Hub with write token.")
    login(token=HF_HUB_TOKEN_WRITE)

    # 1. Load and preprocess data
    raw_datasets = load_pre_split_data(max_samples_per_split=MAX_SAMPLES_DATASET)
    tokenized_datasets, tokenizer = tokenize_datasets(raw_datasets, BASE_MODEL_ID)

    # 2. Configure BitsAndBytes for qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16 if bfloat16 not supported
    )

    # 3. Load base model with quantization
    print(f"Loading base model: {BASE_MODEL_ID} with qLoRA quantization...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" # Automatically distribute model on available GPUs
    )
    print("Base model loaded.")

    # 4. Configure LoRA
    # Common target modules for MarianMT/T5-like models. Adjust if necessary for other architectures.
    # For Helsinki-NLP/opus-mt models (MarianMT based), 'q_proj', 'k_proj', 'v_proj', 'out_proj' are in encoder/decoder attention.
    # 'fc1', 'fc2' are in the FFN layers.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"] # Check model's named parameters for exact matches
    )

    # 5. Apply PEFT to the model
    print("Applying LoRA to the model...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("LoRA applied.")

    # 6. Training Arguments
    # Calculate logging_steps, eval_steps, save_steps to be roughly proportional to dataset size/batch size
    # Aim for logging ~20 times per epoch, eval/save once per epoch or more frequently if epochs are long.
    if tokenized_datasets["train"]:
        steps_per_epoch = len(tokenized_datasets["train"]) // (PER_DEVICE_TRAIN_BATCH_SIZE * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1))
        logging_steps_val = max(1, steps_per_epoch // 20) # Log ~20 times per epoch
    else:
        steps_per_epoch = 200 # Default if no train data (should not happen)
        logging_steps_val = 10


    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=logging_steps_val,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        fp16=False, 
        bf16=True if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported() else False,
        remove_unused_columns=True, 
        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="every_save", 
        hub_token=HF_HUB_TOKEN_WRITE,
        report_to="tensorboard", 
    )

    # 7. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=peft_model, # Pass peft_model here
        label_pad_token_id=tokenizer.pad_token_id
    )

    # 8. Trainer
    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Train
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # 10. Save final adapter and push everything
    print(f"Saving final PEFT adapter to local path: {LOCAL_ADAPTERS_PATH}")
    # The peft_model here is the one that was trained by the Trainer.
    # If load_best_model_at_end=True, trainer.model should be the best one.
    # Saving it directly:
    trainer.model.save_pretrained(LOCAL_ADAPTERS_PATH) 
    
    print("Pushing final model (trainer might have done this) and tokenizer to Hugging Face Hub...")
    # The trainer already pushes adapters with `push_to_hub=True`.
    # This ensures the tokenizer and any other files created by trainer are also pushed.
    # The commit message here will be for the final push after training.
    trainer.push_to_hub(commit_message="End of training: final adapters and tokenizer")
    
    # Explicitly upload the adapter folder to a specific subfolder "peft_lora"
    # This is based on your example of wanting adapters in a specific subfolder.
    # Trainer normally pushes adapters to the root of HF_MODEL_ID.
    api = HfApi()
    print(f"Uploading PEFT adapters from '{LOCAL_ADAPTERS_PATH}' to '{HF_MODEL_ID}/peft_lora' on the Hub...")
    api.upload_folder(
        folder_path=LOCAL_ADAPTERS_PATH,
        path_in_repo="peft_lora", # Store adapters in 'peft_lora' subfolder
        repo_id=HF_MODEL_ID,
        repo_type="model",
        token=HF_HUB_TOKEN_WRITE,
        commit_message="Upload PEFT adapters to peft_lora subfolder"
    )
    print(f"PEFT adapters uploaded to {HF_MODEL_ID}/peft_lora on the Hub.")
    
    # --- Start of LoRA Merging and ONNX Conversion ---
    print("\nAttempting to merge LoRA adapters and convert to ONNX...")
    try:
        # 11.A Merge LoRA adapters
        print("Loading base model for merging (this time without 4-bit quantization)...")
        # Ensure enough RAM/VRAM. Use torch_dtype for precision control.
        # device_map="auto" is usually fine, but "cpu" can be forced if memory is very tight for the full model.
        base_model_for_merging = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True, # Good practice for large models
            device_map="auto", # Or "cpu" to force CPU if VRAM is a concern
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE # Use read or write token
        )
        print(f"Base model ({BASE_MODEL_ID}) loaded for merging.")

        # Load the PEFT model with adapters. Using LOCAL_ADAPTERS_PATH is safest here.
        print(f"Loading PEFT adapters from local path: {LOCAL_ADAPTERS_PATH}")
        # The model object from trainer `trainer.model` is already the PeftModel.
        # If we want to load from disk to be sure:
        # model_with_adapters = PeftModel.from_pretrained(
        #     base_model_for_merging, # The newly loaded base model
        #     LOCAL_ADAPTERS_PATH,    # Path to the saved adapters
        #     token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE,
        #     device_map = "auto" # ensure it's on the same device setup
        # )
        # However, it's more direct to merge from the `trainer.model` IF it's already on a suitable device
        # and if its base model can be "detached" or replaced.
        # Safest is to load adapters onto the fresh base_model_for_merging:
        
        if not os.path.exists(os.path.join(LOCAL_ADAPTERS_PATH, "adapter_config.json")):
             raise FileNotFoundError(f"Adapter config not found at {LOCAL_ADAPTERS_PATH}. Cannot proceed with merging.")

        model_with_adapters = PeftModel.from_pretrained(
            base_model_for_merging, # Fresh base model
            LOCAL_ADAPTERS_PATH,    # Saved adapters
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE
            # is_trainable must be False if you don't plan to train further after loading
        )
        print(f"PEFT adapters loaded from {LOCAL_ADAPTERS_PATH} onto the new base model instance.")

        print("Merging LoRA adapters into the base model...")
        merged_model = model_with_adapters.merge_and_unload()
        print("Adapters merged successfully.")

        # Optional: Save and push the merged model to Hugging Face Hub
        # This model can be used directly without PEFT.
        LOCAL_MERGED_MODEL_PATH = os.path.join(OUTPUT_DIR, "merged_model_artifacts")
        print(f"Saving merged model locally to: {LOCAL_MERGED_MODEL_PATH}")
        os.makedirs(LOCAL_MERGED_MODEL_PATH, exist_ok=True)
        merged_model.save_pretrained(LOCAL_MERGED_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_MERGED_MODEL_PATH) # Save tokenizer with merged model
        print("Merged model and tokenizer saved locally.")

        print(f"Uploading merged model to '{HF_MODEL_ID}/merged' on the Hub...")
        api.upload_folder(
            folder_path=LOCAL_MERGED_MODEL_PATH,
            path_in_repo="merged", # Store merged model in 'merged' subfolder
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Add merged version of the fine-tuned model"
        )
        print(f"Merged model uploaded to {HF_MODEL_ID}/merged on the Hub.")

        # 11.B Convert the MERGED model to ONNX
        print("\nConverting MERGED model to ONNX format...")
        # Define a specific path for ONNX model artifacts from the merged model
        onnx_from_merged_dirname = HF_MODEL_ID.split('/')[-1] + "_onnx_from_merged"
        onnx_model_specific_path = os.path.join(LOCAL_ONNX_MODEL_PATH, onnx_from_merged_dirname)
        os.makedirs(onnx_model_specific_path, exist_ok=True)

        print(f"Exporting merged model (from local path: {LOCAL_MERGED_MODEL_PATH}) to ONNX...")
        # Optimum's ORTModelForSeq2SeqLM can take a local path to a saved model.
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            LOCAL_MERGED_MODEL_PATH, # Path to the saved merged model
            export=True,
            provider="CUDAExecutionProvider", # Uncomment and configure if you have GPU for ONNX
        )
        ort_model.save_pretrained(onnx_model_specific_path)
        # Tokenizer was already saved with merged_model, and from_pretrained for ORTModel uses the config
        # from that path. It's good practice to ensure tokenizer files are in onnx_model_specific_path too.
        # ORTModel.save_pretrained should handle config, but let's ensure tokenizer files are there.
        tokenizer.save_pretrained(onnx_model_specific_path) 
        print(f"ONNX model (from merged) saved locally to: {onnx_model_specific_path}")

        print(f"Uploading ONNX model (from merged) to '{HF_MODEL_ID}/onnx_merged' on the Hub...")
        api.upload_folder(
            folder_path=onnx_model_specific_path,
            path_in_repo="onnx_merged", # Store this ONNX model in 'onnx_merged' subfolder
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Add ONNX version (from merged model)"
        )
        print(f"ONNX model (from merged) uploaded to {HF_MODEL_ID}/onnx_merged on the Hub.")

    except Exception as e:
        print(f"An error occurred during model merging or ONNX conversion from merged model: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping or failed during merging/ONNX conversion from merged model.")
    # --- End of LoRA Merging and ONNX Conversion ---

    # Clean up local artifacts
    if os.path.exists(LOCAL_ADAPTERS_PATH):
        print(f"Cleaning up local PEFT adapter directory: {LOCAL_ADAPTERS_PATH}")
        shutil.rmtree(LOCAL_ADAPTERS_PATH)
    
    # LOCAL_MERGED_MODEL_PATH is inside OUTPUT_DIR, so it will be cleaned with OUTPUT_DIR
    # LOCAL_ONNX_MODEL_PATH contains onnx_model_specific_path
    if os.path.exists(LOCAL_ONNX_MODEL_PATH):
        print(f"Cleaning up local ONNX model directory: {LOCAL_ONNX_MODEL_PATH}")
        shutil.rmtree(LOCAL_ONNX_MODEL_PATH)
        
    if os.path.exists(OUTPUT_DIR): # OUTPUT_DIR contains training checkpoints and merged model artifacts
        print(f"Cleaning up main training output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print("\nTraining and deployment pipeline finished.")


if __name__ == "__main__":
    if not HF_HUB_TOKEN_WRITE or not HF_MODEL_ID:
        print("Critical Error: HF_HUB_TOKEN_WRITE and HF_MODEL_ID must be set in your .env file.")
    else:
        try:
            train()
        except Exception as e:
            print(f"An critical error occurred during the training script execution: {e}")
            import traceback
            traceback.print_exc()