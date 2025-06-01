import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
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
from src.data_load import load_and_split_data
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
    raw_datasets = load_and_split_data(max_samples=MAX_SAMPLES_DATASET)
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
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"] # Typical for T5/MarianMT, adjust if needed
    )

    # 5. Apply PEFT to the model
    print("Applying LoRA to the model...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("LoRA applied.")

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=max(1, int(0.05 * (len(tokenized_datasets['train']) // PER_DEVICE_TRAIN_BATCH_SIZE))), # Log ~20 times per epoch
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save at the end of each epoch
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Or a BLEU score if added to eval
        greater_is_better=False,
        fp16=False, # Not compatible with 4-bit quantization's compute_dtype=bfloat16. If using float16 compute, set to True.
        bf16=True if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else False, # Use bfloat16 if compute_dtype is bfloat16
        remove_unused_columns=True, # Important for PEFT
        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="every_save", # Push every time model is saved (epoch end)
        hub_token=HF_HUB_TOKEN_WRITE,
        report_to="tensorboard", # or "wandb" if you use it
    )

    # 7. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=peft_model,
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
    peft_model.save_pretrained(LOCAL_ADAPTERS_PATH) # Saves adapter_model.safetensors and adapter_config.json
    
    print("Pushing final model, tokenizer, and adapters to Hugging Face Hub...")
    # The trainer already pushes adapters with `push_to_hub=True`.
    # This ensures the tokenizer and any other files created by trainer (like training_args.bin) are also pushed.
    trainer.push_to_hub(commit_message="End of training: final model and tokenizer")
    
    # Explicitly upload the adapter folder if trainer didn't catch it or for redundancy
    api = HfApi()
    api.upload_folder(
        folder_path=LOCAL_ADAPTERS_PATH,
        path_in_repo="adapters", # Store adapters in a subfolder on the Hub
        repo_id=HF_MODEL_ID,
        repo_type="model",
        token=HF_HUB_TOKEN_WRITE,
        commit_message="Upload PEFT adapters"
    )
    print(f"PEFT adapters uploaded to {HF_MODEL_ID}/adapters on the Hub.")
    
    # Clean up local adapter folder
    if os.path.exists(LOCAL_ADAPTERS_PATH):
        shutil.rmtree(LOCAL_ADAPTERS_PATH)


    # 11. Convert to ONNX and push
    print("Converting fine-tuned model to ONNX format...")
    try:
        # For ONNX export, it's usually better to merge LoRA weights first if possible,
        # or load the model with adapters applied.
        # Optimum might handle PEFT models directly. Let's try loading from hub
        # ensuring the read token (or write token if it has read access) is available.
        
        # If HF_HUB_TOKEN_READ is different and needed for private repo
        # login(token=HF_HUB_TOKEN_READ, add_to_git_credential=False)

        # Reload the model with adapters merged for ONNX export if necessary
        # This step depends on how `optimum` handles PEFT models.
        # If it can take the `peft_model` directly or `HF_MODEL_ID` (which now contains adapters):
        
        onnx_model_path = os.path.join(LOCAL_ONNX_MODEL_PATH, HF_MODEL_ID.split('/')[-1] + "_onnx")
        os.makedirs(onnx_model_path, exist_ok=True)

        print(f"Exporting model from {HF_MODEL_ID} (with adapters) to ONNX...")
        # The from_pretrained here will load the base model and automatically apply
        # adapters if they are found in the repo (which they are, pushed by the trainer)
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            HF_MODEL_ID, # This should point to the repo with adapters
            export=True,
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE, # Use read or write token,
            library_name="transformers",
            # provider="CUDAExecutionProvider", # Uncomment if you have GPU for ONNX and want to specify
        )
        ort_model.save_pretrained(onnx_model_path)
        tokenizer.save_pretrained(onnx_model_path) # Save tokenizer with ONNX model for consistency

        print(f"ONNX model saved locally to: {onnx_model_path}")

        print("Uploading ONNX model to Hugging Face Hub...")
        api.upload_folder(
            folder_path=onnx_model_path,
            path_in_repo="onnx", # Store ONNX model in 'onnx' subfolder on the Hub
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Add ONNX version of the model"
        )
        print(f"ONNX model uploaded to {HF_MODEL_ID}/onnx on the Hub.")

        # Clean up local ONNX model folder
        if os.path.exists(LOCAL_ONNX_MODEL_PATH):
            shutil.rmtree(LOCAL_ONNX_MODEL_PATH)

    except Exception as e:
        print(f"Error during ONNX conversion or upload: {e}")
        print("Skipping ONNX part.")

    # Clean up training output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up training output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print("Training, adapter saving, ONNX conversion (if successful), and Hub uploads complete.")


if __name__ == "__main__":
    # Ensure .env is loaded by config.py
    # For this script to run, HF_HUB_TOKEN_WRITE and HF_MODEL_ID must be set in .env
    
    # You might need to be logged into huggingface_hub via CLI first for some operations
    # `huggingface-cli login` with your WRITE token
    
    if not HF_HUB_TOKEN_WRITE or not HF_MODEL_ID:
        print("Error: HF_HUB_TOKEN_WRITE and HF_MODEL_ID must be set in your .env file.")
    else:
        try:
            train()
        except Exception as e:
            print(f"An critical error occurred during the training script: {e}")
            import traceback
            traceback.print_exc()