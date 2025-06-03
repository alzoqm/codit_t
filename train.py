import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import HfApi, login
# from optimum.onnxruntime import ORTModelForSeq2SeqLM # ONNX 사용 안함
import os
import shutil

from src.config import (
    BASE_MODEL_ID, HF_MODEL_ID, HF_HUB_TOKEN_WRITE, HF_HUB_TOKEN_READ,
    NUM_TRAIN_EPOCHS, LEARNING_RATE, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, OUTPUT_DIR,
    QLORA_R, QLORA_ALPHA, QLORA_DROPOUT,
    LOCAL_ADAPTERS_PATH, MAX_SAMPLES_DATASET # LOCAL_ONNX_MODEL_PATH 제거됨
)
from src.data_load import load_pre_split_data
from src.preprocess import tokenize_datasets

def train():
    print("Starting training process...")

    if not HF_HUB_TOKEN_WRITE:
        raise ValueError("HF_HUB_TOKEN_WRITE is not set. Cannot push to Hub.")
    if not HF_MODEL_ID:
        raise ValueError("HF_MODEL_ID is not set. Cannot determine push location.")

    # Hugging Face Hub 로그인
    print(f"Logging in to Hugging Face Hub with write token.")
    login(token=HF_HUB_TOKEN_WRITE)

    # 1. 데이터 로드 및 전처리
    print(f"Loading and preprocessing data (max_samples_per_split: {MAX_SAMPLES_DATASET})...")
    raw_datasets = load_pre_split_data(max_samples_per_split=MAX_SAMPLES_DATASET)
    tokenized_datasets, tokenizer = tokenize_datasets(raw_datasets, BASE_MODEL_ID)
    print("Data loaded and tokenized.")

    # 2. qLoRA를 위한 BitsAndBytes 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )

    # 3. 양자화와 함께 기본 모델 로드
    print(f"Loading base model: {BASE_MODEL_ID} with qLoRA quantization...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Base model loaded.")

    # 4. LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 5. 모델에 PEFT 적용
    print("Applying LoRA to the model...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("LoRA applied.")

    # 6. 학습 인자 설정
    if tokenized_datasets["train"] and len(tokenized_datasets["train"]) > 0:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * max(1, num_gpus)
        steps_per_epoch = len(tokenized_datasets["train"]) // effective_batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        logging_steps_val = max(1, steps_per_epoch // 20 if steps_per_epoch > 20 else 1)
    else:
        print("Warning: Training dataset is empty or not found. Using default steps.")
        steps_per_epoch = 100
        logging_steps_val = 5

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=logging_steps_val,
        eval_strategy="epoch" if tokenized_datasets.get("validation") else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if tokenized_datasets.get("validation") else False,
        metric_for_best_model="eval_loss" if tokenized_datasets.get("validation") else None,
        greater_is_better=False,
        fp16=False,
        bf16=True if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported() else False,
        remove_unused_columns=True,
        push_to_hub=False, # Trainer가 직접 Hub에 푸시하지 않도록 설정
        report_to="tensorboard",
    )

    # 7. 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=peft_model,
        label_pad_token_id=tokenizer.pad_token_id
    )

    # 8. 트레이너
    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if tokenized_datasets["train"] and len(tokenized_datasets["train"]) > 0 else None,
        eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") and len(tokenized_datasets["validation"]) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. 학습
    if tokenized_datasets["train"] and len(tokenized_datasets["train"]) > 0:
        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning complete.")

        # 10. 최종 어댑터를 로컬에 임시 저장 (병합용)
        print(f"Saving final PEFT adapter temporarily to local path: {LOCAL_ADAPTERS_PATH}")
        os.makedirs(LOCAL_ADAPTERS_PATH, exist_ok=True)
        trainer.model.save_pretrained(LOCAL_ADAPTERS_PATH) # 어댑터와 설정 파일 저장
        tokenizer.save_pretrained(LOCAL_ADAPTERS_PATH) # 토크나이저도 함께 저장
        print(f"Temporary PEFT adapter and tokenizer saved to {LOCAL_ADAPTERS_PATH}.")

    else:
        print("Skipping training as no training data is available.")
        print("\nTraining and deployment pipeline finished (training skipped).")
        return


    # --- LoRA 병합 시작 ---
    print("\nAttempting to merge LoRA adapters...")
    try:
        # 11.A LoRA 어댑터 병합
        print("Loading base model for merging (this time without 4-bit quantization)...")
        base_model_for_merging = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE # 공개 모델이면 토큰 불필요할 수 있음
        )
        print(f"Base model ({BASE_MODEL_ID}) loaded for merging.")

        print(f"Loading PEFT adapters from local path: {LOCAL_ADAPTERS_PATH}")
        if not os.path.exists(os.path.join(LOCAL_ADAPTERS_PATH, "adapter_config.json")):
             raise FileNotFoundError(f"Adapter config not found at {LOCAL_ADAPTERS_PATH}. Cannot proceed with merging.")

        # 로컬에 저장된 어댑터로부터 PeftModel 로드
        model_with_adapters = PeftModel.from_pretrained(
            base_model_for_merging,
            LOCAL_ADAPTERS_PATH
        )
        print(f"PEFT adapters loaded from {LOCAL_ADAPTERS_PATH} onto the new base model instance.")

        print("Merging LoRA adapters into the base model...")
        merged_model = model_with_adapters.merge_and_unload()
        print("Adapters merged successfully.")

        # 병합된 모델을 저장할 로컬 경로 (OUTPUT_DIR 내)
        LOCAL_MERGED_MODEL_PATH = os.path.join(OUTPUT_DIR, "merged_model_artifacts")
        print(f"Saving merged model locally to: {LOCAL_MERGED_MODEL_PATH}")
        os.makedirs(LOCAL_MERGED_MODEL_PATH, exist_ok=True)
    
        # 병합된 모델과 토크나이저 저장
        merged_model.save_pretrained(
            LOCAL_MERGED_MODEL_PATH,
            safe_serialization=False # pytorch_model.bin 으로 저장 (필요에 따라 True로 변경 가능)
        )
        # LOCAL_ADAPTERS_PATH에 저장된 토크나이저를 사용하거나, 원본 tokenizer를 다시 저장
        # 여기서는 원본 tokenizer 객체를 다시 저장합니다.
        tokenizer.save_pretrained(LOCAL_MERGED_MODEL_PATH)
        print("Merged model and tokenizer saved locally.")

        # 저장된 파일 확인 (디버깅용)
        expected_bin_file = os.path.join(LOCAL_MERGED_MODEL_PATH, "pytorch_model.bin")
        expected_safetensors_file = os.path.join(LOCAL_MERGED_MODEL_PATH, "model.safetensors")
        
        if not (os.path.exists(expected_bin_file) or os.path.exists(expected_safetensors_file)):
            print(f"CRITICAL WARNING: Model file (pytorch_model.bin or model.safetensors) was NOT found at {LOCAL_MERGED_MODEL_PATH} after saving.")
            print(f"Files in {LOCAL_MERGED_MODEL_PATH}: {os.listdir(LOCAL_MERGED_MODEL_PATH)}")
        else:
            if os.path.exists(expected_bin_file):
                 print(f"Successfully saved {expected_bin_file}.")
            if os.path.exists(expected_safetensors_file):
                 print(f"Successfully saved {expected_safetensors_file}.")


        # 병합된 모델을 Hugging Face Hub의 루트에 업로드
        api = HfApi()
        print(f"Uploading merged model to '{HF_MODEL_ID}' (root) on the Hub...")
        api.upload_folder(
            folder_path=LOCAL_MERGED_MODEL_PATH,
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Add merged version of the fine-tuned model"
            # path_in_repo를 지정하지 않으면 리포지토리 루트에 업로드됩니다.
        )
        print(f"Merged model uploaded to {HF_MODEL_ID} (root) on the Hub.")

    except Exception as e:
        print(f"An error occurred during model merging or Hub upload: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping or failed during merging/uploading merged model.")
    # --- LoRA 병합 끝 ---

    # ONNX 변환 및 관련 업로드/저장 로직은 제거됨

    # 로컬 아티팩트 정리
    if os.path.exists(LOCAL_ADAPTERS_PATH): # 임시 어댑터 폴더 정리
        print(f"Cleaning up local PEFT adapter directory: {LOCAL_ADAPTERS_PATH}")
        shutil.rmtree(LOCAL_ADAPTERS_PATH)
    
    # LOCAL_MERGED_MODEL_PATH는 OUTPUT_DIR 내부에 있으므로 OUTPUT_DIR 정리 시 함께 삭제됨
    # LOCAL_ONNX_MODEL_PATH 관련 정리 로직은 필요 없음

    if os.path.exists(OUTPUT_DIR): # OUTPUT_DIR은 학습 체크포인트 및 임시 병합 모델 아티팩트를 포함
        print(f"Cleaning up main training output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print("\nTraining and deployment pipeline finished.")


if __name__ == "__main__":
    if not HF_HUB_TOKEN_WRITE:
        print("Warning: HF_HUB_TOKEN_WRITE environment variable is not set. Will try to use cached token or prompt for login.")
    if not HF_MODEL_ID:
        print("Critical Error: HF_MODEL_ID environment variable must be set (e.g., 'your-username/your-model-name').")
        exit(1)

    print(f"Configuration for training:")
    print(f"  BASE_MODEL_ID: {BASE_MODEL_ID}")
    print(f"  HF_MODEL_ID (target repo): {HF_MODEL_ID}")
    print(f"  NUM_TRAIN_EPOCHS: {NUM_TRAIN_EPOCHS}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  MAX_SAMPLES_DATASET: {MAX_SAMPLES_DATASET}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")

    try:
        train()
    except ValueError as ve:
        print(f"Configuration or Value Error: {ve}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An critical error occurred during the training script execution: {e}")
        import traceback
        traceback.print_exc()