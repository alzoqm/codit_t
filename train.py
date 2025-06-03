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
        device_map="auto" # 사용 가능한 GPU에 모델 자동 분배
    )
    print("Base model loaded.")

    # 4. LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"] # 모델의 명명된 파라미터를 확인하여 정확한 모듈 이름 사용
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
        if steps_per_epoch == 0: # 데이터셋이 배치 크기보다 작은 경우 대비
            steps_per_epoch = 1
        logging_steps_val = max(1, steps_per_epoch // 20 if steps_per_epoch > 20 else 1) # 에포크당 약 20회 로깅
    else:
        print("Warning: Training dataset is empty or not found. Using default steps.")
        steps_per_epoch = 100 # 학습 데이터가 없을 경우 기본값 (발생해서는 안 됨)
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
        push_to_hub=True,
        hub_model_id=HF_MODEL_ID,
        hub_strategy="every_save",
        hub_token=HF_HUB_TOKEN_WRITE,
        report_to="tensorboard",
    )

    # 7. 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=peft_model, # 여기에 peft_model 전달
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

        # 10. 최종 어댑터 저장 및 모든 것 푸시
        print(f"Saving final PEFT adapter to local path: {LOCAL_ADAPTERS_PATH}")
        os.makedirs(LOCAL_ADAPTERS_PATH, exist_ok=True) # 디렉터리 생성 확인
        trainer.model.save_pretrained(LOCAL_ADAPTERS_PATH)
        
        print("Pushing final model (trainer might have done this) and tokenizer to Hugging Face Hub...")
        trainer.push_to_hub(commit_message="End of training: final adapters and tokenizer")
        
        api = HfApi()
        print(f"Uploading PEFT adapters from '{LOCAL_ADAPTERS_PATH}' to '{HF_MODEL_ID}/peft_lora' on the Hub...")
        api.upload_folder(
            folder_path=LOCAL_ADAPTERS_PATH,
            path_in_repo="peft_lora", # 어댑터를 'peft_lora' 하위 폴더에 저장
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Upload PEFT adapters to peft_lora subfolder"
        )
        print(f"PEFT adapters uploaded to {HF_MODEL_ID}/peft_lora on the Hub.")
    else:
        print("Skipping training as no training data is available.")
        # 학습이 스킵된 경우 어댑터 병합 및 ONNX 변환을 시도하지 않도록 처리할 수 있습니다.
        # 여기서는 간결함을 위해 바로 종료합니다.
        print("\nTraining and deployment pipeline finished (training skipped).")
        return


    # --- LoRA 병합 및 ONNX 변환 시작 ---
    print("\nAttempting to merge LoRA adapters and convert to ONNX...")
    try:
        # 11.A LoRA 어댑터 병합
        print("Loading base model for merging (this time without 4-bit quantization)...")
        base_model_for_merging = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto", # 또는 메모리가 매우 부족하면 "cpu" 강제
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE
        )
        print(f"Base model ({BASE_MODEL_ID}) loaded for merging.")

        print(f"Loading PEFT adapters from local path: {LOCAL_ADAPTERS_PATH}")
        if not os.path.exists(os.path.join(LOCAL_ADAPTERS_PATH, "adapter_config.json")):
             raise FileNotFoundError(f"Adapter config not found at {LOCAL_ADAPTERS_PATH}. Cannot proceed with merging.")

        model_with_adapters = PeftModel.from_pretrained(
            base_model_for_merging, # 새로 로드된 기본 모델
            LOCAL_ADAPTERS_PATH,    # 저장된 어댑터 경로
            token=HF_HUB_TOKEN_READ or HF_HUB_TOKEN_WRITE
        )
        print(f"PEFT adapters loaded from {LOCAL_ADAPTERS_PATH} onto the new base model instance.")

        print("Merging LoRA adapters into the base model...")
        merged_model = model_with_adapters.merge_and_unload()
        print("Adapters merged successfully.")

        LOCAL_MERGED_MODEL_PATH = os.path.join(OUTPUT_DIR, "merged_model_artifacts")
        print(f"Saving merged model locally to: {LOCAL_MERGED_MODEL_PATH}")
        os.makedirs(LOCAL_MERGED_MODEL_PATH, exist_ok=True)
    
        merged_model.save_pretrained(
            LOCAL_MERGED_MODEL_PATH,
            safe_serialization=False
        )
 
        tokenizer.save_pretrained(LOCAL_MERGED_MODEL_PATH) # 병합된 모델과 함께 토크나이저 저장
        print("Merged model and tokenizer saved locally.")

        # 저장된 파일 확인 (디버깅용)
        expected_bin_file = os.path.join(LOCAL_MERGED_MODEL_PATH, "pytorch_model.bin")
        if not os.path.exists(expected_bin_file):
            print(f"CRITICAL WARNING: pytorch_model.bin was NOT found at {expected_bin_file} after saving.")
            print(f"Files in {LOCAL_MERGED_MODEL_PATH}: {os.listdir(LOCAL_MERGED_MODEL_PATH)}")
        else:
            print(f"Successfully saved {expected_bin_file}.")


        print(f"Uploading merged model to '{HF_MODEL_ID}/merged' on the Hub...")
        api.upload_folder(
            folder_path=LOCAL_MERGED_MODEL_PATH,
            path_in_repo="merged", # 병합된 모델을 'merged' 하위 폴더에 저장
            repo_id=HF_MODEL_ID,
            repo_type="model",
            token=HF_HUB_TOKEN_WRITE,
            commit_message="Add merged version of the fine-tuned model"
        )
        print(f"Merged model uploaded to {HF_MODEL_ID}/merged on the Hub.")

        # 11.B 병합된 모델을 ONNX로 변환
        print("\nConverting MERGED model to ONNX format...")
        onnx_from_merged_dirname = HF_MODEL_ID.split('/')[-1] + "_onnx_from_merged"
        onnx_model_specific_path = os.path.join(LOCAL_ONNX_MODEL_PATH, onnx_from_merged_dirname)
        os.makedirs(onnx_model_specific_path, exist_ok=True)

        print(f"Exporting merged model (from local path: {LOCAL_MERGED_MODEL_PATH}) to ONNX...")
        # Optimum의 ORTModelForSeq2SeqLM은 저장된 모델 경로를 사용할 수 있습니다.
        # from_pretrained는 LOCAL_MERGED_MODEL_PATH에서 pytorch_model.bin 또는 model.safetensors를 찾습니다.
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            LOCAL_MERGED_MODEL_PATH, # 저장된 병합 모델 경로
            export=True,
            # provider="CUDAExecutionProvider", # ONNX에 GPU를 사용하려면 주석 해제 및 설정
        )
        ort_model.save_pretrained(onnx_model_specific_path)
        tokenizer.save_pretrained(onnx_model_specific_path) # ONNX 모델과 함께 토크나이저 파일 저장
        print(f"ONNX model (from merged) saved locally to: {onnx_model_specific_path}")

        print(f"Uploading ONNX model (from merged) to '{HF_MODEL_ID}/onnx_merged' on the Hub...")
        api.upload_folder(
            folder_path=onnx_model_specific_path,
            path_in_repo="onnx_merged", # 이 ONNX 모델을 'onnx_merged' 하위 폴더에 저장
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
    # --- LoRA 병합 및 ONNX 변환 끝 ---

    # 로컬 아티팩트 정리
    if os.path.exists(LOCAL_ADAPTERS_PATH):
        print(f"Cleaning up local PEFT adapter directory: {LOCAL_ADAPTERS_PATH}")
        shutil.rmtree(LOCAL_ADAPTERS_PATH)
    
    # LOCAL_MERGED_MODEL_PATH는 OUTPUT_DIR 내부에 있으므로 OUTPUT_DIR과 함께 정리됩니다.
    # LOCAL_ONNX_MODEL_PATH는 onnx_model_specific_path를 포함합니다.
    if os.path.exists(LOCAL_ONNX_MODEL_PATH): # onnx_models 폴더 전체 삭제
        print(f"Cleaning up local ONNX model base directory: {LOCAL_ONNX_MODEL_PATH}")
        shutil.rmtree(LOCAL_ONNX_MODEL_PATH)
        
    if os.path.exists(OUTPUT_DIR): # OUTPUT_DIR은 학습 체크포인트 및 병합된 모델 아티팩트를 포함합니다.
        print(f"Cleaning up main training output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print("\nTraining and deployment pipeline finished.")


if __name__ == "__main__":
    # 환경 변수 설정 확인 (스크립트 실행 전에 .env 파일 등을 통해 설정해야 함)
    if not HF_HUB_TOKEN_WRITE:
        print("Warning: HF_HUB_TOKEN_WRITE environment variable is not set. Will try to use cached token or prompt for login.")
    if not HF_MODEL_ID:
        print("Critical Error: HF_MODEL_ID environment variable must be set (e.g., 'your-username/your-model-name').")
        exit(1) # HF_MODEL_ID는 필수입니다.

    # HF_HUB_TOKEN_WRITE가 없으면 login() 함수가 대화형 로그인을 시도합니다.
    # HF_HUB_TOKEN_READ가 없으면 일부 from_pretrained 호출 시 문제가 발생할 수 있으나,
    # 공개 모델이거나 HF_HUB_TOKEN_WRITE 토큰에 읽기 권한도 있으면 괜찮을 수 있습니다.

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
    except ValueError as ve: # 설정 오류 등을 명확히 처리
        print(f"Configuration or Value Error: {ve}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An critical error occurred during the training script execution: {e}")
        import traceback
        traceback.print_exc()