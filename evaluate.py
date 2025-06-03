# evaluate.py
import pandas as pd
import time
import torch
from datetime import datetime

from src.config import (
    BASE_MODEL_ID,
    HF_MODEL_ID, # This is the repo_id for fine-tuned models
    HF_HUB_TOKEN_READ,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET,
    MAX_SAMPLES_DATASET
)
from src.data_load import load_pre_split_data
from src.translation_utils import (
    get_hf_model_and_tokenizer,
    translate_texts_hf,
    get_onnx_model_and_tokenizer,
    translate_texts_onnx,
    translate_texts_papago,
    calculate_bleu,
    measure_translation_speed
)

def evaluate_models():
    print("Starting model evaluation process...")
    start_time_eval = time.time()

    print("\n--- Loading Test Data ---")
    # ... (데이터 로드 부분은 동일)
    try:
        dataset_dict = load_pre_split_data(
            train_path=None, 
            valid_path=None, 
            max_samples_per_split=MAX_SAMPLES_DATASET or 100 
        )
        if "test" not in dataset_dict or len(dataset_dict["test"]) == 0:
            print("Test data not found or is empty. Aborting evaluation.")
            return
        test_data = dataset_dict["test"]
        ko_texts = [item["ko"] for item in test_data]
        en_references = [[item["en"]] for item in test_data]
        
        num_samples_for_speed_test = min(100, len(ko_texts))
        speed_test_ko_texts = ko_texts[:num_samples_for_speed_test]
        
        print(f"Loaded {len(ko_texts)} samples for BLEU evaluation from test split.")
        print(f"Using {len(speed_test_ko_texts)} samples for speed tests.")

    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ONNX on GPU only if main models are on GPU and HF_MODEL_ID is set (implying fine-tuned model exists)
    onnx_provider = "CUDAExecutionProvider" if device == "cuda" and HF_MODEL_ID else "CPUExecutionProvider"
    
    print(f"\n--- Evaluating Base Model: {BASE_MODEL_ID} ---")
    try:
        base_model, base_tokenizer = get_hf_model_and_tokenizer(
            repo_id=BASE_MODEL_ID, # repo_id for base model
            token=HF_HUB_TOKEN_READ,
            device=device
        )
        # ... (이하 BLEU 및 속도 측정 로직 동일)
        start_trans = time.time()
        base_translations = translate_texts_hf(ko_texts, base_model, base_tokenizer, batch_size=16, device=device)
        time_trans = time.time() - start_trans
        
        base_bleu = calculate_bleu(base_translations, en_references)
        print(f"Base Model BLEU: {base_bleu:.4f}")
        print(f"Base Model Translation Time for {len(ko_texts)} texts: {time_trans:.2f}s")

        avg_total_time, avg_time_per_text = measure_translation_speed(
            translate_texts_hf, (base_model, base_tokenizer), speed_test_ko_texts, batch_size_override=16
        )
        results.append({
            "Model": "Base (Pre-Finetuning)",
            "ID": BASE_MODEL_ID,
            "BLEU": f"{base_bleu:.4f}",
            f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time:.4f}s total, {avg_time_per_text:.6f}s/text",
            "Device": device
        })
        del base_model, base_tokenizer
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        results.append({"Model": "Base (Pre-Finetuning)", "ID": BASE_MODEL_ID, "BLEU": "Error", "Avg Speed": "Error", "Device": device})
        import traceback
        traceback.print_exc()


    print(f"\n--- Evaluating Fine-tuned Merged Model: {HF_MODEL_ID}/merged ---")
    if HF_MODEL_ID:
        merged_model_display_id = f"{HF_MODEL_ID}/merged"
        try:
            # train.py saves merged model and its tokenizer in the "merged" subfolder of HF_MODEL_ID repo
            ft_model, ft_tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,           # Main repo ID
                model_subfolder="merged",      # Subfolder for the merged model files
                tokenizer_repo_id=HF_MODEL_ID, # Tokenizer is also in HF_MODEL_ID
                tokenizer_subfolder="merged",  # In the "merged" subfolder
                token=HF_HUB_TOKEN_READ,
                device=device
            )
            # ... (이하 BLEU 및 속도 측정 로직 동일)
            start_trans = time.time()
            ft_translations = translate_texts_hf(ko_texts, ft_model, ft_tokenizer, batch_size=16, device=device)
            time_trans = time.time() - start_trans
            
            ft_bleu = calculate_bleu(ft_translations, en_references)
            print(f"Fine-tuned Merged Model BLEU: {ft_bleu:.4f}")
            print(f"Fine-tuned Merged Model Translation Time for {len(ko_texts)} texts: {time_trans:.2f}s")

            avg_total_time_ft, avg_time_per_text_ft = measure_translation_speed(
                translate_texts_hf, (ft_model, ft_tokenizer), speed_test_ko_texts, batch_size_override=16
            )
            results.append({
                "Model": "Fine-tuned (Merged)",
                "ID": merged_model_display_id,
                "BLEU": f"{ft_bleu:.4f}",
                f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time_ft:.4f}s total, {avg_time_per_text_ft:.6f}s/text",
                "Device": device
            })
            del ft_model, ft_tokenizer
            if device == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error evaluating fine-tuned merged model ({merged_model_display_id}): {e}")
            results.append({"Model": "Fine-tuned (Merged)", "ID": merged_model_display_id, "BLEU": "Error", "Avg Speed": "Error", "Device": device})
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping fine-tuned merged model evaluation.")
        results.append({"Model": "Fine-tuned (Merged)", "ID": "N/A", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "N/A"})

    print(f"\n--- Evaluating ONNX Model: {HF_MODEL_ID}/onnx_merged ---")
    if HF_MODEL_ID:
        onnx_model_display_id = f"{HF_MODEL_ID}/onnx_merged"
        try:
            # train.py saves ONNX model and its tokenizer in the "onnx_merged" subfolder of HF_MODEL_ID repo
            onnx_model, onnx_tokenizer = get_onnx_model_and_tokenizer(
                model_repo_id=HF_MODEL_ID,         # Main repo ID
                onnx_model_subfolder="onnx_merged", # Subfolder for ONNX model files & its tokenizer
                token=HF_HUB_TOKEN_READ,
                provider=onnx_provider
            )
            # ... (이하 BLEU 및 속도 측정 로직 동일)
            start_trans = time.time()
            onnx_translations = translate_texts_onnx(ko_texts, onnx_model, onnx_tokenizer, batch_size=16)
            time_trans = time.time() - start_trans
            
            onnx_bleu = calculate_bleu(onnx_translations, en_references)
            print(f"ONNX Model BLEU: {onnx_bleu:.4f}")
            print(f"ONNX Model Translation Time for {len(ko_texts)} texts: {time_trans:.2f}s (Provider: {onnx_provider})")
            
            avg_total_time_onnx, avg_time_per_text_onnx = measure_translation_speed(
                translate_texts_onnx, (onnx_model, onnx_tokenizer), speed_test_ko_texts, batch_size_override=16
            )
            results.append({
                "Model": "Fine-tuned (ONNX)",
                "ID": onnx_model_display_id,
                "BLEU": f"{onnx_bleu:.4f}",
                f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time_onnx:.4f}s total, {avg_time_per_text_onnx:.6f}s/text",
                "Device": onnx_provider 
            })
            del onnx_model, onnx_tokenizer
        except Exception as e:
            print(f"Error evaluating ONNX model ({onnx_model_display_id}): {e}")
            results.append({"Model": "Fine-tuned (ONNX)", "ID": onnx_model_display_id, "BLEU": "Error", "Avg Speed": "Error", "Device": onnx_provider})
            import traceback
            traceback.print_exc()
    else:
        print("HF_MODEL_ID not set. Skipping ONNX model evaluation.")
        results.append({"Model": "Fine-tuned (ONNX)", "ID": "N/A", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "N/A"})

    # # --- Papago API Evaluation --- (No changes needed here)
    # print("\n--- Evaluating Papago API ---")
    # # ... (Papago API 로직 동일)
    # if PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET:
    #     try:
    #         start_trans = time.time()
    #         papago_translations = translate_texts_papago(ko_texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    #         time_trans = time.time() - start_trans

    #         valid_papago_translations = [t for t in papago_translations if not t.startswith("[Papago Error")]
    #         papago_bleu = 0.0
    #         if len(valid_papago_translations) == len(en_references):
    #             papago_bleu = calculate_bleu(valid_papago_translations, en_references)
    #             print(f"Papago API BLEU: {papago_bleu:.4f}")
    #         else:
    #             print(f"Papago API: Translated {len(valid_papago_translations)}/{len(ko_texts)} successfully. BLEU score might be affected or not calculated.")

    #         print(f"Papago API Translation Time for {len(ko_texts)} texts: {time_trans:.2f}s")

    #         avg_total_time_papago, avg_time_per_text_papago = measure_translation_speed(
    #              translate_texts_papago, None, speed_test_ko_texts[:min(20, len(speed_test_ko_texts))] 
    #         )
    #         results.append({
    #             "Model": "Papago API",
    #             "ID": "Naver Papago NMT",
    #             "BLEU": f"{papago_bleu:.4f}" if papago_bleu > 0 else "N/A or Partial Error",
    #             f"Avg Speed ({min(20, len(speed_test_ko_texts))} texts)": f"{avg_total_time_papago:.4f}s total, {avg_time_per_text_papago:.6f}s/text",
    #             "Device": "API"
    #         })
    #     except Exception as e:
    #         print(f"Error evaluating Papago API: {e}")
    #         results.append({"Model": "Papago API", "ID": "Naver Papago NMT", "BLEU": "Error", "Avg Speed": "Error", "Device": "API"})
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print("Papago API credentials not set. Skipping Papago API evaluation.")
    #     results.append({"Model": "Papago API", "ID": "Naver Papago NMT", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "API"})


    print("\n\n--- Overall Evaluation Summary ---")
    # ... (결과 출력 및 저장 로직 동일)
    results_df = pd.DataFrame(results)
    print(results_df.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.csv"
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"\nEvaluation results saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    end_time_eval = time.time()
    print(f"\nTotal evaluation script time: {(end_time_eval - start_time_eval)/60:.2f} minutes.")


if __name__ == "__main__":
    evaluate_models()