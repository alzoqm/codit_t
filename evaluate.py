import pandas as pd
import time
import torch
from datetime import datetime
import traceback # For detailed error logging

from src.config import (
    BASE_MODEL_ID,
    HF_MODEL_ID, # This is your target model ID, e.g., "alzoqm/test_model_2"
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

    # --- 1. Load Test Data ---
    print("\n--- Loading Test Data ---")
    try:
        dataset_dict = load_pre_split_data(
            train_path=None, # Don't load train for eval
            valid_path=None, # Don't load valid for eval
            max_samples_per_split=MAX_SAMPLES_DATASET or 100 # Limit for faster evaluation, or None for full
        )
        if "test" not in dataset_dict or len(dataset_dict["test"]) == 0:
            print("Test data not found or is empty. Aborting evaluation.")
            return
        test_data = dataset_dict["test"]
        ko_texts = [item["ko"] for item in test_data]
        en_references = [[item["en"]] for item in test_data] # sacrebleu expects list of lists
        
        num_samples_for_speed_test = min(100, len(ko_texts))
        speed_test_ko_texts = ko_texts[:num_samples_for_speed_test]
        
        print(f"Loaded {len(ko_texts)} samples for BLEU evaluation from test split.")
        print(f"Using {len(speed_test_ko_texts)} samples for speed tests.")

    except Exception as e:
        print(f"Error loading test data: {e}")
        print(traceback.format_exc())
        return

    results = []
    pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
    # For ONNX, prefer CUDA if PyTorch is on CUDA, else CPU. Actual provider determined in get_onnx_model_and_tokenizer
    onnx_preferred_provider = "CUDAExecutionProvider" if pytorch_device == "cuda" else "CPUExecutionProvider"
    
    # --- 2. Evaluate Base Model (Pre-finetuning) ---
    print(f"\n--- Evaluating Base Model: {BASE_MODEL_ID} ---")
    try:
        base_model, base_tokenizer = get_hf_model_and_tokenizer(
            repo_id=BASE_MODEL_ID, # Main repo ID
            token=HF_HUB_TOKEN_READ, 
            device=pytorch_device
        )
        start_trans_base = time.time()
        base_translations = translate_texts_hf(ko_texts, base_model, base_tokenizer, batch_size=16, device=pytorch_device)
        time_trans_base = time.time() - start_trans_base
        base_bleu = calculate_bleu(base_translations, en_references)
        print(f"Base Model BLEU: {base_bleu:.4f}, Translation time: {time_trans_base:.2f}s")

        avg_total_time_b, avg_time_per_text_b = measure_translation_speed(
            translate_texts_hf, (base_model, base_tokenizer), speed_test_ko_texts, batch_size_override=16
        )
        results.append({
            "Model": "Base (Pre-Finetuning)", "ID": BASE_MODEL_ID, "BLEU": f"{base_bleu:.4f}",
            f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time_b:.4f}s total, {avg_time_per_text_b:.6f}s/text",
            "Device": pytorch_device
        })
        del base_model, base_tokenizer
        if pytorch_device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        print(traceback.format_exc())
        results.append({"Model": "Base (Pre-Finetuning)", "ID": BASE_MODEL_ID, "BLEU": "Error", "Avg Speed": "Error", "Device": pytorch_device})

    # --- 3. Evaluate Fine-tuned Model (Merged) ---
    merged_model_display_id = f"{HF_MODEL_ID}/merged" if HF_MODEL_ID else "N/A"
    print(f"\n--- Evaluating Fine-tuned Merged Model: {merged_model_display_id} ---")
    if HF_MODEL_ID:
        try:
            # ft_model, ft_tokenizer = get_hf_model_and_tokenizer(
            #     repo_id=HF_MODEL_ID,          # Main repo ID
            #     model_subfolder="merged",     # Subfolder for the merged model
            #     # Tokenizer for merged should also be in "merged" or root of HF_MODEL_ID.
            #     # translation_utils will try HF_MODEL_ID/merged first for tokenizer if tokenizer_subfolder="merged"
            #     tokenizer_repo_id=HF_MODEL_ID, 
            #     tokenizer_subfolder="merged",
            #     token=HF_HUB_TOKEN_READ,
            #     device=pytorch_device
            # )
            ft_model, ft_tokenizer = get_hf_model_and_tokenizer(
                repo_id=HF_MODEL_ID,          # Main repo ID
                token=HF_HUB_TOKEN_READ,
                device=pytorch_device
            )
            start_trans_ft = time.time()
            ft_translations = translate_texts_hf(ko_texts, ft_model, ft_tokenizer, batch_size=16, device=pytorch_device)
            time_trans_ft = time.time() - start_trans_ft
            ft_bleu = calculate_bleu(ft_translations, en_references)
            print(f"Fine-tuned Merged Model BLEU: {ft_bleu:.4f}, Translation time: {time_trans_ft:.2f}s")

            avg_total_time_ft, avg_time_per_text_ft = measure_translation_speed(
                translate_texts_hf, (ft_model, ft_tokenizer), speed_test_ko_texts, batch_size_override=16
            )
            results.append({
                "Model": "Fine-tuned (Merged)", "ID": merged_model_display_id, "BLEU": f"{ft_bleu:.4f}",
                f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time_ft:.4f}s total, {avg_time_per_text_ft:.6f}s/text",
                "Device": pytorch_device
            })
            del ft_model, ft_tokenizer
            if pytorch_device == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error evaluating fine-tuned merged model ({merged_model_display_id}): {e}")
            print(traceback.format_exc())
            results.append({"Model": "Fine-tuned (Merged)", "ID": merged_model_display_id, "BLEU": "Error", "Avg Speed": "Error", "Device": pytorch_device})
    else:
        print("HF_MODEL_ID not set. Skipping fine-tuned merged model evaluation.")
        results.append({"Model": "Fine-tuned (Merged)", "ID": "N/A", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "N/A"})

    # --- 4. Evaluate ONNX Model ---
    # onnx_model_display_id = f"{HF_MODEL_ID}/onnx_merged" if HF_MODEL_ID else "N/A"
    # print(f"\n--- Evaluating ONNX Model: {onnx_model_display_id} (Preferred Provider: {onnx_preferred_provider}) ---")
    # if HF_MODEL_ID:
    #     actual_onnx_provider_in_use = "N/A" # Placeholder
    #     try:
    #         onnx_model, onnx_tokenizer = get_onnx_model_and_tokenizer(
    #             model_repo_id=HF_MODEL_ID,        # Main repo ID
    #             onnx_model_subfolder="onnx_merged", # Subfolder for ONNX model files
    #             # tokenizer_base_repo_id will default to model_repo_id inside the function
    #             token=HF_HUB_TOKEN_READ,
    #             preferred_provider=onnx_preferred_provider
    #         )
    #         # Get the actual provider used by the ONNX model session if available
    #         actual_onnx_provider_in_use = onnx_model.providers[0] if hasattr(onnx_model, 'providers') and onnx_model.providers else onnx_preferred_provider
            
    #         start_trans_onnx = time.time()
    #         onnx_translations = translate_texts_onnx(ko_texts, onnx_model, onnx_tokenizer, batch_size=1)
    #         time_trans_onnx = time.time() - start_trans_onnx
    #         onnx_bleu = calculate_bleu(onnx_translations, en_references)
    #         print(f"ONNX Model BLEU: {onnx_bleu:.4f}, Translation time: {time_trans_onnx:.2f}s (Using Provider: {actual_onnx_provider_in_use})")
            
    #         avg_total_time_onnx, avg_time_per_text_onnx = measure_translation_speed(
    #             translate_texts_onnx, (onnx_model, onnx_tokenizer), speed_test_ko_texts, batch_size_override=1
    #         )
    #         results.append({
    #             "Model": "Fine-tuned (ONNX)", "ID": onnx_model_display_id, "BLEU": f"{onnx_bleu:.4f}",
    #             f"Avg Speed ({len(speed_test_ko_texts)} texts)": f"{avg_total_time_onnx:.4f}s total, {avg_time_per_text_onnx:.6f}s/text",
    #             "Device": actual_onnx_provider_in_use
    #         })
    #         del onnx_model, onnx_tokenizer
    #     except Exception as e:
    #         print(f"Error evaluating ONNX model ({onnx_model_display_id}): {e}")
    #         print(traceback.format_exc())
    #         results.append({"Model": "Fine-tuned (ONNX)", "ID": onnx_model_display_id, "BLEU": "Error", "Avg Speed": "Error", "Device": actual_onnx_provider_in_use}) # Report actual provider if known, else preferred
    # else:
    #     print("HF_MODEL_ID not set. Skipping ONNX model evaluation.")
    #     results.append({"Model": "Fine-tuned (ONNX)", "ID": "N/A", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "N/A"})

    # # --- 5. Evaluate Papago API ---
    # print("\n--- Evaluating Papago API ---")
    # if PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET:
    #     try:
    #         start_trans_papago = time.time()
    #         papago_translations = translate_texts_papago(ko_texts, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET)
    #         time_trans_papago = time.time() - start_trans_papago

    #         valid_papago_translations = [t for t in papago_translations if not t.startswith("[Papago Error")]
    #         papago_bleu = 0.0
    #         if valid_papago_translations and len(valid_papago_translations) == len(ko_texts) and len(ko_texts) > 0:
    #             papago_bleu = calculate_bleu(valid_papago_translations, en_references)
    #             print(f"Papago API BLEU: {papago_bleu:.4f}, Translation time: {time_trans_papago:.2f}s")
    #         elif len(ko_texts) == 0:
    #              print("Papago API: No texts to translate for BLEU.")
    #         else: # Partial success or all failed
    #             print(f"Papago API: Translated {len(valid_papago_translations)}/{len(ko_texts)} successfully. BLEU score might be affected or not calculated. Time: {time_trans_papago:.2f}s")
            
    #         # Use a smaller, consistent number of samples for Papago speed test due to API call limitations/costs
    #         papago_speed_test_samples = speed_test_ko_texts[:min(20, len(speed_test_ko_texts))]
    #         avg_total_time_papago, avg_time_per_text_papago = measure_translation_speed(
    #              translate_texts_papago, None, papago_speed_test_samples 
    #         )
    #         results.append({
    #             "Model": "Papago API", "ID": "Naver Papago NMT",
    #             "BLEU": f"{papago_bleu:.4f}" if (valid_papago_translations and len(valid_papago_translations) == len(ko_texts) and len(ko_texts) > 0) else "N/A or Partial Error",
    #             f"Avg Speed ({len(papago_speed_test_samples)} texts)": f"{avg_total_time_papago:.4f}s total, {avg_time_per_text_papago:.6f}s/text",
    #             "Device": "API"
    #         })
    #     except Exception as e:
    #         print(f"Error evaluating Papago API: {e}")
    #         print(traceback.format_exc())
    #         results.append({"Model": "Papago API", "ID": "Naver Papago NMT", "BLEU": "Error", "Avg Speed": "Error", "Device": "API"})
    # else:
    #     print("Papago API credentials not set. Skipping Papago API evaluation.")
    #     results.append({"Model": "Papago API", "ID": "Naver Papago NMT", "BLEU": "Skipped", "Avg Speed": "Skipped", "Device": "API"})

    # --- 6. Display Results ---
    print("\n\n--- Overall Evaluation Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.csv"
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"\nEvaluation results saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        print(traceback.format_exc())

    end_time_eval = time.time()
    print(f"\nTotal evaluation script time: {(end_time_eval - start_time_eval)/60:.2f} minutes.")

if __name__ == "__main__":
    # Ensure .env is loaded by src.config when other modules import it.
    # If running evaluate.py directly and it doesn't trigger .env loading elsewhere,
    # you might need to load_dotenv() here or ensure src.config is imported early.
    # from dotenv import load_dotenv
    # load_dotenv()
    evaluate_models()