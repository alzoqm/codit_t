import time
import requests
from typing import List
import random

# FastAPI server URL
API_BASE_URL = "http://localhost:8000/translate"

# Sample Korean sentences (try to get diverse lengths)
KOREAN_SAMPLE_SENTENCES = [
    "안녕하세요, 오늘 날씨가 정말 좋네요.",
    "이것은 한국어 번역 속도 테스트를 위한 샘플 문장입니다.",
    "기계 번역 모델의 성능을 평가하는 것은 매우 중요한 작업이며, 다양한 지표를 사용합니다.",
    "인공지능 기술은 빠르게 발전하고 있으며, 우리 생활의 많은 부분을 변화시키고 있습니다.",
    "백범 김구 선생님은 '나의 소원은 우리나라 대한의 완전한 자주독립이오' 라고 말씀하셨습니다.",
    "복잡하고 긴 문장을 번역할 때 모델이 얼마나 빠르고 정확하게 처리하는지 확인해봅시다.",
    "데이터 프라이버시와 보안은 현대 사회에서 점점 더 중요해지고 있는 문제입니다.",
    "짧은 문장.",
    "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열.",
    "대한민국은 아름다운 사계절을 가진 나라입니다. 봄에는 벚꽃, 여름에는 푸른 바다, 가을에는 단풍, 겨울에는 눈 덮인 산을 볼 수 있습니다."
]


def generate_test_text(num_words_target: int = 100) -> str:
    """Generates a test text of approximately num_words_target words."""
    text = ""
    current_words = 0
    while current_words < num_words_target:
        sentence = random.choice(KOREAN_SAMPLE_SENTENCES)
        text += sentence + " "
        current_words += len(sentence.split())
        if len(text) > 10000: # Safety break for very large targets
             break 
    return text.strip()


def measure_translation_speed(model_type: str, text_to_translate: str, num_requests: int = 5) -> float | None:
    """Measures average translation time for a given model type via API."""
    endpoint_map = {
        "base": f"{API_BASE_URL}/base",
        "finetuned": f"{API_BASE_URL}/finetuned",
        "onnx": f"{API_BASE_URL}/onnx",
        "papago": f"{API_BASE_URL}/papago", # Papago might have rate limits
    }
    if model_type not in endpoint_map:
        print(f"Unknown model type for speed test: {model_type}")
        return None

    total_time = 0
    successful_requests = 0
    
    print(f"  Testing speed for {model_type} with text length {len(text_to_translate.split())} words...")

    for i in range(num_requests):
        try:
            payload = {"text": text_to_translate, "source_lang": "ko", "target_lang": "en"}
            start_time = time.perf_counter()
            response = requests.post(endpoint_map[model_type], json=payload, timeout=60) # Longer timeout for speed test
            response.raise_for_status()
            elapsed_time = time.perf_counter() - start_time
            
            # Get processing_time_seconds from response if available, otherwise use measured RTT
            api_processing_time = response.json().get("processing_time_seconds")
            if api_processing_time is not None:
                total_time += api_processing_time # Use server-reported time if available
            else:
                total_time += elapsed_time # Use client-measured round-trip time

            successful_requests += 1
            # print(f"    Request {i+1}/{num_requests} for {model_type}: {elapsed_time:.4f}s (API time: {api_processing_time if api_processing_time else 'N/A'})")
            time.sleep(0.2) # Small delay between requests
        except requests.RequestException as e:
            print(f"    Request failed for {model_type}: {e}")
            if i == 0: # If first request fails, likely server issue or model not loaded
                return None 
        except KeyError:
             print(f"    Error parsing API response for {model_type}. Response: {response.text}")


    if successful_requests == 0:
        return None
    
    average_time = total_time / successful_requests
    return average_time


def run_speed_evaluation():
    print("Starting translation speed evaluation...")
    print("Ensure the FastAPI server (src/inference/main.py) is running on http://localhost:8000.")
    # input("Press Enter to start speed tests once the server is running...")

    test_text_100_words = generate_test_text(num_words_target=100)
    num_actual_words = len(test_text_100_words.split())
    print(f"\nGenerated test text with approximately {num_actual_words} words.")

    models_to_test = ["base", "finetuned", "onnx"] # Add "papago" if you want to test it, be mindful of quotas

    results = {}

    for model_name in models_to_test:
        print(f"\nMeasuring speed for model: {model_name}")
        avg_time = measure_translation_speed(model_name, test_text_100_words, num_requests=5)
        results[model_name] = avg_time
        if avg_time is not None:
            print(f"  Average translation time for {model_name} ({num_actual_words} words): {avg_time:.4f} seconds")
            if avg_time > 0:
                 words_per_second = num_actual_words / avg_time
                 print(f"  Estimated words per second for {model_name}: {words_per_second:.2f} WPS")
        else:
            print(f"  Could not measure speed for {model_name} (likely model not loaded or API error).")

    print("\n--- Speed Evaluation Summary ---")
    print(f"Text length: {num_actual_words} words")
    for model_name, avg_time in results.items():
        if avg_time is not None:
            wps_info = f"({num_actual_words / avg_time:.2f} WPS)" if avg_time > 0 else ""
            print(f"Model: {model_name:<15} | Avg. Time: {avg_time:.4f} seconds {wps_info}")
        else:
            print(f"Model: {model_name:<15} | Avg. Time: N/A (test failed)")

if __name__ == "__main__":
    try:
        run_speed_evaluation()
    except Exception as e:
        print(f"An error occurred during speed evaluation: {e}")
        import traceback
        traceback.print_exc()