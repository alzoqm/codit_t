import json
import requests
from src.config import PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET

PAPAGO_API_URL = "https://papago.apigw.ntruss.com/nmt/v1/translation"

def translate_with_papago(text: str, source_lang: str = "ko", target_lang: str = "en") -> str | None:
    if not PAPAGO_CLIENT_ID or not PAPAGO_CLIENT_SECRET:
        print("Papago API credentials (PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET) not found in .env")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-NCP-APIGW-API-KEY-ID": PAPAGO_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": PAPAGO_CLIENT_SECRET
    }
    payload = {
        "source": source_lang,
        "target": target_lang,
        "text": text
    }

    try:
        response = requests.post(PAPAGO_API_URL, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        if "message" in result and "result" in result["message"]:
            return result["message"]["result"]["translatedText"]
        else:
            print(f"Papago API error: Unexpected response format. Full response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Papago API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}, Response text: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Papago API error: Could not decode JSON response. Response text: {response.text}")
        return None


if __name__ == "__main__":
    # Example usage (ensure .env has Papago credentials)
    if not PAPAGO_CLIENT_ID or not PAPAGO_CLIENT_SECRET:
        print("Please set PAPAGO_CLIENT_ID and PAPAGO_CLIENT_SECRET in your .env file to test Papago.")
    else:
        sample_text = "안녕하세요, 만나서 반갑습니다."
        translation = translate_with_papago(sample_text)
        if translation:
            print(f"Original: {sample_text}")
            print(f"Papago Translation: {translation}")
        else:
            print("Translation failed.")