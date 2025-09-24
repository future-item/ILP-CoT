import os
import time
import json
import requests
from typing import List, Dict, Any

GPT_IMAGE_API_KEY = os.getenv('GPT_IMAGE_API_KEY', 'YOUR_IMAGE_API_KEY_HERE')
GPT_TEXT_API_KEY = os.getenv('GPT_TEXT_API_KEY', 'YOUR_TEXT_API_KEY_HERE')

API_URL = "YOUR_API_URL"
MODEL_NAME = "gpt-4o-2024-11-20"
MAX_TOKENS = 2000
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 100


def _post_request(payload: Dict[str, Any], api_key: str) -> str:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SECONDS
            )

            if response.status_code == 200:
                response_data = response.json()
                choices = response_data.get("choices", [])
                if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                    output_text = choices[0]["message"]["content"]
                    with open("output_text.txt", "w", encoding="utf-8") as f:
                        f.write(output_text)
                    return output_text
                else:
                    print("API Error: Response format is invalid or choices are empty.")
                    return ""
            else:
                print(f"API Error: Received status code {response.status_code}")
                print(f"Response body: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)

    print("All retry attempts failed. Returning empty string.")
    return ""

def image_to_text(input_images_path: str, image_filenames: List[str], prompt_text: str) -> str:
    if not GPT_IMAGE_API_KEY or 'YOUR_IMAGE_API_KEY_HERE' in GPT_IMAGE_API_KEY:
        print("Error: Image API key is not configured. Please set the GPT_IMAGE_API_KEY environment variable.")
        return ""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    for filename in image_filenames:
        image_path = os.path.join(input_images_path, filename)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_path}
        })

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS
    }

    return _post_request(payload, GPT_IMAGE_API_KEY)


def text_to_text(prompt_text: str) -> str:
    if not GPT_TEXT_API_KEY or 'YOUR_TEXT_API_KEY_HERE' in GPT_TEXT_API_KEY:
        print("Error: Text API key is not configured. Please set the GPT_TEXT_API_KEY environment variable.")
        return ""

    messages = [{"role": "user", "content": prompt_text}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS
    }

    return _post_request(payload, GPT_TEXT_API_KEY)