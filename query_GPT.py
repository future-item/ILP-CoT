import os
import time
import json
import requests


max_retries = 3
retry_delay = 5  # seconds
def image_to_text(input_images_path, image_filenames, prompt_text):
    headers = {
        'Authorization': '',
        'Content-Type': '',
    }
    url = ""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

    for image_filename in image_filenames:
        image_url = os.path.join(input_images_path, image_filename)
        print("Generated image URL:", image_url)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    payload = json.dumps({
        "model": "gpt-4o-2024-11-20",
        "messages": messages,
        "max_tokens": 2000
    })


    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=100)

            if response.status_code == 200:
                response_data = response.json()

                choices = response_data.get("choices", [])
                if choices:
                    output_text = choices[0]["message"]["content"]
                    with open("output_text.txt", "w", encoding="utf-8") as file:
                        file.write(output_text)
                    return output_text
                else:
                    print("No content returned in choices.")
                    return ""
            else:
                print("Error:", response.status_code)
                print("Response:", response.text)

        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print("All retry attempts failed.")
    return ""

def text_to_text(prompt_text):
    headers = {
        'Authorization': '',
        'Content-Type': '',
    }
    url = ""

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    payload = json.dumps({
        "model": "gpt-4o-2024-11-20",
        "messages": messages,
        "max_tokens": 2000
    })

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=60)

            if response.status_code == 200:
                response_data = response.json()

                choices = response_data.get("choices", [])
                if choices:
                    output_text = choices[0]["message"]["content"]
                    with open("output_text.txt", "w", encoding="utf-8") as file:
                        file.write(output_text)
                    return output_text
                else:
                    print("No content returned in choices.")
                    return ""
            else:
                print("Error:", response.status_code)
                print("Response:", response.text)

        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print("All retry attempts failed.")
    return ""