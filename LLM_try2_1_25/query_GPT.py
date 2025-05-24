import os
import time
import json
import requests

####### only query once #########
# def Image_to_text(input_images_path, image_filenames, prompt_text):
#     headers = {
#         'Authorization': 'Bearer sk-vBGhv7oP6p4IfEBDbcmwV2kKX6adCZ9bXO6UJ1iSEXDXclld',
#         'Content-Type': 'application/json',
#     }
#     url = "https://xiaoai.plus/v1/chat/completions"
#
#     messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
#
#     for image_filename in image_filenames:
#         image_url = os.path.join(input_images_path, image_filename)
#         print("Generated image URL:", image_url)  # Debug
#         messages[0]["content"].append({
#             "type": "image_url",
#             "image_url": {"url": image_url}
#         })
#
#     payload = json.dumps({
#         "model": "gpt-4o-2024-08-06",
#         "messages": messages,
#         "max_tokens": 2000
#     })
#
#     try:
#         response = requests.post(url, headers=headers, data=payload, timeout=100)
#
#         if response.status_code == 200:
#             response_data = response.json()
#             # print("Full Response:", json.dumps(response_data, indent=4))  # Debug 完整响应内容
#
#             choices = response_data.get("choices", [])
#             if choices:
#                 output_text = choices[0]["message"]["content"]
#                 # print("Output:", output_text)
#                 # 保存到文件以防内容过长截断
#                 with open("output_text.txt", "w", encoding="utf-8") as file:
#                     file.write(output_text)
#             else:
#                 print("No content returned in choices.")
#                 output_text = ""
#         else:
#             print("Error:", response.status_code)
#             print("Response:", response.text)
#             output_text = ""
#
#     except requests.exceptions.RequestException as e:
#         print("Request failed:", e)
#         output_text = ""
#
#     return output_text
#
# def Text_to_text(prompt_text):
#     headers = {
#         'Authorization': 'Bearer sk-vBGhv7oP6p4IfEBDbcmwV2kKX6adCZ9bXO6UJ1iSEXDXclld',  # 替换为你的 API Key
#         'Content-Type': 'application/json',
#     }
#     url = "https://xiaoai.plus/v1/chat/completions"  # API 端点
#
#     messages = [
#         {"role": "user", "content": prompt_text}
#     ]
#
#     payload = json.dumps({
#         "model": "gpt-4o-2024-08-06",
#         "messages": messages,
#         "max_tokens": 2000
#     })
#
#     try:
#         response = requests.post(url, headers=headers, data=payload, timeout=60)
#
#         if response.status_code == 200:
#             response_data = response.json()
#             # print("Full Response:", json.dumps(response_data, indent=4))
#
#             choices = response_data.get("choices", [])
#             if choices:
#                 output_text = choices[0]["message"]["content"]
#                 # print("Output:", output_text)
#                 with open("output_text.txt", "w", encoding="utf-8") as file:
#                     file.write(output_text)
#             else:
#                 print("No content returned in choices.")
#                 output_text = ""
#         else:
#             print("Error:", response.status_code)
#             print("Response:", response.text)
#             output_text = ""
#
#     except requests.exceptions.RequestException as e:
#         print("Request failed:", e)
#         output_text = ""
#
#     return output_text



max_retries = 3
retry_delay = 5  # seconds
def Image_to_text(input_images_path, image_filenames, prompt_text):
    headers = {
        'Authorization': 'Bearer sk-vBGhv7oP6p4IfEBDbcmwV2kKX6adCZ9bXO6UJ1iSEXDXclld',
        'Content-Type': 'application/json',
    }
    url = "https://xiaoai.plus/v1/chat/completions"

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

    for image_filename in image_filenames:
        image_url = os.path.join(input_images_path, image_filename)
        print("Generated image URL:", image_url)  # Debug
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

def Text_to_text(prompt_text):
    headers = {
        'Authorization': 'Bearer sk-W80mlv5px007rULvZxvWi9piRMVJMOJMe0LsThPSEjAfgQEf',  # 替换为你的 API Key
        'Content-Type': 'application/json',
    }
    url = "https://xiaoai.plus/v1/chat/completions"  # API 端点

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