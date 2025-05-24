import re
import torch
import clip

def select_rules_using_clip(texts,positive_names,negative_names):

    parts = re.split(r'\s*\d+\.\s*', texts)
    texts = [p.strip() for p in parts if p.strip()]

    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    base_url = '/home/pengyf/Third_Work/Work/ILP-CoT-Customization/positive/'
    base_url_n = '/home/pengyf/Third_Work/Work/ILP-CoT-Customization/negative/'


    positive_image_paths = [base_url + name for name in positive_names]
    negative_image_paths = [base_url_n + name for name in negative_names]

    from PIL import Image
    def load_images_from_paths(path_list):
        images = []
        for path in path_list:
            image = Image.open(path).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            images.append(image)
        return torch.cat(images, dim=0)

    positive_images = load_images_from_paths(positive_image_paths)
    negative_images = load_images_from_paths(negative_image_paths)

    with torch.no_grad():
        positive_image_features = model.encode_image(positive_images)
        negative_image_features = model.encode_image(negative_images)

    positive_avg = positive_image_features.mean(dim=0)
    negative_avg = negative_image_features.mean(dim=0)

    # 计算文本embedding并比较
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # 计算分数
    scores = 0.3 * (text_features @ positive_avg) - 0.7 * (text_features @ negative_avg)

    best_score_index = torch.argmax(scores).item()

    return texts[best_score_index]

def extract_metarules(input_str):
    pattern = r"metarule\(\[.*?\],\s*\[.*?\],\s*\[\[.*?\]\]\)\."
    matches = re.findall(pattern, input_str, re.DOTALL)
    clean_matches = []
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip()
        if not cleaned.endswith("."):
            cleaned += "."
        clean_matches.append(cleaned)
    return clean_matches

def extract_prolog_facts(input_str):
    pattern = r"\b[A-Z][a-zA-Z_]+\w*\(\s*[^)]+\s*\)\s*\.(?=\s|$)"

    matches = re.findall(pattern, input_str, re.MULTILINE | re.DOTALL)

    clean_facts = []
    for fact in matches:
        # 标准化空格：移除参数间多余空格，保留必要空格
        cleaned = re.sub(r"\s*,\s*", ", ", fact)  # 规范逗号空格
        cleaned = re.sub(r"\(\s+", "(", cleaned)  # 移除左括号后空格
        cleaned = re.sub(r"\s+\)", ")", cleaned)  # 移除右括号前空格
        cleaned = re.sub(r"\s+", " ", cleaned)  # 合并多余空格
        clean_facts.append(cleaned.strip())

    return clean_facts

def extract_prolog_content(s):
    # 分割两次，先去掉开头标记，再去掉结尾标记
    return s.split('```prolog', 1)[-1].rsplit('```', 1)[0].strip()

