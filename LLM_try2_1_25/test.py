import torch
import clip
import requests
from PIL import Image
from io import BytesIO

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 输入文本
texts = [
    "The plant is oriented towards the light, and its stem is green in color.",
    "The plant is oriented towards the light and is under sunlight.",
    "The plant is oriented towards the light and has yellow-colored petals.",
    "The plant is in an outdoor environment and is oriented towards the light.",
    "The plant has green-colored leaves and is oriented towards the light."
]

# 图片基础URL
base_url = '/home/pengyf/Third_Work/Work/ILP-CoT-Customization/positive/'
base_url_n = '/home/pengyf/Third_Work/Work/ILP-CoT-Customization/negative/'
positive_names = ['7WUCBG.png', '7WUzCJ.png', '7WUtAL.png']
negative_names = ['7WUXRt.png', '7WUIqe.png', '7WUjZb.png']

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
scores = 0.8 * (text_features @ positive_avg) - 0.2 * (text_features @ negative_avg)

# 取分数最大的序号
best_score_index = torch.argmax(scores).item()

print(f"最佳匹配序号: {best_score_index + 1}, 描述: {texts[best_score_index]}")
