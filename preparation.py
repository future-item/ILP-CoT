import torch
import re
from typing import List, Dict, Any, Tuple


def preparation(index: int) -> Tuple[str, str, str, List[str], List[str], List[List[str]], List[List[int]], int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_images_path = ["..."]
    input_images_path_neg = ["..."]
    pos_image_filenames = ["..."]
    neg_image_filenames = ["..."]

    target_classes = [['cat', 'dog'], ['man', 'women', 'child'], ['boy', 'girl'], ['human'], ['cat', 'child'], ['cat'],
                      ['lady', 'chair'], ['dog'], ['child', 'adult'],
                      ['male', 'female'], ['male', 'female'], ['male', 'female'], ['human'], ['mom', 'child'],
                      ['teacher', 'student'], ['pig'], ['human'], ['cow'], ['tree'], ['sunflower'], ['human'],
                      ['teacher', 'student']]
    target_num = [[1, 1], [1, 1, 1], [1, 1], [2], [1, 1], [1], [1, 2], [2], [1, 1],
                  [1, 1], [1, 1], [1, 1], [1], [1, 1], [1, 2], [1], [3], [1], [2], [1], [1], [1, 1]]
    choose_class = index
    input_images_path = input_images_path[choose_class][0]
    input_images_path_neg = input_images_path_neg[choose_class][0]
    pos_image_filenames = pos_image_filenames[choose_class]
    neg_image_filenames = neg_image_filenames[choose_class]
    return device, input_images_path, input_images_path_neg, pos_image_filenames, neg_image_filenames, target_classes, target_num, choose_class
