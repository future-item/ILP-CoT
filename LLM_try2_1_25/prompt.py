import re
import os
import ast
from typing import List
from collections import Counter
import string

prolog_dir = "/home/pengyf/Third_Work/Work/LLM_try2_1_25/PL_File/learn.pl"
output_dir = "/home/pengyf/Third_Work/Work/LLM_try2_1_25/Output"


# def Knowledge_neg(input):
#     prompt = f"""[{input}],The above information are positive info, now we are going to extract negative info from images.
#     The subjects in negative info are [{subjects_names_neg[global_choose_class]}].
#     Based on positive info to find the negative info, and write down the info in the format like positive info"""
#     return prompt
# def Knowledge(input):
#     prompt = f"""[{input}], above are descriptions, write down the above descriptions as prolog form.
#     One sentence may includes many descriptins, dont miss any description!
#     For example:
#     [
#     The roof is always red in a cloudy day and a Stevens is always drive by
#     Description includes:
#     1.roof is red
#     2.steven is driving
#     3.sky is cloudy
#     if subjects is roof, steven,sky, then we would conclude:
#     red(roof). driving(steven).cloudy(sky).
#     ].
#     In case you missed any description, you should check all sentence at least twice.
#     The subjects in each images are [{subjects_names}].
#     Now write down these information in such a prolog format:
#     Description(subject1, subject2, ..., subjectN)
#     Example if Description is golden_fur, and subject is dog:
#     golden_fur(dog).
#     However
#     size(dog,small). is illegal! since small is not subject.
#     size_small(dog) is legal.
#     """
#     return prompt
#
# # def capturer(input):
# #     prompt = f"""based on the description in the following parts [{input}],
# #     firstly write down crucial information in description,
# #     secondely, check each information image by image(dont miss any information).
# #     for example:
# #     hypothesis 1 is: The roof is always red in a cloudy day and a Stevens is always drive by -->
# #     capture:
# #     1.roof is red in image1,image2,image3,
# #     2.steven is driving image1,image2,image3,
# #     3.sky is cloudy image1,image2,image3.
# #     hypothesis 2 is: The sky is blue in summer, and summit is very high -->
# #     capture:
# #     1.sky is blue in image1,image2,image3,
# #     2.mountain in image1,image2,image3,
# #     3.mountain is high in image1,image2,image3.
# #     ...
# #
# #     """
# #     return prompt
#
# def capturer(input):
#     prompt = f"""based on the description in the following parts [{input}],
#     首先，将descriptions切割成一个一个的小知识点。
#     secondely, check each information image by image(dont miss any information).
#     for example:
#     hypothesis 1 is: The roof is always red in a cloudy day and a Stevens is always drive by -->
#     capture:
#     1.roof is red in image1,image2,image3,
#     2.steven is driving image1,image2,image3,
#     3.sky is cloudy image1,image2,image3.
#     hypothesis 2 is: The sky is blue in summer, and summit is very high -->
#     capture:
#     1.sky is blue in image1,image2,image3,
#     2.mountain in image1,image2,image3,
#     3.mountain is high in image1,image2,image3.
#     ...
#
#     """

# def hypothesis_space():
#     prompt = f"""there is a rule consistent cross these images , we are going to find that rule, now based on all images make five descriptions,
#     make each description as concrete as possible,and write down in such a form:
#     Description1: .
#     ##
#     Description2: .
#     ##
#     Description3: .
#     ##
#     Description4: .
#     ##
#     Description5: .
#     ...
#     """
#     return prompt
# def hypothesis_space():
#     prompt = f"""there is a rule consistent cross these images , we are going to find that rule.
#     Now based on all images make five descriptions, make each description as concrete(specific) as possible, and write down in such a form:
#     Description1: .
#     ##
#     Description2: .
#     ##
#     Description3: .
#     ##
#     Description4: .
#     ##
#     Description5: .
#     ...
#     """
#     return prompt

import re

def check_format(input):
    prompt = f"""[{input}] : Above content may have format error, I need help to make clarify
    """
    return prompt
def delete_wrong_info(input):
    prompt = f"""Based on images, delete the info that is not true without explanation.
    info:
    [{input}]
    
    Every words should be in lower case.
    """
    return prompt

def extract_capture_from_hypothesis_space_neg(input):
    pattern = r":-\s*(.*?)\."
    matches = re.findall(pattern, input, re.DOTALL)
    knowledge_list = [match.strip().replace("\n", " ") for match in matches]

###
    # def transform_predicate(predicate_str):
    #     """
    #     将单个谓词字符串转换为将参数替换为数字的形式。
    #     例如： "presentation_bouquet_flowers(Boy, Girl)"  --> "presentation_bouquet_flowers(2)"
    #     """
    #     # 移除两边空白字符后用正则表达式匹配谓词名和参数列表
    #     match = re.match(r"(\w+)\((.*)\)", predicate_str.strip())
    #     if match:
    #         predicate_name = match.group(1)  # 提取谓词名
    #         args_str = match.group(2).strip()  # 提取参数列表字符串
    #         if args_str == '':
    #             num_args = 0
    #         else:
    #             # 按逗号分割参数，同时去除空白字符
    #             args = [arg.strip() for arg in args_str.split(',')]
    #             num_args = len(args)
    #         return f"{predicate_name}({num_args})"
    #     else:
    #         # 如果没有匹配上，直接返回原始字符串（或可选择报错）
    #         return predicate_str
    #
    # def process_list(pred_list):
    #     """
    #     对给定的包含谓词字符串的列表进行转换，
    #     每个谓词中的参数都替换为对应的参数个数数字。
    #     """
    #     new_list = []
    #     for item in pred_list:
    #         # 使用正则表达式按逗号分隔谓词，
    #         # 这里利用正向预查 (?=\w+\()，保证逗号后面是谓词开头
    #         predicates = re.split(r',\s*(?=\w+\()', item)
    #         # 对每个谓词进行转换
    #         transformed = [transform_predicate(pred) for pred in predicates]
    #         # 将转换后的谓词重新组合为一个字符串（以逗号和空格分隔）
    #         new_item = ', '.join(transformed)
    #         new_list.append(new_item)
    #     return new_list

    def transform_predicate(predicate_str, subjects):
        """
        将单个谓词字符串转换为：
          1. 保留谓词名不变。
          2. 对参数列表中的每个参数，如果参数以 subjects 中的任一元素（不区分大小写）开头，
             则将该参数替换为下划线 "_"；否则保持原样。

        例如：
          transform_predicate("give(boy0, girl0, flower)", ['boy','girl'])
             --> "give(_, _, flower)"
        """
        predicate_str = predicate_str.strip()
        # 匹配谓词名称和括号内的参数列表
        match = re.match(r"(\w+)\((.*)\)", predicate_str)
        if match:
            predicate_name = match.group(1)
            args_str = match.group(2).strip()
            if args_str == "":
                # 无参数时直接返回
                new_args = []
            else:
                # 按逗号分割参数，并去掉空白
                args = [arg.strip() for arg in args_str.split(',')]
                new_args = []
                for arg in args:
                    # 检查该参数是否属于 subjects
                    # 这里采用忽略大小写的前缀匹配（例如 "boy0" 以 "boy" 开头）
                    if any(arg.lower().startswith(sub.lower()) for sub in subjects):
                        new_args.append('_')
                    else:
                        new_args.append(arg)
            # 使用转换后的参数列表构造新的谓词字符串
            return f"{predicate_name}({', '.join(new_args)})"
        else:
            # 如果没有匹配上则直接返回原始字符串
            return predicate_str

    def process_list(pred_list, subjects):
        """
        对包含多个谓词字符串的列表进行转换：
          - 每个列表项可能包含多个谓词（用逗号分隔）。
          - 对每个谓词调用 transform_predicate 函数，并传入 subjects 列表。
          - 最后将转换后的谓词重新组合为字符串。
        """
        new_list = []
        for item in pred_list:
            # 利用正则表达式分隔多个谓词
            predicates = re.split(r',\s*(?=\w+\()', item)
            # 对每个谓词进行转换
            transformed = [transform_predicate(pred, subjects) for pred in predicates]
            # 将转换后的谓词重新组合为一个字符串（以逗号和空格连接）
            new_item = ', '.join(transformed)
            new_list.append(new_item)
        return new_list
    knowledge_list = process_list(knowledge_list,chosen_classes)

    def remove_subject_info(predicate_list, subjects):
        # 构造一个全小写的 subjects 集合，便于忽略大小写比较
        subjects_set = {s.lower() for s in subjects}

        processed_list = []

        # 对每个字符串进行处理（原来每个字符串内包含多个 predicate，用逗号分隔）
        for pred_str in predicate_list:
            # 按逗号分割，注意要 strip 去掉额外空白
            predicates = [p.strip() for p in pred_str.split(",")]
            new_predicates = []

            for pred in predicates:
                # 找到第一个左括号 "(" 作为分割点
                idx = pred.find("(")
                if idx == -1:
                    # 如果格式不对（没有括号），则直接保留原字符串
                    new_predicates.append(pred)
                    continue

                # 提取 predicate 名称和参数部分
                name = pred[:idx]
                params = pred[idx:]  # 包括括号及内容

                # 按下划线分割 predicate 名称
                parts = name.split("_")
                # 过滤掉那些和 subjects 中（忽略大小写）匹配的部分
                new_parts = [part for part in parts if part.lower() not in subjects_set]

                # 如果全部部分都被删除，则说明 predicate 名称仅包含 subject 信息
                if not new_parts:
                    # 按题目要求，这样的 predicate 直接删除，不再加入结果中
                    continue

                # 否则，重新拼接新的 predicate 名称
                new_name = "_".join(new_parts)
                new_pred = new_name + params
                new_predicates.append(new_pred)

            # 将处理后的 predicate 用逗号和适当空格拼接回一个字符串
            # 此处使用 ",     " 与原输入格式保持一致
            processed_list.append(",     ".join(new_predicates))

        return processed_list

    knowledge_list = remove_subject_info(knowledge_list,chosen_classes)


    ###

    prompt = f"""Use the knowledge provided in:[{knowledge_list}] to extract knowledge for each image. 
    The numbers in parentheses indicate the required number of subjects that must be present. 
    For example, if the subjects are "cat0" and "dog0," then fur_golden(_) could apply to either fur_golden(cat0) or fur_golden(dog0) based on the information captured from the images.
    For example, if subjects are "man1" and "women1", then smilling(_,_) coule apply to either smilling(man1,women1) or smilling(women1,man1) based on the information captured from the images.
    Now, given the subjects for each image as:[{subjects_names_neg[global_choose_class]}]. Return captured knowledge in prolog format: Description(subject). """

    # prompt = f"""Extract matching content from the image based on the words in the [{knowledge_list}].
    # Return captured knowledge in prolog format: Description(subject).
    # Subjects is in such an order for these images: [{subjects_names_neg[global_choose_class]}].
    # Don't add in new knowledge.
    # Example of Description(subject):
    # Golden_fur(Dog)
    # """

    return prompt

def extract_capture_from_hypothesis_space(input):
    pattern = r":-\s*(.*?)\."
    matches = re.findall(pattern, input, re.DOTALL)
    knowledge_list = [match.strip().replace("\n", " ") for match in matches]

    def remove_subject_info(predicate_list, subjects):
        # 构造一个全小写的 subjects 集合，便于忽略大小写比较
        subjects_set = {s.lower() for s in subjects}

        processed_list = []

        # 对每个字符串进行处理（原来每个字符串内包含多个 predicate，用逗号分隔）
        for pred_str in predicate_list:
            # 按逗号分割，注意要 strip 去掉额外空白
            predicates = [p.strip() for p in pred_str.split(",")]
            new_predicates = []

            for pred in predicates:
                # 找到第一个左括号 "(" 作为分割点
                idx = pred.find("(")
                if idx == -1:
                    # 如果格式不对（没有括号），则直接保留原字符串
                    new_predicates.append(pred)
                    continue

                # 提取 predicate 名称和参数部分
                name = pred[:idx]
                params = pred[idx:]  # 包括括号及内容

                # 按下划线分割 predicate 名称
                parts = name.split("_")
                # 过滤掉那些和 subjects 中（忽略大小写）匹配的部分
                new_parts = [part for part in parts if part.lower() not in subjects_set]

                # 如果全部部分都被删除，则说明 predicate 名称仅包含 subject 信息
                if not new_parts:
                    # 按题目要求，这样的 predicate 直接删除，不再加入结果中
                    continue

                # 否则，重新拼接新的 predicate 名称
                new_name = "_".join(new_parts)
                new_pred = new_name + params
                new_predicates.append(new_pred)

            # 将处理后的 predicate 用逗号和适当空格拼接回一个字符串
            # 此处使用 ",     " 与原输入格式保持一致
            processed_list.append(",     ".join(new_predicates))

        return processed_list

    knowledge_list = remove_subject_info(knowledge_list,chosen_classes)

    prompt = f"""Extract matching content from the image based on the words in the [{knowledge_list}].
    Return captured knowledge in prolog format: Description(subject).
    Subjects is in such an order for these images: [{subjects_names}].
    Don't add in new knowledge.
    Example of Description(subject):
    Golden_fur(Dog) 
    """

    return prompt
def hypothesis_space():
    prompt = f"""the images has subjects [{subjects_names}], there is a rule consistent cross these images , we are going to find that rule. 
    Now based on all images make five descriptions, make each description as concrete(specific) as possible.
    Write down these descriptions in prolog form ( Hypothesis(subjects):- description0(subject),description1(subject),... ).
    An Example with subject (Cat, Dog):
    ...
prolog
Hypothesis_1(Cat, Dog) :-
    environment_indoors(Cat),
    environment_indoors(Dog),
    interaction_on_top(Cat, Dog).


### Rule Set 2: Color Attributes Consistency
- Cats are always striped, and dogs are always golden.
  
prolog
Hypothesis_2(Cat, Dog) :-
    color_striped(Cat),
    (color_golden(Dog);color_blue(Dog)).
    ...
    """
    return prompt

def build_results(chosen_classes, chosen_num, num_images):
    results = []
    indices = [0] * len(chosen_classes)  # 分别记录每个类的索引

    for i in range(num_images):
        result_list = []  # 改成列表
        for cls_idx, (cls, num) in enumerate(zip(chosen_classes, chosen_num)):
            for _ in range(num):
                result_list.append(f"{cls}{indices[cls_idx]}")  # 使用append添加元素
                indices[cls_idx] += 1
        results.append(result_list)  # 最里面的部分是列表

    return results

def Image_to_text(target_classes,choose_class,target_num,image_filenames):
    def generate_subject_var(chosen_classes, num):
        """
        根据 chosen_classes 和 num 生成 subject_var。
        每个类在 subject_var 中的数量由 num 决定。
        """
        subject_var = []
        idx = 0  # 字母索引
        for cls, count in zip(chosen_classes, num):
            if count == 1:
                subject_var.append((string.ascii_uppercase[idx], cls))  # 单个类直接映射
                idx += 1
            else:
                for i in range(count):
                    subject_var.append((string.ascii_uppercase[idx], f"{cls}{i}"))  # 为类创建多个实例
                    idx += 1
        return subject_var
    global subjects_names
    global subjects_names_neg
    global chosen_classes
    global global_choose_class
    global num_images
    global chosen_num
    global subject_var_names
    global subject_var_mapping
    global subject_var_names
    global subject_var

    def generate_labels(strings):
        labels = {}
        for i, string in enumerate(strings):
            labels[string] = chr(65 + i)  # 65 是 ASCII 中 'A' 的值
        return labels
    global_choose_class = choose_class
    num_images = len(image_filenames)
    chosen_classes = target_classes[choose_class]
    chosen_num = target_num[choose_class]
    subjects_names = build_results(chosen_classes, chosen_num, num_images)
    subjects_names_neg = [[['cat3', 'dog3'],['cat4', 'dog4'],['cat5', 'dog5']],[['man3','women3','child3'], ['man4','women4','child4'], ['man5','women5','child5']],[['boy3', 'girl3'],['boy4', 'girl4'],['boy5', 'girl5']],[['human6','human7'],['human8','human9'],['human10','human11']],
                          [['cat3','child3'],['cat4','child4'],['cat5','child5']],[['cat3'],['cat4'],['cat5']],[['lady3','desk6','desk7'],['lady4','desk8','desk9'],['lady5','desk10','desk11']],[['dog6','dog7'],['dog8','dog9'],['dog10','dog11']],[['child3','adult3'],['child4','adult4'],['child5','adult5']],
                          [['male3','female3'],['male4','female4'],['male5','female5']],[['male3','female3'],['male4','female4'],['male5','female5']],[['male3','female3'],['male4','female4'],['male5','female5']],[['human3'],['human4'],['human5']],[['mom3','child3'],['mom4','child4'],['mom5','child5']],[['teacher3','student6','student7'],['teacher4','student8','student9'],['teacher5','student10','student11']],
                          [['pig3'],['pig4'],['pig5']],[['human9','human10','human11'],['human12','human13','human14'],['human15','human16','human17']],[['cow3'],['cow4'],['cow5']],[['tree6','tree7'],['tree8','tree9'],['tree10','tree11']],[['sunflower3'],['sunflower4'],['sunflower5']],[['human3'],['human4'],['human5']],[['teacher3','student3'],['teacher4','student4'],['teacher5','student5']]]
    subject_var = generate_labels(chosen_classes)


    # 生成 subject_var
    subject_var_mapping = generate_subject_var(chosen_classes, chosen_num)
    subject_var_names = [var[0] for var in subject_var_mapping]

    text = f"""In these {num_images} images, the main characters are {chosen_classes}. First, design fifteen possible attribute feature descriptions and fifteen relationship feature descriptions based on the main characters of the three images, with the following specific requirements:
0. Each feature description word is a class, and for each main character in the image, this class feature must be captured (e.g., Class: Action, Specific: walking, running, jumping...; Class: Pose, Specific: looking up, lowering head, extending paw...). If a feature cannot be captured, skip it.
1. design attributes and relationship in the form : Attributes:... ; Relationship:... . Examples: Attributes: color,size,direction... ; Relationship: friends, embrace,...
2. Use the same set of feature description words for each image. If a feature cannot be captured, skip it.
3. Fill in the captured features based on the specific observations and summarize them into Prolog code in the form of facts and relationships. For example, the first image includes {subjects_names[0]}, and the second image includes {subjects_names[1]}. The observed facts should be described in the following format:
(0) noun_verb_preposition_adjective_adverb_verb_noun(subject1, subject2, ..., subjectN).
(1) The order of subjects must be defined as subject1, subject2, ..., subjectN... --> cat, dog, human...
(2) Examples:
    0) on(cat0, dog0).
    1) laugh_at(human0,dog0).
    2) shake_hand(Bob, Linda).
    3) link_shoulder(Linda,Cathy).
    4) play_on_ground(cat0, dog0).
    5) size_relatively_large(dog1).
    6) dancing_together(human0, human1).
notes: Avoid repeated capturing like dog(dog), human(human),species_dog(dog) since these predicates are duplicate with its arguments
"""

    return text

def Negative_prompt_check(output):
    prompt = f"""The following content [{output}] are information extracted from these images. However, these extracted information may captured by mistake. Now help me make sure 
    All the information matches the picture information, delete information does not match the image. finally, return the information as the form consistent with the format in input
    (you dont need to explain what/why you delete) """
    return prompt

def Negative_prompt(Output):
    prompt = f"""The following content ["{Output}"] has two parts, the first parts is the designed words capturer, the second parts is the captured words on positive images.  I want to use these word capturer to capture the information from a few given images, which are negative examples. The subjects of these negative examples are respectively: [{subjects_names_neg[global_choose_class]}]. 
However, the subjects are [{subjects_names_neg[global_choose_class]}], but this does not mean that the subject must appear in the image. For example, if the subject is a woman, but in reality, the image contains no women and only men, in such cases, the men should not be mistakenly identified as women. Pay extra attention to this!
I need to capture the information from these negative examples, and you should follow the steps below:

Firstly, identify whether the subjects exist in each images.
Second, Analyze which descriptive terms were used for the positive examples.
Use the descriptive terms that appeared in the positive examples to describe the negative examples.
If a descriptive term cannot capture content, skip that term.
Try to use as many of the descriptive terms as possible.
After completing the capture, write the content in Prolog code format, consistent with the format used for the positive examples."""
    return prompt

def Change_format_to_prolog(input):
    prompt = f"""Rewrite the knowledge captured from each image in the input: "{input}" content into the following Prolog facts (Inductive Logic Programming facts) format, Delete facts that does not meet requirement:

Subjects: {subjects_names}，
Description(subject1, subject2, ..., subjectN). 
The order of subjects must follow [{subjects_names}]


[Notice : 
Strictly limit the positions where the subject and description can appear!
You should noticed that subject should only shows in bracket and description should only appear outside the bracket! Description(subjects) for example:present_in_front_table(Linda,Cathy).
So, if you encounter the following situation, you need to know how to handle it:
Subject: Linda, Cathy
Knowledge: hold_in_hand(Linda, umbrella).
--> Transform the knowledge into: hold_umbrella_in_hand(Linda).
Knowledge: wood(umbrella).
--> Delete this fact, since there is no subject could replace umbrella 
Knowledge: inside_building(shoes)
--> Delete this fact, since there is no subject could replace shoes ]


The answer returns in following example format:

### Prolog Facts and Relationships

**Image 1: cat0, dog0**

```prolog
expression_eyes_closed(dog0).
size_small_relatively(cat0).
...
```

**Image 2: cat1, dog1**

```prolog
interaction_touch(cat1, dog1).
environment_indoors(cat1).
...
```

**Image 3: cat2, dog2**

```prolog
color_striped(cat2).
color_golden(dog2).
...

"""
    return prompt

def Change_format_to_prolog_neg(input,pos):
    prompt = f"""Rewrite the knowledge captured from each image in the negative input: "{input}" content into the following Prolog facts (Inductive Logic Programming facts) format:

Noun_verb_preposition_adjective_adverb_verb_noun(subject1, subject2, ..., subjectN).
The order of subjects must be defined as subject1, subject2, ..., subjectN... --> cat, dog, human... 
Examples:
dog(dog0).
on(cat0, dog0).
link_shoulder(Linda,Cathy).
shake_hand(Bob, Linda).
play_on_ground(cat0, dog0).
size_relatively_large(dog1).
paws_on_golden_face(cat0, dog0). 

The following is example based on positive input:
{pos}
"""
    return prompt


def Facts_to_relationship(input):
    prompt = f"""The above information was captured from three images, involving {subjects_names}.
Convert comparable facts within each image into relationships. Only compare facts within the same image; do not compare facts across different images. Add the discovered relationships to the original content.
Example:
tall(tree0), short(tree1) → taller(tree0, tree1)
jump_10_inches(Lubio), jump_20_inches(Chaty) → jump_higher(Chaty, Lubio)
    """
    return prompt

def Combine_similar_words(input):
    prompt = f""" ["{input}"] The above content is information captured from three images. 
    Now, I need to unify the attributes and relationship where possible without changing it format. Example:
    smaller(Chacy, Nora), larger(Norman, Pink) --> smaller(Chacy, Nora), smaller(Pink, Norman)
    or larger(Nora, Chacy), larger(Norman, Pink)."""

    return prompt

def Delete_low_frequency_words(input):
    prompt = f"""The content inside ["{input}"] is information from three images. 
    Now, remove all predicates that are used only once from the information. and only return me cleaned version"""

    return prompt

# def Delete_low_entropy_info(input):
#     prompt = f"""The content inside ["{input}"] is information from three images, namely:
#     {subjects_names}, they are subjects in three images separately.
# Now, remove the information that lacks informativeness.
# 1. What lacks informativeness means:
#     1). Overlapping Information Between Predicate and Argument
#     Example:
#     gender_male(male)
#     dog(dog)
#     Explanation:
#     Here, both the predicate and the argument convey the same information. The argument duplicates the predicate's meaning, making one redundant.
#
#     2). More Detailed vs. Less Detailed Predicates
#     Example:
#     standing(dog)
#     standing_tall(dog)
#     Explanation:
#     When the arguments are identical, but one predicate provides more detailed or comprehensive information than the other, the more informative predicate should be retained. In this case, standing_tall(dog) is more descriptive than standing(dog), so the latter should be removed.
#
#     3). Semantically Overlapping Predicates
#     Example:
#     happy(man0)
#     happiness(man0)
#     Explanation:
#     These predicates express overlapping information. Since they represent the same concept, one of them can be removed to avoid redundancy.
#
#     Additional Examples of Lacks Informativeness
#     4). Synonymous Predicates
#     Example:
#     parent_of(man, child)
#     father(man, child)
#     Explanation:
#     Both predicates provide essentially the same information, but father(man, child) is more specific as it includes the gender of the parent. Therefore, parent_of(man, child) can be removed.
#
#     5). Predicate Duplication Across Representations
#     Example:
#     owns(man, car)
#     possession(man, car)
#     Explanation:
#     The two predicates represent the same relationship using different terms. Since they are semantically identical, one of them can be removed to avoid redundancy.
#
#     6). Implicit Information Made Explicit
#     Example:
#     dog(animal)
#     mammal(dog)
#     Explanation:
#     The fact that a dog is a mammal is implicit in the knowledge that dog is an animal. Unless explicitly needed in reasoning, mammal(dog) may be redundant when dog(animal) is already present.
#
#     7). Self-Referential or Circular Definitions
#     Example:
#     knows(john, john)
#     Explanation:
#     A self-referential predicate like knows(john, john) typically provides no useful information unless self-knowledge is explicitly required in the reasoning task. It can often be removed as it adds no new knowledge.
#
#     8). Overlapping Temporal Predicates
#     Example:
#     running(dog, time1)
#     running_fast(dog, time1)
#     Explanation:
#     If running_fast(dog, time1) is present, it already implies that dog is running at time1. Thus, running(dog, time1) becomes redundant and should be removed.
#
#
#
# 2. Lastly, you don't need to explain why you remove any predicate,and you only need to take away each predicate you think needed to be removed.
# For exmaple:
# Input:
# ```prolog
# dog(dog).
# cat(cat).
# play_with(dog,cat).
#
# Output:
# ```prolog
# play_with(dog,cat).
# """
#     return prompt
def Delete_low_entropy_info(input):
    prompt = f"""The content inside ["{input}"] is information from {num_images} images, namely:
    {subjects_names}, they are subjects in {num_images} images separately. Now, remove overlapping Information Between Predicate and Argument
    Example:
    gender_male(male)  
    dog(dog)
    Explanation:
    Here, both the predicate and the argument convey the same information. The argument duplicates the predicate's meaning, making one redundant.


return the output as follow:
For exmaple:
Input: 
```prolog
dog(dog).
cat(cat).
play_with(dog,cat).

Output:
```prolog
play_with(dog,cat).
"""
    return prompt

def Delete_based_on_predicted_rule(predicted_rules,knowledge):
    prompt = f"""[{knowledge}], the above is the knowledge extracted from the image. The following [{predicted_rules}] are a few possible rules summarized and inferred from the image and the knowledge.
Now, we need to delete the predicates in the knowledge that are not used by the possible rules. Finally, return the result without explaining:
For example Input:
```prolog
dog(dog).
cat(cat).
play_with(dog,cat).

For example Output:
```prolog
play_with(dog,cat). """
    return prompt
def Build_meta_rules_prepration(knowledge_pos,hypothesis_space):
    prompt = f"""The knowledge is inside the following list [{knowledge_pos}].
    The description is inside the following list [{hypothesis_space}].
    Now based on five descriptions and knowledge to form five prolog rules, Don't miss any hypothesis, and don't add in new knowledge(you should not add in any extra info)!
    Example:
...
prolog
environment_interaction(Cat, Dog) :-
    environment_indoors(cat, Cat),
    environment_indoors(dog, Dog),
    interaction_on_top(Cat, Dog).


### Rule Set 2: Color Attributes Consistency
- Cats are always striped, and dogs are always golden.
  
prolog
consistent_color_attributes(Cat, Dog) :-
    color_striped(cat, Cat),
    (color_golden(dog, Dog);color_blue(dog,Dog)).
..."""

    return prompt

def Build_meta_rules_prepration(Meta_rules):
    prompt = f""""[{Meta_rules}] The content above consists of five hypotheses. I want to transform these five hypotheses into metarules for use in Metagol. 
Here are some examples of metarules:
metarule([P,Q],[P,A,B],[[Q,A,B]]).
metarule([P,Q,R],[P,A,B],[[Q,A,B],[R,A,B]]).
metarule([P,Q,R],[P,A,B],[[Q,A,C],[R,C,B]]).
In these metarules:
P represents the clause (predicate) being defined.
Q and R represent the predicates used within this clause.  
A,B are parameters, the order of parameters should follow hypotheses
Lastly, return me answer in such form: 
[Metarules: metarule([P,Q],[P,A,B],[[Q,A,B]]).
metarule([P,Q,R],[P,A,B],[[Q,A,B],[R,A,B]]).
metarule([P,Q,R],[P,A,B],[[Q,A,C],[R,C,B]]).] """
    return prompt

def final_answer(Answer):

    query = f"""[{Answer}],From the rules above, find the most appropriate rule to describe these three images. 
    The main subjects of the images are {subjects_names}. The rule should describe as many images as possible while including as much detail as possible. 
    The corresponding relationship is [{subject_var}]."""

    return query

#####

def Build_PL_file(knowlege_base_neg, knowledge_base, metarules):
    import re

    # Helper function to extract predicate name and arity from a fact
    def extract_predicates(kb):
        predicates = {}
        for line in kb.splitlines():
            match = re.match(r'(\w+)\(([^)]*)\)', line)
            if match:
                predicate = match.group(1)
                args = match.group(2).split(',')
                arity = len(args)
                predicates[predicate] = arity
        return predicates

    # Extract predicates and their arities from both knowledge bases
    predicates = extract_predicates(knowledge_base)
    predicates.update(extract_predicates(knowlege_base_neg))

    # Generate body_pred definitions
    body_predicates = [f"body_pred({predicate}/{arity})." for predicate, arity in predicates.items()]

    # Generate facts and rules for the .pl file
    output = []
    output.append(":- use_module('metagol').\n")
    output.append("metagol:max_clauses(1).\n")
    output.append("\n% Facts from the knowledge base\n")
    output.append(knowledge_base + "\n")
    output.append(knowlege_base_neg + "\n")

    # Add body_pred definitions
    output.append("\n% Body predicates\n")
    output.extend(body_predicates)
    output.append("\n")

    # Add metarules
    output.append("\n% Metarules\n")
    output.extend(metarules)
    output.append("\n")

    # Add positive and negative examples

    if isinstance(subjects_names[0], list):  # 如果 subjects_names 是嵌套列表
        pos_examples = [f"f({', '.join(pair)})" for pair in subjects_names]
    else:  # 如果 subjects_names 是普通的一维列表
        pos_examples = [f"f({subject})" for subject in subjects_names]

    # 针对 neg_examples，处理 subjects_names_neg 的情况
    if isinstance(subjects_names_neg[global_choose_class][0], list):  # 如果选中的 subjects_names_neg 是嵌套列表
        neg_examples = [f"f({', '.join(pair)})" for pair in subjects_names_neg[global_choose_class]]
    else:  # 如果选中的 subjects_names_neg 是普通的一维列表
        neg_examples = [f"f({subject})" for subject in subjects_names_neg[global_choose_class]]

    # Add learning task
    output.append("\n% Learning Task\n")
    output.append("a :- Pos = [" + ", ".join(pos_examples) + "],\n")
    output.append("     Neg = [" + ", ".join(neg_examples) + "],\n")
    output.append("     learn(Pos, Neg).\n")

    # Write to .pl file
    pl_file_content = "\n".join(output)

    with open(prolog_dir, "w") as file:
        file.write(pl_file_content)

    print("Prolog file generated successfully as 'generated_file.pl'.")
    return prolog_dir

def convert_to_string_literals(content):
    """
    将未加引号的函数调用形式（如 mood_relaxed(cat0)）转换为字符串形式（如 'mood_relaxed(cat0)'）。
    """
    # 匹配未加引号的函数调用形式，如 mood_relaxed(cat0)
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([a-zA-Z0-9_,\s]*\))'
    # 将匹配到的内容加上引号
    converted_content = re.sub(pattern, r"'\1'", content)
    return converted_content

def remove_empty_lists(nested_list):
    # 遍历并递归处理嵌套列表，去掉空的子列表
    cleaned_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):  # 如果是列表，递归处理
            cleaned_sublist = remove_empty_lists(sublist)
            if cleaned_sublist:  # 如果递归后的子列表非空，添加到结果中
                cleaned_list.append(cleaned_sublist)
        else:
            cleaned_list.append(sublist)  # 如果不是列表，直接添加
    return cleaned_list

#####

def negtive_case(output):
    prompt = f"""下面这段内容["Here are the designed attribute and relationship feature descriptions:

Attributes: color, size, pose, action, facial_expression, body_part_position, texture, direction, fur_length, eyes_closed, ears_position, tail_position, body_orientation, mouth_open, mood.

Relationship: sleeping_on, playing_with, paw_touching, lying_next_to, looking_at, holding_still, friendly_interaction, attentive_to, resting_on, sitting_beside, cuddling_with, near, facing, contact, motion_towards.

### Image 1:

Attributes:
- cat0: color(striped), size(small), pose(lying), facial_expression(relaxed), eyes_closed(true), ears_position(erect), body_part_position(paw_on).
- dog0: color(golden), size(large), pose(lying), facial_expression(relaxed), eyes_closed(true), fur_length(long).

Relationship:
- sleeping_on(cat0, dog0).
- cuddling_with(cat0, dog0).

Prolog Facts:
```prolog
cat(cat0).
dog(dog0).
color_striped(cat0).
size_small(cat0).
pose_lying(cat0).
facial_expression_relaxed(cat0).
eyes_closed_true(cat0).
ears_position_erect(cat0).
body_part_position_paw_on(cat0).
color_golden(dog0).
size_large(dog0).
pose_lying(dog0).
facial_expression_relaxed(dog0).
eyes_closed_true(dog0).
fur_length_long(dog0).
sleeping_on(cat0, dog0).
cuddling_with(cat0, dog0).
```

### Image 2:

Attributes:
- cat1: color(striped), size(small), pose(standing), body_part_position(paw_on).
- dog1: color(golden), size(large), pose(lying), direction(towards), facial_expression(attentive).

Relationship:
- playing_with(cat1, dog1).
- paw_touching(cat1, dog1).

Prolog Facts:
```prolog
cat(cat1).
dog(dog1).
color_striped(cat1).
size_small(cat1).
pose_standing(cat1).
body_part_position_paw_on(cat1).
color_golden(dog1).
size_large(dog1).
pose_lying(dog1).
direction_towards(dog1).
facial_expression_attentive(dog1).
playing_with(cat1, dog1).
paw_touching(cat1, dog1).
```

### Image 3:

Attributes:
- cat2: color(striped), size(small), pose(standing), action(raising_paw), facial_expression(playful).
- dog2: color(golden), size(large), pose(lying), facial_expression(happy), mouth_open(true), body_orientation(front).

Relationship:
- playing_with(cat2, dog2).
- looking_at(cat2, dog2).

Prolog Facts:
```prolog
cat(cat2).
dog(dog2).
color_striped(cat2).
size_small(cat2).
pose_standing(cat2).
action_raising_paw(cat2).
facial_expression_playful(cat2).
color_golden(dog2).
size_large(dog2).
pose_lying(dog2).
facial_expression_happy(dog2).
mouth_open_true(dog2).
body_orientation_front(dog2).
playing_with(cat2, dog2).
looking_at(cat2, dog2).
```"]是构造的十五个attributes描述词和十五个relationship描述词，同时还有使用这些描述词来捕捉图片信息的例子。现在我要使用这些
    描述词来捕捉现在给入的几张图片，这几张图片是负例，我需要捕捉这几张负例上的信息，捕捉完成也写成prolog代码格式，并且和正例格式一致"""
    return prompt

############## temp ################


def convert_to_string_literals(text):
    """
    将未加引号的函数调用转换为字符串形式
    例如：pose_lying(cat0) -> "pose_lying(cat0)"
    """
    # 匹配函数调用形式的正则表达式，例如：pose_lying(cat0)
    pattern = r"(\b[a-zA-Z_]+\([^)]*\))"
    # 用双引号包裹函数调用
    text = re.sub(pattern, r'"\1"', text)
    return text

def remove_empty_lists(conclusion_list):
    """
    移除嵌套列表中的空列表
    """
    if isinstance(conclusion_list, list):
        return [remove_empty_lists(item) for item in conclusion_list if item]
    return conclusion_list


def meta_post_process(output_text):
    global subject_var
    def generate_labels(strings):
        labels = {}
        for i, string in enumerate(strings):
            labels[string] = chr(65 + i)  # 65 是 ASCII 中 'A' 的值
        return labels
    subject_var = generate_labels(chosen_classes)

    query = f"""Do following thing: You have extracted information from images. 
Input:
[{output_text}].

Process:
The Conclusion derived in the above information involve possible combinations of mined rules, where:

0. Each sub-sublist represents a combination of rules and includes multiple predicates.
1. The order of objects is specified as follows: [{subjects_names}], and corresponds to [{subject_var_mapping}]( maps [{chosen_classes}] to variables [{subject_var_names}])，These variables are arguments in metarules.
2. each predicates use a different variable to represent in a clause , however, different predicates can shared the same variable in different clause
3. the variable represent predicates should different from arguments (Suggestions for variable name for clauses_name, such as P, and suggestions for variable names for predicates, such as P1, P2, P3, etc.)
4. same predicates should use same variable in one metarules (could be different in different metarules)

Example 1:
Object & Object order
[dog , cat] & [dog2 , cat2]
Input predicates:
[handing(dog2), play_with(dog2,cat2)]

First, determine how many objects there are, how many predicates exist, and who the arguments of each predicate are.
Since there are two different predicates, the first position is [P, P1, P2] --> P is the name of the clause, and P1 and P2 are all the predicates in this statement.
Since there are two objects in total, the second position is [P, A, B].
Since there are two different predicates, and the arguments of each predicate are 1 and 2, corresponding to A; A, B respectively, the third position is [[P1, A], [P2, A, B]].
note: The variable used for predicates should different from variable used for arguments
metarule([P, P1, P2], [P, A, B], [[P1, A], [P2, A, B]]).

Example 2:
Object & Object order
[bird, worm] & [bird1,worm1]

Input predicates:
[fly(bird1), eat(bird1, worm1), move_forward(worm1)]

First, determine how many objects there are, how many predicates exist, and who the arguments of each predicate are.
Since there are three different predicates, the first position is [P, P1, P2, P3] --> P is the name of the clause, and P1, P2, P3 are all the predicates in this statement.
Since there are two objects in total, the second position is [P, A, B].
Since there are three different predicates, and the arguments of each predicate are 1 , 2 and 3, corresponding to A; A, B; B respectively, the third position is [[P1, A], [P2, A, B], [P3, B]].

Output:
metarule([P, P1, P2, P3], [P, A, B], [[P1, A], [P2, A, B], [P3, B]]).

Lastly, build all metarules based on conclusion in the input parts and then put them together under Meta-rules. 
Exmaple of likely output:
[Meta-rules: 
metarule([P, P1, P2], [P, A, B, C], [[P1, A], [P2, A]]).
metarule([P, P1, P2], [P, A, B, C], [[P1, A], [P2, A, B, C]]).
metarule([P, P1, P2, P3], [P, A, B, C], [[P1, A, C], [P2, A, B],[P3, B, C]]).
]

"""
    return query


def Build_metarules(input_text):
    # 匹配 "Meta-rules:" 后的内容，包括跨行内容
    marker_pattern = r"(?i)Meta-rules:\s*(.*)"  # 匹配 "Meta-rules:" 后的部分，忽略大小写
    marker_match = re.search(marker_pattern, input_text, re.DOTALL)

    if not marker_match:
        return []  # 如果没找到 "Meta-rules:"，返回空列表

    # 提取从 "Meta-rules:" 开始的内容
    content_after_marker = marker_match.group(1)

    # 改进的 metarule 正则表达式：支持多行、换行和括号嵌套结构
    metarule_pattern = r"(?i)metarule\s*\(\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\)\."

    # 提取所有 metarule 表达式
    metarules = re.findall(metarule_pattern, content_after_marker, re.DOTALL)

    # 返回提取到的 metarule 列表
    return metarules

# def complete_metarules(metarule_list):
#     def generate_subject_var(chosen_classes):
#         return list(string.ascii_uppercase[:len(chosen_classes)])
#
#     subject_var = generate_subject_var(chosen_classes)
#
#     updated_rules = []
#     for rule in metarule_list:
#         match = re.search(r'metarule\(\[(.*?)\], \[(.*?)\], \[(.*?)\]\)\.', rule)
#         if match:
#             meta_vars = match.group(1)  # 第一个[]内的变量
#             p_vars = match.group(2)  # 第二个[]内的变量 (主要关心的部分)
#             body = match.group(3)  # 最后 [] 的规则体
#
#             p_var_list = [var.strip() for var in p_vars.split(",")]
#
#             missing_vars = [var for var in subject_var if var not in p_var_list]
#             if missing_vars:
#                 complete_p_vars = p_var_list + missing_vars
#                 new_p_vars = ", ".join(complete_p_vars)
#                 updated_rule = re.sub(r'\[\s*' + re.escape(p_vars) + r'\s*\]', f"[{new_p_vars}]", rule)
#                 updated_rules.append(updated_rule)
#             else:
#                 updated_rules.append(rule)
#         else:
#             updated_rules.append(rule)
#     return updated_rules

def Rule_to_NLP(Final_rules):
    prompt = f"""[{Final_rules}] contains a distilled rule presented in Prolog form, extracted as the sole rule from the three provided images. The main entities in these images are [{chosen_classes}], and there exists a single rule that can describe the relationships between them or their interactions with the environment.
    Now, based on the images and the distilled Prolog rule, describe the image content in one sentence of natural language, focusing on the rule and supplemented by other important details from the images."""

    return prompt

def complete_metarules(metarule_list):
    updated_rules = []
    for rule in metarule_list:
        match = re.search(r'metarule\(\[(.*?)\], \[(.*?)\], \[(.*?)\]\)\.', rule)
        if match:
            meta_vars = match.group(1)  # 第一个[]内的变量
            p_vars = match.group(2)  # 第二个[]内的变量 (主要关心的部分)
            body = match.group(3)  # 最后 [] 的规则体

            p_var_list = [var.strip() for var in p_vars.split(",")]

            # 检查缺失的变量
            missing_vars = [var for var in subject_var_names if var not in p_var_list]
            if missing_vars:
                complete_p_vars = p_var_list + missing_vars
                new_p_vars = ", ".join(complete_p_vars)
                updated_rule = re.sub(r'\[\s*' + re.escape(p_vars) + r'\s*\]', f"[{new_p_vars}]", rule)
                updated_rules.append(updated_rule)
            else:
                updated_rules.append(rule)
        else:
            updated_rules.append(rule)

    # 打印用于调试的映射
    print("Subject variable mapping:", subject_var_mapping)
    return updated_rules



def Save(choose_class,target_classes,NLP_Answer,Answer,knowlege_base,index):
    if 0 <= choose_class < len(target_classes):
        folder_name = f"{'_'.join(target_classes[choose_class])}_{choose_class}"
        folder_path = os.path.join(output_dir, folder_name)

        # 创建目标文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 文件路径
        nlp_answer_file = os.path.join(folder_path, f"NLP_Answer_{index}.txt")
        answer_file = os.path.join(folder_path, f"Answer_{index}.txt")
        knowlege_base_file = os.path.join(folder_path, f"Knowledge_{index}.txt")

        # 写入 NLP_Answer 内容到 NLP_Answer_index 文件
        with open(nlp_answer_file, 'w') as f:
            f.write(NLP_Answer)
        print(f"NLP_Answer file created: {nlp_answer_file}")

        # 写入 Answer 内容到 Answer_index 文件
        with open(answer_file, 'w') as f:
            f.write(Answer)
        print(f"Answer file created: {answer_file}")

        with open(knowlege_base_file, 'w') as f:
            f.write(knowlege_base)
        print(f"Answer file created: {knowlege_base_file}")
    else:
        print(f"Error: choose_class {choose_class} is out of range.")



########################### Old ############################
# def Build_metarules(matches):
#     import re
#     import string
#
#     def parse_predicate(s):
#         s = s.strip()
#         match = re.match(r"(\w+)\((.*)\)", s)
#         if match:
#             predicate_name = match.group(1)
#             arguments = match.group(2).split(',')
#             arguments = [arg.strip() for arg in arguments]
#             return predicate_name, arguments
#         else:
#             raise ValueError(f"Invalid predicate format: {s}")
#
#     def is_subject(entity):
#         return re.match(r'(cat|dog)\d+$', entity) is not None
#
#     def get_next_var_label(var_iter):
#         """Gets the next variable label from an iterator."""
#         try:
#             return next(var_iter)
#         except StopIteration:
#             raise Exception("Ran out of variable labels")
#
#     # Initialize the processed big list
#     processed_big_list = []
#
#     # Process each sublist in big_list
#     for sublist in matches:
#         processed_sublist = []
#         # Process each subsublist
#         for subsublist in sublist:
#             processed_subsublist = []
#
#             # Initialize variable iterators and mappings for this subsublist
#             subject_var_iter = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#             constant_var_iter = iter(['A' + c for c in string.ascii_uppercase])
#             pred_var_iter = iter('SDFGHIJKLMNOPQRSTUVWXYZ')
#
#             # Mappings for this subsublist
#             entity_vars_subjects = {}  # Entity name -> variable label
#             entity_vars_constants = {}  # Constants -> variable label
#             pred_vars = {}  # Predicate name -> variable label
#
#             # Collect all entities to ensure proper mapping
#             entities_set = set()
#             predicates_in_subsublist = []
#             for predicate_str in subsublist:
#                 pred_name, arguments = parse_predicate(predicate_str)
#                 entities_set.update(arguments)
#                 predicates_in_subsublist.append((pred_name, arguments))
#
#             # Assign variables to subject names
#             for entity in entities_set:
#                 if is_subject(entity):
#                     if entity not in entity_vars_subjects:
#                         var_label = get_next_var_label(subject_var_iter)
#                         entity_vars_subjects[entity] = var_label
#
#             # Assign variables to constants (entities not in subjects)
#             for entity in entities_set:
#                 if entity not in entity_vars_subjects and entity not in entity_vars_constants:
#                     var_label = get_next_var_label(constant_var_iter)
#                     entity_vars_constants[entity] = var_label
#
#             # Process each predicate and replace entities with variables
#             for pred_name, arguments in predicates_in_subsublist:
#                 # Assign a unique variable label to the predicate name if not already assigned
#                 if pred_name not in pred_vars:
#                     pred_var = get_next_var_label(pred_var_iter)
#                     pred_vars[pred_name] = pred_var
#                 else:
#                     pred_var = pred_vars[pred_name]
#
#                 # Replace arguments with their variable labels
#                 new_arguments = []
#                 for arg in arguments:
#                     if arg in entity_vars_subjects:
#                         new_arg = entity_vars_subjects[arg]
#                     elif arg in entity_vars_constants:
#                         new_arg = entity_vars_constants[arg]
#                     else:
#                         new_arg = arg  # Should not occur
#                     new_arguments.append(new_arg)
#                 # Create the predicate string with variables
#                 processed_predicate_str = f"{pred_var}[{', '.join(new_arguments)}]"
#                 # Add the processed predicate to the subsublist
#                 processed_subsublist.append(processed_predicate_str)
#
#             # Add the processed subsublist to the sublist
#             processed_sublist.append(processed_subsublist)
#         # Add the processed sublist to the big list
#         processed_big_list.append(processed_sublist)
#
#     def generate_P(subjects: List[str]) -> str:
#         num_subjects = len(subjects)
#         if num_subjects == 0:
#             raise ValueError("subjects 列表不能为空")
#         # 生成变量部分，例如 ['A', 'B'] 对应 'A,B'
#         variables = ','.join([f'{chr(65 + i)}' for i in range(num_subjects)])  # chr(65) = 'A'
#         return f'P[{variables}]'
#
#     def add_P_and_colon(nested: List[List[List[str]]], subjects_names: List[List[str]]) -> List[List[List[str]]]:
#         if not subjects_names or not subjects_names[0]:
#             raise ValueError("subjects_names must contain an none empty list")
#
#         # 遍历每个 outer 和 middle 层级
#         for outer_idx, outer in enumerate(nested):
#             for middle_idx, middle in enumerate(outer):
#                 # 获取对应的 subjects_names
#                 if outer_idx < len(subjects_names):
#                     subjects = subjects_names[outer_idx]
#                 else:
#                     subjects = subjects_names[0]  # 默认使用第一个
#                 P_str = generate_P(subjects)
#                 # 在 subsublist 的开头添加 P_str 和 ':-'
#                 nested[outer_idx][middle_idx] = [P_str, ':-'] + nested[outer_idx][middle_idx]
#
#         return nested
#
#     modified_nested_list = add_P_and_colon(processed_big_list, subjects_names)
#
#     def parse_predicate(predicate_str):
#         pattern = r'(\w+)\[([^\]]+)\]'
#         match = re.match(pattern, predicate_str)
#         if match:
#             pred_name = match.group(1)
#             vars_str = match.group(2)
#             # 按逗号分割变量并去除空格
#             vars_list = [var.strip() for var in vars_str.split(',')]
#             return pred_name, vars_list
#         else:
#             raise ValueError(f"谓词 '{predicate_str}' 格式不正确。")
#
#     def transform_to_metarule(sub_sublist):
#         # 规则的头部
#         head = sub_sublist[0]
#         # ':-' 后面的部分是规则的主体
#         body = sub_sublist[2:]  # 跳过 ':-'
#
#         # 解析头部谓词
#         head_pred, head_vars = parse_predicate(head)
#
#         # 初始化谓词列表和主体谓词
#         predicates = [head_pred]  # 将头部谓词加入列表
#         body_predicates = []
#
#         # 遍历主体谓词
#         for pred_str in body:
#             pred_name, pred_vars = parse_predicate(pred_str)
#             predicates.append(pred_name)
#             body_predicates.append([pred_name] + pred_vars)
#
#         # 移除重复谓词，同时保持顺序
#         unique_predicates = []
#         seen = set()
#         for pred in predicates:
#             if pred not in seen:
#                 unique_predicates.append(pred)
#                 seen.add(pred)
#
#         # 构建 metarule 子句
#         metarule_clause = f"metarule([{', '.join(unique_predicates)}], [{head_pred}, {', '.join(head_vars)}], ["
#
#         # 格式化主体谓词
#         body_parts = []
#         for bp in body_predicates:
#             # 将谓词名和变量用逗号连接
#             body_part = f"[{', '.join(bp)}]"
#             body_parts.append(body_part)
#
#         metarule_clause += ', '.join(body_parts) + "])."
#
#         return metarule_clause
#
#     def process_data(nested_data):
#         metarules = []
#         for i, sublist in enumerate(nested_data):
#             for j, sub_sublist in enumerate(sublist):
#                 try:
#                     metarule = transform_to_metarule(sub_sublist)
#                     metarules.append(metarule)
#                 except ValueError as e:
#                     print(f"  处理子子列表 {j + 1} 时出错: {e}")
#         return metarules
#
#     metarule_clauses = process_data(modified_nested_list)
#
#     def remove_duplicates(metarules):
#         # Create a set to store unique representations of metarules
#         unique_metarules = set()
#         result = []
#
#         for metarule in metarules:
#             # Extract the unique portion of the metarule by splitting and sorting
#             unique_part = metarule.split('[[', 1)[1].rstrip(']).').replace('],', ']|')
#             # Use the sorted representation of the unique part as the key
#             unique_representation = '[[{}]]'.format(
#                 '|'.join(sorted(unique_part.split('|')))
#             )
#
#             # Check if the unique representation is already in the set
#             if unique_representation not in unique_metarules:
#                 unique_metarules.add(unique_representation)
#                 result.append(metarule)  # Keep the original metarule format
#
#         return result
#
#     unique_metarules = remove_duplicates(metarule_clauses)
#
#     # 提取metarule的谓词和参数数量
#     def extract_predicates(rule):
#         # 找到所有子谓词
#         sub_predicates = re.findall(r'\[(\w+)(?:, ([^\]]+))?\]', rule)
#         # 构建谓词: 参数个数映射
#         predicate_count = Counter()
#         for predicate, args in sub_predicates:
#             if args:
#                 # 参数个数通过逗号分割计算
#                 arg_count = len(args.split(', '))
#             else:
#                 arg_count = 0
#             predicate_count[(predicate, arg_count)] += 1
#         return predicate_count
#
#     # 去重的逻辑
#     def remove_duplicate_metarules(rules):
#         unique_rules = []
#         seen_predicates = []
#
#         for rule in rules:
#             # 提取谓词信息
#             predicates = extract_predicates(rule)
#
#             # 检查是否已经存在相同谓词的规则
#             if predicates not in seen_predicates:
#                 seen_predicates.append(predicates)
#                 unique_rules.append(rule)
#
#         return unique_rules
#
#     # 去重后的metarule列表
#     unique_metarules = remove_duplicate_metarules(unique_metarules)
#     return unique_metarules
# def extract_conclusion_list(text):
#     try:
#         # 使用正则表达式提取 Conclusion 后的内容
#         pattern = r"Conclusion:\s*(\[[\s\S]*\])"
#         match = re.search(pattern, text)
#
#         if match:
#             # 提取匹配到的内容
#             conclusion_content = match.group(1)
#
#             # 清理内容：去掉多余的换行符和空格
#             conclusion_content = re.sub(r"\s+", " ", conclusion_content.strip())
#
#             # 转换未加引号的函数调用形式为字符串
#             conclusion_content = convert_to_string_literals(conclusion_content)
#
#             # 使用 ast.literal_eval 解析为 Python 对象
#             conclusion_list = ast.literal_eval(conclusion_content)
#
#             # 移除空列表
#             conclusion_list = remove_empty_lists(conclusion_list)
#
#             return conclusion_list
#
#         return "No match found"
#
#     except Exception as e:
#         # 捕获并返回异常信息
#         return f"Error during parsing: {str(e)}"


def rules_to_nl(rules):
    prompt = f"""The following rules are candidates description [{rules}],The rules has objects [{subjects_names}]
     for example: 
     \"rules = \"f(A,B) :- golden(A),blue(B),play_with(A,B) \" 
     objects = \" dog,cat \"
     translation = dog is color golden, cat is color blue, dog happily play with cat .\"
     there is no \"if-else\" statement in the description.
     Now:
     translate all rules to natural language, and separately output them, the answer only includes a list of separate natural language description """
    return prompt