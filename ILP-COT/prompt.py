import copy
import gc
import re
import os
import test
import tools
import prolog
import base64
import random
import itertools
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
import string


prolog_dir = "..."
output_dir = "..."

def call_chat(prompt_messages):
    openai_api_key = "EMPTY"
    openai_api_base = "..."
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    return client.chat.completions.create(
        model="InternVL3-8B",
        messages=prompt_messages,
        temperature=0.6,
        max_tokens=5050
    ).choices[0].message.content


def verb_transformation(rules_intermediate):
    def build_prompt(rule_intermediate):
        system_messages = system()
        prompt = (
                system_messages
                + [{"role": "user", "content": [{"type": "text", "text": (f'''
                    1. What is a transformation-criterion? It is a minimal form that rewrites a natural-language rule into a pair: a transformation and the condition that triggers this transformation.
                    For example:
                    The red small dog sings → criterion: red_dog; transformation/verb: sing → sing(red_dog)
                    A person in short sleeves sneezes → criterion: person_in_short_sleeves; transformation/verb: sneeze → sneeze(person_in_short_sleeves)
                    At time t_k_0 the digit 8 forms a circle; at time t_k_1 it becomes the digit 9 → criterion: number_8_with_circle; transformation/verb: to_9 → to_9(number_8_with_circle)''')}]}]
                + [{"role": "user", "content": [{"type": "text", "text": (f'''
                    2. Extract the rules from the conclusion below and convert them into the transformation-criterion form. Absolute position is not important; features other than absolute position (especially compositional features formed by groups of digits) matter more:
                     {rule_intermediate}
                    ''')}]}]
        )
        return prompt

    tasks1 = [
        build_prompt(rules_intermediate[i])
        for i in range(len(rules_intermediate))
    ]
    with ThreadPoolExecutor(max_workers=10) as ex:
        verb_captured = list(ex.map(call_chat, tasks1))
    verb_captured = tools.extract_after_last_think(verb_captured)
    verb_captured = tools.extract_transformation_criteria(verb_captured)
    verb_captured = [[tools.convert_token(t) for t in sub] for sub in verb_captured]

    def build_prompt_meta(rule_intermediate, verb):
        system_messages = system()
        prompt = (
                system_messages
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''What are meta-rules? Meta-rules are an abstract format of rules.
    For example:
    There exist rules: "The red small dog sings." "The person wearing short sleeves sneezed."
    There exist predicates: sing__red_dog(A), sneeze__person_in_short_sleeves(A).
    We can assign predicate symbols: Q --> sing__red_dog, R --> sneeze__person_in_short_sleeves, with predicate relation [Q, R].
    Therefore, a meta-rule exists: [[P, Q, R], [P, A], [[Q, A], [R, A]]].''')
            }]
        }]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''Here are the summarized rules: {rule_intermediate}''')
            }]
        }]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''Here are the available predicates in the rule: {verb}''')
            }]
        }]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''I now need to convert the rules into the meta-rule format.''')
            }]
        }]
        )
        return prompt

    tasks2 = [
        build_prompt_meta(rules_intermediate[i], verb_captured[i])
        for i in range(len(rules_intermediate))
    ]
    with ThreadPoolExecutor(max_workers=10) as ex:
        meta_captured = list(ex.map(call_chat, tasks2))
    meta_captured = tools.extract_after_last_think(meta_captured)
    meta_captured = tools.extract_meta_rules_grouped(meta_captured)
    meta_captured = tools.dedup_grouped_meta(meta_captured)
    meta_captured = [[tools.strip_outer_brackets(item) for item in sub] for sub in meta_captured]
    verb_captured = tools.dedup_sublists(verb_captured)

    rules_ilp = Build_PL_file(verb_captured, meta_captured, )
    return rules_ilp, verb_captured


def answer_sub_question(data: List, sub_question: List[str], m: int, n: int):
    def build_prompt(sub_question_text: str, value) -> List[dict]:
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = [
            *system_messages,
            {"role": "user", "content": data_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Based on all given train_k_0 to train_k_1 pairs, "
                            f"answer the following question {sub_question_text}. "
                            "Provide a concise final answer."
                        ),
                    }
                ],
            },
        ]
        return prompt

    tasks = [
        build_prompt(sub_question[i * m + j], data[i])
        for i in range(len(data))
        for j in range(m)
        for _ in range(n)
    ]

    with ThreadPoolExecutor(max_workers=10) as executor:
        output_answer = list(executor.map(call_chat, tasks))

    output_answer = tools.extract_after_last_think(output_answer)
    block = m * n
    return output_answer, block


def rule_induction(data, rule_list, generation):
    def build_prompt(value):
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = (
                system_messages
                + [{"role": "user", "content": data_prompt}]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f'''The above are example inputs, each showing a transformation from t_k_0 to t_k_1. Based on all such changes, summarize the transformation pattern during the process and try to propose three rules that apply to all cases.''')
            }]
        }]
        )
        return prompt

    tasks = [build_prompt(data[i]) for i in range(len(data))]
    with ThreadPoolExecutor(max_workers=5) as ex:
        hypothesis = list(ex.map(call_chat, tasks))
    hypothesis = tools.extract_after_last_think(hypothesis)
    rules_ilp, rules_intermediate = verb_transformation(hypothesis)

    def build_prompt_rules(value, rule_ilp):
        system_messages = system()

        data_prompt = ARC_AGI(value)
        prompt = (
                system_messages
                + [{"role": "user", "content": data_prompt}]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''
                        1. Main question: Based on the changes from t_k_0 to t_k_1, summarize the transformation pattern during the process. (single/unique rule)
                        ...
                    ''')
            }]
        }]
                + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (f'''
                        2. Below are clues found via an Inductive Logic Programming system that you may reference.
                        3. Each predicate in the rule has the form verb__criterion. "criterion" specifies what should be selected in t_k_0, and "verb" specifies what transformation to apply to the selected content.
                        4. [...{rule_ilp}...]
                        5. Your final and sole objective is to answer the main question and provide the single transformation rule from t_k_0 to t_k_1.
                    ''')
            }]
        }]
        )

        return prompt

    tasks = [build_prompt_rules(data[i], rules_ilp[i]) for i in range(len(data))]
    with ThreadPoolExecutor(max_workers=10) as ex:
        rules = list(ex.map(call_chat, tasks))
    rules = tools.extract_after_last_think(rules)

    return rules, rules_intermediate, rules_ilp


def image_matrix(data: List):
    """
    Analyzes image matrices to induce transformation rules.
    """

    def build_prompt(value) -> List[dict]:
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = [
            *system_messages,
            {"role": "user", "content": data_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The above are example inputs, each showing a transformation "
                            "from t_k_0 to t_k_1. Now, based on all t_k_0 to t_k_1 changes, "
                            "summarize the transformation pattern during the process and "
                            "try to propose three rules that apply to all cases."
                        ),
                    }
                ],
            },
        ]
        return prompt

    tasks1 = [build_prompt(item) for item in data]
    with ThreadPoolExecutor(max_workers=5) as executor:
        hypothesis = list(executor.map(call_chat, tasks1))

    hypothesis = tools.extract_after_last_think(hypothesis)
    rules_ilp, rules_intermediate = verb_transformation(hypothesis)

    def build_prompt_rules(value, rule_ilp: str) -> List[dict]:
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = [
            *system_messages,
            {"role": "user", "content": data_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'''
                    1. Main question: Based on the changes from t_k_0 to t_k_1, summarize the transformation pattern during the process. (single/unique rule)
                    ...
                    '''
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'''
                    2. Below are clues found via an Inductive Logic Programming system that you may reference.
                    3. Each predicate in the rule is composed as verb__criterion. "criterion" specifies what should be selected in t_k_0, and "verb" specifies what transformation to apply to the selected content.
                    4. [...{rule_ilp}...]
                    5. Your sole objective is to answer the main question and provide the single transformation rule from t_k_0 to t_k_1.'''
                        ),
                    }
                ],
            },
        ]
        return prompt

    tasks2 = [build_prompt_rules(data[i], rules_ilp[i]) for i in range(len(data))]
    with ThreadPoolExecutor(max_workers=10) as executor:
        rules = list(executor.map(call_chat, tasks2))

    rules = tools.extract_after_last_think(rules)

    return rules, rules_intermediate, rules_ilp


def _log_progress(file_path: Path, messages: list):
    print("\n" + "#" * 30)
    for msg in messages:
        print(msg)
    print("#" * 30)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f, redirect_stdout(f):
        print("\n" + "#" * 30)
        for msg in messages:
            print(msg)
        print("#" * 30)


def _find_all_missing_indices(results_list: list, total_items: int = 5) -> set:
    full_set = set(range(total_items))
    all_missing = set()
    for result in results_list:
        missing_in_this_run = full_set - set(result)
        all_missing.update(missing_in_this_run)
    return all_missing


def inference(data, continue_indices, num_list, seed, max_iterations, save_dir):
    iteration = 0
    TOTAL_ITEMS = 5

    current_continue_indices = copy.deepcopy(continue_indices)
    intermediate_continue_indices = copy.deepcopy(continue_indices)
    rule_list = list(range(len(data)))
    generation = list(range(len(data)))

    while iteration < max_iterations:
        rules, rules_intermediate, rules_ilp = rule_induction(data, rule_list, generation)

        file_name, train_inputs, train_outputs, test_input, test_output = tools.getdata(data)

        indices = [random.randrange(len(sublist)) for sublist in train_inputs]
        train_input = [train_inputs[i][idx] for i, idx in enumerate(indices)]
        train_output = [train_outputs[i][idx] for i, idx in enumerate(indices)]

        current_continue_indices, generation = test_generation(
            file_name, train_output, train_input, rules, current_continue_indices, rules_ilp
        )
        intermediate_continue_indices, _ = test_generation(
            file_name, train_output, train_input, rules_intermediate, intermediate_continue_indices, rules_ilp
        )

        iteration += 1

        for i, item in enumerate(rule_list):
            if isinstance(item, int) and i not in current_continue_indices:
                rule_list[i] = rules_ilp[i]

        log_messages = [
            "Train Generation Progress",
            f"Converged rules: {TOTAL_ITEMS - len(current_continue_indices)} / {TOTAL_ITEMS}"
        ]
        _log_progress(Path("logs/train_generation.log"), log_messages)

    NUM_TEST_RUNS = 3
    test_results = []
    ablation_results = []

    final_rules, _, _ = rule_induction(data, rule_list, generation)
    _, _, test_input, test_output = tools.getdata(data)[1:]

    for i in range(NUM_TEST_RUNS):
        print(f"\n--- Starting Test Run {i + 1}/{NUM_TEST_RUNS} ---")

        test_run_indices, _ = test_generation(
            file_name, test_output, test_input, final_rules, copy.deepcopy(continue_indices), seed
        )
        test_results.append(test_run_indices)
        print(f"Test run {i + 1} results: {test_run_indices}")

        ablation_run_indices = test_generation_ablation(
            data, test_output, test_input, copy.deepcopy(continue_indices), seed
        )
        ablation_results.append(ablation_run_indices)
        print(f"Ablation run {i + 1} results: {ablation_run_indices}")

    final_test_indices = _find_all_missing_indices(test_results, TOTAL_ITEMS)
    final_ablation_indices = _find_all_missing_indices(ablation_results, TOTAL_ITEMS)

    log_messages = [
        "Final Test Generation Results",
        f"Total successful (Test): {len(final_test_indices)}",
        f"Total successful (Ablation): {len(final_ablation_indices)}",
        f"Successful indices (Test): {sorted(list(final_test_indices))}",
        f"Successful indices (Ablation): {sorted(list(final_ablation_indices))}"
    ]
    _log_progress(Path("logs/final_test_results.log"), log_messages)


def test_generation(data, test_output, test_input, rules, continue_indices, seed):
    prompt_list = []
    for index, value in enumerate(data):
        system_messages = system()
        prompt = (
                system_messages
                + [{"role": "user", "content": [{"type": "text", "text": (
            f"""Using the following summarized rule, derive the output from the input: \"{rules[index]}\" """)}]}]
                + [{"role": "user", "content": [{"type": "text", "text": (
            f'''Here is the input we need to consider: {test_input[index]}. Based on this input, generate the output. Do not include any other content.''')}]}])
        prompt_list.append(prompt)

    with ThreadPoolExecutor(max_workers=8) as executor:
        generation = list(executor.map(call_chat, prompt_list))
    generation = tools.extract_after_last_think(generation)
    extracted_matrices = tools.extract_all_matrices_with_padding(generation, test_output)
    comparison_results = tools.compare(extracted_matrices, test_output)
    converted_list = [int(value) for value in comparison_results]
    continue_indices_train_ = [index for index, value in enumerate(converted_list) if value == 0]
    continue_indices_train = list(set(continue_indices) & set(continue_indices_train_))
    return continue_indices_train, extracted_matrices


def test_generation_ablation(data, test_output, test_input, continue_indices, seed):
    prompt_list = []
    for index, value in enumerate(data):
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = (
                system_messages
                + [{"role": "user", "content": data_prompt}]
                + [{"role": "user", "content": [{"type": "text", "text": (
            f"""From the above input–output examples, infer the transformation rule. Then, using that rule, produce the output for the following input:""")}]}]
                + [{"role": "user", "content": [{"type": "text", "text": (
            f'''Here is the input we need to consider: {test_input[index]}. Based on this input, generate the output. Do not include any other content.''')}]}])
        prompt_list.append(prompt)

    with ThreadPoolExecutor(max_workers=8) as executor:
        generation = list(executor.map(call_chat, prompt_list))
    generation = tools.extract_after_last_think(generation)
    extracted_matrices = tools.extract_all_matrices_with_padding(generation, test_output)
    comparison_results = tools.compare(extracted_matrices, test_output)
    converted_list = [int(value) for value in comparison_results]
    continue_indices_train_ = [index for index, value in enumerate(converted_list) if value == 0]
    continue_indices_train = list(set(continue_indices) & set(continue_indices_train_))
    return continue_indices_train


def train_generation(data, train_outputs, train_inputs, rules, continue_indices, seed):
    prompt_list = []
    n = random.randint(1, 2)

    for index, value in enumerate(data):
        system_messages = system()
        prompt = (
                system_messages
                + [{"role": "user", "content": [{"type": "text", "text": (
            f"""Using the following summarized rule, derive the output from the input: \"{rules[index]}\" """)}]}]
                + [{"role": "user", "content": [{"type": "text", "text": (
            f'''Here is the input we need to consider: {train_inputs[index][-n]}. Based on this input, generate the output. Do not include any other content.''')}]}])
        prompt_list.append(prompt)

    with ThreadPoolExecutor(max_workers=8) as executor:
        generation = list(executor.map(call_chat, prompt_list))

    train_output_ = [t[-n] for t in train_outputs]
    extracted_matrices = tools.extract_matrices(generation, train_output_)
    comparison_results = tools.compare(extracted_matrices, train_output_)
    converted_list = [int(value) for value in comparison_results]
    continue_indices_train_ = [index for index, value in enumerate(converted_list) if value == 0]
    continue_indices_train = list(set(continue_indices) & set(continue_indices_train_))
    return continue_indices_train


def ARC_AGI_pos(data):
    content = []
    for i, (inp, out) in enumerate(zip(data['train_inputs'], data['train_outputs'])):
        content.extend([
            {"type": "text", "text": f"train_t_{i}_0："},
            {"type": "text", "text": f"{inp}"},
            {"type": "text", "text": f"train_t_{i}_1："},
            {"type": "text", "text": f"{out}"},
        ])
    return content


def system():
    messages = [{"role": "system", "content": "You are a helpful logic assistant."}]
    return messages


def ARC_AGI(data):
    content = []
    for i, (inp, out) in enumerate(zip(data['train_inputs'], data['train_outputs'])):
        content.extend([
            {"type": "text", "text": f"train_t_{i}_0："},
            {"type": "text", "text": f"{inp}"},
            {"type": "text", "text": f"train_t_{i}_1："},
            {"type": "text", "text": f"{out}"},
        ])
    return content


def pil_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def Build_meta_rules_prepration():
    meta_text = 1
    return meta_text


def Build_PL_file(knowledge_base, metarules):
    tools.prolog_write(knowledge_base, metarules)
    import glob

    folder = "PL_File"
    rules = []

    for pl_path in sorted(glob.glob(os.path.join(folder, "kb_*.pl"))):
        rules.append(tools.capture_prolog_output(pl_path))
    rules = tools.top_rules(rules)
    return rules


def final_answer(Answer):
    query = f"""{Answer}"""
    return query


def Rule_to_NLP(Final_rules):
    prompt = f"""[{Final_rules}] contains a distilled rule presented in Prolog form, extracted as the sole rule from the provided images. The main entities in these images are , and there exists a single rule that can describe the relationships between them or their interactions with the environment.
    Now, based on the images and the distilled Prolog rule, describe the image content in one sentence of natural language, focusing on the rule and supplemented by other important details from the images."""

    return prompt


def sample(data_pre_set, filenames):
    def is_valid_entry(item):
        if not (isinstance(item, list) and len(item) == 2):
            return False

        first_element = item[0]
        if not (isinstance(first_element, list) and len(first_element) > 0):
            return False

        value = first_element[0]
        return isinstance(value, str) and value.strip() not in ('', '#')

    final_sub_answers = []
    final_sub_questions = []

    for fname in filenames:
        if fname not in data_pre_set:
            continue

        group = data_pre_set[fname]
        all_lists = [item for subdict in group.values() for item in subdict]
        all_valid_lists = [item for item in all_lists if is_valid_entry(item)]

        sampled_answers = random.sample(all_valid_lists, 20)
        first_sublists = [entry[0][0] for entry in sampled_answers]
        final_sub_answers.extend(first_sublists)

        sampled_questions = random.sample(all_valid_lists, 10)
        first_questions = [entry[0][0] for entry in sampled_questions]
        final_sub_questions.extend(first_questions)

    return final_sub_answers, final_sub_questions


def rule_induction_masked(data, rule_masked):
    def build_initial_prompt(value):
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = [
            *system_messages,
            {"role": "user", "content": data_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '''
                        1. Question: Based on the changes from t_k_0 to t_k_1, summarize the transformation pattern during the process.
                        2. Summarize the pattern in no more than five sentences.
                    '''
                        ),
                    }
                ],
            },
        ]
        return prompt

    tasks1 = [build_initial_prompt(item) for item in data]
    with ThreadPoolExecutor(max_workers=10) as executor:
        rules0 = list(executor.map(call_chat, tasks1))

    rules0 = tools.extract_after_last_think(rules0)

    def build_refinement_prompt(value, masked_rule, rule_first):
        system_messages = system()
        data_prompt = ARC_AGI(value)
        prompt = [
            *system_messages,
            {"role": "user", "content": data_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'''
                        1. Question: Based on the changes from t_k_0 to t_k_1, summarize the pattern of changes during the process.
                        2. Below are the possible rules I inferred from two observations:
                        1) "{masked_rule}"
                        2) "{rule_first}"
                        3. I believe my summary does not explain all input-output pairs. Please help me restate the rule.
                    '''
                        ),
                    }
                ],
            },
        ]
        return prompt

    tasks2 = [
        build_refinement_prompt(data[i], rule_masked[i], rules0[i])
        for i in range(len(rule_masked))
    ]
    with ThreadPoolExecutor(max_workers=10) as executor:
        rules = list(executor.map(call_chat, tasks2))

    rules = tools.extract_after_last_think(rules)

    return rules


def mask_input(data):
    masked_i_o = [t["masked_I_O"] for t in data]

    region_keys = [
                      "changed_region",
                      "unchanged_region",
                      "left_half",
                      "upper_half",
                  ] + [f"random_object_region{i}" for i in range(11)]

    all_inputs = [[item[key] for item in masked_i_o] for key in region_keys]

    def build_prompt(value):
        return [
            *system(),
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f'''
                    In train-inputs, there are multiple matrices. Through some transformation, each one maps to a single matrix in train-outputs. Here, 0 denotes background and other numbers form objects.
                    \" {value} \"
                    Based on your observations, identify the transformation rule and summarize it in no more than five sentences.
                    '''
                    )
                }]
            }
        ]

    tasks = [build_prompt(value) for value in chain(*all_inputs)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        masked = list(executor.map(call_chat, tasks))

    masked = tools.extract_after_last_think(masked)

    num_regions = len(region_keys)
    data = data * num_regions

    return masked, data


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

    def _transform_predicate_args(predicate_str, subjects):
        predicate_str = predicate_str.strip()
        match = re.match(r"(\w+)\((.*)\)", predicate_str)
        if not match:
            return predicate_str

        predicate_name = match.group(1)
        args_str = match.group(2).strip()
        if not args_str:
            return f"{predicate_name}()"

        args = [arg.strip() for arg in args_str.split(",")]
        new_args = []
        for arg in args:
            is_subject = any(arg.lower().startswith(sub.lower()) for sub in subjects)
            new_args.append("_" if is_subject else arg)

        return f"{predicate_name}({', '.join(new_args)})"

    def _process_predicate_list(pred_list, subjects):
        new_list = []
        for item in pred_list:
            predicates = re.split(r",\s*(?=\w+\()", item)
            transformed = [
                _transform_predicate_args(pred, subjects) for pred in predicates
            ]
            new_item = ", ".join(transformed)
            new_list.append(new_item)
        return new_list

    knowledge_list = _process_predicate_list(knowledge_list, chosen_classes)

    def _remove_subject_from_names(predicate_list, subjects):
        subjects_set = {s.lower() for s in subjects}
        processed_list = []

        for pred_str in predicate_list:
            predicates = [p.strip() for p in pred_str.split(",")]
            new_predicates = []

            for pred in predicates:
                idx = pred.find("(")
                if idx == -1:
                    new_predicates.append(pred)
                    continue

                name = pred[:idx]
                params = pred[idx:]
                parts = name.split("_")
                new_parts = [part for part in parts if part.lower() not in subjects_set]

                if not new_parts:
                    continue

                new_name = "_".join(new_parts)
                new_predicates.append(new_name + params)

            processed_list.append(",     ".join(new_predicates))
        return processed_list

    knowledge_list = _remove_subject_from_names(knowledge_list, chosen_classes)

    prompt = (
        f"""Use the knowledge provided in:[{knowledge_list}] to extract knowledge for each image. 
        The numbers in parentheses indicate the required number of subjects that must be present. 
        For example, if the subjects are "cat0" and "dog0," then fur_golden(_) could apply to either fur_golden(cat0) or fur_golden(dog0) based on the information captured from the images.
        For example, if subjects are "man1" and "women1", then smilling(_,_) coule apply to either smilling(man1,women1) or smilling(women1,man1) based on the information captured from the images.
        Now, given the subjects for each image as:[{subjects_names_neg[global_choose_class]}]. Return captured knowledge in prolog format: Description(subject). """
    )

    return prompt

def extract_capture_from_hypothesis_space(input):
    pattern = r":-\s*(.*?)\."
    matches = re.findall(pattern, input, re.DOTALL)
    knowledge_list = [match.strip().replace("\n", " ") for match in matches]

    def _remove_subject_from_names(predicate_list, subjects):
        subjects_set = {s.lower() for s in subjects}
        processed_list = []

        for pred_str in predicate_list:
            predicates = [p.strip() for p in pred_str.split(",")]
            new_predicates = []

            for pred in predicates:
                idx = pred.find("(")
                if idx == -1:
                    new_predicates.append(pred)
                    continue

                name = pred[:idx]
                params = pred[idx:]
                parts = name.split("_")
                new_parts = [part for part in parts if part.lower() not in subjects_set]

                if not new_parts:
                    continue

                new_name = "_".join(new_parts)
                new_predicates.append(new_name + params)

            processed_list.append(",     ".join(new_predicates))
        return processed_list

    knowledge_list = _remove_subject_from_names(knowledge_list, chosen_classes)

    prompt = (
        f"""Extract matching content from the image based on the words in the [{knowledge_list}].
        Return captured knowledge in prolog format: Description(subject).
        Subjects is in such an order for these images: [{subjects_names}].
        Don't add in new knowledge.
        Example of Description(subject):
        Golden_fur(Dog) 
        """
    )

    return prompt


def hypothesis_space():
    prompt = (
        f"""the images has subjects [{subjects_names}], there is a rule consistent cross these images , we are going to find that rule. 
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
    )
    return prompt

def build_results(chosen_classes, chosen_num, num_images):
    results = []
    indices = [0] * len(chosen_classes)

    for i in range(num_images):
        result_list = []
        for cls_idx, (cls, num) in enumerate(zip(chosen_classes, chosen_num)):
            for _ in range(num):
                result_list.append(f"{cls}{indices[cls_idx]}")
                indices[cls_idx] += 1
        results.append(result_list)

    return results


def Image_to_text(target_classes, choose_class, target_num, image_filenames):
    def generate_subject_var(chosen_classes, num):
        subject_var = []
        idx = 0
        for cls, count in zip(chosen_classes, num):
            if count == 1:
                subject_var.append((string.ascii_uppercase[idx], cls))
                idx += 1
            else:
                for i in range(count):
                    subject_var.append((string.ascii_uppercase[idx], f"{cls}{i}"))
                    idx += 1
        return subject_var

    def generate_labels(strings):
        labels = {}
        for i, s in enumerate(strings):
            labels[s] = chr(65 + i)
        return labels

    global subjects_names, subjects_names_neg, chosen_classes
    global global_choose_class, num_images, chosen_num
    global subject_var_names, subject_var_mapping, subject_var

    global_choose_class = choose_class
    num_images = len(image_filenames)
    chosen_classes = target_classes[choose_class]
    chosen_num = target_num[choose_class]
    subjects_names = build_results(chosen_classes, chosen_num, num_images)

    subjects_names_neg = [
        [['cat3', 'dog3'], ['cat4', 'dog4'], ['cat5', 'dog5']],
        [['man3', 'women3', 'child3'], ['man4', 'women4', 'child4'], ['man5', 'women5', 'child5']],
        [['boy3', 'girl3'], ['boy4', 'girl4'], ['boy5', 'girl5']],
        [['human6', 'human7'], ['human8', 'human9'], ['human10', 'human11']],
        [['cat3', 'child3'], ['cat4', 'child4'], ['cat5', 'child5']],
        [['cat3'], ['cat4'], ['cat5']],
        [['lady3', 'desk6', 'desk7'], ['lady4', 'desk8', 'desk9'], ['lady5', 'desk10', 'desk11']],
        [['dog6', 'dog7'], ['dog8', 'dog9'], ['dog10', 'dog11']],
        [['child3', 'adult3'], ['child4', 'adult4'], ['child5', 'adult5']],
        [['male3', 'female3'], ['male4', 'female4'], ['male5', 'female5']],
        [['male3', 'female3'], ['male4', 'female4'], ['male5', 'female5']],
        [['male3', 'female3'], ['male4', 'female4'], ['male5', 'female5']],
        [['human3'], ['human4'], ['human5']],
        [['mom3', 'child3'], ['mom4', 'child4'], ['mom5', 'child5']],
        [['teacher3', 'student6', 'student7'], ['teacher4', 'student8', 'student9'], ['teacher5', 'student10', 'student11']],
        [['pig3'], ['pig4'], ['pig5']],
        [['human9', 'human10', 'human11'], ['human12', 'human13', 'human14'], ['human15', 'human16', 'human17']],
        [['cow3'], ['cow4'], ['cow5']],
        [['tree6', 'tree7'], ['tree8', 'tree9'], ['tree10', 'tree11']],
        [['sunflower3'], ['sunflower4'], ['sunflower5']],
        [['human3'], ['human4'], ['human5']],
        [['teacher3', 'student3'], ['teacher4', 'student4'], ['teacher5', 'student5']],
    ]

    subject_var = generate_labels(chosen_classes)
    subject_var_mapping = generate_subject_var(chosen_classes, chosen_num)
    subject_var_names = [var[0] for var in subject_var_mapping]

    text = (
        f"""In these {num_images} images, the main characters are {chosen_classes}. First, design fifteen possible attribute feature descriptions and fifteen relationship feature descriptions based on the main characters of the three images, with the following specific requirements:
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
    )
    return text


def Negative_prompt_check(output):
    prompt = (
        f"""The following content [{output}] are information extracted from these images. However, these extracted information may captured by mistake. Now help me make sure 
        All the information matches the picture information, delete information does not match the image. finally, return the information as the form consistent with the format in input
        (you dont need to explain what/why you delete) """
    )
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


def Change_format_to_prolog_neg(input, pos):
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


def Delete_based_on_predicted_rule(predicted_rules, knowledge):
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


def Build_meta_rules_prepration(knowledge_pos, hypothesis_space):
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

def Build_PL_file(knowledge_base_neg, knowledge_base, metarules):
    def _extract_predicates(kb):
        predicates = {}
        for line in kb.splitlines():
            match = re.match(r"(\w+)\(([^)]*)\)", line)
            if match:
                predicate = match.group(1)
                args_str = match.group(2)
                arity = len(args_str.split(",")) if args_str else 0
                predicates[predicate] = arity
        return predicates

    def _format_examples(data_source):
        if not data_source:
            return []
        if isinstance(data_source[0], list):
            return [f"f({', '.join(pair)})" for pair in data_source]
        else:
            return [f"f({subject})" for subject in data_source]

    predicates = _extract_predicates(knowledge_base)
    predicates.update(_extract_predicates(knowledge_base_neg))

    body_predicates = [
        f"body_pred({pred}/{arity})." for pred, arity in predicates.items()
    ]

    pos_examples_str = ", ".join(_format_examples(subjects_names))
    neg_examples_str = ", ".join(
        _format_examples(subjects_names_neg[global_choose_class])
    )

    output_lines = [
        ":- use_module('metagol').",
        "metagol:max_clauses(1).",
        "\n% Facts from the knowledge base",
        knowledge_base,
        knowledge_base_neg,
        "\n% Body predicates",
        *body_predicates,
        "\n% Metarules",
        *metarules,
        "\n% Learning Task",
        f"a :- Pos = [{pos_examples_str}],",
        f"     Neg = [{neg_examples_str}],",
        "     learn(Pos, Neg).",
    ]

    pl_file_content = "\n".join(output_lines)

    with open(prolog_dir, "w") as f:
        f.write(pl_file_content)

    print(f"Prolog file generated successfully at '{prolog_dir}'.")
    return prolog_dir

def convert_to_string_literals(content):
    _PATTERN = r"([a-zA-Z_][a-zA-Z0-9_]*\([a-zA-Z0-9_,\s]*\))"
    converted_content = re.sub(_PATTERN, r"'\1'", content)
    return converted_content

def remove_empty_lists(nested_list):
    cleaned_list = []
    for item in nested_list:
        if isinstance(item, list):
            cleaned_sublist = remove_empty_lists(item)
            if cleaned_sublist:
                cleaned_list.append(cleaned_sublist)
        else:
            cleaned_list.append(item)
    return cleaned_list

def negative_case(output):
    prompt = f"""The following content ["Here are the designed attribute and relationship feature descriptions:

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
```"]contains the designed fifteen attribute descriptors and fifteen relationship descriptors, along with examples of using them to capture information from images.
Now, use these descriptors to capture the information from the newly provided images. These images are negative examples. After capturing the information from these negative examples, write it in Prolog code format, ensuring it is consistent with the format of the positive examples."""
    return prompt

def convert_to_string_literals(text):
    pattern = r"(\b[a-zA-Z_]+\([^)]*\))"
    text = re.sub(pattern, r'"\1"', text)
    return text

def remove_empty_lists(conclusion_list):
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
    _MARKER_PATTERN = r"(?i)Meta-rules:\s*(.*)"
    marker_match = re.search(_MARKER_PATTERN, input_text, re.DOTALL)

    if not marker_match:
        return []

    content_after_marker = marker_match.group(1)

    _METARULE_PATTERN = r"(?i)metarule\s*\(\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\)\."
    metarules = re.findall(_METARULE_PATTERN, content_after_marker, re.DOTALL)

    return metarules


def rule_to_nlp(Final_rules):
    prompt = f"""[{Final_rules}] contains a distilled rule presented in Prolog form, extracted as the sole rule from the three provided images. The main entities in these images are [{chosen_classes}], and there exists a single rule that can describe the relationships between them or their interactions with the environment.
    Now, based on the images and the distilled Prolog rule, describe the image content in one sentence of natural language, focusing on the rule and supplemented by other important details from the images."""

    return prompt

def complete_metarules(metarule_list):
    _METARULE_PATTERN = re.compile(r"metarule\(\[(.*?)\], \[(.*?)\], \[(.*?)\]\)\.")
    updated_rules = []

    for rule in metarule_list:
        match = _METARULE_PATTERN.search(rule)
        if not match:
            updated_rules.append(rule)
            continue

        meta_vars, p_vars, body = match.groups()
        p_var_list = [var.strip() for var in p_vars.split(",")]

        missing_vars = [var for var in subject_var_names if var not in p_var_list]
        if missing_vars:
            complete_p_vars = p_var_list + missing_vars
            new_p_vars_str = ", ".join(complete_p_vars)
            updated_rule = f"metarule([{meta_vars}], [{new_p_vars_str}], [{body}])."
            updated_rules.append(updated_rule)
        else:
            updated_rules.append(rule)

    print("Subject variable mapping:", subject_var_mapping)
    return updated_rules


def Save(choose_class, target_classes, NLP_Answer, Answer, knowlege_base, index):
    def _write_file(filepath, content, label):
        with open(filepath, "w") as f:
            f.write(content)
        print(f"{label} file created: {filepath}")

    if 0 <= choose_class < len(target_classes):
        folder_name = f"{'_'.join(target_classes[choose_class])}_{choose_class}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        nlp_answer_file = os.path.join(folder_path, f"NLP_Answer_{index}.txt")
        answer_file = os.path.join(folder_path, f"Answer_{index}.txt")
        knowlege_base_file = os.path.join(folder_path, f"Knowledge_{index}.txt")

        _write_file(nlp_answer_file, NLP_Answer, "NLP_Answer")
        _write_file(answer_file, Answer, "Answer")
        _write_file(knowlege_base_file, knowlege_base, "Knowledge")
    else:
        print(f"Error: choose_class {choose_class} is out of range.")


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
