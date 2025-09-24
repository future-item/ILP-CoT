import ast
import json
import math
import os
import pickle
import re
import subprocess
import sys
import torch
import clip
from PIL import Image
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

POSITIVE_SIMILARITY_WEIGHT = 0.3
NEGATIVE_SIMILARITY_WEIGHT = 0.7
URL_POS = '...'
URL_NEG = '...'
_ROW_RE = re.compile(r"\s*\[[\d,\s-]+\]\s*,?\s*$")
_MATRIX_KEYWORDS = ["final output", "final result", "test_output"]
_NAIVE_META = "[[P, Q], [P, A], [[Q, A]]]"
_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_TAIL_SINGLE_RE = re.compile(r"^\[\s*([A-Za-z]+)\s*,\s*A\s*\]$")
_HEAD_RE = re.compile(r"^\[\s*[A-Za-z]+\s*,\s*A\s*\]$")
_PREDLIST_RE = re.compile(r"^\[\s*[A-Za-z]+(?:\s*,\s*[A-Za-z]+)+\s*\]$")
_TRANSFORMATION_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\([^()]+\)")
_PRED_PAT = re.compile(r"\s*([a-zA-Z0-9_]+\(.*?\))\s*")
_TOKEN_RE = re.compile(r"\b[A-Za-z]+\b")
_ABLATION_OUT_DIR = "./result/ablation"
_ABLATION_OUT_FILE = "results.json"


def _printed_matrices(text: str) -> List[Tuple[int, Any]]:
    mats, lines = [], text.splitlines(keepends=True)
    abs_pos = 0
    buf, buf_start = [], None

    for line in lines + ["\n"]:
        if _ROW_RE.match(line):
            buf.append(line)
            if buf_start is None:
                buf_start = abs_pos
        else:
            if len(buf) >= 2:
                try:
                    rows = [ast.literal_eval(l.split("#")[0]) for l in buf]
                    if all(isinstance(r, list) for r in rows):
                        mats.append((buf_start, rows))
                except Exception:
                    pass
            buf, buf_start = [], None
        abs_pos += len(line)
    return mats


def _lists_in_block(block: str) -> List[Tuple[int, Any]]:
    singles = []
    level = 0
    start = None
    for i, ch in enumerate(block):
        if ch == "[":
            if level == 0:
                start = i
            level += 1
        elif ch == "]":
            level -= 1
            if level == 0 and start is not None:
                snippet = block[start: i + 1]
                try:
                    obj = ast.literal_eval(snippet)
                    if isinstance(obj, list):
                        singles.append((start, obj))
                except Exception:
                    pass
                start = None

    if len(singles) < 2:
        return singles

    merged: List[Tuple[int, Any]] = []
    buf: List[Any] = []
    buf_pos = None
    row_len = None

    def flush():
        if not buf:
            return
        if len(buf) > 1:
            merged.append((buf_pos, buf.copy()))
        else:
            merged.append((buf_pos, buf[0]))
        buf.clear()

    last_end = None
    for pos, list_item in singles:
        is_row = list_item and all(not isinstance(x, list) for x in list_item)
        if is_row:
            cur_len = len(list_item)
            contiguous = last_end is None or re.fullmatch(r"\s*", block[last_end:pos])
            same_width = row_len is None or cur_len == row_len
            if buf and not (contiguous and same_width):
                flush()
            if not buf:
                buf_pos = pos
                row_len = cur_len
            buf.append(list_item)
        else:
            flush()
            merged.append((pos, list_item))
        last_end = pos + len(str(list_item))

    flush()
    return merged


def _extract_matrices(text: str) -> List[Tuple[int, Any]]:
    mats: List[Tuple[int, Any]] = []
    for m in re.finditer(r"```(?:\w+)?\s*([\s\S]*?)```", text):
        for rel_pos, obj in _lists_in_block(m.group(1)):
            mats.append((m.start() + rel_pos, obj))
    for m in re.finditer(r"\[\[[\s\S]*?\]\]", text):
        try:
            obj = ast.literal_eval(m.group(0))
            if isinstance(obj, list):
                mats.append((m.start(), obj))
        except Exception:
            pass

    return sorted(mats, key=lambda t: t[0])


def extract_matrix(text: str) -> List[List[int]]:
    mats = _extract_matrices(text)
    if not mats:
        return [[0]]
    if len(mats) == 1:
        return mats[0][1]

    chosen = None
    for pos, mat in mats:
        prev = text[:pos].lower()
        if any(k.lower() in prev for k in _MATRIX_KEYWORDS):
            chosen = mat
    return chosen or mats[-1][1]


def batch_extract(str_list: List[str]) -> List[List[List[int]]]:
    return [extract_matrix(s) for s in str_list]


def compare(
        list1_extracted: List[List[Any]], list2_answers: List[Any]
) -> List[bool]:
    results = []
    for found_matrices_for_pos, truth_matrix in zip(list1_extracted, list2_answers):
        is_correct = any(
            found_matrix == truth_matrix for found_matrix in found_matrices_for_pos
        )
        results.append(is_correct)
    return results


def find_matching_sublist_indices(list1, list2):
    continue_indices_train = []
    for idx, (sublist1, sublist2) in enumerate(zip(list1, list2)):
        if not (sublist1 and sublist2):
            continue

        matrix1 = sublist1[0]
        matrix2 = sublist2[0]
        if isinstance(matrix1, list) and isinstance(matrix2, list):
            rows1 = len(matrix1)
            cols1 = len(matrix1[0]) if rows1 > 0 else 0
            rows2 = len(matrix2)
            cols2 = len(matrix2[0]) if rows2 > 0 else 0
            if rows1 == rows2 and cols1 == cols2:
                continue_indices_train.append(idx)
    return continue_indices_train


def count_exact_matches_list(
        result_cot_ilp, result_cot, result_wo_cot, train_output
):
    num_correct_cot_ilp, _ = count_exact_matches(result_cot_ilp, train_output)
    num_correct_cot, _ = count_exact_matches(result_cot, train_output)
    num_correct, _ = count_exact_matches(result_wo_cot, train_output)
    return num_correct_cot_ilp, num_correct_cot, num_correct


def check_equal_structure(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            check_equal_structure(ai, bi) for ai, bi in zip(a, b)
        )
    return a == b


def count_exact_matches(pred, gt):
    correct_indices = [
        idx for idx, (p, g) in enumerate(zip(pred, gt)) if check_equal_structure(p, g)
    ]
    return len(correct_indices), correct_indices


def adjust_output_to_label(label, output):
    adjusted_output = []
    for label_matrix, output_matrix in zip(label, output):
        target_rows = len(label_matrix)
        if target_rows == 0:
            adjusted_output.append([])
            continue
        target_cols = len(label_matrix[0])
        adjusted_matrix = []

        for r in range(target_rows):
            if r < len(output_matrix):
                row = []
                for c in range(target_cols):
                    if c < len(output_matrix[r]):
                        row.append(output_matrix[r][c])
                    else:
                        row.append(0)
                adjusted_matrix.append(row)
            else:
                adjusted_matrix.append([0] * target_cols)
        adjusted_output.append(adjusted_matrix)
    return adjusted_output


def extract_all_matrices_with_padding(
        input_strings: List[str], answer_matrices: List[List[List[Union[int, float]]]]
) -> List[List[List[Union[int, float]]]]:
    list_of_all_matrices = []
    pattern_matrix = r"(\[\s*(?:\[[^\[\]]*\]\s*,?\s*)+\])"
    pattern_single_sequence = r"(\[(?:[^\[\]]*)\](?:\s*,\s*\[(?:[^\[\]]*)\])+)"

    for idx, s in enumerate(input_strings):
        matrices_in_string = []
        captured_spans = []

        for m in re.finditer(pattern_matrix, s, re.DOTALL):
            span = m.span()
            mat_str = m.group(1)
            try:
                mat = ast.literal_eval(mat_str)
                if isinstance(mat, list) and all(isinstance(row, list) for row in mat):
                    matrices_in_string.append(mat)
                    captured_spans.append(span)
            except (ValueError, SyntaxError):
                continue

        def is_overlap(span):
            return any(
                span[0] < cspan[1] and span[1] > cspan[0] for cspan in captured_spans
            )

        for m in re.finditer(pattern_single_sequence, s, re.DOTALL):
            if is_overlap(m.span()):
                continue
            single_str = f"[{m.group(1)}]"
            try:
                mat = ast.literal_eval(single_str)
                if isinstance(mat, list) and all(isinstance(row, list) for row in mat):
                    matrices_in_string.append(mat)
            except (ValueError, SyntaxError):
                continue

        if not matrices_in_string:
            try:
                shape_provider = answer_matrices[idx]
                num_rows = len(shape_provider)
                num_cols = len(shape_provider[0]) if num_rows > 0 else 0
                zero_matrix = [[0] * num_cols for _ in range(num_rows)]
                matrices_in_string = [zero_matrix]
            except (IndexError, TypeError):
                matrices_in_string = [[]]

        list_of_all_matrices.append(matrices_in_string)
    return list_of_all_matrices


def flag_one_to_one(item) -> int:
    if item is None:
        return 0

    if isinstance(item, dict):
        for params in item.values():
            if isinstance(params, Iterable) and not isinstance(params, (str, bytes)):
                if len(params) != 1:
                    return 0
        return 1

    if isinstance(item, str):
        headers = re.findall(r"\[([^\[\]]+?)\]_\[([^\[\]]+?)\]:", item)
        if not headers:
            return 0
        mapping = {}
        for pred, param in headers:
            mapping.setdefault(pred, set()).add(param)
        return 1 if all(len(v) == 1 for v in mapping.values()) else 0

    return 0


def getdata(data):
    keys = ["filename", "train_inputs", "train_outputs", "test_input", "test_output"]
    return tuple([item[key] for item in data] for key in keys)


def _save_pickle(data: Any, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def save_info_accuracy(
        num_correct_cot_ilp_list,
        num_correct_cot_list,
        num_correct_wo_cot_list,
        transformation_criterion_save,
        save_dir,
):
    result_dir = os.path.join(save_dir, "result")
    _save_pickle(
        num_correct_cot_ilp_list,
        os.path.join(result_dir, "num_correct_cot_ilp_list.pkl"),
    )
    _save_pickle(
        num_correct_cot_list, os.path.join(result_dir, "num_correct_cot_list.pkl")
    )
    _save_pickle(
        num_correct_wo_cot_list,
        os.path.join(result_dir, "num_correct_wo_cot_list.pkl"),
    )
    _save_pickle(
        transformation_criterion_save,
        os.path.join(result_dir, "transformation_criterion_save.pkl"),
    )


def save_question(Question, save_dir):
    result_dir = os.path.join(save_dir, "result")
    _save_pickle(Question, os.path.join(result_dir, "Question.pkl"))


def custom_collate_fn(batch, processor, process_vision_info):
    formatted_batch = []
    for chat_instance in batch:
        new_instance = []
        for message in chat_instance:
            content = message["content"]
            if isinstance(content, list):
                content_str = "".join(
                    part["text"] for part in content if part["type"] == "text"
                )
            else:
                content_str = content
            new_message = {"role": message["role"], "content": content_str}
            new_instance.append(new_message)
        formatted_batch.append(new_instance)

    texts = [
        processor.apply_chat_template(
            example, tokenize=False, add_generation_prompt=True
        )
        for example in formatted_batch
    ]
    image_inputs = [process_vision_info(example)[0] for example in batch]

    model_inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    return model_inputs


def text_generate_no_truncation_test(
        batch, model, processor, tokenizer, stop_sign_indices, decoded_outputs, max_new_tokens=256
):
    def flatten_msg(msg):
        content = msg["content"]
        if isinstance(content, list):
            pieces = []
            for blk in content:
                if blk.get("type") == "text":
                    pieces.append(blk["text"])
                elif blk.get("type") == "image_url":
                    pieces.append(f"[IMAGE: {blk['url']}]")
            msg["content"] = "".join(pieces)
        return msg

    for conv in batch:
        for msg in conv:
            flatten_msg(msg)

    prompts = tokenizer(
        [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in batch
        ],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    prompt_lens = prompts["attention_mask"].sum(dim=1)

    generated_ids = model.generate(
        **prompts, max_new_tokens=5000, pad_token_id=tokenizer.eos_token_id
    )
    replies = []
    for i, output_ids in enumerate(generated_ids):
        gen_ids = output_ids[prompt_lens[i]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        replies.append(reply)

    return extract_after_last_think(replies)


def extract_after_last_think(strings):
    results = []
    for s in strings:
        last_think_end = s.rfind("</think>")
        if last_think_end == -1:
            results.append(s)
        else:
            start_index = last_think_end + len("</think>")
            extracted = s[start_index:]
            if extracted.startswith("\n\n"):
                extracted = extracted[2:]
            elif extracted.startswith("\n"):
                extracted = extracted[1:]
            results.append(extracted)
    return results


def updates_list(
        rules_list,
        sub_answer_list,
        sub_question_list,
        meta_question_list,
        sub_question,
        sub_answer,
        meta_question,
        rules,
):
    if not rules_list:
        num_rules = len(rules)
        rules_list.extend([[] for _ in range(num_rules)])
        sub_answer_list.extend([[] for _ in range(num_rules)])
        sub_question_list.extend([[] for _ in range(num_rules)])
        meta_question_list.extend([[] for _ in range(num_rules)])

    for i in range(len(rules)):
        rules_list[i].append(rules[i])
        sub_answer_list[i].append(sub_answer[i])
        sub_question_list[i].append(sub_question[i])
        meta_question_list[i].append(meta_question[i])
    return rules_list, sub_answer_list, sub_question_list, meta_question_list


def save_list(
        rules_list,
        sub_answer_list,
        sub_question_list,
        meta_question_list,
        file_name,
        continue_indices,
        save_dir,
):
    def load_qa(directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        json_path = os.path.join(directory_path, "q_a.json")
        if not os.path.exists(json_path):
            print(f"[INFO] File {json_path} does not exist. Creating a new one.")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return {}
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_qa(data: dict, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        json_path = os.path.join(directory_path, "q_a.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_record(
            qa_dict: dict, file_name: str, rules_list, sub_answer, sub_question, meta_question
    ):
        if file_name not in qa_dict:
            qa_dict[file_name] = {
                "rules_list": [],
                "sub_answer": [],
                "sub_question": [],
                "meta_question": [],
            }
        qa_dict[file_name]["rules_list"].append(rules_list)
        qa_dict[file_name]["sub_answer"].append(sub_answer)
        qa_dict[file_name]["sub_question"].append(sub_question)
        qa_dict[file_name]["meta_question"].append(meta_question)

    def process_batch(
            qa_path: str,
            indices_to_process: Sequence[int],
            all_rules,
            all_sub_answers,
            all_sub_questions,
            all_meta_questions,
            all_filenames,
    ):
        qa_data = load_qa(qa_path)
        for i in indices_to_process:
            add_record(
                qa_data,
                file_name=all_filenames[i],
                rules_list=all_rules[i],
                sub_answer=all_sub_answers[i],
                sub_question=all_sub_questions[i],
                meta_question=all_meta_questions[i],
            )
        save_qa(qa_data, qa_path)

    good_indices = sorted(set(range(len(file_name))) - set(continue_indices))
    process_batch(
        os.path.join(save_dir, "Q_A"),
        good_indices,
        rules_list,
        sub_answer_list,
        sub_question_list,
        meta_question_list,
        file_name,
    )

    bad_indices = sorted(list(set(continue_indices)))
    process_batch(
        os.path.join(save_dir, "Q_A_bad"),
        bad_indices,
        rules_list,
        sub_answer_list,
        sub_question_list,
        meta_question_list,
        file_name,
    )


def prolog_write(knowledge_base, metarules):
    def split_predicate(atom: str):
        m = re.match(r"\s*([a-zA-Z0-9_]+)\s*\(([^)]*)\)", atom)
        if not m:
            return atom.strip(), 0
        name = m.group(1).strip()
        args = [x for x in m.group(2).split(",") if x.strip() != ""]
        return name, len(args)

    out_dir = Path(__file__).resolve().parent / "PL_File"
    out_dir.mkdir(exist_ok=True)

    for idx, predicates in enumerate(knowledge_base, start=1):
        lines = [":- use_module('metagol').", "metagol:max_clauses(1).", ""]
        if predicates:
            lines.append("% ---------- background facts ----------")
            lines.extend([f"{atom}." for atom in predicates])
            lines.append("")
            lines.append("% ---------- body_pred declarations ----------")
            body_lines = []
            seen = set()
            for atom in predicates:
                name, arity = split_predicate(atom)
                sig = f"{name}/{arity}"
                if sig not in seen:
                    body_lines.append(f"body_pred({sig}).")
                    seen.add(sig)
            lines.extend(body_lines)
            lines.append("")

        if metarules[idx - 1]:
            lines.append("% ---------- metarules ----------")
            lines.extend([f"metarule({mr.rstrip('.')})." for mr in metarules[idx - 1]])
            lines.append("")

        lines.extend(
            ["% ---------- learning task ----------", "a :-", "    Pos = [f(b)],", "    Neg = [],",
             "    learn(Pos,Neg).", ""]
        )
        file_path = out_dir / f"kb_{idx:02d}.pl"
        file_path.write_text("\n".join(lines), encoding="utf-8")


def capture_prolog_output(file_dir, timeout=5):
    temp_script_path = "temp_prolog_query.py"
    temp_code = f"""
from pyswip import Prolog
try:
    prolog = Prolog()
    prolog.consult("{file_dir}")
    query = "a."
    list(prolog.query(query))
except Exception as e:
    print(f"Prolog Error: {{e}}")
"""
    with open(temp_script_path, "w") as f:
        f.write(temp_code)

    try:
        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = process.stdout
        f_rules = re.findall(r"f\([A-Z](?:,[A-Z])*?\):-.*?\.", output)
        return "\n".join(f_rules)
    except subprocess.TimeoutExpired:
        return f"Query exceeded timeout of {timeout} seconds."
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)


def dedup_sublists(list_of_lists):
    deduped = []
    for sub in list_of_lists:
        seen = set()
        new_sub = []
        for item in sub:
            if item not in seen:
                seen.add(item)
                new_sub.append(item)
        deduped.append(new_sub)
    return deduped


def strip_outer_brackets(s: str) -> str:
    if s.startswith("[[") and s.endswith("]]"):
        return s[1:-1]
    return s


def dedup_grouped_meta(grouped: List[List[str]]) -> List[List[str]]:
    result: List[List[str]] = []
    for sub in grouped:
        seen: set[str] = set()
        unique: List[str] = []
        for rule in sub:
            key = canonical_meta_rule(rule)
            if key not in seen:
                seen.add(key)
                unique.append(rule)
        if not unique:
            unique.append(_NAIVE_META)
        result.append(unique)
    return result


def canonical_meta_rule(rule: str) -> str:
    mapping: dict[str, str] = {}

    def repl(m: re.Match):
        tok = m.group(0)
        if tok == "A":
            return tok
        if tok not in mapping:
            mapping[tok] = f"V{len(mapping) + 1}"
        return mapping[tok]

    return _TOKEN_RE.sub(repl, rule.strip())


def extract_meta_rules_grouped(text_blocks: List[str | List[str]]) -> List[List[str]]:
    grouped: List[List[str]] = []
    for block in text_blocks:
        if isinstance(block, list):
            block = "\n".join(block)

        metas_this_doc = []
        for frag in _CODE_BLOCK_RE.findall(block):
            lines = frag.splitlines()
            if lines and not lines[0].lstrip().startswith("[["):
                frag = "\n".join(lines[1:])

            i = 0
            n = len(frag)
            while i < n:
                if frag.startswith("[[", i):
                    depth = 0
                    for j in range(i, n):
                        if frag[j] == "[":
                            depth += 1
                        elif frag[j] == "]":
                            depth -= 1
                            if depth == 0:
                                cand = frag[i: j + 1]
                                fmt = _normalize_meta(cand)
                                if fmt and fmt not in metas_this_doc:
                                    metas_this_doc.append(fmt)
                                i = j
                                break
                    else:
                        break
                i += 1
        if not metas_this_doc:
            metas_this_doc.append(_NAIVE_META)
        grouped.append(metas_this_doc)
    return grouped


def _normalize_meta(raw: str) -> str | None:
    raw = raw.strip()
    if not (raw.startswith("[[") and raw.endswith("]]")):
        return None

    inner = raw[1:-1]
    parts = _split_top_level(inner)
    if len(parts) != 3:
        return None

    preds, head, tail = parts
    if not _PREDLIST_RE.fullmatch(preds):
        return None
    if not _HEAD_RE.fullmatch(head):
        return None

    if _TAIL_SINGLE_RE.fullmatch(tail):
        tail = f"[{tail}]"
    else:
        pattern = r"\[\s*(?:\[\s*[A-Za-z]+\s*,\s*A\s*\]\s*,\s*)*\[\s*[A-Za-z]+\s*,\s*A\s*\]\s*\]"
        if not re.fullmatch(pattern, tail):
            return None
    return f"[[{preds[1:-1]}], {head}, {tail}]"


def _split_top_level(lst: str) -> List[str]:
    parts, depth, buf = [], 0, ""
    for ch in lst:
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        buf += ch
    parts.append(buf.strip())
    return parts


def extract_transformation_criteria(blocks: List[str]) -> List[List[str]]:
    results = []
    for paragraph in blocks:
        matches = _TRANSFORMATION_PATTERN.findall(paragraph)
        results.append(matches)
    return results


def convert_token(token: str) -> str:
    if "(" not in token or ")" not in token:
        return token
    name, args = token.split("(", 1)
    args = args.rstrip(")")
    args_clean = re.sub(r"[\s,]", "", args)
    return f"{name}__{args_clean}(b)"


def canonical_prolog_line(line):
    if ":-" not in line:
        return line.strip()
    head, body = line.split(":-", 1)
    body = body.strip().rstrip(".")
    preds = [m.group(1) for m in _PRED_PAT.finditer(body)]
    if len(set(preds)) != len(preds):
        return None
    preds_sorted = ",".join(sorted(preds))
    return f"{head.strip()} :- {preds_sorted}."


def top_rules(data, k=3):
    res = []
    for block in data:
        raw = [
            ln.strip()
            for ln in (block.splitlines() if isinstance(block, str) else block)
            if ln.strip()
        ]
        total = len(raw)
        if total < k:
            res.append([])
            continue

        canon_map = {}
        for ln in raw:
            can = canonical_prolog_line(ln)
            if can:
                canon_map[ln] = can

        freq = Counter(canon_map.values())
        scores = {c: -math.log2(cnt / total) for c, cnt in freq.items()}
        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], -len(set(_PRED_PAT.findall(x[0]))), x[0]),
        )

        picked, seen = [], set()
        for can, _ in ranked:
            if can in seen:
                continue
            for ln, c in canon_map.items():
                if c == can:
                    picked.append(ln)
                    seen.add(can)
                    break
            if len(picked) == k:
                break
        res.append(picked)
    return res


def save_extra(filenames, continue_indices_ablation, continue_indices_masked):
    def indices_to_dict(indices: List[int], all_filenames: List[str]) -> Dict[str, str]:
        return {all_filenames[i]: "not-pass" for i in indices}

    def build_labeled_dicts(
            ablation: List[List[List[int]]], masked: List[List[int]], all_filenames: List[str]
    ) -> Dict[str, Dict[str, str]]:
        labelled = {}
        for i, sub in enumerate(ablation):
            for j, subsub in enumerate(sub):
                key = f"ablation_{i}_{j}"
                labelled[key] = indices_to_dict(subsub, all_filenames)
        for k, sub in enumerate(masked):
            key = f"masked_{k}"
            labelled[key] = indices_to_dict(sub, all_filenames)
        return labelled

    def load_existing(path: str) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_labeled_dicts(new_dicts: Dict[str, Dict[str, str]]) -> None:
        os.makedirs(_ABLATION_OUT_DIR, exist_ok=True)
        path = os.path.join(_ABLATION_OUT_DIR, _ABLATION_OUT_FILE)
        existing = load_existing(path)
        existing.update(new_dicts)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    labeled = build_labeled_dicts(
        continue_indices_ablation, continue_indices_masked, filenames
    )
    save_labeled_dicts(labeled)


def select_rules_using_clip(
        rules_text,
        positive_filenames,
        negative_filenames,
        positive_images_dir=URL_POS,
        negative_images_dir=URL_NEG
):
    parts = re.split(r'\s*\d+\.\s*', rules_text)
    candidate_rules = [p.strip() for p in parts if p.strip()]
    if not candidate_rules:
        return ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    def load_images_from_paths(base_dir, filenames):
        images = []
        for name in filenames:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                images.append(image_tensor)
        if not images:
            raise FileNotFoundError(f"No images found in directory: {base_dir} with provided filenames.")
        return torch.cat(images, dim=0)

    positive_images = load_images_from_paths(positive_images_dir, positive_filenames)
    negative_images = load_images_from_paths(negative_images_dir, negative_filenames)

    with torch.no_grad():
        positive_image_features = model.encode_image(positive_images).float()
        negative_image_features = model.encode_image(negative_images).float()

        positive_image_features /= positive_image_features.norm(dim=-1, keepdim=True)
        negative_image_features /= negative_image_features.norm(dim=-1, keepdim=True)

    positive_avg_features = positive_image_features.mean(dim=0)
    negative_avg_features = negative_image_features.mean(dim=0)

    text_tokens = clip.tokenize(candidate_rules).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity_to_pos = text_features @ positive_avg_features
    similarity_to_neg = text_features @ negative_avg_features

    scores = (POSITIVE_SIMILARITY_WEIGHT * similarity_to_pos) - (NEGATIVE_SIMILARITY_WEIGHT * similarity_to_neg)
    best_score_index = torch.argmax(scores).item()

    return candidate_rules[best_score_index]


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


def extract_prolog_content(s):
    return s.split('```prolog', 1)[-1].rsplit('```', 1)[0].strip()
