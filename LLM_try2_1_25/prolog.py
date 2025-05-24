import subprocess
import sys
import os
import re
import threading
from collections import defaultdict


def capture_prolog_output(file_dir, timeout=100):  # 设置默认超时时间为 5 秒
    temp_code = f'''
from pyswip import Prolog
prolog = Prolog()
prolog.consult("{file_dir}")
query = "a."
list(prolog.query(query))
'''

    with open('temp_prolog_query.py', 'w') as f:
        f.write(temp_code)

    try:
        def run_subprocess():
            nonlocal result
            result = subprocess.run(
                [sys.executable, 'temp_prolog_query.py'],
                capture_output=True,
                text=True
            )
        result = None
        query_thread = threading.Thread(target=run_subprocess)
        query_thread.start()
        query_thread.join(timeout)
        if query_thread.is_alive():
            return f"Query exceeded timeout of {timeout} seconds."
        output = result.stdout if result else ""
        f_rules = re.findall(r'f\([A-Z](?:,[A-Z])*?\):-.*?\.', output)

        cleaned_output = '\n'.join(f_rules)
        return cleaned_output

    finally:
        if os.path.exists('temp_prolog_query.py'):
            os.remove('temp_prolog_query.py')


def remove_duplicate_predicates(prolog_str):
    # 按行分割
    rules = prolog_str.strip().split('\n')
    cleaned_rules = []

    for rule in rules:
        # 获取谓词部分 (:-后面的部分)
        predicates_part = rule.split(':-')[1] if ':-' in rule else ''
        # 分割所有谓词
        predicates = [p.strip() for p in predicates_part.split(',')]

        # 检查是否有重复谓词
        if len(predicates) == len(set(predicates)):
            cleaned_rules.append(rule)

    # 重新组合成字符串
    return '\n'.join(cleaned_rules)


def post_rules_process(input_text):
    # 处理输入数据
    rules = set()
    for line in input_text.strip().splitlines():
        conditions = line.split(':-')[1].rstrip('.').split(',')
        conditions = tuple(sorted(set(conditions)))  # 排序去重
        rules.add(conditions)

    # 合并逻辑：将含公共项的规则合并
    combined_rules = defaultdict(set)
    for rule in rules:
        key = frozenset(rule)
        combined_rules[key].update(rule)

    # 去掉完全被其他规则覆盖的规则
    final_rules = set()
    for rule_set in combined_rules.values():
        is_subsumed = any(rule_set < other for other in combined_rules.values() if rule_set != other)
        if not is_subsumed:
            final_rules.add(tuple(sorted(rule_set)))
    return final_rules