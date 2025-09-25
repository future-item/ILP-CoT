import subprocess
import sys
import os
import re


TEMP_SCRIPT_NAME = ''

def capture_prolog_output(prolog_file_path: str, timeout: int = 100) -> str:
    script_content = f'''
from pyswip import Prolog
prolog = Prolog()
prolog.consult("{prolog_file_path}")
query = "a."
list(prolog.query(query))
'''
    with open(TEMP_SCRIPT_NAME, 'w') as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            [sys.executable, TEMP_SCRIPT_NAME],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        return f"Query exceeded timeout of {timeout} seconds."
    finally:
        if os.path.exists(TEMP_SCRIPT_NAME):
            os.remove(TEMP_SCRIPT_NAME)

    found_rules = re.findall(r'f\([A-Z](?:,[A-Z])*?\):-.*?\.', output)
    return '\n'.join(found_rules)

def post_rules_process(rules_str: str) -> set:
    if not rules_str:
        return set()

    normalized_rules = set()
    for line in rules_str.strip().splitlines():
        if ':-' in line:
            conditions_part = line.split(':-')[1].rstrip('.').split(',')
            conditions_tuple = tuple(sorted(set(p.strip() for p in conditions_part)))
            if conditions_tuple:
                normalized_rules.add(conditions_tuple)

    final_rules = set()
    for rule_a in normalized_rules:
        is_subsumed = False
        for rule_b in normalized_rules:
            if rule_a != rule_b and set(rule_a).issubset(set(rule_b)):
                is_subsumed = True
                break
        if not is_subsumed:
            final_rules.add(rule_a)

    return final_rules