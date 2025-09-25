import re
import tools
import prompt
import prolog
import argparse
import query_GPT
import preparation

PROLOG_TIMEOUT_MESSAGE = "'Query exceeded timeout of 100 seconds.'"

def run_learning_process(num_classes: int, max_attempts: int):
    for attempt in range(1, max_attempts + 1):
        print(f"--- Starting Learning Attempt {attempt}/{max_attempts} ---")

        device, pos_images_path, neg_images_path, pos_filenames, neg_filenames, target_classes, target_num, choose_class = preparation.preparation(num_classes)

        default = prompt.Image_to_text(target_classes, choose_class, target_num, target_classes[choose_class])
        default_capturer = query_GPT.image_to_text(pos_images_path, pos_filenames, default)

        facts_proposal = prompt.facts_proposal_space(default_capturer)
        facts_proposal_text = query_GPT.image_to_text(pos_images_path, pos_filenames, facts_proposal)

        capture_prompt_pos = prompt.extract_capture_from_proposal_space(facts_proposal_text)
        captured_info_pos = query_GPT.image_to_text(pos_images_path, pos_filenames, capture_prompt_pos)

        delete_prompt_pos = prompt.delete_wrong_info(captured_info_pos)
        filtered_pos_text = query_GPT.image_to_text(pos_images_path, pos_filenames, delete_prompt_pos)
        filtered_pos = tools.extract_prolog_content(filtered_pos_text)

        capture_prompt_neg = prompt.extract_capture_from_proposal_space_neg(facts_proposal_text)
        captured_info_neg = query_GPT.image_to_text(neg_images_path, neg_filenames, capture_prompt_neg)

        delete_prompt_neg = prompt.delete_wrong_info(captured_info_neg)
        filtered_neg_text = query_GPT.image_to_text(neg_images_path, neg_filenames, delete_prompt_neg)
        filtered_neg = tools.extract_prolog_content(filtered_neg_text)

        meta_text_prompt = prompt.Build_meta_rules_prepration(facts_proposal_text)
        meta_rules_text = query_GPT.text_to_text(meta_text_prompt)
        meta_rules = tools.extract_metarules(meta_rules_text)

        prolog_dir_content = prompt.Build_PL_file(filtered_pos, filtered_neg, meta_rules)
        prolog_answer = prolog.capture_prolog_output(prolog_dir_content)

        if (not prolog_answer) or (prolog_answer == PROLOG_TIMEOUT_MESSAGE):
            predicates = re.findall(r'([a-zA-Z0-9_]+\([^)]*\))\.', filtered_pos)
            temp_list = []
            for predicate in predicates:
                check_prompt = prompt.check_facts(predicate)
                Binary_answer = query_GPT.image_to_text(pos_images_path, pos_filenames, check_prompt)
                temp_list.append(Binary_answer)
            filtered_list2 = [item for flag, item in zip(temp_list, predicates) if flag == 'Yes']
            filtered_pos = "\n".join(f"{item}." for item in filtered_list2)
            prolog_dir_content = prompt.Build_PL_file(filtered_pos, filtered_neg, meta_rules)
            prolog_answer = prolog.capture_prolog_output(prolog_dir_content)

        if (not prolog_answer) or (prolog_answer == PROLOG_TIMEOUT_MESSAGE):
            facts_proposal = prompt.facts_proposal_space(default_capturer)
            facts_proposal_text = query_GPT.image_to_text(pos_images_path, pos_filenames, facts_proposal)

            capture_prompt_pos = prompt.extract_capture_from_proposal_space(facts_proposal_text)
            captured_info_pos = query_GPT.image_to_text(pos_images_path, pos_filenames, capture_prompt_pos)

            delete_prompt_pos = prompt.delete_wrong_info(captured_info_pos)
            filtered_pos_text = query_GPT.image_to_text(pos_images_path, pos_filenames, delete_prompt_pos)
            filtered_pos = tools.extract_prolog_content(filtered_pos_text)

            capture_prompt_neg = prompt.extract_capture_from_proposal_space_neg(facts_proposal_text)
            captured_info_neg = query_GPT.image_to_text(neg_images_path, neg_filenames, capture_prompt_neg)

            delete_prompt_neg = prompt.delete_wrong_info(captured_info_neg)
            filtered_neg_text = query_GPT.image_to_text(neg_images_path, neg_filenames, delete_prompt_neg)
            filtered_neg = tools.extract_prolog_content(filtered_neg_text)

            meta_text_prompt = prompt.Build_meta_rules_prepration(facts_proposal_text)
            meta_rules_text = query_GPT.text_to_text(meta_text_prompt)
            meta_rules = tools.extract_metarules(meta_rules_text)

            prolog_dir_content = prompt.Build_PL_file(filtered_pos, filtered_neg, meta_rules)
            prolog_answer = prolog.capture_prolog_output(prolog_dir_content)

        if not prolog_answer or prolog_answer == PROLOG_TIMEOUT_MESSAGE:
            print(f"Prolog query failed or timed out on attempt {attempt}. Retrying...")
            continue

        processed_answer = prolog.post_rules_process(prolog_answer)
        print("\nProcessed Prolog Answer:")
        print(processed_answer)

        if not processed_answer:
            final_answer_prompt = prompt.final_answer(processed_answer)
            final_answer_rule = query_GPT.image_to_text(pos_images_path, pos_filenames, final_answer_prompt)
            nlp_prompt = prompt.rule_to_nlp(final_answer_rule)
            nlp_answer = query_GPT.image_to_text(pos_images_path, pos_filenames, nlp_prompt)
        elif isinstance(processed_answer, list) and len(processed_answer) == 1:
            answer_to_nl_prompt = prompt.rules_to_nl(processed_answer)
            nlp_answer = query_GPT.image_to_text(pos_images_path, pos_filenames, answer_to_nl_prompt)
        else:
            answer_to_nl_prompt = prompt.rules_to_nl(processed_answer)
            rules_as_text = query_GPT.image_to_text(pos_images_path, pos_filenames, answer_to_nl_prompt)
            nlp_answer = tools.select_rules_using_clip(rules_as_text, pos_filenames, neg_filenames)

        print("\n--- Final Natural Language Rule ---")
        print(nlp_answer)
        print("\nLearning process completed successfully.")
        return

    print(f"\nLearning process failed after {max_attempts} attempts.")


def main():
    parser = argparse.ArgumentParser(
        description="Run an Inductive Logic Programming (ILP) learning process on image data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--num-classes",
        type=int,
        default=2,
        help="The number of target classes for the preparation step."
    )
    parser.add_argument(
        "-m", "--max-attempts",
        type=int,
        default=5,
        help="The maximum number of times to attempt the learning loop if it fails."
    )
    args = parser.parse_args()

    run_learning_process(args.num_classes, args.max_attempts)


if __name__ == "__main__":
    main()
