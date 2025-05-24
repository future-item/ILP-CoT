import os
import tools
import prompt
import prolog
import query_GPT
import Prepration

# test = query_GPT.Text_to_text("knock,knock, anyone here?")

rule_is_learning = True
maximum_learning = 0
while rule_is_learning:
    ################################### prepration #####################################
    device,input_images_path,input_images_path_neg,pos_image_filenames,neg_image_filenames,target_classes,target_num,choose_class = Prepration.prepration(2)
    ################################### Prepare Question ###############################
    prompt_image_to_text = prompt.Image_to_text(target_classes,choose_class,target_num,pos_image_filenames)
    ################################### LLM ############################################
    ## make hypothesis space
    prompt_hypothesis = prompt.hypothesis_space()
    output_text_hypothesis = query_GPT.Image_to_text(input_images_path,pos_image_filenames,prompt_hypothesis)
    print(output_text_hypothesis)

    ## check format
    prompt_check =  prompt.check_format(output_text_hypothesis)
    output_text_hypothesis

    ## capturer
    prompt_capturer = prompt.extract_capture_from_hypothesis_space(output_text_hypothesis)
    output_text_captured_info = query_GPT.Image_to_text(input_images_path,pos_image_filenames,prompt_capturer)
    print("output_text_captured_info")
    print(output_text_captured_info)

    prompt_capturer = prompt.extract_capture_from_hypothesis_space_neg(output_text_hypothesis)
    output_text_captured_info_neg = query_GPT.Image_to_text(input_images_path_neg,neg_image_filenames,prompt_capturer)
    print("output_text_captured_info_neg")
    print(output_text_captured_info_neg)


    prompt_delete_pos = prompt.delete_wrong_info(output_text_captured_info)
    output_filtered_pos = query_GPT.Image_to_text(input_images_path,pos_image_filenames,prompt_delete_pos)
    output_filtered_pos = tools.extract_prolog_content(output_filtered_pos)
    print("output_filtered_pos")
    print(output_filtered_pos)

    prompt_delete_neg = prompt.delete_wrong_info(output_text_captured_info_neg)
    output_filtered_neg = query_GPT.Image_to_text(input_images_path_neg,neg_image_filenames,prompt_delete_neg)
    output_filtered_neg = tools.extract_prolog_content(output_filtered_neg)

    print("output_filtered_neg")
    print(output_filtered_neg)

# ####################   ILP
    ## Build meta rules
    meta_text = prompt.Build_meta_rules_prepration(output_text_hypothesis)
    output_text_metarules = query_GPT.Text_to_text(meta_text)
    metarules =tools.extract_metarules(output_text_metarules)

    print(output_text_metarules)
    print(metarules)

    ## PL_File
    prolog_dir = prompt.Build_PL_file(output_filtered_pos,output_filtered_neg,metarules)

    ##############
    Answer = prolog.capture_prolog_output(prolog_dir)
    print("Answer: ")
    print(Answer)

    if (not Answer or Answer == "'Query exceeded timeout of 100 seconds.'") and maximum_learning<5:
        maximum_learning = maximum_learning+1
        print(Answer)
        print("Answer is empty or timed out. Restarting the loop.")
        continue

    # Answer = prolog.remove_duplicate_predicates(Answer)
    Answer = prolog.post_rules_process(Answer)
    print("Answer")
    print(Answer)

    ## 最终确定规则
    if not Answer:
        Answer = prompt.final_answer(Answer)
        Final_Answer = query_GPT.Image_to_text(input_images_path, pos_image_filenames, Answer)
        NLP_prompt = prompt.Rule_to_NLP(Final_Answer)
        NLP_Answer = query_GPT.Image_to_text(input_images_path, pos_image_filenames, NLP_prompt)
    elif isinstance(Answer, list) and len(Answer) == 1:
        Answer_to_nl = prompt.rules_to_nl(Answer)
        NLP_Answer = query_GPT.Image_to_text(input_images_path, pos_image_filenames, Answer_to_nl)
    else:
        Answer_to_nl = prompt.rules_to_nl(Answer)
        Answer_to_nl = query_GPT.Image_to_text(input_images_path, pos_image_filenames, Answer_to_nl)
        NLP_Answer = tools.select_rules_using_clip(Answer_to_nl, pos_image_filenames, neg_image_filenames)

    rule_is_learning = False
    print(1)

# # #########
