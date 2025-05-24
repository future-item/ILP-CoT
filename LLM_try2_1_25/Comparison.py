import query_GPT
import Prepration
import os
from pathlib import Path

# Root directories where the files will be saved
base_path_with_neg = "/home/pengyf/Third_Work/Work/LLM_try2_1_9/Gpt_pure"
base_path_without_neg = "/home/pengyf/Third_Work/Work/LLM_try2_1_9/gpt_test"
########################### LLM find rules on pos_neg_case ###########################

### prepration ###
for i in range(22):
    for j in range(5):
        device,input_images_path,input_images_path_neg,pos_image_filenames,neg_image_filenames,target_classes,target_num,choose_class = Prepration.prepration(i)
    ###  ###
        prompt_image_to_text = f"""Task Overview
        Objective: Deduce a set of five possible rules based on three images, which apply to both the main characters and the background content.
        
        Context
        Main Characters:
        
        Types: {target_classes[choose_class]}
        Quantities: {target_num[choose_class]}
        Assumption:
        
        All three images share the same set of rules.
        Rule Deduction Guidelines
        Source of Rules:
        
        All elements in the rules must originate from the images themselves.
        Imaginary or speculative elements are not allowed.
        Form of Rules:
        
        No restriction on the format or structure of the rules.
        Reasoning methods are unrestricted.
        Output Format:
        
        Rules must be expressed in natural language.
        Logical Consistency:
        
        The reasoning process and the resulting rules must be logically sound and coherent.
        Expected Output
        Number of Rules: Exactly 5.
        Scope of Rules:
        Attributes: Rules may involve the attributes of the main characters (e.g., size, color, shape).
        Relationships: Rules may address the relationships between the main characters (e.g., spatial arrangements, interactions).
        Character-Background Relationships: Rules may consider the interaction or relationship between the main characters and the background elements."""

        output_text_0 = query_GPT.Image_to_text(input_images_path,pos_image_filenames,prompt_image_to_text)

        prompt_image_to_text = f"""
        The following content is rules that build from Pos case
        [{output_text_0}]
        
        Task Overview
        Objective: Narrow down the hypothesized rules using negative examples and identify a rule that applies only to the positive examples but not to the negative examples.
        
        Context
        Positive Examples:
        
        Previously, five possible rules were hypothesized based on positive examples.
        Negative Examples:
        
        Three images are now provided as negative examples.
        Definition: A rule is invalid if it can apply to the negative examples.
        Workflow
        Step 1: Analysis of Negative Examples
        
        Review the three negative examples.
        Identify whether each hypothesized rule applies to any of the negative examples.
        Step 2: Filter Rules
        
        Eliminate any rule that is applicable to the negative examples.
        Focus on rules that are strictly applicable to the positive examples.
        Step 3: Final Rule Selection
        
        Select a rule that satisfies the condition: applies only to the positive examples and does not apply to the negative examples.
        Step 4: Reasoning for Elimination
        
        Provide explanations for why the other rules were invalidated (i.e., why they applied to the negative examples).
        Expected Output
        Final Rule:
        
        The one rule that is valid for positive examples but not for negative examples.
        Eliminated Rules:
        
        A list of the eliminated rules.
        Clear and logical explanations for why each rule was invalidated.
        
        lastly return only one rule and its description (only one sentence) to the image that conclude the image appropriately without showing how you get this rule and without showing other rules"""

        # with_neg
        output_text_1 = query_GPT.Image_to_text(input_images_path_neg,neg_image_filenames,prompt_image_to_text)

        # without_neg
        prompt_image_to_text = f""" The following content is rules that build from Pos case
        [{output_text_0}], now based on image and the rules learned from pos case, conclude one rules that appropriately described the three images, only one sentence"""
        output_text_2 = query_GPT.Image_to_text(input_images_path,pos_image_filenames,prompt_image_to_text)

  # Create folder name based on target_classes[choose_class]
        folder_name = f"{'_'.join(target_classes[choose_class])}_{choose_class}"

        # Create full paths for the two base directories
        dir_with_neg = os.path.join(base_path_with_neg, folder_name)
        dir_without_neg = os.path.join(base_path_without_neg, folder_name)

        # Create directories if they don't exist
        os.makedirs(dir_with_neg, exist_ok=True)
        os.makedirs(dir_without_neg, exist_ok=True)

        # Define file names
        file_name_with_neg = f"Answer_{j}.txt"
        file_name_without_neg = f"Answer_{j}.txt"

        # Write output_text_1 to the file in the "with_neg" folder
        with open(os.path.join(dir_with_neg, file_name_with_neg), "w", encoding="utf-8") as file_with_neg:
            file_with_neg.write(output_text_1)

        # Write output_text_2 to the file in the "without_neg" folder
        with open(os.path.join(dir_without_neg, file_name_without_neg), "w", encoding="utf-8") as file_without_neg:
            file_without_neg.write(output_text_2)