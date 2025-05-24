import torch
import re

def prepration(index):
    device = "cuda" if torch.cuda.is_available() else "cpu"

##################### 3 pos 3 neg
    # 输入图片的路径
    input_images_path = [["https://s21.ax1x.com/2025/01/03"],["https://v1.ax1x.com/2025/01/23/"],["https://s21.ax1x.com/2025/01/04"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/08"],["https://s21.ax1x.com/2025/01/08"],
                         ["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"]]
    input_images_path_neg = [["https://s21.ax1x.com/2024/12/22"],["https://v1.ax1x.com/2025/01/23/"],["https://s21.ax1x.com/2025/01/04"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/07"],["https://s21.ax1x.com/2025/01/08"],["https://s21.ax1x.com/2025/01/08"],["https://s21.ax1x.com/2025/01/08"],
                             ["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"],["https://v1.ax1x.com/2025/01/08"]]
    pos_image_filenames = [['pEp8kcT.png','pEp8Chq.png','pEp8F3V.png'],['7WUCBG.png','7WUzCJ.png','7WUtAL.png'],['pEp77a8.png','pEp7IqP.png','pEp7TVf.png'],['pE9hnWn.png','pE9huzq.png','pE9hmJs.png'],['pE949k4.png','pE94S7F.png','pE9hz0U.png'],['pE9bsJS.png','pE9bBIf.png','pE9bri8.png'],['pE9LeBR.png','pE9LZu9.png','pE9LmH1.png'],['pE9v0ds.png','pE9vwZj.png','pE9vaLQ.png'],['pE9zmHH.png','pE9zeDe.png','pE9zZuD.png'],
                           ['7VoxBY.png','7Vo8J9.png','7VohAh.png'],['7VovDs.png','7Vocuq.png','7Voafa.png'],['7VoRgJ.png','7VoPTL.png','7VoYYG.png'],['7VoXlP.png','7VoCfe.png','7VojVw.png'],['7Vo6w3.png','7VoiYf.png','7Vosdc.png'],['7VooV4.png','7W9Bm9.png','7W99Xh.png'],['7W9lis.png','7W93Sq.png','7W90wU.png'],
                           ['7W9w8L.png','7W9yrJ.png','7W9umV.png'],['7W9xWP.png','7W9G7e.png','7W9hsb.png'],['7W9Dyc.png','7W9vef.png','7W9ctQ.png'],['7W9Pcm.png','7W9rph.png','7W9Rj4.png'],['7W9Mca.png','7W9fn7.png','7W9IEs.png'],['7W9e1B.png','7W9dHG.png','7W9FoJ.png']]
    neg_image_filenames = [['pAXKt6x.png','pAXKanK.png','pAXKNX6.png'],['7WUXRt.png','7WUIqe.png','7WUjZb.png'],['pEp7Xxs.png','pEp7vMn.png','pEp7O2j.png'],['pE9hYFJ.png','pE9hNWR.png','pE9htY9.png'],['pE9baqI.png','pE9bwZt.png','pE9b0dP.png'],['pE9LEjJ.png','pE9LAc4.png','pE9Lk3F.png'],['pE9vJRf.png','pE9vGJP.png','pE9v8it.png'],['pE9vriq.png','pE9vsJ0.png','pE9vyWV.png'],['pE9zM4I.png','pE9zuEd.png','pE9zKUA.png'],
                           ['7Vo5gH.png','7VoARZ.png','7VoVZU.png'],['7VoSAI.png','7VoOl7.png','7VoUJV.png'],['7Vozut.png','7VordB.png','7Vo4Ob.png'],['7VoMTO.png','7VonJ6.png','7VofhQ.png'],['7VoZi5.png','7Voe7m.png','7VoFOj.png'],['7W9mrH.png','7W9ThY.png','7W9pdZ.png'],['7W9E7a.png','7W9NXI.png','7W9QW7.png'],
                           ['7W9gSt.png','7W9HeG.png','7W9byB.png'],['7W9W8O.png','7W9Vm6.png','7W95jw.png'],['7W9OU3.png','7W9Ssj.png','7W91E5.png'],['7W9XUU.png','7W9JyZ.png','7W9j2q.png'],['7W96zL.png','7W9spI.png','7W92GV.png'],['7WkBce.png','7WkTnP.png','7WkpLw.png']]


#####
    target_classes = [['cat', 'dog'], ['man','women','child'],['boy', 'girl'],['human'],['cat','child'],['cat'], ['lady', 'chair'],['dog'],['child','adult'],
                      ['male','female'],['male','female'],['male','female'],['human'],['mom','child'],['teacher','student'],['pig'],['human'],['cow'],['tree'],['sunflower'],['human'],['teacher','student']]
    target_num = [[1,1], [1,1,1], [1,1], [2], [1,1],[1], [1, 2],[2],[1,1],
                  [1,1],[1,1],[1,1],[1],[1,1],[1,2],[1],[3],[1],[2],[1],[1],[1,1]]
    choose_class = index
    input_images_path = input_images_path[choose_class][0]
    input_images_path_neg = input_images_path_neg[choose_class][0]
    pos_image_filenames = pos_image_filenames[choose_class]
    neg_image_filenames = neg_image_filenames[choose_class]
    return device,input_images_path,input_images_path_neg,pos_image_filenames,neg_image_filenames,target_classes,target_num,choose_class

def extract_entailments(llm_response):
    # Define a regular expression to match Prolog rules and extract the entailment body
    rule_pattern = re.compile(
        r'(\w+\(.*?\))\s*:-\s*(.*?)\.',  # Matches `head :- body.` structure
        re.DOTALL  # Enable multiline matching for rule bodies
    )

    # Find all rules in the LLM response
    matches = rule_pattern.findall(llm_response)

    entailments = []
    for _, body in matches:
        # Split the body into individual predicates, handle nested parentheses and formatting
        predicates = re.split(r',(?![^\(\)]*\))', body)  # Split by ',' but ignore commas inside parentheses
        predicates = [predicate.strip() for predicate in predicates]  # Clean up whitespace
        entailments.append(predicates)

    return entailments

def extract_prolog_from_text(text):
    # Regular expression to match Prolog code blocks
    prolog_blocks = re.findall(r"```prolog(.*?)```", text, re.DOTALL)

    # Cleaning up extracted blocks and concatenating them
    prolog_code = "\n\n".join(block.strip() for block in prolog_blocks)

    return prolog_code





