"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    

    prompt_template_v2 = "You are an "
    
    
    
    prompt_template_v1 = ("You are an expert with lot of knowledge, you learn fast based on context in question1 & question2.\n"
    "You have been provided with 3 questions with the same context\n"
    "*****Anwer only with 'Yes or No' ONLY for the final question based on the context derived from answers of question1 and question2.\n"
    " - Refer to the all the 3 examples carefully and then answer the question under 'Now Solve'*****\n"

    " - Think step-by-step internally but do not show reasoning.\n\n"
    "######Example 1:\n"
    "Question1: Who is Rabindranath Tagore?\n"
    "Ans1: Rabindranath Tagore (1861–1941) was an influential Indian figure. He was a poet, writer, playwright, composer, philosopher, social reformer, and painter\n"
    "Question2: Where was he born?\n"
    "Ans2: Rabindranath Tagore was born in Calcutta (now Kolkata), India.\n"
    "Question3: Is it in America?\n"
    "Ans3: No\n\n"
    "######Example 2:\n"
    "Question1: Who is Rabindranath Tagore?\n"
    "Ans1: Rabindranath Tagore (1861–1941) was an influential Indian figure. He was a poet, writer, playwright, composer, philosopher, social reformer, and painter\n"
    "Question2: Where was he born?\n"
    "Ans2: Rabindranath Tagore was born in Calcutta (now Kolkata), India.\n"
    "Question3: Is it in India?\n"
    "Ans3: Yes\n\n"
    "######Example 3:\n"
    "Question1: What's a star in a planetary system?\n"
    "Ans1: In a planetary system, the star is the massive, central celestial body around which all other objects—including planets, moons, asteroids, and comets—orbit\n"
    "Question2: What's it called in the solar system?\n"
    "Ans2: In our solar system, the central star is called the Sun"
    "Question3: Is it Sun?\n"
    "Ans3: Yes\n\n"
    "Now Solve the following:\n"
    f"Question1: {questions[0]}\n"
    "Ans1:")

    f"Question2: {questions[1]}\n"
    f"Question3: {questions[2]}\n"


    ###########################V3 sol##################################

#todo: remove
    #import time

    

    prompt_template = ("Answer the following questions which are related to each other\n"
                      
         f"Question1: {questions[0]}\n")
    
    #print(f"prompt_template1==== {prompt_template}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"device = {device}")
    model.to(device)

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)

    
    

    yes = tokenizer.encode("Yes", return_tensors="pt", add_special_tokens=False)[0].item()
    no = tokenizer.encode("No", return_tensors="pt", add_special_tokens=False)[0].item()

    #todo: remove
    # start_time = time.perf_counter()

    
    answer1_token = model.generate(input_ids, max_new_tokens=20, use_cache=True)
    answer1 = tokenizer.decode(answer1_token[0], skip_special_tokens=True)
    
    prompt_template = prompt_template+f"Ans1: {answer1}\n"+f"Question2: {questions[1]}\nAns2:"
    #print(f"prompt_template2==== {prompt_template}")

#todo: remove
    # elapsed_time = time.perf_counter() - start_time

    # print(f"Execution time: {elapsed_time} seconds")

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
    answer2_token = model.generate(input_ids, max_new_tokens=20, use_cache=True)
    answer2 = tokenizer.decode(answer2_token[0], skip_special_tokens=True)
    
    prompt_template = prompt_template+f"{answer2}\n"+f"Question3: {questions[2]} (Only Answer with 'Yes' or 'No')\nAns3:"
    # print(f"prompt_template3==== {prompt_template}")
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
    #todo: remove
    #start_time = time.perf_counter()

    answer3 = model.generate(input_ids, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1, use_cache=True)

    #todo: remove
    # elapsed_time = time.perf_counter() - start_time

    # print(f"Logit Execution time: {elapsed_time} seconds")

    logit_stack = torch.stack(answer3.scores, dim=1)

    
    logit_y = logit_stack[0][0][yes].item()
    logit_n = logit_stack[0][0][no].item()


    if logit_n > logit_y:
        return "NO"
    else:
        return "YES"



    ##################################V2 sol####################################

    # prompt_template = ("Answer the following questions, use the below instructions. \n"
    #                    " - ***For the third question, you must respond ONLY with the word 'Yes' or 'No'***\n"
    #      f"Question1: {questions[0]}\n"
    # f"Question2: {questions[1]}\n"
    # f"Question3: {questions[2]}\n"
    # "Answer:")

    # print(f"prompt_template==== {prompt_template}")

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model.to(device)

    # input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)

    # yes = tokenizer.encode("Yes", return_tensors="pt", add_special_tokens=False)[0].item()
    # no = tokenizer.encode("No", return_tensors="pt", add_special_tokens=False)[0].item()

  

    # outputs = model.generate(input_ids, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)

    # logit_stack = torch.stack(outputs.scores, dim=1)

    
    # logit_y = logit_stack[0][0][yes].item()
    # logit_n = logit_stack[0][0][no].item()


    # if logit_n > logit_y:
    #     return "NO"
    # else:
    #     return "YES"

    # return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return "YES"

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':
    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())
    """ End to call """