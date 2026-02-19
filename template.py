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

    model.eval()

    with torch.inference_mode():
        prompt_template = ("Answer the following related question(s)\n" f"Question: {questions[0]}\nAns:")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
        answer1_token = model.generate(input_ids, max_new_tokens=20, use_cache=True)
        answer1 = tokenizer.decode(answer1_token[0], skip_special_tokens=True)

        prompt_template = f"{prompt_template}{answer1}\nQuestion: {questions[1]}\nAns:"

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
        answer2_token = model.generate(input_ids, max_new_tokens=20, use_cache=True)
        answer2 = tokenizer.decode(answer2_token[0], skip_special_tokens=True)

        prompt_template = f"{prompt_template}{answer2}\nQuestion3: {questions[2]} (***Only Answer with 'Yes' or 'No'****)\nAns:"
        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
        answer3 = model.generate(input_ids, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1, use_cache=True)
        logit_stack = torch.stack(answer3.scores, dim=1)

        yes = tokenizer.encode("Yes", return_tensors="pt", add_special_tokens=False)[0].item()
        no = tokenizer.encode("No", return_tensors="pt", add_special_tokens=False)[0].item()

        logit_y = logit_stack[0][0][yes].item()
        logit_n = logit_stack[0][0][no].item()

        if logit_n > logit_y:
            return "NO"
        else:
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