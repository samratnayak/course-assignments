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
    1. Generate answer for the first question.
    2. Generate answer for the second question using the first answer as context.
    3. Generate YES/NO for the third question using the second answer as context.
    4. Return strictly YES or NO (uppercase).
    '''

    model.eval()

    with torch.inference_mode():
        #todo: remove
        #import time 
        #start = time.perf_counter()

        q1, q2, q3 = questions

        # ---- Answer 1 ----
        #todo: remove
        #print(f"Q1: {q1} Q2: {q2} Q3: {q3}\n    {q1}")
        input_ids_1 = tokenizer(q1, return_tensors="pt", truncation=True).input_ids
        output_1 = model.generate(
            input_ids_1,
            max_new_tokens=20,
            do_sample=False
        )
        ans1 = tokenizer.decode(output_1[0], skip_special_tokens=True).strip()

        # ---- Answer 2 (one-shot with ans1) ----
        prompt_2 = f"Question: {q1}\nAnswer: {ans1}\n\nQuestion: {q2}\nAnswer:"
        #todo: remove
        #print(f"   {prompt_2}")
        input_ids_2 = tokenizer(prompt_2, return_tensors="pt", truncation=True).input_ids
        output_2 = model.generate(
            input_ids_2,
            max_new_tokens=20,
            do_sample=False
        )
        ans2 = tokenizer.decode(output_2[0], skip_special_tokens=True).strip()

        # ---- Answer 3 (deterministic YES/NO) ----
        prompt_3 = f"<Context>\n{q2}\nAnswer:{ans2}\n</Context>\nUse the details under <Context> to answer the Question with 'YES' or 'NO' only:\nQuestion:{q3}\nAns:"
        #todo: remove
        #print(f"    {prompt_3}")
        input_ids_3 = tokenizer(prompt_3, return_tensors="pt", truncation=True).input_ids
        output_3 = model.generate(input_ids_3, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=2)
        logit_stack = torch.stack(output_3.scores, dim=1)

        # Token IDs for YES and NO
        yes = tokenizer.encode("Yes", return_tensors="pt", add_special_tokens=False)[0].item()
        no = tokenizer.encode("No", return_tensors="pt", add_special_tokens=False)[0].item()

        logit_y = logit_stack[0][0][yes].item()
        logit_n = logit_stack[0][0][no].item()

        final_output = "YES" if logit_y > logit_n else "NO"
        #print(f"{(time.perf_counter() - start)}")

    return final_output

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