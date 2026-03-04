"""
This program is build with Flan-T5-XL LLM to be able to answer a question in YES/NO using the provided context as in-context learning. 

> The program accepts two parameters provided as a command line input. 
> The two inputs represent the context and the question.
> The question output is deterministic i.e. its either YES or NO. You are required to use logits to extract the output.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.
> Note that the assignment and evaluation test cases are carefully sampled from the model itself, eliminating any chance of hallucination.

Syntax: python template.py <CONTEXT> <QUESTION>

The following example is given for your reference:

 Terminal Input: python assignment.py 'Albert has been working on his project all week. He finished the final report today and submitted it to his manager before the deadline.' 'Did Albert submit his project report on time?'
Terminal Output: YES

 Terminal Input: python assignment.py 'Albert has been working on his project all week. He finished the final report today and submitted it to his manager after the deadline.' 'Did Albert submit his project report on time?'
Terminal Output: NO

 Terminal Input: 'John started watering his plants every morning this week.' 'Did John water his plants yesterday morning?'
Terminal Output: YES

 Terminal Input: 'John started watering his plants every morning this week.' 'Did John water his plants last month?'
Terminal Output: NO

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

#####
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model, tokenizer, context, question): 
    
    #todo: remove
    import time 
    start = time.perf_counter()

    prompt = f"Context: {context}\nQuestion: {question}\nThink carefully based on the information in context and Answer ONLY with Yes or No:"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    #todo: remove
    # print(f"prompt: {prompt}\n")

    # decoder_input_ids starting with the pad_token_id is standard for T5 generation
    outputs = model.generate(input_ids=inputs, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    logit_stack = torch.stack(outputs.scores, dim=1)

    yes = tokenizer.encode("Yes", return_tensors="pt", add_special_tokens=False)[0].item()
    no = tokenizer.encode("No", return_tensors="pt", add_special_tokens=False)[0].item()

    logit_y = logit_stack[0][0][yes].item()
    logit_n = logit_stack[0][0][no].item()

    final_output = "YES" if logit_y > logit_n else "NO"

    #todo: delete
    # outputs = model.generate(input_ids=inputs, max_new_tokens=1)
    # print(f"output: {tokenizer.decode(outputs[0], skip_special_tokens=True).strip()}")

    #todo: delete
    print(f"{(time.perf_counter() - start)}")

    # 4. Format the output to be exactly YES or NO 
    return final_output


"""
ALERT: * * * No changes are allowed below this comment  * * *
"""
if __name__ == '__main__':

    context = sys.argv[1].strip().lower()
    question = sys.argv[2].strip().lower()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,context,question)
    print(out.strip())

    """ End to call """