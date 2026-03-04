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
    # 1. Engineer the prompt using the query and the context [cite: 12, 13, 17]
    # Flan-T5 responds well to structured instructions
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer with YES or NO:"

    # 2. Tokenize the prompt [cite: 5]
    inputs = tokenizer(prompt, return_tensors="pt")

    # 3. Generate output for the prompt using logits 
    # We look at the first generated token's logits to see if 'YES' or 'NO' is more likely
    with torch.no_grad():
        # decoder_input_ids starting with the pad_token_id is standard for T5 generation
        outputs = model(input_ids=inputs.input_ids, 
                        decoder_input_ids=torch.tensor([[model.config.pad_token_id]]))
        logits = outputs.logits[0, -1, :] # Get logits for the first predicted token

    # Map the tokens for 'YES' and 'NO'
    # We use the tokenizer to find the specific IDs for these words
    yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]

    # Compare the logit scores for both tokens 
    if logits[yes_token_id] > logits[no_token_id]:
        final_output = "YES"
    else:
        final_output = "NO"

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