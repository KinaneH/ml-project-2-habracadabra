# importing modules
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from Experiments.LLama_exp import detailed_prompt
from LLama_exp import detailed_prompt

#export PYTHONPATH="${PYTHONPATH}:/Users/malamud/ML_course/projects/project2/"
# initialize tokenizer and model from pretrained GPT2 model from Huggingface
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

# sentence
sequence = detailed_prompt("I hate worms.")
# encoding sentence for model to process
inputs = tokenizer.encode(sequence, return_tensors='pt')
# generating text
outputs = model.generate(inputs, max_length=200, do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
# decoding text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# printing output
print(text)