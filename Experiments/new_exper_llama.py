import torch
from transformers import pipeline

model_id = "sentiment-analysis"

pipe = pipeline(model_id=model_id, task="sentiment-analysis")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8" #"meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
prompt = 'Please classify as positive or negative: The world war 3 is coming. Answer in one word: Positive or negative.'
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(inputs.input_ids, max_length=100)

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
breakpoint()

#python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
