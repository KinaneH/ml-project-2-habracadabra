import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
# from transformers import AutoTokenizer, AutoModel
#
# # Specify the model name
# model_name = 'gpt2'# "distilbert-base-uncased" # "bert-base-uncased"
#
# # Download tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# # Load tokenizer and model
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# Assign a pad token
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
#
# # Resize model embeddings if a new pad token is added
# model.resize_token_embeddings(len(tokenizer))


print("Model and tokenizer downloaded successfully!")
import json

def clean_tweet(tweet):
    return tweet.replace("\n", " ").replace("\t", " ").replace('<user>', '').replace('<url>', '').replace('...', '')

def simple_prompt(tweet):
    tweet = clean_tweet(tweet)
    prompt = (f"Classify the following tweet as Positive "
              f"or Negative. Always output just one word, Positive or Negative:\nTweet: \"{tweet}\"\nSentiment:")
    return prompt

def detailed_prompt(tweet):
    tweet = clean_tweet(tweet)
    prompt = (f"Classify the following tweet as Positive "
              f"or Negative. Always output a detailed explanation why you think this is positive or negative. Positive or Negative:\nTweet: \"{tweet}\"\nSentiment:")
    return prompt

# Preview the DataFrame

def classify_sentiment(tweet, tokenizer, model):
    # Define the prompt

    prompt = detailed_prompt(tweet) #simple_prompt(tweet)
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(inputs.input_ids, max_length=100)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Forward pass to get the hidden states
    outputs = model(**inputs, output_hidden_states=True)

    # Extract the last hidden layer
    last_hidden_layer = outputs.hidden_states[-1]  # The last hidden layer

    # Print the shape of the last hidden layer
    # Shape: [batch_size, sequence_length, hidden_size]
    print("Last hidden layer shape:", last_hidden_layer.shape)
    breakpoint()

    return response

    # Extract sentiment from the response
    # if "Positive" in response:
    #     return 1
    # elif "Negative" in response:
    #     return -1
    # else:
    #     return 0

def classify_with_gpt2(tweet, tokenizer, model):
    # Define the prompt
    prompt = detailed_prompt(tweet)
    # Tokenize the input with padding and truncation
    # Assign a pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Tokenize the input with padding and truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # Enable padding
        truncation=True,  # Truncate long sequences
        max_length=100  # Optional: Set maximum length
    )

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Pass attention mask
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )

    # Decode the output and extract sentiment
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model Response: {response}")

    # Simple heuristic to classify sentiment
    if "Positive" in response:
        return "Positive"
    elif "Negative" in response:
        return "Negative"
    elif "Neutral" in response:
        return "Neutral"
    else:
        return "Unknown"


if __name__ == '__main__':
    # todo always start with this !!! huggingface-cli login
    # Load the CSV file
    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path = os.path.join(path, 'test_data.txt')
    # Process the file manually to split each line on the first comma
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split on the first comma
            split_line = line.strip().split(",", 1)
            if len(split_line) == 2:  # Ensure the row has both ID and Text
                rows.append(split_line)

    # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows, columns=["ID", "Text"]).set_index("ID")

    # # Load the tokenizer and model
    # model_path = '/Users/malamud/.llama/checkpoints/Llama3.2-1B-Instruct-int4-qlora-eo8/'
    # # Generate a config.json file for the specified model
    #
    # config_data = {
    #     "architectures": ["LlamaForCausalLM"],
    #     "model_type": "llama",
    #     "hidden_size": 2048,
    #     "intermediate_size": 8192,
    #     "num_attention_heads": 16,
    #     "num_hidden_layers": 24,
    #     "vocab_size": 32000,
    #     "max_position_embeddings": 4096,
    #     "initializer_range": 0.02,
    #     "rms_norm_eps": 1e-6,
    #     "pad_token_id": 0,
    #     "eos_token_id": 2,
    #     "bos_token_id": 1,
    #     "torch_dtype": "float16",
    #     "use_cache": True
    # }
    #
    # # Save to a JSON file
    # file_path = os.path.join(model_path, 'config.json')
    # with open(file_path, "w") as f:
    #     json.dump(config_data, f, indent=4)
    #
    # breakpoint()
    #
    #


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


    for i in range(df.shape[0]):
        tweet = df.iloc[i, 0]
        sentiment = classify_sentiment(tweet, tokenizer, model)