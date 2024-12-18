import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Experiments.twitter_exp import preprocess
import torch


def apply_model_to_tweet(tokenizer, model, tweet: str, max_length=500):
    """
    Applies the pre-trained model to a single tweet to obtain its output logits.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the pre-trained model.
        model (torch.nn.Module): The pre-trained transformer model.
        tweet (str): The tweet text to be analyzed.
        max_length (int, optional): The maximum sequence length for tokenization. Defaults to 500.

    Returns:
        torch.Tensor: The output logits from the model.
    """
    # Tokenize the input tweet with specified parameters
    encoded_input = tokenizer(
        tweet,
        return_tensors='pt',          # Return PyTorch tensors
        max_length=max_length,        # Maximum sequence length
        padding='max_length',         # Pad sequences to the maximum length
        truncation=True               # Truncate sequences longer than max_length
    )

    # Print the actual length of the tokenized input (for debugging purposes)
    print(f"Tokenized input length: {encoded_input['input_ids'].shape[1]}")

    # Ensure that the tokenized input does not exceed the maximum length
    if encoded_input['input_ids'].shape[1] > max_length:
        encoded_input['input_ids'] = encoded_input['input_ids'][:, :max_length]
        encoded_input['attention_mask'] = encoded_input['attention_mask'][:, :max_length]

    try:
        # Forward pass through the model to obtain outputs
        output = model(**encoded_input)
    except Exception as e:
        # If an error occurs during the forward pass, enter the debugger
        print(f"Error during model inference: {e}")
        breakpoint()

    return output


def load_data(path_train_pos, path_train_neg):
    """
    Loads and preprocesses positive and negative tweet datasets from specified file paths.

    Args:
        path_train_pos (str): File path to the positive tweets `.txt` file.
        path_train_neg (str): File path to the negative tweets `.txt` file.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - pos_train (pd.DataFrame): DataFrame of positive tweets with their labels.
            - neg_train (pd.DataFrame): DataFrame of negative tweets with their labels.
    """
    # Read positive tweets from the specified file, assuming each line is a tweet
    pos_train = pd.read_csv(
        path_train_pos,
        sep='\r',                  # Use carriage return as the separator
        names=['Text']             # Assign column name 'Text' to the tweet content
    )
    # Insert a new column 'Target' with label 1 for positive sentiment
    pos_train.insert(1, 'Target', 1)

    # Read negative tweets from the specified file, assuming each line is a tweet
    neg_train = pd.read_csv(
        path_train_neg,
        sep='\r',                  # Use carriage return as the separator
        names=['Text']             # Assign column name 'Text' to the tweet content
    )
    # Insert a new column 'Target' with label -1 for negative sentiment
    neg_train.insert(1, 'Target', -1)

    # Print the number of records loaded for positive and negative tweets
    print(f"Number of positive tweets loaded: {len(pos_train)}")
    print(f"Number of negative tweets loaded: {len(neg_train)}")

    return pos_train, neg_train
