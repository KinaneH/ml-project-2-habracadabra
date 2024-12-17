import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Data_Cleaning.cleaning import clean_tweet
from src.helpers.data_loader import create_dataloader, TextDataset
from src.helpers.helper import load_data


if __name__ == "__main__":
    # Specify the name of the pretrained model
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Initialize the tokenizer using the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the pretrained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Define the path to the directory containing Twitter datasets
    base_path = os.path.join(
        os.path.expanduser('~'), 
        'ML_course', 
        'projects', 
        'project2', 
        'Data', 
        'twitter-datasets'
    )
    
    # Define the file paths for negative and positive training data
    file_path_neg = os.path.join(base_path, 'train_neg_full.txt')
    file_path_pos = os.path.join(base_path, 'train_pos_full.txt')

    # Load the positive and negative datasets into separate DataFrames
    pos_set, neg_set = load_data(
        path_train_pos=file_path_pos, 
        path_train_neg=file_path_neg
    )
    
    # Set parameters for tokenization and batching
    max_length = 128   # Maximum sequence length for tokenization
    batch_size = 100   # Number of samples per batch

    # Flag to determine which sentiment dataset to evaluate
    positive_sentiment = True
    if not positive_sentiment:
        df = neg_set  # Use negative sentiment dataset
    else:
        df = pos_set  # Use positive sentiment dataset
    
    # Extract a single tweet from the dataset for inspection (optional)
    tweet = df.loc[0, 'Text']
    
    # Apply the cleaning function to preprocess all tweets in the dataset
    df['Text'] = df['Text'].apply(clean_tweet)

    # Create a TextDataset object for handling the data
    dataset = TextDataset(df, tokenizer, max_length)
    
    # Generate a DataLoader to facilitate batch processing
    dataloader = create_dataloader(df, tokenizer, max_length, batch_size)
    
    # Determine the computation device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize counters for tracking correct predictions and total samples
    correct_predictions = 0
    total_samples = 0
    
    # Iterate over each batch in the DataLoader
    for batch in dataloader:
        # Move input tensors to the selected device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)  # Ground truth labels
        
        # Perform a forward pass through the model without computing gradients
        with torch.no_grad():
            start_time = time.time()  # Record the start time of the computation
            
            # Obtain the model outputs (logits)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            
            # Determine correct predictions based on the sentiment flag
            if positive_sentiment:
                # For positive sentiment, check if the score for class 2 exceeds class 0
                correct = (outputs[:, 2] - outputs[:, 0] > 0).sum().item()
            else:
                # For negative sentiment, check if the score for class 2 is less than class 0
                correct = (outputs[:, 2] - outputs[:, 0] < 0).sum().item()
            
            # Accumulate the number of correct predictions
            correct_predictions += correct
            
            # Accumulate the total number of samples processed
            total_samples += outputs.shape[0]
            
            end_time = time.time()  # Record the end time of the computation
            
            # Calculate and display the time taken to process the current batch
            batch_time = end_time - start_time
            print(f'Processed batch of size {batch_size} in {batch_time:.2f} seconds')
            
            # Calculate and display the current accuracy
            current_accuracy = correct_predictions / total_samples
            print(f'Current Accuracy: {current_accuracy:.4f}')
