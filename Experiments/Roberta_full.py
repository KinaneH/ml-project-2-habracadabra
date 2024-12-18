import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

from src.data_cleaning.cleaning import clean_tweet
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
    base_path = os.path.join(os.path.expanduser('~'), 'twitter-datasets')
    
    # Define the file paths for negative and positive training data
    file_path_neg = os.path.join(base_path, 'train_neg_full.txt')
    file_path_pos = os.path.join(base_path, 'train_pos_full.txt')

    # Load the positive and negative datasets into separate DataFrames
    pos_set, neg_set = load_data(
        path_train_pos=file_path_pos, 
        path_train_neg=file_path_neg
    )
    
    # Combine both positive and negative DataFrames
    df = pd.concat([pos_set, neg_set], ignore_index=True)

    # Set parameters for tokenization and batching
    max_length = 128   # Maximum sequence length for tokenization
    batch_size = 400   # Number of samples per batch

    # Clean all tweets in the combined dataset
    df['Text'] = df['Text'].apply(clean_tweet)

    # Create a TextDataset object for handling the data
    dataset = TextDataset(df, tokenizer, max_length)
    
    # Generate a DataLoader to facilitate batch processing
    dataloader = create_dataloader(df, tokenizer, max_length, batch_size)
    
    # Determine the computation device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize counters and accumulators for predictions and labels
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    # Set model to evaluation mode
    model.eval()
    
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            
            # Predict the class by taking the argmax over the logits
            preds = torch.argmax(outputs, dim=1)
            
            # Compare predictions to targets to count correct predictions
            correct = (preds == targets).sum().item()
            
            # Accumulate results
            correct_predictions += correct
            total_samples += outputs.shape[0]
            
            # Store predictions and targets for F1 score calculation
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
            end_time = time.time()  # Record the end time of the computation
            
            # Calculate and display the time taken to process the current batch
            batch_time = end_time - start_time
            print(f'Processed batch of size {outputs.shape[0]} in {batch_time:.2f} seconds')
            
            # Calculate and display the current accuracy
            current_accuracy = correct_predictions / total_samples
            print(f'Current Accuracy: {current_accuracy:.4f}')

    # Once all batches are processed, print the final accuracy
    final_accuracy = correct_predictions / total_samples
    print(f'Final Accuracy on combined positive and negative sets: {final_accuracy:.4f}')
    
    # Compute and print the F1 score
    # If you only want to consider positive (2) and negative (0) classes, specify labels=[0,2].
    f1 = f1_score(all_targets, all_preds, average='macro', labels=[0, 2])
    print(f'Final Macro-F1 Score (on negative and positive classes): {f1:.4f}')
