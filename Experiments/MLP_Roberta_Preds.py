import csv

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os

from Models.MLPwithText import CustomMLP
from src.helpers.data_loader import create_dataloader, TextDataset
from src.helpers.helper import create_csv_submission


if __name__ == "__main__":

    # Define paths and model names
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')

    # Initialize tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pretrained_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

    # Define model parameters
    hidden_size = 768  # For RoBERTa-base
    hidden_dim = 128
    output_dim = 1
    lr = 1e-4
    max_length = 128
    batch_size = 800

    #file name
    name = (f'mlp_ls={lr}_max_len={max_length}'
            f'_batch={batch_size}_{MODEL_NAME.replace("/", "")}'
            f'_hidden_size={hidden_size}_hidden_dim={hidden_dim}')


    # Initialize the MLP model
    mlp = CustomMLP(input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=output_dim)

    # Load the saved state dictionary
    model_path = os.path.join(path, f'{name}.pth')
    mlp.load_state_dict(torch.load(model_path, map_location='cpu'))

    # If GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp.to(device)
    mlp.eval()  # Set the model to evaluation mode

    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path_df = os.path.join(path, 'test_data.txt')

    test_df  = pd.read_csv(file_path_df, sep = '\r',  names = ['Text'])
    test_df.insert(1, 'Target', 0)


    test_dataset = TextDataset(test_df, tokenizer, max_length)
    test_dataloader = create_dataloader(test_df, tokenizer, max_length, batch_size=1)  # batch_size can be 1 for inference


    all_preds = []

    ######################################
    # Run inference
    ######################################
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Get outputs from pretrained model
            outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

            # Extract the CLS embedding (assuming you used CLS token during training)
            cls_embeddings = last_hidden_states[:, 0, :]  # (batch_size, hidden_size)

            # Forward pass through the MLP
            logits = mlp(cls_embeddings).squeeze()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions (threshold at 0.5)
            preds_binary = (probs >= 0.5).float()
            #to -1 1
            preds_converted = (preds_binary * 2) - 1

            # Store predictions
            all_preds.extend(preds_converted.cpu().numpy().flatten().tolist())



    print("Predictions:", all_preds)
    # Create IDs for submission
    # If test_df and all_preds have the same length, we can do:
    ids = range(1, len(all_preds) + 1)

    # Create CSV submission
    output_filename = "submission_sofiya.csv"
    create_csv_submission(ids, all_preds, output_filename)
    print(f"Submission file created: {output_filename}")
