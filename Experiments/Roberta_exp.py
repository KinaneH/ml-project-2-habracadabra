import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from Data_Cleaning.cleaning import clean_tweet
from helpers.data_loader import TextDataset, create_dataloader
from helpers.helper import load_data

# Example usage
if __name__ == "__main__":
    # Load pretrained model and tokenizer
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path_neg = os.path.join(path, 'train_neg.txt')
    file_path_pos = os.path.join(path, 'train_pos.txt')

    # Process the file manually to split each line on the first comma

    # Create a DataFrame from the processed rows
    pos_set, neg_set = load_data(path_train_pos=file_path_pos, path_train_neg=file_path_neg)
    # Tokenizer, DataLoader, and Dataset
    max_length = 128
    batch_size = 100

    positive_sentiment = True
    if not positive_sentiment:
        df = neg_set
    else:
        df = pos_set
    tweet = df.loc[0, 'Text']

    df['Text'] = df['Text'].apply(clean_tweet)

    dataset = TextDataset(df, tokenizer, max_length)
    dataloader = create_dataloader(df, tokenizer, max_length, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    count = 0
    sentiment = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)

        # Forward pass through the pretrained model
        with torch.no_grad():
            t1 = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            #
            if positive_sentiment:
                correct =  (outputs[:, 2] - outputs[:, 0] > 0).sum().item()
            else:
                correct = (outputs[:, 2] - outputs[:, 0] < 0).sum().item()

            sentiment += correct
            count += outputs.shape[0]

            t2 = time.time()
            print(f'computing model for {batch_size} took {t2 - t1} seconds')
            print(f'accuracy so far is {sentiment / count}')

