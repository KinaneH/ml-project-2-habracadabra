import os
import sys
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

from Models.MLPwithText import CustomMLP, train_model, evaluate_model
from src.helpers.data_loader import create_dataloader, TextDataset
from src.helpers.helper import load_data
from src.cfg import *


if __name__ == "__main__":
    # Load pretrained model and tokenizer
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pretrained_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

    path = os.path.join(PROJECT_ROOT, 'data', 'weights')

    # Define MLP (input_dim = hidden size of pretrained model)
    hidden_size = 768  # For RoBERTa-base
    mlp = CustomMLP(input_dim=hidden_size, hidden_dim=128, output_dim=1)  # Binary classification

    # Load data
    pos_set, neg_set = load_data(path_train_pos=TRAIN_POS_FULL_PATH, path_train_neg=TRAIN_NEG_FULL_PATH)

    df1 = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)

    # Shuffle the DataFrame to mix positive and negative samples
    df = df1.sample(frac=1).reset_index(drop=True)

    # Convert targets from -1/+1 to 0/1
    df['Target'] = (df['Target'] + 1) / 2
    df['Target'] = df['Target'].astype(float)

    # Split the DataFrame into training and validation sets
    random_seed = 42
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=random_seed)

    # Tokenizer, DataLoader, and Dataset
    max_length = 128
    batch_size = 100

    train_dataset = TextDataset(train_df, tokenizer, max_length)
    val_dataset = TextDataset(val_df, tokenizer, max_length)

    train_dataloader = create_dataloader(train_df, tokenizer, max_length, batch_size)
    val_dataloader = create_dataloader(val_df, tokenizer, max_length, batch_size)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1

    hidden_size = 768  # For RoBERTa-base
    hidden_dim = 128

    lr = 1e-4

    train_model(train_dataloader, pretrained_model, mlp, optimizer, loss_fn, device,  num_epochs = num_epochs)
    name = (f'mlp_ls={lr}_max_len={max_length}'
            f'_batch={batch_size}_{MODEL_NAME.replace("/", "")}'
            f'_hidden_size={hidden_size}_hidden_dim={hidden_dim}')

    torch.save(mlp.state_dict(), os.path.join(path, f'{name}.pth'))
    print("saved successfully")

    #evaluate_model(val_dataloader, pretrained_model, mlp, device)