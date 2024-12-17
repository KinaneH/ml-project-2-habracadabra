import os
import sys
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd

# Adjust the project root path so that Python can find modules outside the current directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

# Import custom modules and functions.
from Models.MLPwithText import CustomMLP, train_model, evaluate_model
from src.helpers.data_loader import create_dataloader, TextDataset
from src.helpers.helper import load_data
from src.cfg import *  # This likely contains file paths and other configuration constants.

if __name__ == "__main__":
    # Define the name of the pretrained model to use. 
    # Here, "cardiffnlp/twitter-roberta-base-sentiment-latest" is a sentiment analysis model based on RoBERTa.
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Load the tokenizer and the pretrained model from Hugging Face Transformers.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set output_hidden_states=True if we need access to hidden states, though it may not be strictly necessary 
    # if we only want the last hidden state.
    pretrained_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

    # Define a path to save or load model weights.
    path = os.path.join(PROJECT_ROOT, 'data', 'weights')

    # Define the hidden size for the model's embeddings. For RoBERTa-base, the hidden size is usually 768.
    hidden_size = 768
    
    # Instantiate the MLP classifier that will be trained on top of the pretrained embeddings.
    # Input_dim matches the transformer hidden size, output_dim=1 for binary classification.
    mlp = CustomMLP(input_dim=hidden_size, hidden_dim=128, output_dim=1)

    # Load the dataset. This assumes `load_data` returns two sets (positive, negative).
    # TRAIN_POS_FULL_PATH and TRAIN_NEG_FULL_PATH are constants defined in src.cfg.
    pos_set, neg_set = load_data(path_train_pos=TRAIN_POS_FULL_PATH, path_train_neg=TRAIN_NEG_FULL_PATH)

    # Combine positive and negative sets into one DataFrame.
    df1 = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)

    # Shuffle the combined DataFrame so that the order of positive and negative examples is random.
    df = df1.sample(frac=1).reset_index(drop=True)

    # Convert targets from -1/+1 to 0/1 since a binary classification layer often expects 0/1 labels.
    df['Target'] = (df['Target'] + 1) / 2
    df['Target'] = df['Target'].astype(float)

    # Split the data into training and validation sets to check performance during training.
    random_seed = 42
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=random_seed)

    # Set the maximum sequence length for tokenization.
    max_length = 128
    # Set the batch size for DataLoaders.
    batch_size = 900

    # Create custom datasets. TextDataset likely tokenizes the text and stores input IDs, attention masks, etc.
    train_dataset = TextDataset(train_df, tokenizer, max_length)
    val_dataset = TextDataset(val_df, tokenizer, max_length)

    # Create DataLoaders from the datasets. These help in batching data and shuffling for training.
    train_dataloader = create_dataloader(train_df, tokenizer, max_length, batch_size)
    val_dataloader = create_dataloader(val_df, tokenizer, max_length, batch_size)

    # Define an optimizer for the MLP's parameters. Adam with a learning rate of 1e-4 is common.
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Define a loss function. BCEWithLogitsLoss is used for binary classification when using raw logits.
    loss_fn = nn.BCEWithLogitsLoss()

    # Determine the device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the number of training epochs.
    num_epochs = 4

    # Redefine some variables for clarity (optional repetition for clarity)
    hidden_size = 768
    hidden_dim = 128
    lr = 1e-4

    # Train the model on the training data. The train_model function likely handles 
    # the loop and updates the MLP parameters.
    train_model(train_dataloader, pretrained_model, mlp, optimizer, loss_fn, device, num_epochs=num_epochs)

    # Generate a name for the saved model file that includes hyperparameters and model name for easier identification.
    name = (f'mlp_ls={lr}_max_len={max_length}'
            f'_batch={batch_size}_{MODEL_NAME.replace("/", "")}'
            f'_hidden_size={hidden_size}_hidden_dim={hidden_dim}')

    # Save the trained MLP model's weights.
    torch.save(mlp.state_dict(), os.path.join(path, f'{name}.pth'))
    print("saved successfully")

    # Evaluate the model on the validation dataset to check performance (e.g., accuracy).
    evaluate_model(val_dataloader, pretrained_model, mlp, device)
