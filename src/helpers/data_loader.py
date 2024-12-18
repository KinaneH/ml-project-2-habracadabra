import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# A custom dataset class to handle text inputs and their targets for model training
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        # Extract 'Text' column values as a list of strings
        self.texts = dataframe['Text'].tolist()
        # Extract 'Target' column values as a list of labels (usually 0/1 or -1/+1)
        self.targets = dataframe['Target'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text at the given index
        encoding = self.tokenizer(
            self.texts[idx],             # The text string to be tokenized
            padding="max_length",        # Pad sequences to the max_length
            truncation=True,             # Truncate sequences to max_length if they exceed it
            max_length=self.max_length,  # Maximum sequence length
            return_tensors="pt"          # Return PyTorch tensors
        )
        # Convert the target to a PyTorch tensor of type float
        target = torch.tensor(self.targets[idx], dtype=torch.float)

        # Return a dictionary containing the input IDs, attention mask, and target
        # Squeeze removes extra dimensions from the returned tensors (batch dimension)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': target
        }

# A helper function to create a DataLoader from a given dataframe
def create_dataloader(dataframe, tokenizer, max_length, batch_size):
    # Create a TextDataset instance
    dataset = TextDataset(dataframe, tokenizer, max_length)
    # Create a DataLoader that provides batches of data from the dataset
    # shuffle=True ensures the data is shuffled each epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
