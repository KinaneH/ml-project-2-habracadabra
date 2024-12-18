import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text inputs and their corresponding targets.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the data. It must have at least two columns: 'Text' and 'Target'.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer from Hugging Face Transformers to tokenize the text data.
        max_length (int): The maximum length for tokenizing the text sequences.

    Attributes:
        texts (List[str]): List of text samples extracted from the 'Text' column of the dataframe.
        targets (List[float]): List of target labels extracted from the 'Target' column of the dataframe.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for processing text data.
        max_length (int): The maximum sequence length for tokenization.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        # Extract 'Text' column values as a list of strings
        self.texts = dataframe['Text'].tolist()
        # Extract 'Target' column values as a list of labels (usually 0/1 or -1/+1)
        self.targets = dataframe['Target'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized text and corresponding target for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): Token IDs of the input text.
                - 'attention_mask' (torch.Tensor): Attention mask for the input text.
                - 'target' (torch.Tensor): The target label for the input text.
        """
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

def create_dataloader(dataframe, tokenizer, max_length, batch_size):
    """
    Creates a PyTorch DataLoader from a pandas DataFrame for model training.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the data. It must have at least two columns: 'Text' and 'Target'.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer from Hugging Face Transformers to tokenize the text data.
        max_length (int): The maximum length for tokenizing the text sequences.
        batch_size (int): Number of samples per batch to load.

    Returns:
        DataLoader: A PyTorch DataLoader that provides batches of data from the TextDataset. The data is shuffled each epoch.
    """
    # Create a TextDataset instance
    dataset = TextDataset(dataframe, tokenizer, max_length)
    # Create a DataLoader that provides batches of data from the dataset
    # shuffle=True ensures the data is shuffled each epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
