import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe['Text'].tolist()
        self.targets = dataframe['Target'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': target
        }

# Example usage
def create_dataloader(dataframe, tokenizer, max_length, batch_size):
    dataset = TextDataset(dataframe, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':

    # Load tokenizer (e.g., for RoBERTa)
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Define parameters
    max_length = 128  # Maximum sequence length
    batch_size = 32   # Batch size

    # Example dataframe (replace with your actual dataframe)
    import pandas as pd
    data = {'text': ["I love this!", "I hate this!"], 'Target': [1, -1]}
    df = pd.DataFrame(data)

    # Create DataLoader
    dataloader = create_dataloader(df, tokenizer, max_length, batch_size)

    # Iterate over batches
    for batch in dataloader:
        print(batch)
        # Access input_ids, attention_mask, and target
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        breakpoint()
