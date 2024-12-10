import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader

from Experiments.experiment1_gpt import outputs
from helpers.data_loader import create_dataloader, TextDataset
from helpers.helper import load_data
import os

##vim or cat file to see outputs or errors


# Define the MLP
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define a simple training loop
def train_model(dataloader, pretrained_model, mlp, optimizer, loss_fn, device, num_epochs,use_cls='cls'):
    pretrained_model.to(device)
    mlp.to(device)

    pretrained_model.eval()  # Freeze the pretrained model

    for epoch in range(num_epochs):
        mlp.train()

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model
            with torch.no_grad():
                outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
            # Extract CLS token (first token) embedding
            if use_cls == 'cls':
                cls_embeddings = last_hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)
            elif use_cls == 'mean_pooling':
                cls_embeddings = last_hidden_states.mean(dim=1)
            elif use_cls == 'gpt':
                # GPT models are compatible with this
                cls_embeddzings = last_hidden_states[:, -1, :]

            # Forward pass through the MLP
            logits = mlp(cls_embeddings)  # Shape: (batch_size, output_dim)

            # Compute loss
            loss = loss_fn(logits.squeeze(), targets.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

def evaluate_model(dataloader, pretrained_model, mlp, device, use_cls='cls'):
    pretrained_model.eval()
    mlp.eval()

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model
            outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

            # Extract CLS token embedding
            cls_embeddings = last_hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)

            # Forward pass through the MLP
            logits = mlp(cls_embeddings).squeeze()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions (threshold at 0.5)
            preds = (probs >= 0.5).float()



            # Update counts
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy




# Example usage
if __name__ == "__main__":
    # Load pretrained model and tokenizer
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pretrained_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

    # Define MLP (input_dim = hidden size of pretrained model)
    hidden_size = 768  # For RoBERTa-base
    hidden_dim = 128
    mlp = CustomMLP(input_dim=hidden_size, hidden_dim=hidden_dim, output_dim=1)  # Binary classification

    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path_neg = os.path.join(path, 'train_neg.txt')
    file_path_pos = os.path.join(path, 'train_pos.txt')

    # Process the file manually to split each line on the first comma

    # Create a DataFrame from the processed rows
    pos_set, neg_set = load_data(path_train_pos=file_path_pos, path_train_neg=file_path_neg)
    df = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert targets from -1/+1 to 0/1
    df['Target'] = (df['Target'] + 1) / 2
    df['Target'] = df['Target'].astype(float)
    # Tokenizer, DataLoader, and Dataset
    max_length = 128
    batch_size = 100

    dataset = TextDataset(df, tokenizer, max_length)
    dataloader = create_dataloader(df, tokenizer, max_length, batch_size)

    # Define optimizer and loss function
    lr = 1e-4
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(dataloader, pretrained_model, mlp, optimizer, loss_fn, device)


    name = (f'mlp_ls={lr}_max_len={max_length}'
            f'_batch={batch_size}_{MODEL_NAME.replace("/", "")}'
            f'_hidden_size={hidden_size}_hidden_dim={hidden_dim}')

    torch.save(mlp.state_dict(), os.path.join(path, f'{name}.pth'))
    breakpoint()
    # Evaluate the model

