import os
import sys
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader

# Dynamically add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(PROJECT_ROOT)

from Experiments.experiment1_gpt import outputs
from src.helpers.data_loader import create_dataloader, TextDataset
from src.helpers.helper import load_data

# Define a simple MLP (Multi-Layer Perceptron) classifier that takes 
# in embeddings from a pretrained model and outputs predictions.
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomMLP, self).__init__()
        # Fully-connected layer that transforms the input embeddings to a hidden representation.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # ReLU activation function to introduce non-linearity.
        self.relu = nn.ReLU()
        # Output layer that maps the hidden representation to the final prediction dimension.
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass: input -> FC1 -> ReLU -> FC2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the training loop function.
# This function takes a DataLoader, a pretrained model (e.g., a transformer),
# our MLP classifier, an optimizer, a loss function, and various other parameters
# such as the device (CPU/GPU), the number of epochs, and a flag for how to extract embeddings.
def train_model(dataloader, pretrained_model, mlp, optimizer, loss_fn, device, num_epochs, use_cls='cls'):
    # Move the pretrained model and MLP to the specified device (GPU if available, else CPU).
    pretrained_model.to(device)
    mlp.to(device)

    # Put the pretrained model in evaluation mode. We do this because we are using 
    # the pretrained model as a feature extractor (i.e., we do not fine-tune it).
    pretrained_model.eval()

    # Iterate over the specified number of epochs.
    for epoch in range(num_epochs):
        # Set the MLP to training mode so that parameters can be updated.
        mlp.train()

        # Iterate over batches provided by the DataLoader.
        for batch in dataloader:
            # Extract inputs (input_ids, attention_mask) and targets (labels) from the batch.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model to get embeddings.
            # No gradient calculation for the pretrained model, as it's frozen.
            with torch.no_grad():
                outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                # `last_hidden_state` is the final layer's hidden states from the pretrained model.
                last_hidden_states = outputs.last_hidden_state  # shape: (batch_size, seq_length, hidden_size)

            # Depending on `use_cls`, we may extract embeddings differently.
            # By default, 'cls' means we take the [CLS] token embedding (first token).
            if use_cls == 'cls':
                # CLS token embedding is typically the first token representation.
                cls_embeddings = last_hidden_states[:, 0, :]  # shape: (batch_size, hidden_size)
            elif use_cls == 'mean_pooling':
                # Mean pooling across the sequence dimension.
                cls_embeddings = last_hidden_states.mean(dim=1)
            elif use_cls == 'gpt':
                # For GPT-like models, we might consider the last token embedding.
                # Note: There's a small typo in the original code, should be cls_embeddings not cls_embeddzings.
                cls_embeddings = last_hidden_states[:, -1, :]

            # Forward pass through the MLP classifier to get logits.
            logits = mlp(cls_embeddings)  # shape: (batch_size, output_dim)

            # Compute the loss. Assuming a binary classification with a single output neuron,
            # the `logits` are squeezed and then compared with float targets.
            loss = loss_fn(logits.squeeze(), targets.float())

            # Backpropagation steps.
            optimizer.zero_grad()  # Reset gradients
            loss.backward()         # Compute gradients
            optimizer.step()        # Update parameters

            # Print out the loss for monitoring training progress.
            print(f"Loss: {loss.item()}")

# Define an evaluation function to assess the model's performance on a validation or test set.
def evaluate_model(dataloader, pretrained_model, mlp, device, use_cls='cls'):
    # Set models to evaluation mode.
    pretrained_model.eval()
    mlp.eval()

    total_samples = 0
    correct_predictions = 0

    # Disable gradient calculations for evaluation.
    with torch.no_grad():
        # Iterate over the evaluation DataLoader.
        for batch in dataloader:
            # Extract inputs and targets.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model to get embeddings.
            outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

            # By default, we use the CLS token embedding.
            cls_embeddings = last_hidden_states[:, 0, :]  # shape: (batch_size, hidden_size)

            # Pass the embeddings through the MLP to get logits.
            logits = mlp(cls_embeddings).squeeze()

            # For binary classification, apply a sigmoid to get probabilities between [0, 1].
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions using a 0.5 threshold.
            preds = (probs >= 0.5).float()

            # Compare predictions to targets and count correct predictions.
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

    # Compute accuracy as a metric.
    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
