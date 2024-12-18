import os
import sys
import random
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from Experiments.experiment1_gpt import outputs
from helpers.data_loader import create_dataloader, TextDataset
from helpers.helper import load_data

# Dynamically add the project root to the Python path to ensure modules can be imported correctly
PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

# Define the CustomMLP class, a simple Multi-Layer Perceptron for binary classification
class CustomMLP(nn.Module):
    """
    A custom Multi-Layer Perceptron (MLP) with one hidden layer for binary classification tasks.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Dimension of the output layer (typically 1 for binary classification).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomMLP, self).__init__()
        # First fully connected layer transforming input_dim to hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # ReLU activation function introduces non-linearity
        self.relu = nn.ReLU()
        # Second fully connected layer transforming hidden_dim to output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = self.fc1(x)       # Apply first linear transformation
        x = self.relu(x)      # Apply ReLU activation
        x = self.fc2(x)       # Apply second linear transformation
        return x              # Return the logits


def train_model(dataloader, pretrained_model, mlp, optimizer, loss_fn, device, num_epochs, use_cls='cls'):
    """
    Trains the MLP on top of a frozen pretrained transformer model.
    
    Args:
        dataloader (DataLoader): DataLoader for training data.
        pretrained_model (nn.Module): Pretrained transformer model (e.g., RoBERTa).
        mlp (nn.Module): Custom MLP model for classification.
        optimizer (torch.optim.Optimizer): Optimizer for training the MLP.
        loss_fn (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        device (torch.device): Device to run the training on (CPU or GPU).
        num_epochs (int): Number of training epochs.
        use_cls (str, optional): Strategy to extract embeddings ('cls', 'mean_pooling', 'gpt'). Defaults to 'cls'.
    
    Returns:
        None
    """
    # Move models to the specified device
    pretrained_model.to(device)
    mlp.to(device)

    # Set the pretrained model to evaluation mode to freeze its parameters
    pretrained_model.eval()

    # Record the start time for tracking total training duration
    start_time = time.time()

    # Iterate over the specified number of epochs
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record the start time of the current epoch
        mlp.train()  # Set the MLP to training mode

        total_loss = 0  # Initialize cumulative loss for the epoch
        batch_count = len(dataloader)  # Total number of batches in the dataloader

        # Iterate over each batch in the dataloader
        for batch_idx, batch in enumerate(dataloader):
            # Move batch data to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model without tracking gradients
            with torch.no_grad():
                outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

            # Extract embeddings based on the specified strategy
            if use_cls == 'cls':
                cls_embeddings = last_hidden_states[:, 0, :]  # Extract [CLS] token embedding
            elif use_cls == 'mean_pooling':
                cls_embeddings = last_hidden_states.mean(dim=1)  # Apply mean pooling across tokens
            elif use_cls == 'gpt':
                cls_embeddings = last_hidden_states[:, -1, :]  # Extract last token embedding for GPT models
            else:
                raise ValueError(f"Unknown use_cls value: {use_cls}")

            # Forward pass through the MLP to obtain logits
            logits = mlp(cls_embeddings)  # Shape: (batch_size, output_dim)

            # Compute the loss between logits and target labels
            loss = loss_fn(logits.squeeze(), targets.float())
            total_loss += loss.item()  # Accumulate the loss

            # Backpropagation steps
            optimizer.zero_grad()  # Reset gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters

            # Calculate elapsed time for the current batch
            batch_elapsed_time = time.time() - epoch_start_time
            # Estimate remaining time for the current epoch
            epoch_remaining_time = batch_elapsed_time / (batch_idx + 1) * (batch_count - batch_idx - 1)

            # Calculate total elapsed time across all epochs
            total_elapsed_time = time.time() - start_time
            # Estimate average time per epoch
            average_epoch_time = total_elapsed_time / (epoch + (batch_idx + 1) / batch_count)
            # Estimate total training time based on average epoch time
            total_estimated_time = average_epoch_time * num_epochs
            # Estimate remaining time for all epochs
            total_remaining_time = total_estimated_time - total_elapsed_time

            # Print training progress for the current batch
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{batch_count}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Epoch Remaining Time: {str(timedelta(seconds=int(epoch_remaining_time)))}," 
                  f" Total Remaining Time: {str(timedelta(seconds=int(total_remaining_time)))}")

        # Calculate average loss for the epoch
        epoch_loss = total_loss / batch_count
        # Calculate elapsed time for the epoch
        epoch_elapsed_time = time.time() - epoch_start_time
        # Print epoch summary
        print(f"Epoch {epoch + 1} completed in {str(timedelta(seconds=int(epoch_elapsed_time)))} with "
              f"Avg Loss: {epoch_loss:.4f}")

    # Calculate total training time
    total_training_time = time.time() - start_time
    print(f"Training completed in {str(timedelta(seconds=int(total_training_time)))}")


def evaluate_model(dataloader, pretrained_model, mlp, device, use_cls='cls'):
    """
    Evaluates the trained MLP model on validation/test data.
    
    Args:
        dataloader (DataLoader): DataLoader for evaluation data.
        pretrained_model (nn.Module): Pretrained transformer model.
        mlp (nn.Module): Trained MLP model for classification.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        use_cls (str, optional): Strategy to extract embeddings ('cls', 'mean_pooling', 'gpt'). Defaults to 'cls'.
    
    Returns:
        float: Accuracy of the model on the evaluation data.
    """
    # Set models to evaluation mode
    pretrained_model.eval()
    mlp.eval()

    total_samples = 0
    correct_predictions = 0

    all_preds = []    # List to store all predictions
    all_targets = []  # List to store all ground truth labels

    batch_count = len(dataloader)  # Total number of batches
    start_time = time.time()       # Start time for evaluation

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()  # Record batch start time

            # Move batch data to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through the pretrained model
            outputs = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

            # Extract embeddings based on the specified strategy
            if use_cls == 'cls':
                cls_embeddings = last_hidden_states[:, 0, :]  # Extract [CLS] token embedding
            elif use_cls == 'mean_pooling':
                cls_embeddings = last_hidden_states.mean(dim=1)  # Apply mean pooling across tokens
            elif use_cls == 'gpt':
                cls_embeddings = last_hidden_states[:, -1, :]  # Extract last token embedding for GPT models
            else:
                raise ValueError(f"Unknown use_cls value: {use_cls}")

            # Forward pass through the MLP to obtain logits
            logits = mlp(cls_embeddings).squeeze()  # Shape: (batch_size,)

            # Apply sigmoid activation to convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
            preds = (probs >= 0.5).float()

            # Update counts for accuracy calculation
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

            # Append predictions and targets for F1 score calculation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Calculate elapsed time and estimate remaining time
            elapsed_time = time.time() - start_time
            batches_completed = batch_idx + 1
            average_time_per_batch = elapsed_time / batches_completed
            remaining_batches = batch_count - batches_completed
            remaining_time = average_time_per_batch * remaining_batches

            # Print evaluation progress for the current batch
            print(f"Batch {batch_idx + 1}/{batch_count}, Accuracy so far: {correct_predictions / total_samples:.4f}, "
                  f"Remaining Time: {str(timedelta(seconds=int(remaining_time)))}")

    # Calculate final accuracy and F1 score
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_targets, all_preds, average="binary")

    # Calculate total evaluation time
    total_time = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    return accuracy, f1



