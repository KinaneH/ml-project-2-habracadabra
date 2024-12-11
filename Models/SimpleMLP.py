import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.module import T
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
sys.path.append('../')


##the loss function we take is the binary cross entropy

# Define a flexible MLP model
class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1, bias=True, weight_std=0.01):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_layers (list of int): A list where each value represents the number of neurons in that hidden layer.
            output_dim (int): Number of output neurons (default is 1 for binary classification).
            bias (bool): Whether to use bias in linear layers.
            weight_std (float): Standard deviation for weight initialization.
        """
        super(FlexibleMLP, self).__init__()

        # Create a list to hold all layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_layers[0], bias=bias))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias))
            layers.append(nn.ReLU())

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim, bias=bias))
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

        # Initialize weights with the given standard deviation
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=weight_std)
                if bias:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, train_loader, criterion, optimizer, tokenizer, llm_model):
        # Training loop
        for batch in train_loader:
            # this is the loop ver mini-batches. There are train_data_size / batch_size batches of data
            # Forward pass
            # each batch is a bunch of twitter messages, tokenized
            input_tokens = batch['input_ids']
            targets = batch['target']
            [llm_model(**input_token).hidden_states[-1].flatten() for input_token in input_tokens]


            outputs = self.model(inputs)
            # Loss function is the choice of the modeler, we chose binary cross entropy which
            # is common for 2 class data but there might be other choices
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()  # This is to forget all previously computed on other data chunks
            loss.backward()  # This computes the gradient of the current mini batch
            optimizer.step()  # move parameters (weights + biases)
            # in the direction of the gradient computed using loss.backward
        return loss.item()

    def train_model(self, lr: float, epochs: int, optimizer_name: str='Adam'):
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)  # lr is the learning rate
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss = self.train_one_epoch(train_loader, criterion, optimizer)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")




if __name__ == '__main__':
    input_dim = 20  # Number of input features
    hidden_layers = [2]  # List of hidden layers sizes
    output_dim = 1  # Number of outputs for binary classification
    bias = True  # Whether to use bias in linear layers

    # Fixing the random seed
    seed = 42  # You can choose any integer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Example dataset (replace with your actual data)
    X_train = torch.randn(1000, input_dim)  # 1000 samples, input_dim features
    true_functin_weights = torch.randn(input_dim, output_dim) # we just generate some random function
    bla = 0.
    y_train =  1. * (X_train @ true_functin_weights + bla > 0)
    print(f'we have {y_train.mean()} fraction of positive labels in the training set')

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)

    # batch_size = mini batch
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    learning_rates = [10 ** i for i in range(-10, 2)]
    accuracies = []

    # Create the model
    weight_stds =  [10 ** i for i in range(-10, 2)] # Standard deviation for weights initialization
    #model = FlexibleMLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, bias=bias,
                        #weight_std=weight_std)

    for weight_std in weight_stds :
        model = FlexibleMLP(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, bias=bias,
                            weight_std=weight_std)
        for lr in learning_rates:
            model.train_model(lr=lr, epochs=10)
            #Example to evaluate model (you'll need your own test data)
            model.eval()
            with torch.no_grad():
                # Example dataset (replace with your actual data)
                X_test = torch.randn(100, input_dim)  # 100 samples, input_dim features
                y_test = 1. * (X_test @ true_functin_weights + bla > 0)

                outputs = model(X_test)
                predicted = (outputs > 0.5).float()
                accuracy = (predicted == y_test).float().mean()
                accuracies.append(accuracy.item())
                concatenated_tensor = torch.cat((predicted, y_test), dim=1)
                print(f'Accuracy: {accuracy.item():.4f}')
        breakpoint()
        print(accuracies)
