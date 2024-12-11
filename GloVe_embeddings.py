# importing modules because relative imports don't work for me
import torch

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv 
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# from Experiments.twitter_exp import preprocess


def apply_model_to_tweet(tokenizer, model, tweet: str, max_length=500):
    encoded_input = tokenizer(tweet,
                              return_tensors='pt',
                                             max_length=max_length,
                              padding='max_length',
                              truncation=True)
    print(encoded_input['input_ids'].shape[1])
    if encoded_input['input_ids'].shape[1] > max_length:
        encoded_input['input_ids'] = encoded_input['input_ids'][:, :max_length]
        encoded_input['attention_mask'] = encoded_input['attention_mask'][:, :max_length]
    try:
        output = model(**encoded_input)
    except:
        breakpoint()
    return output

def load_data(path_train_pos, path_train_neg):
    """
    Arg:
        path_train_pos : path for positive tweets file .txt
        path_train_neg : path for negative tweets file .txt

    Return:
        pos_train : positive tweets as pandas dataframes
        neg_train : negative tweets as pandas dataframes
    """
    pos_train = pd.read_csv(path_train_pos, sep = '\r',  names = ['Text'])
    pos_train.insert(1, 'Target', 1)
    neg_train = pd.read_csv(path_train_neg, sep = '\r',  names = ['Text'])
    neg_train.insert(1, 'Target', -1)
    print('Records of positive tweets ', len(pos_train))
    print('Records of negative tweets ', len(neg_train))
    return pos_train , neg_train

def split_data(pos_train , neg_train, ratio):
    """
    Arg:
        pos_train : positive tweets as pandas dataframes
        neg_train : negative tweets as pandas dataframes
        ratio : ratio for which we wish to split data into traing and testing set

    Return:
        X_train : traing set data points
        X_test : testing set data points
        y_train : training labels
        y_test : testing labels
    """
    
    #Merge Pos and Neg => Create Train_set
    train_set= pd.concat([pos_train, neg_train])
    X=  train_set.tweet
    y= train_set.target
    #SPLIT: Set same random_state to reproduce same result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=13)

    print("Train_set Info: SIZE= {size}, POSITIVE Tweets ={pos:0.2f}%, NEGATIVE Tweets = {neg:0.2f}%".format( size= len(X_train),
                                                                           pos = len(y_train[y_train == 1])*100/len(X_train),
                                                                           neg = len(y_train[y_train == -1])*100/len(X_train)))

    print("Test_set Info: SIZE= {size}, POSITIVE Tweets ={pos:0.2f}%, NEGATIVE Tweets = {neg:0.2f}%".format( size= len(X_test),
                                                                           pos = len(y_test[y_test == 1])*100/len(X_test),
                                                                           neg = len(y_test[y_test == -1])*100/len(X_test)))
    return X_train, X_test, y_train, y_test

def get_accuracy(y_test, y_pred):
    """
    Arg:
        y_test : true labels of test set
        y_pred : predicted labels 
        
    Return:
        Accuracy score rounded at 4 digits
    """ 
    return round(accuracy_score(y_test, y_pred), 4)


def evaluate_model(model,X_test ,y_pred, y_test) :
    """
    Given a local test split, this function is used to evaluate base line models.
    It displays a confusion matrix and a ROC curve.
    
    Arg:
        model : Base line model
        X_test :  Test set
        y_pred : predicted labels
        y_test : true labels of test set
        
    """ 
    #Show Confusion Matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in c_matrix.flatten() / np.sum(c_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(c_matrix, annot = labels, fmt = '',  xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show() 
    
    ##Show ROC
    # generate a no skill prediction: equivalent to a random guess
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    model_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    model_probs = model_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    model_auc = roc_auc_score(y_test, model_probs)
    # summarize scores
    #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (model_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()    
    
def plot_history(history):
    """
    For tensorflow.keras deep learning model, this function plots:
    - evolution of the accuracy as a function of epochs for both the training and validation set 
    - evolution of the loss as a function of epochs for both the training and validation set 
    """ 
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
print("End of definitions reached")

# Load dataset
def load_data(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    labels = [label] * len(tweets)
    return tweets, labels

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter

# Load positive and negative datasets
data_path = r'C:\Users\jackp\Repositories\ml-project-2-habracadabra\data'
pos_tweets, pos_labels = load_data(data_path + '\\train_pos.txt', 1)  # 1 for positive
neg_tweets, neg_labels = load_data(data_path + '\\train_neg.txt', 0)  # 0 for negative

# Combine datasets
tweets = pos_tweets + neg_tweets
labels = pos_labels + neg_labels

# Tokenize tweets
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in tweets]

# Build vocabulary
all_words = [word for tweet in tokenized_tweets for word in tweet]
vocab = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(vocab.most_common(20000))}  # Top 20k words

# Convert tweets to sequences
def encode_tweet(tweet, vocab):
    return [vocab[word] for word in tweet if word in vocab]

sequences = [encode_tweet(tweet, vocab) for tweet in tokenized_tweets]

# Pad sequences
def pad_sequences(sequences, maxlen=50):
    return np.array([seq[:maxlen] + [0] * max(0, maxlen - len(seq)) for seq in sequences])

padded_sequences = pad_sequences(sequences, maxlen=50)

# Convert labels to NumPy array
labels = np.array(labels)

# Split dataset
x_train, x_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

print(f"Positive samples: {np.sum(y_train == 1)}")
print(f"Negative samples: {np.sum(y_train == 0)}")


def load_glove_embeddings(glove_file_path, embedding_dim):
    embedding_index = {}  # This is your GloVe embeddings dictionary
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]  # The word (e.g., "king")
            vector = np.asarray(values[1:], dtype='float32')  # The vector for the word
            embedding_index[word] = vector
    return embedding_index

print("Beginning of GloVe embeddings reached")

# Load GloVe embeddings
glove_path = r"C:\Users\jackp\Desktop\OneDrive\Documents\EPFL\MA3\Machine Learning\Project2\glove.6B\glove.6B.50d.txt"
embedding_dim = 50
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)


# Step 4: Creating the Embedding Matrix
# In this step, glove_embeddings (a.k.a. embedding_index) is passed to the function create_embedding_matrix to map pre-trained GloVe vectors to your vocabulary (vocab).
def create_embedding_matrix(vocab, embedding_dim, glove_embeddings):
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))  # +1 for padding token
    for word, idx in vocab.items():  # For each word in your dataset's vocabulary
        if idx < len(vocab) + 1:  # Ensure we don’t exceed vocabulary size
            embedding_vector = glove_embeddings.get(word)  # Get the GloVe vector for the word
            if embedding_vector is not None:  # If the word is in GloVe
                embedding_matrix[idx] = embedding_vector  # Assign the GloVe vector
    return embedding_matrix

# Create embedding matrix
embedding_matrix = create_embedding_matrix(vocab, embedding_dim, glove_embeddings)



import torch
from torch.utils.data import Dataset, DataLoader

class TweetDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)  # Convert to tensor
        self.labels = torch.FloatTensor(labels)      # Convert labels to tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create dataset objects
train_dataset = TweetDataset(x_train, y_train)
val_dataset = TweetDataset(x_val, y_val)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("Beginning of LSTM reached")

# Now, let’s define the BiLSTM model with the pre-trained embedding matrix.
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, dropout):
        super(BiLSTMModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Embedding layer with pre-trained GloVe weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False
        )

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectionality

    def forward(self, x):
        embedded = self.embedding(x)          # Input shape: (batch_size, seq_length)
        lstm_out, _ = self.bilstm(embedded)   # Output shape: (batch_size, seq_length, hidden_dim*2)
        last_hidden = lstm_out[:, -1, :]      # Take the last time step's hidden states
        dropped = self.dropout(last_hidden)  # Apply dropout
        output = self.fc(dropped)            # Final output
        return output

# Train model and evaluate performance
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                # Compute accuracy
                preds = torch.round(torch.sigmoid(outputs.squeeze()))
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Hyperparametrization and model initialization
# Define hyperparameters
hidden_dim = 128
output_dim = 1  # Binary classification
dropout = 0.1
epochs = 1
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss, and optimizer
model = BiLSTMModel(embedding_matrix, hidden_dim, output_dim, dropout)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

print("Model trained")

# Evaluate the model on the test data
def predict(model, tweet, vocab, maxlen=50, device="cpu"):
    # Tokenize and encode the tweet
    tokens = word_tokenize(tweet.lower()) # ignoring capitalization could bring issues
    sequence = [vocab.get(word, 0) for word in tokens]  # Use 0 for unknown words
    padded_sequence = pad_sequences([sequence], maxlen=maxlen)

    # Convert to tensor
    input_tensor = torch.LongTensor(padded_sequence).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()

    return -1 if prediction <= 0.5 else 1

print("Model evaluated")

# Test the model on the test set
test_filepath = r"C:\Users\jackp\Repositories\ml-project-2-habracadabra\data\test_data.txt"
with open(test_filepath, 'r', encoding='utf-8') as file:
    tweets = file.readlines()

predictions = []
for tweet in tweets:
    print(tweet)
    pred = predict(model, tweet, vocab, device=device)
    print(pred)
    predictions.append(pred)

# Create submission file
test_ids = np.arange(len(predictions))
create_csv_submission(test_ids, predictions, os.path.join(data_path, 'test_submission2.csv'))

# Example prediction
example_tweet = "its whatever . in a terrible mood "
print(predict(model, example_tweet, vocab, device=device))

print("End of file reached")
