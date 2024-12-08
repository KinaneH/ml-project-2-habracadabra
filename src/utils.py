import os
import sys
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Dynamically add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from src.cfg import *
from src.data_cleaning.cleaning import clean_tweet



# Preprocessing
def preprocess_train_data(pos_tweets, neg_tweets, show_lengths=False, show_samples=False):
    # Convert DataFrames to lists of text
    pos_tweets_lst = pos_tweets["Text"].tolist()
    neg_tweets_lst = neg_tweets["Text"].tolist()
    
    # Combine tweets and labels
    tweets = pos_tweets_lst + neg_tweets_lst
    labels = [1] * len(pos_tweets_lst) + [-1] * len(neg_tweets_lst)

    # Debug lengths to ensure alignment
    if show_lengths:

        print("-----------------------------------------------------------")
        print(f"Number of positive tweets: {len(pos_tweets_lst)}")
        print(f"Number of negative tweets: {len(neg_tweets_lst)}")
        print("-----------------------------------------------------------\n")
        print("-----------------------------------------------------------")
        print(f"Total tweets: {len(tweets)}")
        print(f"Total labels: {len(labels)}")
        print("-----------------------------------------------------------\n")

    # Clean tweets
    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]

    # Debug random samples
    if show_samples:
        random_tweets = random.sample(tweets, 10)
        random_cleaned_tweets = [clean_tweet(tweet) for tweet in random_tweets]
        print("{:<50} | {:<50}".format("Original Tweet", "Cleaned Tweet"))
        print("=" * 82)
        for original, cleaned in zip(random_tweets, random_cleaned_tweets):
            print("{:<50} | {:<50}".format(original[:50], cleaned[:50]))  # Truncate for readability

    # Split into training and validation sets
    return train_test_split(cleaned_tweets, labels, test_size=0.2, random_state=SEED)

def preprocess_test_data(test_path):
    # Load test data
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = f.readlines()

    # Split data into IDs and texts
    test_ids = []
    test_texts = []
    for line in test_data:
        split_line = line.split(",", 1)  # Split at the first comma
        test_ids.append(split_line[0])  # First part is the ID
        test_texts.append(split_line[1])  # Remaining part is the tweet text

    # Clean the test texts
    cleaned_test_texts = [clean_tweet(tweet) for tweet in test_texts]

    return test_ids, cleaned_test_texts


# Define models
def get_model(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=10000)
    elif model_name == "naive_bayes":
        return MultinomialNB()
    elif model_name == "svm":
        return LinearSVC(max_iter=10000)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

# Evaluate model
def evaluate_model(model, X_val, y_val, metric='f1'):
    y_val_pred = model.predict(X_val)
    if metric == 'f1':
        return f1_score(y_val, y_val_pred, average='binary')
    elif metric == 'accuracy':
        return accuracy_score(y_val, y_val_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

# Save predictions
def save_submission(test_ids, predictions, output_path):
    submission = pd.DataFrame({"Id": test_ids, "Prediction": predictions})
    submission.to_csv(output_path, index=False)
