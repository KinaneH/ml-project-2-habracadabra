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



# Preprocessing function for training data
def preprocess_train_data(pos_tweets, neg_tweets, show_lengths=False, show_samples=False, seed_for_split=1):
    """
    Preprocess positive and negative tweet data for training.

    Args:
        pos_tweets (pd.DataFrame): DataFrame containing positive tweets in a column named "Text".
        neg_tweets (pd.DataFrame): DataFrame containing negative tweets in a column named "Text".
        show_lengths (bool): If True, prints the number of positive, negative, and total tweets.
        show_samples (bool): If True, prints random samples of original and cleaned tweets.
        seed_for_split (int): Random seed for reproducibility when splitting data into train/test sets.

    Returns:
        tuple: Train/test splits for cleaned tweets and their corresponding labels.
    """
    # Convert the text data from DataFrames to simple lists for easier processing
    pos_tweets_lst = pos_tweets["Text"].tolist()
    neg_tweets_lst = neg_tweets["Text"].tolist()

    # Combine the positive and negative tweets into a single list
    # Create corresponding labels: 1 for positive tweets, -1 for negative tweets
    tweets = pos_tweets_lst + neg_tweets_lst
    labels = [1] * len(pos_tweets_lst) + [-1] * len(neg_tweets_lst)

    # Optionally, print the number of positive, negative, and total tweets
    if show_lengths:
        print("-----------------------------------------------------------")
        print(f"Number of positive tweets: {len(pos_tweets_lst)}")
        print(f"Number of negative tweets: {len(neg_tweets_lst)}")
        print("-----------------------------------------------------------\n")
        print("-----------------------------------------------------------")
        print(f"Total tweets: {len(tweets)}")
        print(f"Total labels: {len(labels)}")
        print("-----------------------------------------------------------\n")

    # Clean all tweets using a helper function (assumes clean_tweet is defined elsewhere)
    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]

    # Optionally, show random examples of original and cleaned tweets for debugging
    if show_samples:
        random_tweets = random.sample(tweets, 10)
        random_cleaned_tweets = [clean_tweet(tweet) for tweet in random_tweets]
        print("{:<50} | {:<50}".format("Original Tweet", "Cleaned Tweet"))
        print("=" * 82)
        for original, cleaned in zip(random_tweets, random_cleaned_tweets):
            # Truncate tweets for better readability when printing
            print("{:<50} | {:<50}".format(original[:50], cleaned[:50]))

    # Split the data into training and validation sets
    # Use an 80/20 split and set a random seed for consistency
    return train_test_split(cleaned_tweets, labels, test_size=0.2, random_state=seed_for_split)


def preprocess_test_data(test_path):
    """
    Preprocess test data by extracting IDs and text, then cleaning the text.

    Args:
        test_path (str): Path to the test data file. 
                         The file is expected to have lines in the format: ID,TweetText.

    Returns:
        tuple: A tuple containing:
               - test_ids (list): List of tweet IDs.
               - cleaned_test_texts (list): List of cleaned tweet texts.
    """
    # Open the test data file for reading
    with open(test_path, "r", encoding="utf-8") as f:
        # Read all lines from the file
        test_data = f.readlines()

    # Lists to store IDs and tweet texts separately
    test_ids = []
    test_texts = []

    # Process each line in the test data
    for line in test_data:
        # Split the line into two parts: ID and text (split at the first comma)
        split_line = line.split(",", 1)
        test_ids.append(split_line[0])  # The first part is the ID
        test_texts.append(split_line[1])  # The second part is the tweet text

    # Clean the tweet texts using the helper function (assumes clean_tweet is defined elsewhere)
    cleaned_test_texts = [clean_tweet(tweet) for tweet in test_texts]

    # Return the IDs and cleaned texts as separate lists
    return test_ids, cleaned_test_texts



# Function to define and return a machine learning model based on its name
def get_model(model_name):
    """
    Retrieve a machine learning model based on the specified name.

    Args:
        model_name (str): The name of the model to retrieve. Options are:
                          - "logistic_regression"
                          - "naive_bayes"
                          - "svm"

    Returns:
        sklearn model: An instance of the specified machine learning model.

    Raises:
        ValueError: If the provided model name is not recognized.
    """
    # Check the provided model name and return the corresponding model
    if model_name == "logistic_regression":
        # Logistic Regression with a high iteration limit for convergence
        return LogisticRegression(max_iter=10000)
    elif model_name == "naive_bayes":
        # Multinomial Naive Bayes for categorical or text-based features
        return MultinomialNB()
    elif model_name == "svm":
        # Linear Support Vector Classifier with a high iteration limit for convergence
        return LinearSVC(max_iter=10000)
    else:
        # Raise an error if the model name is not valid
        raise ValueError(f"Unknown model: {model_name}")

    

# Function to evaluate our machine learning model using a specified metric
def evaluate_model(model, X_val, y_val, metric='f1'):
    """
    Evaluate the performance of a model on validation data.

    Args:
        model: The trained machine learning model to be evaluated.
        X_val (array-like): Validation feature data.
        y_val (array-like): True labels for the validation data.
        metric (str): The metric to use for evaluation. Options are:
                      - 'f1': F1 score (default)
                      - 'accuracy': Accuracy score

    Returns:
        float: The computed metric score (F1 or accuracy).

    Raises:
        ValueError: If an unsupported metric is provided.
    """
    # Predict labels for the validation data
    y_val_pred = model.predict(X_val)

    # Calculate the requested evaluation metric
    if metric == 'f1':
        # Compute the F1 score (default) for binary classification
        return f1_score(y_val, y_val_pred, average='binary')
    elif metric == 'accuracy':
        # Compute the accuracy score
        return accuracy_score(y_val, y_val_pred)
    else:
        # Raise an error if the provided metric is not supported
        raise ValueError(f"Unsupported metric: {metric}")


# Function to save model predictions in a CSV file for submission
def save_submission(test_ids, predictions, output_path):
    """
    Save predictions along with their corresponding test IDs to a CSV file.

    Args:
        test_ids (list): A list of IDs corresponding to the test data.
        predictions (list): A list of predicted values for the test data.
        output_path (str): The file path where the submission CSV will be saved.

    Returns:
        None
    """
    # Create a DataFrame to store the test IDs and predictions
    submission = pd.DataFrame({
        "Id": test_ids,         # Column for test IDs
        "Prediction": predictions  # Column for predicted values
    })

    # Save the DataFrame to a CSV file without the index
    submission.to_csv(output_path, index=False)

    # Optionally, you can print a confirmation message to ensure the file was saved
    print(f"Submission file saved to: {output_path}")

