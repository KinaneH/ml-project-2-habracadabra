import os

import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

from helpers.helper import load_data, apply_model_to_tweet
from twitter_exp import preprocess, score_to_sentiment

if __name__ == "__main__":
    # Specify the pretrained sentiment analysis model from Hugging Face.
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Load tokenizer and configuration associated with the selected model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    
    # Load the PyTorch model for sequence classification.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)  # Uncomment if you want to save the model locally.

    # Define the path to the directory containing the Twitter datasets.
    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    
    # Define the file paths for negative and positive training data.
    file_path_neg = os.path.join(path, 'train_neg.txt')
    file_path_pos = os.path.join(path, 'train_pos.txt')

    # Load the positive and negative datasets.
    # `load_data` is assumed to return two DataFrames: pos_set and neg_set.
    pos_set, neg_set = load_data(path_train_pos=file_path_pos, path_train_neg=file_path_neg)
    
    # Concatenate the positive and negative DataFrames.
    df = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)
    
    # Initialize a new column in the DataFrame to store the computed sentiment scores.
    df['sentiment'] = 0
    
    # Iterate over the DataFrame rows and apply the model to each tweet.
    for i, ind in enumerate(df.index):
        # Extract the tweet text from the DataFrame.
        tweet = df.loc[ind, 'Text']
        
        # Preprocess the tweet (e.g., handling usernames, links).
        tweet = preprocess(tweet)
        
        # Apply the model to the tweet and get the raw output logits.
        output = apply_model_to_tweet(tokenizer, model, tweet)
        
        # Extract logits and apply softmax to get probabilities for each sentiment class.
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        print(scores)  # Debug print to see the scores for the tweet.
        
        # Convert the scores to a final sentiment label (-1 or 1).
        sentiment = score_to_sentiment(scores)
        print(sentiment)  # Debug print to see the derived sentiment.
        
        # Store the computed sentiment in the DataFrame.
        df.loc[ind, 'sentiment'] = sentiment


    # save `df` to a CSV for further analysis or submission.
    df.to_csv("annotated_tweets.csv", index=False)
