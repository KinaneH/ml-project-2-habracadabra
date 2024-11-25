import os

import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

from helpers.helper import load_data, apply_model_to_tweet
from twitter_exp import preprocess, score_to_sentiment


if __name__ == "__main__":
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    #model.save_pretrained(MODEL)

    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path_neg = os.path.join(path, 'train_neg.txt')
    file_path_pos = os.path.join(path, 'train_pos.txt')

    # Process the file manually to split each line on the first comma

    # Create a DataFrame from the processed rows
    pos_set, neg_set = load_data(path_train_pos=file_path_pos, path_train_neg=file_path_neg)
    df = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)
    df['sentiment'] = 0
    for i, ind in enumerate(df.index):
        tweet = df.loc[ind, 'Text']
        tweet =  preprocess(tweet)
        output = apply_model_to_tweet(tokenizer, model, tweet)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        print(scores)
        sentiment = score_to_sentiment(scores)
        print(sentiment)
        df.loc[ind, 'sentiment'] = sentiment

    breakpoint()

    ##SAVE TO CSV




