import os

import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

from helpers.helper import load_data, apply_model_to_tweet
from twitter_exp import preprocess, score_to_sentiment

#export PYTHONPATH="${PYTHONPATH}:/Users/malamud/ML_course/projects/project2/"

if __name__ == "__main__":

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config)
    #model.save_pretrained(MODEL)

    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path_neg = os.path.join(path, 'train_neg.txt')
    file_path_pos = os.path.join(path, 'train_pos.txt')

    # Process the file manually to split each line on the first comma

    # Create a DataFrame from the processed rows
    pos_set, neg_set = load_data(path_train_pos=file_path_pos, path_train_neg=file_path_neg)
    df = pd.concat([pos_set, neg_set], ignore_index=True, axis=0)

    cls = 'cls'
    hidden_states = pd.DataFrame()
    for i, ind in enumerate(df.index):
        tweet = df.loc[ind, 'Text']
        tweet =  preprocess(tweet)
        output = apply_model_to_tweet(tokenizer, model, tweet, max_length=50)
        hs_s = output.last_hidden_state
        print(hs_s.shape)
        hidden_states[ind] = hs_s.detach().numpy().flatten()
    hidden_states.T['Target'] = df['Target']
    hidden_states.to_csv(os.path.join(path, f'hidden_states_{model}_{cls}.csv'))
