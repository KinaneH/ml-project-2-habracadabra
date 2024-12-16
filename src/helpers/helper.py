import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from Experiments.twitter_exp import preprocess


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
