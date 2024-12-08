import numpy as np
import pandas as pd
import string
import re
import os
import sys


# Dynamically add the project root to the Python path
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
project_root = os.path.abspath(os.path.join(current_file_dir))
sys.path.append(project_root)

from dictionaries import *

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict



def clean_tweet(tweet):
    """
    Takes a tweet as argument and returns cleaned version.
    Note that the order of the calls is relevant.
    """
    #lower case all words
    tweet = tweet.lower()
    #Normalize to 2 repetition 
    tweet = handle_repeating_char(tweet)
    #Translate emoticons
    tweet = handle_emoticons(tweet)
    #apply pre cleaning from ekphrasis
    tweet = handle_slang(tweet)
    #use nltk to unpack hashtags + spell correctro
    tweet = clean_processor(tweet)    
    #unpack slang words
    tweet = handle_slang(tweet)
    # replace any number by "number"
    tweet = replace_numbers(tweet)
    #replace ! and ? by "exclamation" and "question" resp
    tweet = replace_exclamation(tweet)
    tweet = replace_question(tweet)
    #remove all punctuation left
    tweet = remove_punct(tweet)
    #lemmatize
    tweet = word_tokenize(tweet)
    tweet = lemmatizer(tweet)
    #tweet = ' '.join(word for word in tweet)
    tweet = ' '.join(tweet) 
    tweet = ' '.join(tweet.split())  

    return tweet 

def remove_punct(text):
    """
    remove all punctuation defined by string.punctuation
    """
    text  = ''.join([char for char in text if char not in string.punctuation])
    return text

def replace_numbers(text):
    """
    Replace any numbre occurance by 'number' 
    """
    return re.sub('[0-9]+', ' number ', text)

def replace_exclamation(text):
    """
    Replace any '!' occurance by 'exclamation' 
    """
    return re.sub(r'(\!)+', ' exclamation ', text)

def replace_question(text):
    """
    Replace any '?' occurance by 'question' 
    """
    return re.sub(r'(\?)+', ' question ', text)

def unpack_hashtag(text):
    """
    Returns unpacked version of a hashtag
    """
    words = text.split()
    return ' '.join([Segmenter("twitter").segment(word=w[1:]) if (w[0] == '#') else w for w in words])

def remove_stop_words(text):
    """
    Returns a text without any stop word.
    List of stop words is provided from nltk.corpus
    """
    text= text.lower()
    stop_words = set(stopwords.words('english'))
    #word_tokens = word_tokenize(text)
    filtered_sentence = ' '.join([w for w in text.split() if not w in stop_words])
    return filtered_sentence

def handle_repeating_char(text):
    """
    Normalize to 2 repetitions of a single char.
    When a char is repeated at least 2 times, keep only 2 repetitions.
    e.g. "goood" becomes "good"
    """
    return re.sub(r'(.)\1+', r'\1\1', text)

def lemmatizer(data):
    """
    Returns lemmatized version of a sentence 
    """
    lm = WordNetLemmatizer()
    return [lm.lemmatize(w) for w in data]  

def handle_emoticons(text):
    """
    Replace laugh expressions such as "haha","hihi" and "hehe" by 'laugh'
    Replace emojis with explicit meaning e.g. ':)' becomes 'happy'
    """
    text = re.sub('(h+ah+a*h*)+', "<laugh>", text)
    text = re.sub('(h*eh+e*h*)+', "<laugh>", text)
    text= re.sub('(h*ih+i*h*)+', "<laugh>", text)
    return ' '.join(emojis[w] if w in emojis else w for w in text.split())

#
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user','time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used for word segmentation
    segmenter="twitter",
    # corpus from which the word statistics are going to be used for spell correction
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    #list of dictionaries, for replacing tokens extracted from the text,
    #with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def clean_processor(text) : 
    text = " ".join(text_processor.pre_process_doc(text))
    return text

def handle_slang(text):
    """
    Use the slang dict form ekphrasis to replace slang contractions
    e.g. "2mrow" becomes "tomorrow"
    """
    return ' '.join(slangdict[w] if w in slangdict else w for w in text.split())

if __name__ == '__main__':
    #export PYTHONPATH="${PYTHONPATH}:/Users/malamud/ML_course/projects/project2/"
    slang = pd.read_pickle('Data_Cleaning/slangdict.pickle')

    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    # Read the file into a DataFrame
    file_path = os.path.join(path, 'test_data.txt')
    # Process the file manually to split each line on the first comma
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split on the first comma
            split_line = line.strip().split(",", 1)
            if len(split_line) == 2:  # Ensure the row has both ID and Text
                rows.append(split_line)

    # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows, columns=["ID", "Text"]).set_index("ID")

    breakpoint()