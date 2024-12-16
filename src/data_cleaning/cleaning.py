import numpy as np
import pandas as pd
import string
import re
import os
import sys

# Dynamically add the project root directory to the Python path
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Determine the directory of the current script
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
    Cleans a tweet by applying a series of preprocessing steps.
    
    The cleaning process includes:
    1. Converting text to lowercase.
    2. Reducing repeated characters to a maximum of two.
    3. Translating emoticons to their textual representations.
    4. Handling slang terms.
    5. Processing text using Ekphrasis.
    6. Replacing numerical values with the word "number".
    7. Converting exclamation and question marks to words.
    8. Removing remaining punctuation.
    9. Tokenizing and lemmatizing the text.
    
    Args:
        tweet (str): The original tweet text.
    
    Returns:
        str: The cleaned and processed tweet.
    """
    # Convert all characters to lowercase
    tweet = tweet.lower()
    
    # Limit repeated characters to two instances
    tweet = handle_repeating_char(tweet)
    
    # Convert emoticons to their corresponding text
    tweet = handle_emoticons(tweet)
    
    # Replace slang terms with their standard equivalents
    tweet = handle_slang(tweet)
    
    # Apply Ekphrasis preprocessing
    tweet = clean_processor(tweet)
    
    # Replace slang terms again to ensure thorough coverage
    tweet = handle_slang(tweet)
    
    # Substitute any numeric values with the word "number"
    tweet = replace_numbers(tweet)
    
    # Convert exclamation marks to the word "exclamation"
    tweet = replace_exclamation(tweet)
    
    # Convert question marks to the word "question"
    tweet = replace_question(tweet)
    
    # Remove any remaining punctuation marks
    tweet = remove_punct(tweet)
    
    # Tokenize and lemmatize the text
    tweet = word_tokenize(tweet)
    tweet = lemmatizer(tweet)
    
    # Reconstruct the tweet from tokens and remove any extra whitespace
    tweet = ' '.join(tweet) 
    tweet = ' '.join(tweet.split())  
    
    return tweet 


def remove_punct(text):
    """
    Eliminates all punctuation from the provided text.
    
    Args:
        text (str): The text from which punctuation will be removed.
    
    Returns:
        str: Text without any punctuation.
    """
    text = ''.join([char for char in text if char not in string.punctuation])
    return text


def replace_numbers(text):
    """
    Replaces all numerical digits in the text with the word 'number'.
    
    Args:
        text (str): The text containing numbers to be replaced.
    
    Returns:
        str: Text with numbers replaced by the word "number".
    """
    return re.sub('[0-9]+', ' number ', text)


def replace_exclamation(text):
    """
    Replaces one or more exclamation marks with the word 'exclamation'.
    
    Args:
        text (str): The text containing exclamation marks.
    
    Returns:
        str: Text with exclamation marks replaced by "exclamation".
    """
    return re.sub(r'(\!)+', ' exclamation ', text)


def replace_question(text):
    """
    Replaces one or more question marks with the word 'question'.
    
    Args:
        text (str): The text containing question marks.
    
    Returns:
        str: Text with question marks replaced by "question".
    """
    return re.sub(r'(\?)+', ' question ', text)


def unpack_hashtag(text):
    """
    Expands hashtags by segmenting the concatenated words.
    
    For example, '#GoodMorning' becomes 'good morning'.
    
    Args:
        text (str): The text containing hashtags to be unpacked.
    
    Returns:
        str: Text with unpacked hashtags.
    """
    words = text.split()
    return ' '.join([Segmenter("twitter").segment(word=w[1:]) if (w[0] == '#') else w for w in words])


def remove_stop_words(text):
    """
    Removes all English stop words from the provided text.
    
    Args:
        text (str): The text from which stop words will be removed.
    
    Returns:
        str: Text without stop words.
    """
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    filtered_sentence = ' '.join([w for w in text.split() if w not in stop_words])
    return filtered_sentence


def handle_repeating_char(text):
    """
    Reduces consecutive repeated characters in the text to a maximum of two.
    
    For example, "goood" becomes "good".
    
    Args:
        text (str): The text with potential character repetitions.
    
    Returns:
        str: Text with repeated characters limited to two instances.
    """
    return re.sub(r'(.)\1+', r'\1\1', text)


def lemmatizer(data):
    """
    Lemmatizes each word in the provided list of tokens.
    
    Args:
        data (list): A list of word tokens.
    
    Returns:
        list: A list of lemmatized words.
    """
    lm = WordNetLemmatizer()
    return [lm.lemmatize(w) for w in data]  


def handle_emoticons(text):
    """
    Transforms laugh expressions and emojis into standardized textual representations.
    
    - Converts variations of "haha", "hehe", and "hihi" to '<laugh>'.
    - Replaces emojis with their corresponding meanings based on a predefined dictionary.
    
    Args:
        text (str): The text containing emoticons and emojis.
    
    Returns:
        str: Text with emoticons and emojis replaced by standardized terms.
    """
    # Replace various laugh expressions with the tag '<laugh>'
    text = re.sub('(h+ah+a*h*)+', "<laugh>", text)
    text = re.sub('(h*eh+e*h*)+', "<laugh>", text)
    text = re.sub('(h*ih+i*h*)+', "<laugh>", text)
    
    # Replace emojis using the 'emojis' dictionary; leave unchanged if not found
    return ' '.join(emojis[w] if w in emojis else w for w in text.split())


# Initialize the Ekphrasis text processor with specified configurations
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
    # Annotations to be added to the text
    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
    # Segmenter for word segmentation within hashtags
    segmenter="twitter",
    # Spell corrector using Twitter-specific corrections
    corrector="twitter",
    unpack_hashtags=True,          # Enable segmentation of hashtags into separate words
    unpack_contractions=True,      # Expand contractions (e.g., "can't" to "can not")
    spell_correct_elong=True,      # Correct elongated words (e.g., "soooo" to "soo")
    # Tokenizer that processes text into tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # Dictionaries for replacing specific tokens, such as emoticons
    dicts=[emoticons]
)


def clean_processor(text):
    """
    Applies the Ekphrasis text processor to the input text.
    
    Args:
        text (str): The raw text to be processed.
    
    Returns:
        str: The processed text after applying Ekphrasis transformations.
    """
    text = " ".join(text_processor.pre_process_doc(text))
    return text


def handle_slang(text):
    """
    Replaces slang abbreviations with their full-form equivalents using Ekphrasis's slang dictionary.
    
    For example, "2morrow" becomes "tomorrow".
    
    Args:
        text (str): The text containing slang terms.
    
    Returns:
        str: Text with slang terms expanded to their standard forms.
    """
    return ' '.join(slangdict[w] if w in slangdict else w for w in text.split())


if __name__ == '__main__':
    # Load the slang dictionary from a pickle file
    slang = pd.read_pickle('Data_Cleaning/slangdict.pickle')

    # Define the path to the Twitter datasets directory
    path = os.path.join(os.path.expanduser('~'), 'ML_course', 'projects', 'project2', 'Data', 'twitter-datasets')
    
    # Specify the path to the test data file
    file_path = os.path.join(path, 'test_data.txt')
    
    # Initialize a list to store processed rows
    rows = []
    
    # Open and read the test data file
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split each line at the first comma to separate ID and Text
            split_line = line.strip().split(",", 1)
            if len(split_line) == 2:  # Ensure both ID and Text are present
                rows.append(split_line)

    # Create a DataFrame from the processed rows with 'ID' as the index
    df = pd.DataFrame(rows, columns=["ID", "Text"]).set_index("ID")
    

