from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# A helper function to convert raw sentiment scores into a binary sentiment label (-1 or 1).
# The logic here assumes that if the score for "negative" class (index 0) is greater or equal
# to that of the "positive" class (index 2), return -1, else return +1.
def score_to_sentiment(scores):
    return -1 * (scores[0] >= scores[2]) + 1 * (scores[2] > scores[0])

# A simple text preprocessing function:
# - Replaces usernames starting with "@" with "@user".
# - Replaces any token starting with "http" with the token "http".
# This prepares the text for the model, handling common Twitter artifacts.
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

if __name__ == "__main__":
    # Specify the pretrained model to use.
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Load the tokenizer and configuration from the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    
    # Load the pretrained PyTorch model for sequence classification.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Example text to analyze
    text = "Covid cases are increasing fast!"
    text = preprocess(text)  # Preprocess the text (handle usernames/URLs)
    
    # Tokenize the input text and prepare it as model input tensors.
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Run the text through the model to get raw output logits.
    output = model(**encoded_input)
    
    # Extract the output logits for the single input sample.
    scores = output[0][0].detach().numpy()
    
    # Apply softmax to convert logits into probabilities.
    scores = softmax(scores)
    
    # (Optional) TF model loading code is commented out here, as we only use PyTorch above.
    # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)
    # text = "Covid cases are increasing fast!"
    # encoded_input = tokenizer(text, return_tensors='tf')
    # output = model(encoded_input)
    # scores = output[0][0].numpy()
    # scores = softmax(scores)

    # The model's config contains label mappings: config.id2label maps class indices to string labels.
    # We sort the scores in descending order to see which sentiment is most likely.
    ranking = np.argsort(scores)[::-1]  # Sort and reverse to get highest probability first.
    
    # Print out the top labels and their probabilities.
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = scores[ranking[i]]
        print(f"{i+1}) {label} {np.round(float(score), 4)}")
