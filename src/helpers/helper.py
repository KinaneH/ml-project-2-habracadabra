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