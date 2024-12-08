# Configuration for file paths, hyperparameters, and settings

# Random seed
SEED = 27

# File paths
TRAIN_POS_PATH = "../data/train_pos.txt"
TRAIN_NEG_PATH = "../data/train_neg.txt"
TRAIN_POS_FULL_PATH = "../data/train_pos_full.txt"
TRAIN_NEG_FULL_PATH = "../data/train_neg_full.txt"

TEST_PATH = "../data/test_data.txt"
SUBMISSION_PATH = "submission.csv"

# Hyperparameters for different models
PARAM_GRID = {
    'logistic_regression': {
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'vectorizer__min_df': [1, 5, 10],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2', None]
    },
    'naive_bayes': {
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'vectorizer__min_df': [1, 5, 10],
        'classifier__alpha': [0.001, 0.01, 0.1, 1, 10]
    },

    'svm': {
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'vectorizer__min_df': [1, 5, 10],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
}

# Model options
MODELS = ['logistic_regression', 'naive_bayes', 'svm']