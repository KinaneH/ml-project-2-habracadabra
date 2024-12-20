# ml-project-2-habracadabra

## Introduction
This project focuses on sentiment classification of tweets, leveraging both traditional machine learning and transformer-based approaches. We first implemented a pipeline using TF-IDF features as input for three classical classification algorithms: Naive Bayes, Logistic Regression, and Support Vector Machines (SVM), with hyperparameter tuning performed through grid search and evaluation across multiple seeds. To try to further improve performance, we then tested a pre-trained Twitter-specific RoBERTa model and designed a lightweight Multi-Layer Perceptron (MLP) on top of its embeddings for binary sentiment classification.

This project was created by Sofiya Malamud (sofiya.malamud@epfl.ch), Jack Pulcrano (jack.pulcrano@epfl.ch), and Kinane Habra (kinane.habra@epfl.ch) for the course CS-433: Machine Learning at the EPFL.

## Files description
- **classifiers**  
    - **submissions_classifiers:** A directory containing CSV files with predictions for the test set, formatted for submission. Each file corresponds to a specific classifier and random seed.

    - **`classifiers.ipynb`:** This notebook implements the pipeline for training and evaluating classical machine learning classifiers using TF-IDF features. Specifically, we test three classifiers: Naive Bayes, Logistic Regression, and Support Vector Machines. The pipeline includes data preprocessing, hyperparameter tuning via GridSearchCV, evaluation on the validation set, and generation of predictions for submission.

    - **`run.ipynb`:** This notebook contains the code for our optimized solution to the project challenge. 
    - **`visuals.ipynb`:** This notebook includes the code used to generate Figure 1 featured in the report.
    - **`final_submission.csv`:** Final predictions using a linear SVM (C=1) with TfidfVectorizer (ngram_range=(1,3), min_df=1) and with the seed set to 27.
   

- **Models** 
    - **`MLPwithText.py`:** Constructs our MLP and functions for training and evaluating an MLP classifier on text data using embeddings from a pretrained language model. It includes loading text files, tokenizing inputs, extracting CLS-based embeddings, and applying a simple MLP head for classification. The pipeline also covers generating predictions.


- **Experiments** 
    - **`train_model_on_small_set.py`:** Integrates a pretrained sentiment model with an MLP to load, tokenize, embed, train, save weights, and evaluate on a validation set. 
    - **`Roberta.py`:** Uses a pretrained sentiment model on tweets (load, clean, tokenize) and reports accuracy on positive or negative sets.
    - **`Roberta_full.py`:** Similar to Roberta.py but reports both accuracy and F1 score on concatenated positive and negative datasets.
    - **`MLP_Roberta_Preds.py`:** Uses a pretrained language model to embed tweets and an MLP classifier for predictions. Converts predictions to -1/1 and saves them as CSV.
    - **`twitter_exp.py`:** Applies a pretrained RoBERTa sentiment model to raw tweets, including text cleaning, tokenization, obtaining sentiment probabilities, and printing top sentiment labels with probabilities. This allowed to test how to integrate and run inference using a Hugging Face Transformers sentiment model on custom text inputs.
    - **`Classify_using_twitter_roberta_sentiment.py`:**  Loads and cleans positive and negative tweet datasets, applies a pretrained RoBERTa sentiment model, and saves binary-labeled (-1 or 1) tweets in annotated_tweets.csv. Used for initial experiments to understand the functionning of RoBERTa.

- **data**
    - **`train_neg.txt, train_pos.txt`:** Reduced labeled training datasets containing 100,000 tweets each for the negative and positive classes.
    - **`train_neg_full.txt, train_pos_full.txt`:** Full labeled training datasets, each containing 1.25 million tweets for the negative and positive classes, respectively.
    - **`test_data.txt`:** Test set containing 10,000 tweets.
    - **`mlp_ls=0.0001_max_len=128_batch=800_cardiffnlptwitter-roberta-base-sentiment-latest_hidden_size=768_hidden_dim=128.pth`:** The saved model weights used for test set evaluation.

- **src**
   - **`data_cleaning`:**
       - **`cleaning.py`:** Applies a series of functions (e.g removing punctuation, lemmatization...) to clean the twitter datasets.
       - **`data_loader.py`:** Reads and loads the necessary data into a dataframe.
       - **`dictionaries.py`:** Contains the ekphrasis dictionary used to replace symbols with words.
   - **`helpers`:**
       - **`data_loader.py`:** Provides TextDataset and create_dataloader. TextDataset tokenizes text and converts inputs/labels to PyTorch tensors. create_dataloader builds a DataLoader for efficient batching and shuffling.
       - **`helper.py`:** Provides functions to load and preprocess labeled positive and negative tweets from text files and to apply a pre-trained transformer model to individual tweets for sentiment classification.

    - **`cfg.py`:** A configuration file that defines file paths, random seeds, and hyperparameter grids for Logistic Regression, Naive Bayes, and SVM classifiers.
    - **`utils_classifier.py`:** A utility script with functions for data loading, preprocessing, model selection, evaluation, and saving predictions, designed to streamline classifier experiments.



- **`ml-project-2-habracadabra.pdf`:** This is the PDF report of the project, summarizing our results.




## References for Packages  

- **NumPy**: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*, 357–362. [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)  
- **Pandas**: McKinney, W., & others. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51–56).  
- **Scikit-Learn**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... others. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(Oct), 2825–2830.  
- **NLTK**: Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python: Analyzing text with the Natural Language Toolkit*. O'Reilly Media, Inc.  
- **Ekphrasis**: Baziotis, C., Nikolaos, A., & Papaloukas, C. (2017). DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)* (pp. 747–754). [https://aclanthology.org/S17-2126](https://aclanthology.org/S17-2126)  
- **Torch**: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035). Curran Associates, Inc. Retrieved from [http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)  
- **Transformers**: Muralidharan, S., Sreenivas, S. T., Joshi, R., Chochowski, M., Patwary, M., Shoeybi, M., Catanzaro, B., Kautz, J., & Molchanov, P. (2024). Compact language models via pruning and knowledge distillation. *arXiv preprint arXiv:2407.14679*. [https://arxiv.org/abs/2407.14679](https://arxiv.org/abs/2407.14679)
- **Twitter RoBERTa Model**: Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Universal Evaluation Benchmark for Tweet Classification. Findings of the Association for Computational Linguistics: EMNLP 2020, 1644–1650. https://aclanthology.org/2020.findings-emnlp.148/ (Model available at https://huggingface.co/cardiffnlp/twitter-roberta-base)



