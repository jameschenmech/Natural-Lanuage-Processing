# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:33:06 2018

@author: James
"""
import pandas as pd
import numpy as np

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer

# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


#import the fake news file which has 4 columns, for supervised learning
df = pd.read_csv('fake_or_real_news.csv')

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size=0.33, random_state=53)

# =============================================================================
# #Generate a token count matrix
# =============================================================================
# Initialize a CountVectorizer object: count_vectorizer
#Creates the matrix of word counts
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
#Using tranform only to ignore all words not found in the fit_transform above
count_test = count_vectorizer.transform(X_test)


# =============================================================================
# #Generate Term Frequency Inverse Document Frequency nrmalaiton to a sparse matrix
# =============================================================================
# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer =TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# =============================================================================
# #Create data frames from the count vertorizer and TfidfVectorizer
# =============================================================================
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Calculate the difference in columns: difference
difference = set(count_df) - set(tfidf_df)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

# =============================================================================
# # Use Naive Bayes classifer
# =============================================================================
# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

# =============================================================================
# #  Naive Bayes classificaiton on TfidVectorizer
# =============================================================================
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)


# =============================================================================
# #  Improve the Naive Bayes model by playing with different Alphas
# =============================================================================
# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()

# =============================================================================
# # Inspect the model
# =============================================================================
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
