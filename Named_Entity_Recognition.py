# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:57:21 2018

@author: James
"""

# Import necessary modules
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import regexp_tokenize, TweetTokenizer
#import re 
#import matplotlib.pyplot as plt
#from collections import Counter
#from nltk.stem import WordNetLemmatizer
#from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
#import itertools
#from gensim.models.tfidfmodel import TfidfModel
import matplotlib.pyplot as plt


with open('articles.txt',encoding="utf8") as file:
   article = file.read()

   
# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
            
# =============================================================================
# #Charting practice
# =============================================================================
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=False)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]

# Create the pie chart
plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()




