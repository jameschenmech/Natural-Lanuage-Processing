# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:07:20 2018

@author: James
"""
import nltk
nltk.download('punkt')

path = r"C:\Users\junch\Documents\Python Data Science\NLP\Wikipedia articles"
myreader = nltk.corpus.reader.PlaintextCorpusReader(path, r".*\.txt")


files = myreader.fileids()

file_text=list()

#could not locate a way to get the corpus reader to give me the files
#in a list of list format
#create my own
for index, f in enumerate(files):
    print(f)
    file_text.append(myreader.raw(f))


print(len(file_text))


from nltk.tokenize import sent_tokenize, word_tokenize
print(word_tokenize(file_text[0]))
print(sent_tokenize(file_text[1]))