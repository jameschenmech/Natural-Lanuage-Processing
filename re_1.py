# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:48:45 2017

@author: James
"""
import re
my_string = "Let's write RegEx!"
pattern = r"\w+"
print(re.findall(pattern, my_string))

my_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"
sentence_endings=r"[.,?!]"
print(re.split(sentence_endings,my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))