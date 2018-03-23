# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:40:58 2017

@author: James
"""

import re

#print(re.search('cd','abcde'))

match_digits_and_words = ('(\d+|\w+)')

match = re.findall(match_digits_and_words, 'He has 11 cats')

#print(match)

my_str = 'match lowercase spaces nums like 12, but no commas'

#print(re.match('[a-z0-9 ]+', my_str))

my_str = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
pattern = r"(#\d)"
print(re.search(pattern, my_str))

        