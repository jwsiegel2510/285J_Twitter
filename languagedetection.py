from __future__ import absolute_import, division, print_function
import pickle
import numpy as np
import math
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA, TruncatedSVD, LatentDirichletAllocation
import time
import re  # regex

import codecs
import os
from langdetect import detect

raw_text = pickle.load(open('raw_text_data.pkl','rb'), errors='ignore')

# Make sure NaNs turn into strings
# (We probably don't want this in the long run)
raw_text = [str(x) for x in raw_text]
print("Number of Samples:", len(raw_text))

clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                            word.lower() for word in tweet.split() # go word by word and keep them if...
                                    if not word.startswith('@') and # they don't start with @, #, or http
                                    not word.startswith('#') and
                                    not word.startswith('http')]
                                ) # force ascii encoding, ignore weird characters just in case
                            for tweet in raw_text]

for tweet in clean_text:
    try:
        if str(detect(tweet)) == 'de':
            print(tweet)
    except:
        pass
