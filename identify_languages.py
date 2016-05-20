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

#: A default list of languages for :func:`get_stop_words`.
available_languages = [
    'ar',
    'ca',
    'cs',
    'de',
    'el',
    'en',
    'es',
    'fr',
    'no',
    'pt',
    'ru',
    'tr',
    ]


def get_stop_words(*lang_codes): # Gets stop words for each language.
    '''Get a set of words that should generally be ignored.

    Returns a set of strings that are stop words for every
    (two-letter, lowercase) language code passed as a positional
    parameter.  If no parameters are specified, then the returned set
    includes all languages for which data is available, listed out in
    :data:`available_languages`.

    '''
    if len(lang_codes) == 0:
        lang_codes = available_languages

    this_dir = os.path.dirname(__file__)

    stop_words = set()
    for lang_code in lang_codes:
        path = os.path.join(this_dir, 'stopwords-{0}.txt'.format(lang_code))
        with codecs.open(path, 'r', 'utf-8-sig') as f:
            for line in f:
                stop_words.add(line.strip())
    return stop_words


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
                                ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                            for tweet in raw_text]


Language_Tweet_Mat = np.zeros((len(available_languages),len(clean_text)))

for (j,lang) in enumerate(available_languages): #counts for each tweet the number of stop words of each language and arranges this in a matrix. Probably not very efficient...
    lang_set = set(get_stop_words(lang))
    length = len(get_stop_words(lang))
    for (i,tweet) in enumerate(clean_text):
        stop_words_in_tweet = [word for word in tweet.split() if word.decode('ascii') in lang_set]
        Language_Tweet_Mat[j,i] = float(len(stop_words_in_tweet))/length

for (i,tweet) in enumerate(clean_text):
    q = 0;
    for (j,lang) in enumerate(available_languages):
        q = q+Language_Tweet_Mat[j,i]*Language_Tweet_Mat[j,i]
    if q > 0:
        for (j,lang) in enumerate(available_languages):
            Language_Tweet_Mat[j,i] = Language_Tweet_Mat[j,i]/math.sqrt(q)

pickle.dump(Language_Tweet_Mat, open('language_tweet_mat.pkl','wb'))

for tweet in clean_text[:50]:
    print(tweet)
    print(available_languages[np.argmax(Language_Tweet_Mat[:,clean_text.index(tweet)])])

(W, H) = pickle.load(open('../NMF/NMF_500_topics_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))

Language_Topic_Mat = np.dot(Language_Tweet_Mat,W)

Topic_Language_Mat = Language_Topic_Mat.T

pickle.dump(Topic_Language_Mat, open('topic_language_mat.pkl','wb'))

sorted = [available_languages[np.argmax(x)] for x in Topic_Language_Mat]
for (i,x) in enumerate(sorted):
    print(i)
    print(x)
