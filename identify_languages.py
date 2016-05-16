from __future__ import absolute_import, division, print_function
import pickle
import numpy as np

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
    'fi',
    'fr',
    'hu',
    'it',
    'ja',
    'kr',
    'nl',
    'no',
    'pl',
    'pt',
    'ru',
    'sk',
    'sv',
    'tr',
    'zh',
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
                            re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                    word.replace('"','').lower()) # force lower case, remove double quotes
                                for word in tweet.split() # go word by word and keep them if...
                                    if not word.startswith('@') and # they don't start with @, #, or http
                                    not word.startswith('#') and
                                    not word.startswith('http')]
                                ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                            for tweet in raw_text]


Language_Tweet_Mat = np.zeros((len(available_languages),len(clean_text)))

for lang in available_languages: #counts for each tweet the number of stop words of each language and arranges this in a matrix. Probably not very efficient...
    for tweet in clean_text:

        stop_words_in_tweet = [word for word in tweet.split() if word.decode('ascii') in set(get_stop_words(lang))]
        Language_Tweet_Mat[available_languages.index(lang),clean_text.index(tweet)] = len(stop_words_in_tweet)

pickle.dump(raw_data, open('language_tweet_mat.pkl','wb'))

for tweet in clean_text[:50]:
    print(tweet)
    print(available_languages[np.argmax(Language_Tweet_Mat[:,clean_text.index(tweet)])])
